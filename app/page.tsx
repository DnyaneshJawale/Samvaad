"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { SentenceEngine } from "../lib/sentenceEngine";
import type { SentenceEngineResult } from "../lib/sentenceEngine";
import {
  analyzeHands,
  classifyHand,
  normalizeHandedness,
  type DetectedHandSummary,
  type HandDetectionResult,
} from "../lib/islRecognizer";

// ─── Types ────────────────────────────────────────────────────────────────────

type CameraStatus = "idle" | "requesting" | "live" | "blocked" | "error";
type ModelStatus = "idle" | "loading" | "ready" | "error";
type RecognitionState =
  | "idle"
  | "loading"
  | "listening"
  | "signing"
  | "speaking"
  | "error";
type SpeechMode = "word" | "sentence";

type PersistedSession = {
  speechEnabled: boolean;
  autoCommitEnabled: boolean;
  transcript: string;
  speechMode: SpeechMode;
  lifetimeWordCount: number;
  onboardingDismissed: boolean;
};

// ─── Constants ────────────────────────────────────────────────────────────────
// Palette is deliberate, not the generic dark/neon default:
//   ink       — background
//   parchment — text, the "page" your words are written on
//   saffron   — voice: speech, the primary action
//   banyan    — calm detection: a hand is seen and understood
//   indigo    — the watching lens: idle, attentive, not yet sure
//   clay      — trouble

const STORAGE_KEY = "samvaad:session:v6";
const LOGO_SRC = "/assets/logo.png";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const MODEL_URL = "/models/hand_landmarker.task";
const MIRROR_PREVIEW = true;
const HISTORY_SIZE = 8;
const STABLE_THRESHOLD = 4;
const MIN_ACTIVE_CONFIDENCE = 0.82;
const FINALIZE_TICK_MS = 300;
const FLASH_MS = 900;
const THREAD_MAX_WORDS = 10;

const SUPPRESSED_LOG_PARTS = [
  "Created TensorFlow Lite XNNPACK delegate for CPU",
  "OpenGL error checking is disabled",
  "Feedback manager requires a model with a single signature inference",
  "Using NORM_RECT without IMAGE_DIMENSIONS is only supported for the square ROI",
];

// ─── Pure helpers ─────────────────────────────────────────────────────────────

function buildCandidateText(
  primaryHand: DetectedHandSummary | null,
  stableGesture?: string
): string {
  const text =
    (typeof primaryHand?.stableLabel === "string"
      ? primaryHand.stableLabel.trim()
      : "") ||
    (typeof stableGesture === "string" ? stableGesture.trim() : "");

  if (
    !text ||
    text === "—" ||
    text === "Searching..." ||
    text === "Searching." ||
    text === "No hand detected"
  ) {
    return "";
  }
  return text;
}

function clearCanvas(canvas: HTMLCanvasElement | null) {
  const ctx = canvas?.getContext("2d");
  if (!ctx || !canvas) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function stripTrailingPunctuation(word: string): string {
  return word.replace(/[.!?,;:]+$/, "");
}

// Pulls out the most recently finalized sentence from a running transcript,
// so "full sentence" speech mode can speak just what's new.
function extractLastSentence(text: string): string {
  const trimmed = text.trim();
  if (!trimmed) return "";
  const parts = trimmed.split(/(?<=[.!?])\s+/).filter(Boolean);
  return parts[parts.length - 1] ?? trimmed;
}

function drawHandOverlay(
  ctx: CanvasRenderingContext2D,
  hand: { x: number; y: number; z: number }[],
  label: string,
  handedness: string,
  w: number,
  h: number
) {
  const CONNECTIONS: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
  ];

  const px = (x: number) => (MIRROR_PREVIEW ? w - x * w : x * w);
  const py = (y: number) => y * h;
  const clamp = (v: number, lo: number, hi: number) =>
    Math.min(hi, Math.max(lo, v));

  ctx.save();
  ctx.lineWidth = 2.5;
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(150,196,178,0.95)"; // banyan, light
  ctx.fillStyle = "rgba(110,168,142,0.95)"; // banyan, deeper

  const pts = hand.map((p) => ({ x: px(p.x), y: py(p.y) }));

  for (const [s, e] of CONNECTIONS) {
    const a = pts[s], b = pts[e];
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const p of pts) {
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  const txt = `${handedness} · ${label}`;
  ctx.font = "500 12px system-ui, sans-serif";
  const padX = 8,
    bh = 24,
    bw = ctx.measureText(txt).width + padX * 2;
  const bx = clamp(Math.min(...pts.map((p) => p.x)), 6, w - bw - 6);
  const by = clamp(
    Math.min(...pts.map((p) => p.y)) - bh - 6,
    6,
    h - bh - 6
  );
  ctx.fillStyle = "rgba(11,14,20,0.88)"; // ink
  ctx.strokeStyle = "rgba(232,163,61,0.55)"; // saffron, soft
  ctx.lineWidth = 1;
  ctx.fillRect(bx, by, bw, bh);
  ctx.strokeRect(bx, by, bw, bh);
  ctx.fillStyle = "#F4D9A0"; // saffron-tinted parchment
  ctx.fillText(txt, bx + padX, by + 17);
  ctx.restore();
}

// ─── State label / colour helpers ─────────────────────────────────────────────

function stateDotClass(state: RecognitionState): string {
  switch (state) {
    case "signing":   return "bg-[#5C8C76]";
    case "speaking":  return "bg-[#E8A33D] animate-pulse motion-reduce:animate-none";
    case "listening": return "bg-[#5B6CB8] animate-pulse motion-reduce:animate-none";
    case "loading":   return "bg-[#F3EFE6]/70 animate-pulse motion-reduce:animate-none";
    case "error":     return "bg-[#C1503D]";
    default:          return "bg-[#F3EFE6]/20";
  }
}

function statePingClass(state: RecognitionState): string {
  switch (state) {
    case "speaking":  return "bg-[#E8A33D]";
    case "listening": return "bg-[#5B6CB8]";
    case "loading":   return "bg-[#F3EFE6]";
    default:          return "";
  }
}

function stateLabel(state: RecognitionState): string {
  switch (state) {
    case "signing":   return "Sign detected";
    case "speaking":  return "Speaking";
    case "listening": return "Listening for signs";
    case "loading":   return "Loading model";
    case "error":     return "Error";
    default:          return "Camera off";
  }
}

function stateTextClass(state: RecognitionState): string {
  switch (state) {
    case "signing":  return "text-[#9FCDB7]";
    case "speaking": return "text-[#F4C268]";
    case "error":    return "text-[#E08066]";
    default:         return "text-[#F3EFE6]/45";
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Home() {
  // ── Refs (shared with RAF loop / stable callbacks) ──────────────────────────
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const rafRef = useRef<number | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const handLandmarkerPromiseRef = useRef<Promise<HandLandmarker> | null>(null);
  const cameraActiveRef = useRef(false);
  const lastVideoTimeRef = useRef(-1);
  const speechEnabledRef = useRef(false);
  const speechModeRef = useRef<SpeechMode>("word");
  const speechQueueRef = useRef<string[]>([]);
  const speechBusyRef = useRef(false);
  const speechLastTextRef = useRef("");
  const autoCommitRef = useRef(true);
  const engineRef = useRef<SentenceEngine | null>(new SentenceEngine());
  const handHistoriesRef = useRef<Record<string, string[]>>({});
  const flashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const removedFlashTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const transcriptRef = useRef("");
  // Shadow-refs for keyboard handler — keep stable listener, avoid 60fps re-registration
  const primaryHandRef = useRef<DetectedHandSummary | null>(null);
  const stableGestureRef = useRef("—");
  const cameraStatusRef = useRef<CameraStatus>("idle");
  const helpOpenRef = useRef(false);
  const settingsOpenRef = useRef(false);
  const onboardingOpenRef = useRef(false);

  // ── State ───────────────────────────────────────────────────────────────────
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>("idle");
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [errorMessage, setErrorMessage] = useState("");
  const [rawGesture, setRawGesture] = useState("—");
  const [stableGesture, setStableGesture] = useState("—");
  const [gestureConfidence, setGestureConfidence] = useState(0);
  const [handsDetected, setHandsDetected] = useState(0);
  const [primaryHand, setPrimaryHand] = useState<DetectedHandSummary | null>(null);
  const [speechEnabled, setSpeechEnabled] = useState(false);
  const [speechMode, setSpeechMode] = useState<SpeechMode>("word");
  const [autoCommitEnabled, setAutoCommitEnabled] = useState(true);
  const [transcript, setTranscript] = useState("");
  const [logoBroken, setLogoBroken] = useState(false);
  const [copyFeedback, setCopyFeedback] = useState<"idle" | "copied" | "error">("idle");
  const [isHydrated, setIsHydrated] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [committedFlash, setCommittedFlash] = useState("");
  const [removedFlash, setRemovedFlash] = useState("");
  const [lifetimeWordCount, setLifetimeWordCount] = useState(0);
  const [onboardingDismissed, setOnboardingDismissed] = useState(false);
  const [onboardingOpen, setOnboardingOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);

  // ── Ref sync effects ────────────────────────────────────────────────────────
  useEffect(() => { speechEnabledRef.current = speechEnabled; }, [speechEnabled]);
  useEffect(() => { speechModeRef.current = speechMode; }, [speechMode]);
  useEffect(() => { autoCommitRef.current = autoCommitEnabled; }, [autoCommitEnabled]);
  useEffect(() => { primaryHandRef.current = primaryHand; }, [primaryHand]);
  useEffect(() => { stableGestureRef.current = stableGesture; }, [stableGesture]);
  useEffect(() => { cameraStatusRef.current = cameraStatus; }, [cameraStatus]);
  useEffect(() => { transcriptRef.current = transcript; }, [transcript]);
  useEffect(() => { helpOpenRef.current = helpOpen; }, [helpOpen]);
  useEffect(() => { settingsOpenRef.current = settingsOpen; }, [settingsOpen]);
  useEffect(() => { onboardingOpenRef.current = onboardingOpen; }, [onboardingOpen]);

  // ── Onboarding lifecycle ─────────────────────────────────────────────────────
  // First-ever visit: show the introduction once. Starting the camera for the
  // first time also counts as "got it" — we don't nag a returning signer.
  useEffect(() => {
    if (isHydrated && !onboardingDismissed) setOnboardingOpen(true);
  }, [isHydrated, onboardingDismissed]);

  useEffect(() => {
    if (cameraStatus === "live") {
      setOnboardingDismissed(true);
      setOnboardingOpen(false);
    }
  }, [cameraStatus]);

  const closeOnboarding = useCallback(() => {
    setOnboardingOpen(false);
    setOnboardingDismissed(true);
  }, []);

  const replayOnboarding = useCallback(() => {
    setOnboardingOpen(true);
  }, []);

  // ── Derived recognition state ───────────────────────────────────────────────
  // NOTE: "blocked" (permission denied) and modelStatus "error" are both error states
  const recognitionState: RecognitionState = (() => {
    if (cameraStatus === "error" || cameraStatus === "blocked" || modelStatus === "error") {
      return "error";
    }
    if (cameraStatus !== "live") {
      return modelStatus === "loading" ? "loading" : "idle";
    }
    if (modelStatus === "loading") return "loading";
    if (isSpeaking) return "speaking";
    if (handsDetected > 0) return "signing";
    return "listening";
  })();

  // ── Derived display values ──────────────────────────────────────────────────
  const liveCandidate = buildCandidateText(primaryHand, stableGesture);
  const wordCount = transcript
    ? transcript.trim().split(/\s+/).filter(Boolean).length
    : 0;
  const confidencePct = Math.round(
    (primaryHand?.confidence ?? gestureConfidence) * 100
  );
  const allCommittedWords = transcript
    ? transcript.trim().split(/\s+/).filter(Boolean)
    : [];
  const threadWords = allCommittedWords.slice(-THREAD_MAX_WORDS);
  const threadTruncated = allCommittedWords.length > THREAD_MAX_WORDS;
  const ping = statePingClass(recognitionState);

  // ── Log suppression ─────────────────────────────────────────────────────────
  useEffect(() => {
    const orig = {
      error: console.error,
      warn: console.warn,
      info: console.info,
    };
    const suppress = (args: unknown[]) => {
      const msg =
        typeof args[0] === "string"
          ? args[0]
          : args[0] instanceof Error
          ? args[0].message
          : String(args[0] ?? "");
      return SUPPRESSED_LOG_PARTS.some((p) => msg.includes(p));
    };
    console.error = (...a: unknown[]) => {
      if (!suppress(a)) orig.error(...(a as Parameters<typeof console.error>));
    };
    console.warn = (...a: unknown[]) => {
      if (!suppress(a)) orig.warn(...(a as Parameters<typeof console.warn>));
    };
    console.info = (...a: unknown[]) => {
      if (!suppress(a)) orig.info(...(a as Parameters<typeof console.info>));
    };
    return () => {
      console.error = orig.error;
      console.warn = orig.warn;
      console.info = orig.info;
    };
  }, []);

  // ── Hydration ───────────────────────────────────────────────────────────────
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const p = JSON.parse(raw) as Partial<PersistedSession>;
        if (typeof p.speechEnabled === "boolean") setSpeechEnabled(p.speechEnabled);
        if (typeof p.autoCommitEnabled === "boolean")
          setAutoCommitEnabled(p.autoCommitEnabled);
        if (typeof p.transcript === "string") {
          setTranscript(p.transcript);
          engineRef.current?.loadTranscript(p.transcript);
        }
        if (p.speechMode === "word" || p.speechMode === "sentence") {
          setSpeechMode(p.speechMode);
        }
        if (
          typeof p.lifetimeWordCount === "number" &&
          Number.isFinite(p.lifetimeWordCount)
        ) {
          setLifetimeWordCount(Math.max(0, Math.floor(p.lifetimeWordCount)));
        }
        if (typeof p.onboardingDismissed === "boolean") {
          setOnboardingDismissed(p.onboardingDismissed);
        }
      }
    } catch {
      // ignore
    } finally {
      setIsHydrated(true);
    }
  }, []);

  // ── Persistence ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!isHydrated) return;
    try {
      const toSave: PersistedSession = {
        speechEnabled,
        autoCommitEnabled,
        transcript,
        speechMode,
        lifetimeWordCount,
        onboardingDismissed,
      };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave));
    } catch {
      // ignore
    }
  }, [
    isHydrated,
    speechEnabled,
    autoCommitEnabled,
    transcript,
    speechMode,
    lifetimeWordCount,
    onboardingDismissed,
  ]);

  // ── Speech ──────────────────────────────────────────────────────────────────

  const cancelSpeech = useCallback(() => {
    if (typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
    speechQueueRef.current = [];
    speechBusyRef.current = false;
    speechLastTextRef.current = "";
    setIsSpeaking(false);
  }, []);

  const drainSpeechQueue = useCallback(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    if (speechBusyRef.current) return;
    const next = speechQueueRef.current.shift();
    if (!next) {
      setIsSpeaking(false);
      return;
    }
    speechBusyRef.current = true;
    setIsSpeaking(true);
    const utt = new SpeechSynthesisUtterance(next);
    utt.rate = 0.95;
    utt.pitch = 1.0;
    utt.lang = "en-IN";
    utt.onend = () => {
      speechBusyRef.current = false;
      if (speechQueueRef.current.length === 0) setIsSpeaking(false);
      drainSpeechQueue();
    };
    utt.onerror = () => {
      speechBusyRef.current = false;
      setIsSpeaking(false);
      drainSpeechQueue();
    };
    window.speechSynthesis.speak(utt);
    speechLastTextRef.current = next;
  }, []);

  const speakText = useCallback(
    (text: string, replaceQueue = false) => {
      if (typeof window === "undefined" || !window.speechSynthesis) return;
      const cleaned = text.trim();
      if (!cleaned || cleaned === "—") return;
      if (replaceQueue) {
        speechQueueRef.current = [cleaned];
        speechBusyRef.current = false;
        window.speechSynthesis.cancel();
      } else {
        // Deduplicate: skip if this exact text is already the last thing in queue
        if (
          cleaned === speechLastTextRef.current &&
          speechQueueRef.current.length === 0
        ) {
          return;
        }
        speechQueueRef.current.push(cleaned);
      }
      drainSpeechQueue();
    },
    [drainSpeechQueue]
  );

  // ── Engine result handler ───────────────────────────────────────────────────
  // Per-word commits flash + (optionally) speak immediately. Sentence-level
  // finalization is handled by the finalize tick below, so a single word is
  // never spoken twice no matter which speech mode is active.

  const applyEngineResult = useCallback(
    (result: SentenceEngineResult) => {
      setTranscript(result.committedSentence);

      if (result.shouldSpeak && result.speechText && !result.isFinalized) {
        setCommittedFlash(result.speechText);
        if (flashTimerRef.current) clearTimeout(flashTimerRef.current);
        flashTimerRef.current = setTimeout(() => setCommittedFlash(""), FLASH_MS);

        setLifetimeWordCount((c) => c + 1);

        if (speechEnabledRef.current && speechModeRef.current === "word") {
          speakText(result.speechText);
        }
      }
    },
    [speakText]
  );

  const processLiveText = useCallback(
    (text: string, forceCommit = false) => {
      const engine = engineRef.current;
      if (!engine) return;
      const result = engine.processPrediction(text, {
        now: Date.now(),
        autoCommitEnabled: autoCommitRef.current,
        forceCommit,
      });
      applyEngineResult(result);
    },
    [applyEngineResult]
  );

  // ── Finalization tick ───────────────────────────────────────────────────────
  // Handles the "user paused signing" → auto-period case when RAF is inactive.
  // In "full sentence" speech mode, this is also where the just-completed
  // sentence is read aloud — once, as a whole, instead of word by word.
  useEffect(() => {
    const timer = window.setInterval(() => {
      const engine = engineRef.current;
      if (!engine) return;
      const result = engine.processPrediction("", {
        now: Date.now(),
        autoCommitEnabled: autoCommitRef.current,
      });
      if (result.isFinalized && result.committedSentence) {
        setTranscript(result.committedSentence);
        if (speechEnabledRef.current && speechModeRef.current === "sentence") {
          const sentence = extractLastSentence(result.committedSentence);
          if (sentence) speakText(sentence, true);
        }
      }
    }, FINALIZE_TICK_MS);
    return () => window.clearInterval(timer);
  }, [speakText]);

  // ── Transcript actions ──────────────────────────────────────────────────────

  const clearTranscript = useCallback(() => {
    engineRef.current?.clear();
    setTranscript("");
    setCommittedFlash("");
    setRemovedFlash("");
    cancelSpeech();
    setCopyFeedback("idle");
  }, [cancelSpeech]);

  const backspaceTranscript = useCallback(() => {
    const prevWords = transcriptRef.current.trim().split(/\s+/).filter(Boolean);
    const next = engineRef.current?.backspace() ?? "";
    const nextWords = next.trim().split(/\s+/).filter(Boolean);
    const removed =
      prevWords.length > nextWords.length ? prevWords[prevWords.length - 1] : "";

    setTranscript(next);
    setCommittedFlash("");
    setCopyFeedback("idle");

    if (removed) {
      setRemovedFlash(removed);
      if (removedFlashTimerRef.current) clearTimeout(removedFlashTimerRef.current);
      removedFlashTimerRef.current = setTimeout(() => setRemovedFlash(""), FLASH_MS);
    }
  }, []);

  const copyTranscript = async () => {
    const text = transcript.trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopyFeedback("copied");
      setTimeout(() => setCopyFeedback("idle"), 1500);
    } catch {
      setCopyFeedback("error");
      setTimeout(() => setCopyFeedback("idle"), 1500);
    }
  };

  const downloadTranscript = () => {
    const text = transcript.trim();
    if (!text) return;
    const blob = new Blob([`${text}\n`], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "samvaad-transcript.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  const resetSession = useCallback(() => {
    setSpeechEnabled(false);
    setAutoCommitEnabled(true);
    setSpeechMode("word");
    engineRef.current?.clear();
    setTranscript("");
    setCommittedFlash("");
    setRemovedFlash("");
    cancelSpeech();
    setCopyFeedback("idle");
  }, [cancelSpeech]);

  // ── Camera + model ──────────────────────────────────────────────────────────

  const stopDetectionLoop = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  };

  // Stopping the camera never clears the transcript — a signer may want to
  // hold the screen up to someone after they've finished signing. Only an
  // explicit Clear or Reset touches the transcript.
  const cleanupCameraResources = useCallback(
    (resetUi: boolean) => {
      cameraActiveRef.current = false;
      stopDetectionLoop();
      lastVideoTimeRef.current = -1;

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }

      const video = videoRef.current;
      if (video) {
        video.pause();
        video.srcObject = null;
        video.removeAttribute("src");
        video.load();
      }

      clearCanvas(canvasRef.current);
      cancelSpeech();

      if (resetUi) {
        setCameraStatus("idle");
        setErrorMessage("");
        setRawGesture("—");
        setStableGesture("—");
        setGestureConfidence(0);
        setHandsDetected(0);
        setPrimaryHand(null);
        handHistoriesRef.current = {};
      }
    },
    [cancelSpeech]
  );

  const loadHandLandmarker = async (): Promise<HandLandmarker> => {
    if (handLandmarkerRef.current) return handLandmarkerRef.current;
    if (handLandmarkerPromiseRef.current) return handLandmarkerPromiseRef.current;

    setModelStatus("loading");

    const promise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      return HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    })();

    handLandmarkerPromiseRef.current = promise;

    try {
      const lm = await promise;
      handLandmarkerRef.current = lm;
      setModelStatus("ready");
      return lm;
    } catch (err) {
      setModelStatus("error");
      setErrorMessage(
        err instanceof Error ? err.message : "Failed to load hand model."
      );
      throw err;
    } finally {
      handLandmarkerPromiseRef.current = null;
    }
  };

  const updateRecognition = (result: HandDetectionResult) => {
    const snapshot = analyzeHands(result, handHistoriesRef.current, {
      mirrorPreview: MIRROR_PREVIEW,
      historySize: HISTORY_SIZE,
      stableThreshold: STABLE_THRESHOLD,
      minActiveConfidence: MIN_ACTIVE_CONFIDENCE,
    });
    setHandsDetected(snapshot.handsDetected);
    setPrimaryHand(snapshot.primaryHand);
    setRawGesture(snapshot.rawGesture);
    setStableGesture(snapshot.stableGesture);
    setGestureConfidence(snapshot.gestureConfidence);
    return snapshot;
  };

  const drawResults = (result: HandDetectionResult) => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!video || !canvas || !ctx) return;
    const w = video.videoWidth,
      h = video.videoHeight;
    if (!w || !h) return;
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    const lms = result.landmarks ?? [];
    const hdns = result.handedness ?? [];
    const sorted = lms
      .map((hand, i) => {
        const side = normalizeHandedness(
          hdns[i]?.[0]?.categoryName ?? `Hand ${i + 1}`
        );
        const gesture = classifyHand(hand).label;
        const cx =
          hand.reduce((s, p) => s + p.x, 0) / Math.max(hand.length, 1);
        return { hand, side, gesture, viewX: MIRROR_PREVIEW ? 1 - cx : cx };
      })
      .sort((a, b) => a.viewX - b.viewX);

    for (const item of sorted) {
      drawHandOverlay(ctx, item.hand, item.gesture, item.side, w, h);
    }
  };

  const startDetectionLoop = useCallback(() => {
    stopDetectionLoop();
    const loop = () => {
      const video = videoRef.current;
      const lm = handLandmarkerRef.current;
      if (!video || !lm || !cameraActiveRef.current) {
        rafRef.current = requestAnimationFrame(loop);
        return;
      }
      if (
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        video.videoHeight > 0 &&
        video.currentTime !== lastVideoTimeRef.current
      ) {
        lastVideoTimeRef.current = video.currentTime;
        try {
          const result = lm.detectForVideo(
            video,
            performance.now()
          ) as HandDetectionResult;
          const snapshot = updateRecognition(result);
          drawResults(result);
          const candidate = buildCandidateText(
            snapshot.primaryHand,
            snapshot.stableGesture
          );
          processLiveText(candidate, false);
        } catch (err) {
          setErrorMessage(
            err instanceof Error ? err.message : "Hand detection failed."
          );
          setCameraStatus("error");
          cleanupCameraResources(false);
          return;
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  }, [processLiveText, cleanupCameraResources]);

  const startCamera = useCallback(async () => {
    try {
      setErrorMessage("");
      setCameraStatus("requesting");
      cleanupCameraResources(false);

      if (!navigator.mediaDevices?.getUserMedia) {
        setCameraStatus("error");
        setErrorMessage("This browser does not support camera access.");
        return;
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;

      const video = videoRef.current;
      if (!video) {
        setCameraStatus("error");
        setErrorMessage("Video element is not ready.");
        return;
      }

      video.srcObject = stream;
      video.muted = true;
      video.playsInline = true;
      video.autoplay = true;

      await new Promise<void>((resolve) => {
        const onLoaded = () => {
          video.removeEventListener("loadedmetadata", onLoaded);
          resolve();
        };
        video.addEventListener("loadedmetadata", onLoaded);
      });

      await video.play();
      cameraActiveRef.current = true;
      setCameraStatus("live");
      await loadHandLandmarker();
      if (cameraActiveRef.current) startDetectionLoop();
    } catch (err) {
      setCameraStatus("blocked");
      setErrorMessage(
        err instanceof Error ? err.message : "Unable to access the camera."
      );
    }
  }, [cleanupCameraResources, startDetectionLoop]);

  const stopCamera = useCallback(
    () => cleanupCameraResources(true),
    [cleanupCameraResources]
  );

  // ── Keyboard shortcuts ──────────────────────────────────────────────────────
  // Uses shadow-refs so this effect never needs to re-register. While any
  // overlay (help, settings, onboarding) is open, only Escape is honoured —
  // everything else is swallowed so reading a shortcut list doesn't trigger one.

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      const overlayOpen =
        helpOpenRef.current || settingsOpenRef.current || onboardingOpenRef.current;
      if (overlayOpen) {
        if (e.key === "Escape") {
          e.preventDefault();
          setHelpOpen(false);
          setSettingsOpen(false);
          closeOnboarding();
        }
        return;
      }

      switch (e.key) {
        case "Backspace":
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            backspaceTranscript();
          }
          break;
        case "Escape":
          e.preventDefault();
          clearTranscript();
          break;
        case " ":
          e.preventDefault();
          {
            const liveText = buildCandidateText(
              primaryHandRef.current,
              stableGestureRef.current
            );
            if (liveText) processLiveText(liveText, true);
          }
          break;
        case "v":
        case "V":
          if (!e.ctrlKey && !e.metaKey) setSpeechEnabled((p) => !p);
          break;
        case "a":
        case "A":
          if (!e.ctrlKey && !e.metaKey) setAutoCommitEnabled((p) => !p);
          break;
        case "s":
        case "S":
          if (!e.ctrlKey && !e.metaKey) {
            const cs = cameraStatusRef.current;
            if (cs !== "live" && cs !== "requesting") void startCamera();
          }
          break;
        case "?":
          e.preventDefault();
          setHelpOpen(true);
          break;
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backspaceTranscript, clearTranscript, processLiveText, startCamera, closeOnboarding]);

  // ── Cleanup on unmount ──────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      cleanupCameraResources(false);
      if (flashTimerRef.current) clearTimeout(flashTimerRef.current);
      if (removedFlashTimerRef.current) clearTimeout(removedFlashTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ── Hydration skeleton ──────────────────────────────────────────────────────

  if (!isHydrated) {
    return (
      <main className="h-screen w-full overflow-hidden bg-[#0B0E14] text-[#F3EFE6] flex items-center justify-center">
        <div className="text-center space-y-3">
          <div className="mx-auto h-10 w-10 rounded-xl border border-[#F3EFE6]/10 bg-[#F3EFE6]/[0.04] animate-pulse motion-reduce:animate-none" />
          <p className="text-sm text-[#F3EFE6]/40 font-serif">Samvaad is waking up…</p>
        </div>
      </main>
    );
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <main
      className="h-screen max-xl:h-auto overflow-hidden max-xl:overflow-y-auto bg-[#0B0E14] text-[#F3EFE6] flex flex-col gap-3 p-3 selection:bg-[#E8A33D]/30"
    >
      {/* ── Onboarding ─────────────────────────────────────────────────────── */}
      {onboardingOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 backdrop-blur-sm p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="onboarding-title"
        >
          <div className="w-full max-w-md rounded-3xl border border-[#F3EFE6]/10 bg-[#10141C] p-6 sm:p-8 shadow-2xl">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-[#E8A33D]/85 mb-2">
              Welcome
            </p>
            <h1
              id="onboarding-title"
              className="font-serif text-2xl sm:text-[28px] text-[#F3EFE6] leading-snug mb-3"
            >
              Samvaad means dialogue.
            </h1>
            <p className="text-sm text-[#F3EFE6]/65 leading-relaxed mb-5">
              You sign. Samvaad finds the words and speaks them aloud — so
              the person in front of you can simply listen.
            </p>
            <ol className="space-y-3 mb-5">
              {[
                ["Start your camera", "Give it a clear view of your hands."],
                ["Sign naturally", "Hold a sign for a moment and it becomes a word."],
                ["Turn on voice", "Samvaad says each word aloud as it lands."],
              ].map(([title, detail]) => (
                <li key={title} className="flex gap-3 text-sm">
                  <span className="shrink-0 mt-0.5 h-5 w-5 rounded-full border border-[#E8A33D]/40 text-[#E8A33D] text-[11px] flex items-center justify-center font-medium">
                    {title === "Start your camera" ? 1 : title === "Sign naturally" ? 2 : 3}
                  </span>
                  <span>
                    <span className="font-medium text-[#F3EFE6]">{title}.</span>{" "}
                    <span className="text-[#F3EFE6]/55">{detail}</span>
                  </span>
                </li>
              ))}
            </ol>
            <p className="text-[11px] text-[#F3EFE6]/40 leading-relaxed mb-6 border-t border-[#F3EFE6]/8 pt-4">
              Only the shape of your hands is sent off for recognition —
              never your video.
            </p>
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => {
                  closeOnboarding();
                  void startCamera();
                }}
                className="flex-1 rounded-full bg-[#E8A33D] px-4 py-2.5 text-sm font-semibold text-[#0B0E14] hover:bg-[#F4C268] transition-colors"
              >
                Start signing
              </button>
              <button
                type="button"
                onClick={closeOnboarding}
                className="text-sm text-[#F3EFE6]/45 hover:text-[#F3EFE6]/80 transition-colors px-2"
              >
                Skip
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Help ───────────────────────────────────────────────────────────── */}
      {helpOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 backdrop-blur-sm p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="help-title"
        >
          <div className="w-full max-w-sm rounded-3xl border border-[#F3EFE6]/10 bg-[#10141C] p-6 shadow-2xl max-h-[85vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h2 id="help-title" className="font-serif text-xl text-[#F3EFE6]">
                Shortcuts &amp; how it works
              </h2>
              <button
                type="button"
                onClick={() => setHelpOpen(false)}
                aria-label="Close help"
                className="text-[#F3EFE6]/40 hover:text-[#F3EFE6] text-lg leading-none px-1"
              >
                ×
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2 mb-5">
              {[
                ["Space", "Commit sign"],
                ["Backspace", "Remove word"],
                ["Esc", "Clear all"],
                ["V", "Voice toggle"],
                ["S", "Start camera"],
                ["A", "Auto toggle"],
                ["?", "This help"],
              ].map(([key, action]) => (
                <div
                  key={key}
                  className="flex items-center gap-2 rounded-lg border border-[#F3EFE6]/8 bg-black/20 px-2.5 py-1.5 text-[11px]"
                >
                  <kbd className="rounded bg-[#F3EFE6]/10 px-1.5 py-0.5 font-mono text-[#F3EFE6]/80 shrink-0">
                    {key}
                  </kbd>
                  <span className="text-[#F3EFE6]/55 truncate">{action}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-[#F3EFE6]/55 leading-relaxed mb-3">
              Samvaad watches your hands for a steady moment before it
              commits a word — that small pause is what keeps the transcript
              accurate. A longer pause closes the sentence with a period,
              automatically.
            </p>
            <p className="text-xs text-[#F3EFE6]/35 leading-relaxed">
              Only the shape of your hands is sent for recognition — your
              camera feed itself never leaves this device.
            </p>
          </div>
        </div>
      )}

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="shrink-0 rounded-2xl border border-[#F3EFE6]/8 bg-[#F3EFE6]/[0.035] px-4 py-2.5 flex items-center justify-between gap-3 flex-wrap">
        {/* Brand */}
        <div className="flex items-center gap-3 min-w-0">
          {!logoBroken ? (
            <img
              src={LOGO_SRC}
              alt="Samvaad logo"
              className="h-8 w-auto object-contain shrink-0"
              onError={() => setLogoBroken(true)}
            />
          ) : (
            <div className="h-8 w-8 rounded-lg border border-[#E8A33D]/25 bg-[#E8A33D]/10 flex items-center justify-center text-xs font-bold text-[#E8A33D] shrink-0 font-serif">
              S
            </div>
          )}
          <div className="min-w-0 hidden sm:block">
            <p className="font-serif text-sm font-semibold leading-none text-[#F3EFE6]">
              Samvaad
              <span className="sr-only"> — Indian Sign Language communication aid</span>
            </p>
            <p className="text-[10px] leading-tight text-[#F3EFE6]/35 mt-0.5 tracking-wide">
              Hands that speak
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {/* State pill */}
          <span
            role="status"
            aria-live="polite"
            className={`inline-flex items-center gap-1.5 rounded-full border border-[#F3EFE6]/8 bg-black/25 px-2.5 py-1 text-[11px] font-medium ${stateTextClass(
              recognitionState
            )}`}
          >
            <span className="relative inline-flex h-1.5 w-1.5 shrink-0">
              {ping && (
                <span
                  className={`absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping motion-reduce:animate-none ${ping}`}
                />
              )}
              <span
                className={`relative inline-flex h-1.5 w-1.5 rounded-full ${stateDotClass(
                  recognitionState
                )}`}
              />
            </span>
            {stateLabel(recognitionState)}
          </span>

          {/* Model pill — only meaningful when not idle */}
          {(modelStatus === "loading" || modelStatus === "ready" || modelStatus === "error") && (
            <span className="hidden md:inline-flex items-center gap-1 rounded-full border border-[#F3EFE6]/8 bg-black/25 px-2.5 py-1 text-[11px] text-[#F3EFE6]/40">
              Model{" "}
              <span
                className={
                  modelStatus === "ready"
                    ? "text-[#9FCDB7]"
                    : modelStatus === "error"
                    ? "text-[#E08066]"
                    : "text-[#F3EFE6]/70"
                }
              >
                {modelStatus}
              </span>
            </span>
          )}

          {/* Voice toggle */}
          <button
            type="button"
            onClick={() => setSpeechEnabled((p) => !p)}
            aria-pressed={speechEnabled}
            title="Toggle voice output (V)"
            className={`rounded-full border px-3 py-1 text-[11px] font-medium transition-colors ${
              speechEnabled
                ? "border-[#E8A33D]/35 bg-[#E8A33D]/15 text-[#F4C268]"
                : "border-[#F3EFE6]/8 bg-transparent text-[#F3EFE6]/40 hover:text-[#F3EFE6]/70"
            }`}
          >
            Voice {speechEnabled ? "ON" : "OFF"}
          </button>

          {/* Auto-commit toggle */}
          <button
            type="button"
            onClick={() => setAutoCommitEnabled((p) => !p)}
            aria-pressed={autoCommitEnabled}
            title="Toggle auto word-build (A)"
            className={`rounded-full border px-3 py-1 text-[11px] font-medium transition-colors ${
              autoCommitEnabled
                ? "border-[#5C8C76]/30 bg-[#5C8C76]/12 text-[#9FCDB7]"
                : "border-[#F3EFE6]/8 bg-transparent text-[#F3EFE6]/40 hover:text-[#F3EFE6]/70"
            }`}
          >
            Auto {autoCommitEnabled ? "ON" : "OFF"}
          </button>

          {/* Settings */}
          <div className="relative">
            <button
              type="button"
              onClick={() => setSettingsOpen((p) => !p)}
              aria-pressed={settingsOpen}
              aria-haspopup="dialog"
              title="Settings"
              className={`rounded-full border px-3 py-1 text-[11px] font-medium transition-colors ${
                settingsOpen
                  ? "border-[#F3EFE6]/25 bg-[#F3EFE6]/10 text-[#F3EFE6]"
                  : "border-[#F3EFE6]/8 bg-transparent text-[#F3EFE6]/40 hover:text-[#F3EFE6]/70"
              }`}
            >
              Settings
            </button>

            {settingsOpen && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={() => setSettingsOpen(false)}
                  aria-hidden="true"
                />
                <div
                  role="dialog"
                  aria-label="Settings"
                  className="absolute right-0 top-full z-50 mt-2 w-72 max-w-[90vw] rounded-2xl border border-[#F3EFE6]/10 bg-[#10141C] p-4 shadow-2xl"
                >
                  <p className="text-[10px] font-semibold uppercase tracking-widest text-[#F3EFE6]/35 mb-2">
                    Speech style
                  </p>
                  <div className="flex gap-1.5 mb-4 rounded-full bg-black/30 p-1">
                    {(["word", "sentence"] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => setSpeechMode(mode)}
                        aria-pressed={speechMode === mode}
                        className={`flex-1 rounded-full px-2.5 py-1.5 text-[11px] font-medium transition-colors ${
                          speechMode === mode
                            ? "bg-[#E8A33D] text-[#0B0E14]"
                            : "text-[#F3EFE6]/45 hover:text-[#F3EFE6]/75"
                        }`}
                      >
                        {mode === "word" ? "Each word" : "Full sentence"}
                      </button>
                    ))}
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setSettingsOpen(false);
                      replayOnboarding();
                    }}
                    className="w-full text-left text-xs text-[#F3EFE6]/55 hover:text-[#F3EFE6] transition-colors mb-4"
                  >
                    Replay the introduction →
                  </button>
                  <div className="flex items-center justify-between rounded-xl border border-[#F3EFE6]/8 bg-black/20 px-3 py-2 mb-3">
                    <span className="text-[11px] text-[#F3EFE6]/45">
                      Words spoken, ever
                    </span>
                    <span className="text-sm font-semibold font-serif text-[#F3EFE6]">
                      {lifetimeWordCount.toLocaleString()}
                    </span>
                  </div>
                  <p className="text-[10px] text-[#F3EFE6]/30 leading-relaxed">
                    Camera frames never leave your device — only hand
                    positions do.
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Help */}
          <button
            type="button"
            onClick={() => setHelpOpen(true)}
            title="Help (?)"
            className="rounded-full border border-[#F3EFE6]/8 bg-transparent px-3 py-1 text-[11px] font-medium text-[#F3EFE6]/40 hover:text-[#F3EFE6]/70 transition-colors"
          >
            Help
          </button>
        </div>
      </header>

      {/* ── Content grid ───────────────────────────────────────────────────── */}
      <div className="flex-1 min-h-0 grid gap-3 xl:grid-cols-[1fr_370px]">

        {/* ── Transcript panel — HERO ─────────────────────────────────────── */}
        <section
          className="flex flex-col min-h-0 rounded-2xl border border-[#F3EFE6]/8 bg-[#F3EFE6]/[0.035] p-4 sm:p-5"
          aria-label="Transcript"
          aria-describedby="transcript-hint"
        >
          {/* Panel header */}
          <div className="shrink-0 flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-[#F3EFE6]/35">
                Transcript
              </span>
              {wordCount > 0 && (
                <span className="rounded-full bg-black/25 border border-[#F3EFE6]/6 px-2 py-0.5 text-[10px] text-[#F3EFE6]/40">
                  {wordCount} {wordCount === 1 ? "word" : "words"}
                </span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                onClick={backspaceTranscript}
                disabled={!transcript}
                title="Remove last word (Backspace)"
                aria-label="Remove last word"
                className="rounded-lg border border-[#F3EFE6]/8 bg-transparent px-2.5 py-1 text-[11px] text-[#F3EFE6]/40 hover:text-[#F3EFE6]/75 hover:bg-[#F3EFE6]/6 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
              >
                ← Back
              </button>
              <button
                type="button"
                onClick={clearTranscript}
                disabled={!transcript}
                title="Clear all (Escape)"
                aria-label="Clear transcript"
                className="rounded-lg border border-[#F3EFE6]/8 bg-transparent px-2.5 py-1 text-[11px] text-[#F3EFE6]/40 hover:text-[#F3EFE6]/75 hover:bg-[#F3EFE6]/6 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
              >
                Clear
              </button>
            </div>
          </div>

          {/* Live recognition indicator */}
          <div
            className="shrink-0 mb-3 rounded-xl border border-[#F3EFE6]/6 bg-black/20 px-4 py-2.5 flex items-center gap-3"
            style={{ minHeight: "44px" }}
          >
            <span className={`h-2 w-2 rounded-full shrink-0 ${stateDotClass(recognitionState)}`} />
            <div className="flex-1 min-w-0">
              {liveCandidate ? (
                <span className="text-sm text-[#F3EFE6]/70">
                  Signing:{" "}
                  <span className="font-semibold text-[#9FCDB7]">{liveCandidate}</span>
                  {confidencePct > 0 && (
                    <span className="ml-2 text-[11px] text-[#F3EFE6]/30">
                      {confidencePct}%
                    </span>
                  )}
                </span>
              ) : (
                <span className="text-sm text-[#F3EFE6]/35">
                  {stateLabel(recognitionState)}
                </span>
              )}
            </div>
            {committedFlash && (
              <span
                aria-live="assertive"
                className="shrink-0 text-xs font-semibold text-[#9FCDB7] bg-[#5C8C76]/12 border border-[#5C8C76]/25 rounded-full px-2 py-0.5"
              >
                ✓ {committedFlash}
              </span>
            )}
            {!committedFlash && removedFlash && (
              <span
                aria-live="polite"
                className="shrink-0 text-xs font-medium text-[#F3EFE6]/45 bg-[#F3EFE6]/6 border border-[#F3EFE6]/12 rounded-full px-2 py-0.5"
              >
                − {removedFlash}
              </span>
            )}
          </div>

          {/* Thread — recently committed words, with the live candidate glowing at the tip */}
          {(threadWords.length > 0 || liveCandidate) && (
            <div className="shrink-0 mb-3">
              <p className="text-[9px] font-semibold uppercase tracking-widest text-[#F3EFE6]/25 mb-1.5 px-1">
                Recently signed
              </p>
              <div className="relative flex items-center gap-2.5 overflow-x-auto pb-1 pt-1 px-1 -mx-1">
                <div
                  aria-hidden="true"
                  className="absolute left-2 right-2 top-1/2 h-px -translate-y-1/2 bg-gradient-to-r from-transparent via-[#F3EFE6]/15 to-transparent pointer-events-none"
                />
                {threadTruncated && (
                  <span className="relative z-10 shrink-0 text-[#F3EFE6]/25 text-xs px-1">
                    …
                  </span>
                )}
                {threadWords.map((word, i) => (
                  <button
                    key={`${i}-${word}`}
                    type="button"
                    onClick={() => speakText(stripTrailingPunctuation(word), false)}
                    title="Hear this word again"
                    className="relative z-10 shrink-0 rounded-full border border-[#F3EFE6]/10 bg-[#0B0E14] px-3 py-1 text-xs text-[#F3EFE6]/65 hover:border-[#E8A33D]/40 hover:text-[#F3EFE6] transition-colors"
                  >
                    {word}
                  </button>
                ))}
                {liveCandidate && (
                  <span
                    aria-hidden="true"
                    className="relative z-10 shrink-0 rounded-full border border-[#E8A33D]/50 bg-[#E8A33D]/10 px-3 py-1 text-xs text-[#F4C268] animate-pulse motion-reduce:animate-none"
                  >
                    {liveCandidate}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* ── Transcript text — HERO ── */}
          <div
            className="flex-1 min-h-0 overflow-y-auto rounded-xl border border-[#F3EFE6]/6 bg-black/30 p-5 sm:p-6"
            role="region"
            aria-label="Transcript text"
          >
            {transcript ? (
              <p
                className="font-serif text-2xl sm:text-3xl font-light leading-relaxed text-[#F3EFE6] tracking-wide"
                aria-live="polite"
                aria-atomic="false"
              >
                {transcript}
              </p>
            ) : (
              <div
                id="transcript-hint"
                className="h-full flex flex-col items-center justify-center text-center gap-3"
              >
                <p className="font-serif text-lg text-[#F3EFE6]/55">
                  {cameraStatus === "live" ? "Show a sign to begin." : "Nothing said yet."}
                </p>
                <p className="text-xs text-[#F3EFE6]/35 max-w-xs leading-relaxed">
                  {cameraStatus === "live" ? (
                    <>
                      Hold a sign steady and it lands here. Press{" "}
                      <kbd className="bg-[#F3EFE6]/10 border border-[#F3EFE6]/8 rounded px-1 py-0.5 text-[#F3EFE6]/55 font-mono text-[10px]">
                        Space
                      </kbd>{" "}
                      to commit one by hand.
                    </>
                  ) : (
                    <>
                      Press{" "}
                      <kbd className="bg-[#F3EFE6]/10 border border-[#F3EFE6]/8 rounded px-1 py-0.5 font-mono text-[10px]">
                        S
                      </kbd>{" "}
                      or use Start Camera below to begin.
                    </>
                  )}
                </p>
                <button
                  type="button"
                  onClick={replayOnboarding}
                  className="text-[11px] text-[#E8A33D]/75 hover:text-[#E8A33D] underline underline-offset-2 mt-1"
                >
                  How this works
                </button>
              </div>
            )}
          </div>

          {/* Action buttons */}
          <div className="shrink-0 mt-3 flex flex-wrap gap-2" role="toolbar" aria-label="Transcript actions">
            <button
              type="button"
              onClick={() => {
                const t = buildCandidateText(
                  primaryHandRef.current,
                  stableGestureRef.current
                );
                if (t) processLiveText(t, true);
              }}
              disabled={!liveCandidate}
              title="Commit current sign now (Space)"
              aria-label="Commit current sign"
              className="rounded-full border border-[#5C8C76]/30 bg-[#5C8C76]/12 px-3 py-1.5 text-xs font-medium text-[#9FCDB7] hover:bg-[#5C8C76]/20 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
            >
              Commit Sign
            </button>

            <button
              type="button"
              onClick={() => speakText(transcript, true)}
              disabled={!transcript}
              aria-label="Read transcript aloud"
              className="rounded-full border border-[#F3EFE6]/8 bg-transparent px-3 py-1.5 text-xs font-medium text-[#F3EFE6]/65 hover:bg-[#F3EFE6]/6 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
            >
              Speak All
            </button>

            <button
              type="button"
              onClick={copyTranscript}
              disabled={!transcript}
              aria-label="Copy transcript to clipboard"
              className="rounded-full border border-[#F3EFE6]/8 bg-transparent px-3 py-1.5 text-xs font-medium text-[#F3EFE6]/65 hover:bg-[#F3EFE6]/6 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
            >
              {copyFeedback === "copied"
                ? "Copied!"
                : copyFeedback === "error"
                ? "Failed"
                : "Copy"}
            </button>

            <button
              type="button"
              onClick={downloadTranscript}
              disabled={!transcript}
              aria-label="Download transcript as text file"
              className="rounded-full border border-[#F3EFE6]/8 bg-transparent px-3 py-1.5 text-xs font-medium text-[#F3EFE6]/65 hover:bg-[#F3EFE6]/6 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
            >
              Save
            </button>

            <button
              type="button"
              onClick={resetSession}
              aria-label="Reset session"
              className="rounded-full border border-[#F3EFE6]/8 bg-transparent px-3 py-1.5 text-xs font-medium text-[#F3EFE6]/35 hover:text-[#F3EFE6]/70 hover:bg-[#F3EFE6]/6 transition-colors ml-auto"
            >
              Reset
            </button>
          </div>
        </section>

        {/* ── Right column ─────────────────────────────────────────────────── */}
        <aside
          className="flex flex-col min-h-0 gap-3 xl:overflow-y-auto"
          aria-label="Camera and recognition"
        >
          {/* Camera feed */}
          <div className="shrink-0 rounded-2xl border border-[#F3EFE6]/8 bg-[#F3EFE6]/[0.035] p-3">
            <div
              className="relative rounded-xl overflow-hidden bg-black"
              style={{ aspectRatio: "16/9" }}
            >
              <video
                ref={videoRef}
                className={`absolute inset-0 h-full w-full object-cover z-10 transition-opacity duration-300 ${
                  cameraStatus === "live" ? "opacity-100" : "opacity-0"
                }`}
                style={{ transform: "scaleX(-1)" }}
                autoPlay
                muted
                playsInline
                aria-hidden="true"
              />
              <canvas
                ref={canvasRef}
                className={`absolute inset-0 h-full w-full z-20 pointer-events-none transition-opacity duration-300 ${
                  cameraStatus === "live" ? "opacity-100" : "opacity-0"
                }`}
                aria-hidden="true"
              />

              {/* Hand placement guide — fades out the moment a hand is seen */}
              {cameraStatus === "live" && (
                <div
                  aria-hidden="true"
                  className={`absolute inset-0 z-30 flex flex-col items-center justify-center gap-2 pointer-events-none transition-opacity duration-500 ${
                    handsDetected === 0 ? "opacity-100" : "opacity-0"
                  }`}
                >
                  <div className="h-24 w-24 sm:h-28 sm:w-28 rounded-2xl border-2 border-dashed border-[#F3EFE6]/25" />
                  <p className="text-[11px] text-[#F3EFE6]/55 bg-black/45 rounded-full px-2.5 py-1">
                    Bring a hand into view
                  </p>
                </div>
              )}

              {/* Placeholder when not live */}
              {cameraStatus !== "live" && (
                <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 p-4 text-center">
                  {cameraStatus === "requesting" || modelStatus === "loading" ? (
                    <>
                      <div className="h-5 w-5 border-2 border-[#F3EFE6]/15 border-t-[#E8A33D] rounded-full animate-spin motion-reduce:animate-none" />
                      <p className="text-xs text-[#F3EFE6]/35">
                        {cameraStatus === "requesting"
                          ? "Requesting camera access..."
                          : "Loading AI model..."}
                      </p>
                    </>
                  ) : (
                    <>
                      <svg
                        className="h-6 w-6 text-[#F3EFE6]/20"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                        strokeWidth={1.5}
                        aria-hidden="true"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
                        />
                      </svg>
                      <p className="text-xs text-[#F3EFE6]/30">
                        {cameraStatus === "blocked"
                          ? "Camera blocked — allow access and retry"
                          : "Camera inactive"}
                      </p>
                    </>
                  )}
                </div>
              )}

              {/* Live badge */}
              {cameraStatus === "live" && (
                <div className="absolute left-2 top-2 z-30 rounded-full border border-[#5C8C76]/20 bg-black/50 px-2 py-0.5 text-[10px] text-[#9FCDB7] backdrop-blur-sm">
                  ● Live
                </div>
              )}

              {/* Hand count badge */}
              {handsDetected > 0 && cameraStatus === "live" && (
                <div className="absolute right-2 top-2 z-30 rounded-full border border-[#F3EFE6]/10 bg-black/50 px-2 py-0.5 text-[10px] text-[#F3EFE6]/55 backdrop-blur-sm">
                  {handsDetected} hand{handsDetected > 1 ? "s" : ""}
                </div>
              )}
            </div>

            {/* Camera controls */}
            <div className="mt-2.5 flex gap-2">
              <button
                type="button"
                onClick={startCamera}
                disabled={cameraStatus === "requesting" || modelStatus === "loading"}
                title="Start camera (S)"
                className="flex-1 rounded-full border border-[#E8A33D]/30 bg-[#E8A33D]/12 px-3 py-1.5 text-xs font-medium text-[#F4C268] hover:bg-[#E8A33D]/20 transition-colors disabled:cursor-not-allowed disabled:opacity-40"
              >
                {cameraStatus === "live" ? "Restart" : "Start Camera"}
              </button>
              <button
                type="button"
                onClick={stopCamera}
                disabled={cameraStatus === "idle" || cameraStatus === "requesting"}
                className="rounded-full border border-[#F3EFE6]/8 bg-transparent px-3 py-1.5 text-xs font-medium text-[#F3EFE6]/45 hover:text-[#F3EFE6]/75 hover:bg-[#F3EFE6]/6 transition-colors disabled:opacity-25 disabled:cursor-not-allowed"
              >
                Stop
              </button>
            </div>
          </div>

          {/* Recognition details */}
          <div className="flex-1 min-h-0 rounded-2xl border border-[#F3EFE6]/8 bg-[#F3EFE6]/[0.035] p-4 flex flex-col gap-3 overflow-y-auto">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-[#F3EFE6]/30 shrink-0">
              Recognition
            </p>

            {/* Current sign — prominent display */}
            <div className="shrink-0 rounded-xl border border-[#F3EFE6]/6 bg-black/25 px-4 py-3">
              <p className="text-[10px] text-[#F3EFE6]/30 mb-1">Current sign</p>
              <p className="font-serif text-2xl font-semibold text-[#BFE0D0] leading-tight truncate">
                {(primaryHand?.stableLabel ?? stableGesture) || "—"}
              </p>

              {confidencePct > 0 && (
                <div className="mt-2">
                  <div className="flex items-center justify-between text-[10px] text-[#F3EFE6]/30 mb-1">
                    <span>Confidence</span>
                    <span>{confidencePct}%</span>
                  </div>
                  <div className="h-1 w-full rounded-full bg-black/40">
                    <div
                      className={`h-1 rounded-full transition-all duration-200 ${
                        confidencePct >= 80
                          ? "bg-[#5C8C76]"
                          : confidencePct >= 60
                          ? "bg-[#E8A33D]"
                          : "bg-[#F3EFE6]/30"
                      }`}
                      style={{ width: `${confidencePct}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            {/* Stat rows */}
            <div className="shrink-0 space-y-1.5 text-[11px]">
              {[
                { label: "Best guess", value: primaryHand?.rawLabel ?? rawGesture },
                { label: "Hands seen", value: String(handsDetected) },
                { label: "Hand", value: primaryHand?.handedness ?? "—" },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="flex items-center justify-between rounded-lg border border-[#F3EFE6]/6 bg-black/20 px-3 py-1.5"
                >
                  <span className="text-[#F3EFE6]/35">{label}</span>
                  <span className="text-[#F3EFE6]/75 font-medium truncate max-w-[60%] text-right">
                    {value}
                  </span>
                </div>
              ))}
            </div>

            {/* Error display */}
            {errorMessage && (
              <div
                role="alert"
                className="shrink-0 rounded-xl border border-[#C1503D]/25 bg-[#C1503D]/10 px-3 py-2.5 text-xs text-[#E08066] leading-relaxed"
              >
                {errorMessage}
              </div>
            )}

            {/* Keyboard reference — compact */}
            <div className="shrink-0 pt-3 border-t border-[#F3EFE6]/6">
              <div className="flex items-center justify-between mb-2">
                <p className="text-[10px] text-[#F3EFE6]/30">Keyboard shortcuts</p>
                <button
                  type="button"
                  onClick={() => setHelpOpen(true)}
                  className="text-[10px] text-[#E8A33D]/70 hover:text-[#E8A33D]"
                >
                  More
                </button>
              </div>
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px] text-[#F3EFE6]/35">
                {[
                  ["Space", "Commit sign"],
                  ["Backspace", "Remove word"],
                  ["Esc", "Clear all"],
                  ["V", "Voice toggle"],
                  ["S", "Start camera"],
                  ["A", "Auto toggle"],
                ].map(([key, action]) => (
                  <div key={key} className="flex items-center gap-1.5">
                    <kbd className="bg-black/30 border border-[#F3EFE6]/8 rounded px-1.5 py-0.5 text-[#F3EFE6]/45 font-mono shrink-0">
                      {key}
                    </kbd>
                    <span className="truncate">{action}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </main>
  );
}