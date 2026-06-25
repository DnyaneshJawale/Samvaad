type Landmark = {
  x: number;
  y: number;
  z: number;
};

type HandClassification = {
  categoryName?: string;
  score?: number;
};

export type HandDetectionResult = {
  landmarks?: Landmark[][];
  handedness?: HandClassification[][];
};

export type DetectedHandSummary = {
  id: string;
  handedness: string;
  rawLabel: string;
  stableLabel: string;
  confidence: number;
  viewX: number;
};

export interface RecognitionOptions {
  mirrorPreview?: boolean;
  historySize?: number;
  stableThreshold?: number;
  minActiveConfidence?: number;
}

export interface RecognitionSnapshot {
  hands: DetectedHandSummary[];
  primaryHand: DetectedHandSummary | null;
  handsDetected: number;
  rawGesture: string;
  stableGesture: string;
  latestHandedness: string;
  gestureConfidence: number;
}

const DEFAULT_HISTORY_SIZE = 8;
const DEFAULT_STABLE_THRESHOLD = 4;

const FRAME_SEQUENCE_LENGTH = 30;

const AI_SERVER_URL =
  process.env.NEXT_PUBLIC_AI_SERVER_URL ?? "http://127.0.0.1:8000/predict";

const MIN_INFERENCE_GAP_MS = 280;
const PREDICTION_TTL_MS = 1800;

const LIVE_CONFIDENCE_THRESHOLD = 0.70;
const STABLE_REPEAT_THRESHOLD = 3;
const STABLE_AVG_CONFIDENCE_THRESHOLD = 0.74;
const MIN_RAW_CONFIDENCE = 0.55;

type ServerPrediction = {
  label: string;
  confidence: number;
  timestamp: number;
};

type PredictionResponse =
  | {
      success: true;
      label: string;
      confidence: number;
      class_index?: number;
      top_k?: Array<{ label: string; confidence: number; index: number }>;
    }
  | {
      success: false;
      error: string;
    };

let frameBuffer: number[][] = [];
let lastLivePrediction: ServerPrediction | null = null;
let stablePrediction: ServerPrediction | null = null;
let predictionHistory: ServerPrediction[] = [];
let inferenceInFlight = false;
let lastInferenceAt = 0;
let serverOfflineUntil = 0;

const handHistories = new Map<string, string[]>();

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function normalizeHandedness(label: string): string {
  const value = (label || "").toLowerCase();

  if (value.includes("left")) return "Left";
  if (value.includes("right")) return "Right";

  return "Hand";
}

function isFingerExtended(hand: Landmark[], tipIndex: number, pipIndex: number): boolean {
  return hand[tipIndex].y < hand[pipIndex].y;
}

function classifyFallback(hand: Landmark[]): { label: string; confidence: number } {
  const index = isFingerExtended(hand, 8, 6);
  const middle = isFingerExtended(hand, 12, 10);
  const ring = isFingerExtended(hand, 16, 14);
  const pinky = isFingerExtended(hand, 20, 18);

  const count = [index, middle, ring, pinky].filter(Boolean).length;

  if (count === 0) return { label: "Fist", confidence: 0.96 };
  if (count === 4) return { label: "Open palm", confidence: 0.95 };
  if (index && !middle && !ring && !pinky) return { label: "Point", confidence: 0.92 };
  if (index && middle && !ring && !pinky) return { label: "Two fingers", confidence: 0.91 };
  if (index && middle && ring && !pinky) return { label: "Three fingers", confidence: 0.90 };

  return { label: "Hand detected", confidence: 0.80 };
}

function deriveStablePrediction(): ServerPrediction | null {
  const now = Date.now();
  const recent = predictionHistory.filter((item) => now - item.timestamp <= PREDICTION_TTL_MS);

  if (recent.length < STABLE_REPEAT_THRESHOLD) {
    return null;
  }

  const grouped = new Map<
    string,
    { count: number; confidenceSum: number; lastTimestamp: number }
  >();

  for (const item of recent) {
    if (item.confidence < MIN_RAW_CONFIDENCE) continue;

    const current = grouped.get(item.label) ?? {
      count: 0,
      confidenceSum: 0,
      lastTimestamp: 0,
    };

    current.count += 1;
    current.confidenceSum += item.confidence;
    current.lastTimestamp = Math.max(current.lastTimestamp, item.timestamp);

    grouped.set(item.label, current);
  }

  if (grouped.size === 0) {
    return null;
  }

  let bestLabel = "";
  let bestCount = 0;
  let bestAvgConfidence = 0;
  let bestTimestamp = 0;

  for (const [label, stats] of grouped.entries()) {
    const avgConfidence = stats.confidenceSum / stats.count;

    const isBetter =
      stats.count > bestCount ||
      (stats.count === bestCount && avgConfidence > bestAvgConfidence) ||
      (stats.count === bestCount &&
        avgConfidence === bestAvgConfidence &&
        stats.lastTimestamp > bestTimestamp);

    if (isBetter) {
      bestLabel = label;
      bestCount = stats.count;
      bestAvgConfidence = avgConfidence;
      bestTimestamp = stats.lastTimestamp;
    }
  }

  if (
    bestCount >= STABLE_REPEAT_THRESHOLD &&
    bestAvgConfidence >= STABLE_AVG_CONFIDENCE_THRESHOLD
  ) {
    return {
      label: bestLabel,
      confidence: bestAvgConfidence,
      timestamp: bestTimestamp,
    };
  }

  return null;
}

function recordPrediction(label: string, confidence: number): void {
  const prediction: ServerPrediction = {
    label,
    confidence,
    timestamp: Date.now(),
  };

  lastLivePrediction = prediction;
  predictionHistory.push(prediction);

  if (predictionHistory.length > 10) {
    predictionHistory = predictionHistory.slice(-10);
  }

  stablePrediction = deriveStablePrediction();
}

function emptyHandVector(): number[] {
  return Array.from({ length: 63 }, () => 0);
}

function isZeroVec(vec: number[]): boolean {
  return vec.every((v) => v === 0);
}

function landmarksToVector(handLandmarks: Landmark[]): number[] {
  const vec: number[] = [];

  for (const lm of handLandmarks) {
    vec.push(lm.x, lm.y, lm.z);
  }

  if (vec.length !== 63) return emptyHandVector();

  return vec;
}

function buildFrameFeature(result: HandDetectionResult): number[] {
const leftVec: number[] = emptyHandVector();
const rightVec: number[] = emptyHandVector();

  const landmarksList = result.landmarks ?? [];
  const handednessList = result.handedness ?? [];

  if (landmarksList.length === 0) {
    return [...leftVec, ...rightVec];
  }

  for (let i = 0; i < landmarksList.length; i++) {
    const handLandmarks = landmarksList[i];
    const side = normalizeHandedness(
      handednessList[i]?.[0]?.categoryName ?? "Hand"
    ).toLowerCase();

    const vec = landmarksToVector(handLandmarks);

    if (side === "left") {
      for (let j = 0; j < 63; j++) leftVec[j] = vec[j];
    } else if (side === "right") {
      for (let j = 0; j < 63; j++) rightVec[j] = vec[j];
    } else {
      if (isZeroVec(leftVec)) {
        for (let j = 0; j < 63; j++) leftVec[j] = vec[j];
      } else if (isZeroVec(rightVec)) {
        for (let j = 0; j < 63; j++) rightVec[j] = vec[j];
      }
    }
  }

  return [...leftVec, ...rightVec];
}

async function dispatchPrediction(sequence: number[][]): Promise<void> {
  if (inferenceInFlight) return;

  const now = Date.now();
  if (now < serverOfflineUntil) return;
  if (now - lastInferenceAt < MIN_INFERENCE_GAP_MS) return;

  inferenceInFlight = true;
  lastInferenceAt = now;

  try {
    const response = await fetch(AI_SERVER_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sequence,
      }),
    });

    if (!response.ok) {
      serverOfflineUntil = Date.now() + 3000;
      return;
    }

    const data = (await response.json()) as PredictionResponse;

    if (!data.success) {
      serverOfflineUntil = Date.now() + 3000;
      return;
    }

    recordPrediction(String(data.label), clamp(Number(data.confidence ?? 0), 0, 1));
  } catch {
    serverOfflineUntil = Date.now() + 3000;
  } finally {
    inferenceInFlight = false;
  }
}

function getLivePrediction(): ServerPrediction | null {
  if (!lastLivePrediction) return null;

  const age = Date.now() - lastLivePrediction.timestamp;
  if (age > PREDICTION_TTL_MS) return null;

  return lastLivePrediction;
}

function getStablePrediction(): ServerPrediction | null {
  if (!stablePrediction) return null;

  const age = Date.now() - stablePrediction.timestamp;
  if (age > PREDICTION_TTL_MS) return null;

  return stablePrediction;
}

export function classifyHand(hand: Landmark[]): { label: string; confidence: number } {
  const live = getLivePrediction();
  const stable = getStablePrediction();

  if (stable && stable.confidence >= LIVE_CONFIDENCE_THRESHOLD) {
    return {
      label: stable.label,
      confidence: stable.confidence,
    };
  }

  if (live && live.confidence >= LIVE_CONFIDENCE_THRESHOLD) {
    return {
      label: live.label,
      confidence: live.confidence,
    };
  }

  return classifyFallback(hand);
}

function clearSequenceState() {
  frameBuffer = [];
  lastLivePrediction = null;
  stablePrediction = null;
  predictionHistory = [];
}

export function analyzeHands(
  result: HandDetectionResult,
  histories: Record<string, string[]>,
  options: RecognitionOptions = {}
): RecognitionSnapshot {
  const mirrorPreview = options.mirrorPreview ?? true;
  const historySize = options.historySize ?? DEFAULT_HISTORY_SIZE;
  const stableThreshold = options.stableThreshold ?? DEFAULT_STABLE_THRESHOLD;

  const landmarksList = result.landmarks ?? [];
  const handednessList = result.handedness ?? [];

  if (landmarksList.length === 0) {
    clearSequenceState();

    return {
      hands: [],
      primaryHand: null,
      handsDetected: 0,
      rawGesture: "No hand detected",
      stableGesture: "No hand detected",
      latestHandedness: "—",
      gestureConfidence: 0,
    };
  }

  const hands: DetectedHandSummary[] = landmarksList.map((hand, index) => {
    const rawHandedness = normalizeHandedness(
      handednessList[index]?.[0]?.categoryName ?? `Hand ${index + 1}`
    );

    const fallback = classifyFallback(hand);

    const centerX =
      hand.reduce((sum, point) => sum + point.x, 0) / Math.max(hand.length, 1);
    const viewX = mirrorPreview ? 1 - centerX : centerX;

    const historyKey = rawHandedness.toLowerCase() || `hand-${index}`;
    const existingHistory = histories[historyKey] ?? handHistories.get(historyKey) ?? [];
    const updatedHistory = [...existingHistory, fallback.label].slice(-historySize);

    histories[historyKey] = updatedHistory;
    handHistories.set(historyKey, updatedHistory);

    const stableLabel = deriveStablePrediction()?.label ?? deriveStableLabel(updatedHistory, stableThreshold);

    return {
      id: `${historyKey}-${index}`,
      handedness: rawHandedness,
      rawLabel: fallback.label,
      stableLabel,
      confidence: fallback.confidence,
      viewX,
    };
  });

  const orderedByScreen = [...hands].sort((a, b) => a.viewX - b.viewX);
  const heuristicPrimary = [...orderedByScreen].sort(
    (a, b) => b.confidence - a.confidence || a.viewX - b.viewX
  )[0];

  const feature = buildFrameFeature(result);
  if (feature.some((value) => value !== 0)) {
    frameBuffer.push(feature);
    if (frameBuffer.length > FRAME_SEQUENCE_LENGTH) {
      frameBuffer.shift();
    }
  }

  if (frameBuffer.length === FRAME_SEQUENCE_LENGTH) {
    void dispatchPrediction([...frameBuffer]);
  }

  const live = getLivePrediction();
  const stable = getStablePrediction();

  const rawGesture = live?.label ?? heuristicPrimary?.rawLabel ?? "—";
  const stableGesture = stable?.label ?? "Searching...";
  const gestureConfidence = stable?.confidence ?? live?.confidence ?? heuristicPrimary?.confidence ?? 0;

  const primaryHand: DetectedHandSummary | null = stable
    ? {
        id: "model-primary",
        handedness: heuristicPrimary?.handedness ?? "Hand",
        rawLabel: rawGesture,
        stableLabel: stableGesture,
        confidence: gestureConfidence,
        viewX: heuristicPrimary?.viewX ?? 0.5,
      }
    : live
      ? {
          id: "model-primary",
          handedness: heuristicPrimary?.handedness ?? "Hand",
          rawLabel: rawGesture,
          stableLabel: "Searching...",
          confidence: gestureConfidence,
          viewX: heuristicPrimary?.viewX ?? 0.5,
        }
      : heuristicPrimary
        ? {
            ...heuristicPrimary,
            stableLabel: "Searching...",
          }
        : null;

  return {
    hands: orderedByScreen,
    primaryHand,
    handsDetected: orderedByScreen.length,
    rawGesture,
    stableGesture,
    latestHandedness: primaryHand?.handedness ?? "—",
    gestureConfidence,
  };
}

function deriveStableLabel(history: string[], stableThreshold: number): string {
  const recent = history.filter(Boolean).slice(-DEFAULT_HISTORY_SIZE);

  if (recent.length === 0) return "Searching...";

  const counts = new Map<string, number>();

  for (const label of recent) {
    if (label === "—" || label === "Searching..." || label === "No hand detected") {
      continue;
    }
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  if (counts.size === 0) {
    return recent[recent.length - 1] ?? "—";
  }

  let bestLabel = "—";
  let bestCount = 0;

  for (const [label, count] of counts.entries()) {
    if (count > bestCount) {
      bestCount = count;
      bestLabel = label;
    }
  }

  return bestCount >= stableThreshold ? bestLabel : "Searching...";
}