# SAMVAAD — Project Audit

Audit date: 2026-06-17
Scope: full repository at `C:\Backup\Samvaad\samvaad` (working tree, uncommitted changes included). Read-only analysis — no files were modified to produce this report.

---

## 1. Current Architecture

SAMVAAD is a two-process system, not a single deployable app:

1. **Next.js 16.2.3 / React 19 frontend** (`app/`, `lib/`) — a single client-side page (`app/page.tsx`) that owns the camera, MediaPipe hand landmark detection, on-canvas hand drawing, transcript building, and speech synthesis. There is exactly one route (`/`); no API routes, no server actions, no middleware.
2. **Standalone Python FastAPI inference server** (`ai_server.py`) — loads a Keras LSTM model (`trained_model/samvaad_isl_model.keras`) and exposes a single `POST /predict` endpoint. It is not started by Next.js, not proxied through it, has no Dockerfile/process manager, and is reached only via a hardcoded `http://127.0.0.1:8000/predict` default.

These two processes are connected by one `fetch()` call in `lib/islRecognizer.ts`. There is no shared build, no monorepo tooling, no shared types between the Python and TypeScript sides (the 126-feature/30-frame contract is duplicated by hand in both languages).

Supporting Python tooling (`tools/`) forms an offline, manually-run pipeline (dataset prep → training → tfjs export) that is disconnected from both runtime processes except via the files it drops into `trained_model/` and (intended, but not actually populated) `public/models/isl/`.

There is no database, no auth, no persistence layer beyond the browser's `localStorage`. The "backend" is purely a model server, not an application backend.

A notable artifact: `public.zip` (a snapshot from 2026-04-22) shows an originally-planned `src/components/`, `src/hooks/`, `src/lib/`, `src/types/` structure that was abandoned — the project collapsed to a flat `app/` + `lib/` layout with one 1358-line page component.

---

## 2. Frontend Architecture

- **Framework**: Next.js App Router, `"use client"` on the only meaningful page. `app/layout.tsx` is a minimal HTML shell (title "Samvaad", description "Indian Sign Language to Text and Speech"). `app/error.tsx`, `app/global-error.tsx`, `app/loading.tsx` are boilerplate Next.js convention files with matching dark-theme styling but no real recovery logic beyond `reset()` / `window.location.reload()`.
- **Styling**: Tailwind CSS v4 via `@tailwindcss/postcss`, imported once in `app/globals.css` (`@import "tailwindcss"`). All styling in `page.tsx` is inline utility classes; no design tokens, no component library, no shared style module.
- **Single component, 1358 lines**: `app/page.tsx` contains:
  - A second, parallel implementation of `SentenceEngine` (lines 144–347) duplicated almost verbatim from `lib/sentenceEngine.ts` (see §10 and §12 — this is dead-code duplication, not reuse).
  - All camera lifecycle management (`startCamera`/`stopCamera`/`cleanupCameraResources`).
  - MediaPipe `HandLandmarker` loading (`loadHandLandmarker`).
  - The `requestAnimationFrame` detection loop (`startDetectionLoop`).
  - Hand-skeleton canvas drawing (`drawHand`, `drawResults`).
  - Speech queue management (`speakText`, `drainSpeechQueue`).
  - Transcript actions (copy/download/backspace/clear/reset).
  - All UI rendering (header, live camera panel, recognition panel, transcript panel, status panel).
- **No component decomposition**: there are zero child components. Everything — video feed, skeleton overlay, recognition readout, transcript editor, status badges — is one JSX tree inside `Home()`.
- **Console patching**: a `useEffect` globally monkey-patches `console.error/warn/info` for the component's lifetime to suppress specific MediaPipe/TFLite log strings (`SUPPRESSED_LOG_PARTS`). This is a global side effect with no scoping protection — any other code running while this component is mounted has its console output filtered too.
- **Layout**: `xl:grid-cols-[minmax(0,2.1fr)_minmax(360px,0.9fr)]`, `h-screen`/`min-h-0` flex chain — built to be non-scrolling on desktop. On narrow viewports the `<aside>` becomes `overflow-y-auto`, i.e. it intentionally scrolls below a breakpoint, but there's no horizontal/mobile-specific layout, and the 420px-min-height camera box plus stacked panels will overflow on small phone screens.
- **No automated tests**: no test framework configured, no `*.test.*`/`*.spec.*` files, no Playwright/Cypress config.
- **No React state library**: state is plain `useState`/`useRef`, ~16 state variables and ~10 refs in one component.

---

## 3. Backend Architecture

`ai_server.py` (139 lines) is the entire backend:

- Loads exactly one model at import time (module-level, not lazy): `trained_model/samvaad_isl_model.keras`, and its labels from `trained_model/label_encoder.json` (61 classes). If either file is missing, the script raises and FastAPI never starts — there's no fallback or health-degraded mode.
- `SEQUENCE_SHAPE = (30, 126)` is hardcoded and must match the frontend's `FRAME_SEQUENCE_LENGTH`/`FRAME_FEATURE_DIMENSION` constants in `lib/islRecognizer.ts` — there is no shared schema; a change on one side silently breaks the other at runtime (return becomes `{"success": false, "error": "Expected sequence shape..."}`, swallowed by the frontend as "server offline" for 3s, see §7).
- Two endpoints only: `GET /` (status/label count) and `POST /predict` (body: `{"sequence": number[][]}` → label/confidence/top_k via `np.argmax`).
- CORS is wide open: `allow_origins=["*"]`, `allow_credentials=True`, all methods/headers — acceptable for a local dev tool, not for any public deployment.
- No authentication, no rate limiting, no request size caps (a client could submit an arbitrarily large `sequence` array; FastAPI/pydantic will accept any list of lists, and `np.asarray` will only fail downstream on shape mismatch — large payloads are not rejected early).
- No logging beyond two `print()` statements at startup (model load, label count). No structured logs, no request tracing, no metrics.
- No `requirements.txt`, `Pipfile`, `pyproject.toml`, or pinned versions anywhere — the only evidence of a working install is `__pycache__/ai_server.cpython-311.pyc` (Python 3.11) and the import list itself (`fastapi`, `pydantic`, `tensorflow`, `numpy`). Runtime dependency reproducibility is entirely undocumented.
- No process manager / ASGI server invocation is defined (no `uvicorn ai_server:app` command in any script, README, or Procfile) — a developer has to know to run it themselves.
- `trained_model_4class/` and `trained_alnum_model/` exist on disk but are **not loaded by `ai_server.py` at all** — they are orphaned experiment artifacts (see §4).

---

## 4. AI Training Pipeline

There are two materially different model lineages, plus one fully orphaned one:

### a) Continuous-sign LSTM model (`tools/train_isl_model.py` → `trained_model/`)
- Input: `(30, 126)` sequences (30 frames × [left-hand 63 + right-hand 63] landmark floats), loaded from `manifest.csv` + `.npy` sequence files produced by the dataset pipeline (§5).
- Architecture: `Masking → Bidirectional LSTM(96) → BatchNorm → Dropout(0.35) → Bidirectional LSTM(64) → BatchNorm → Dropout(0.35) → Dense(128, relu) → Dropout(0.25) → Dense(num_classes, softmax)`.
- Per-class stratified train/val split (`split_all_samples_per_class`) with singleton classes forced into train only.
- Class-balanced weights (`compute_class_weight`, capped at 20.0).
- `Adam(lr=1e-3, clipnorm=1.0)`, `sparse_categorical_crossentropy`, callbacks: `ModelCheckpoint(best val_accuracy)`, `EarlyStopping(patience=12)`, `ReduceLROnPlateau(patience=4)`.
- Result (`trained_model/classification_report.txt`): **100% precision/recall/f1 on every one of 61 classes**, 726 validation samples, except `brinjal` (0.96 f1) and `giraffe` (0.96 f1) which are also near-perfect. This is discussed as a major red flag in §12.
- 61 classes are vocabulary words/phrases (`hello`, `thank you`, animal names, vegetable names, `good morning`, `what is your name`, etc.) — this is a fixed, closed vocabulary, not free ISL grammar.
- One class (`still`) has only 31/61 samples — a data collection gap not flagged anywhere except the raw counts.

### b) 4-class "hard" LSTM model (`tools/train_isl_4class_model.py` → `trained_model_4class/`)
- Same architecture family but deeper (`LSTM(128)→LSTM(64)→Dense(128)→Dense(64)`), trained on a hardcoded 4-class subset (`bear, break, brinjal, budget` — phonetically/visually similar words, presumably chosen to stress-test confusable signs).
- Adds explicit data augmentation (`augment_sequence`: temporal roll ±3 frames, scale jitter 0.96–1.04, Gaussian noise σ=0.003, 1% feature dropout) applied 6× per sample (`expand_training_set`) before training — augmentation is **only** applied to the training split, correctly excluding validation.
- Result: again **100% on all 4 classes**, 48 validation samples. Comment in script: `"Next step: point ai_server.py to this new model folder for live testing."` — this was never done; `ai_server.py` still points at `trained_model/`.

### c) Orphaned alphanumeric image model (`trained_alnum_model/`)
- `alnum_image_model.keras` classifies 26 single-letter classes (`a`–`z`) with 14,347 samples, again **100% accuracy** on 2,869 validation samples.
- The script that produced it, `tools/train_alnum_image_model.py`, is **0 bytes** — empty file. The training code is gone; only the artifacts (model, label map, plots, history CSV) remain. This model cannot be regenerated, audited, or understood from the current repo, and it is loaded by nothing.
- Filename ("image" model, not "sequence") implies a different input modality (static per-frame image classification for fingerspelling letters) than the two LSTM models above — i.e., a third, incompatible architecture exists in the repo with no integration path and no recoverable source.

### d) TFJS export tool (`tools/export_tfjs_model.py`)
- Converts `trained_model/samvaad_isl_model.keras` + `label_encoder.json` into a TFJS `model.json` bundle at `public/models/isl/`, intended to let the frontend run inference client-side using the already-installed `@tensorflow/tfjs` dependency.
- **This has never been run successfully against the current `public/` tree**: `public/models/` contains only `hand_landmarker.task`, no `isl/` subfolder. The `@tensorflow/tfjs` npm dependency is installed but unused by any `import` in the codebase (confirmed: no `tfjs` import anywhere under `app/` or `lib/`). The actual inference path is the FastAPI server, not TFJS, despite TFJS being declared as a dependency.

**Training reproducibility**: none of the three models has a documented dataset source-of-truth checked into the repo (the raw video dataset lives in `~/Downloads/archive`, hardcoded as a default path in `tools/prepare_isl_dataset_v2.py`, and the processed sequences live in `~/Downloads/processed_isl`/`C:\Users\dvjaw\Downloads\processed_isl`, hardcoded as the default `--dataset-dir` in both training scripts). The dataset itself is not in version control.

---

## 5. Dataset Processing Pipeline

Two generations of the same idea exist side by side, both unused by anything else once their `.npy` outputs are produced:

- **v1 — `tools/prepare_isl_dataset.py`**: uses the legacy `mediapipe.solutions.hands` API (`mp_hands.Hands(...)`, `static_image_mode=False`). Walks `--dataset-root/Sample Videos` and `--dataset-root/Video_Dataset/<label>/*.mp4`, derives a label per video from folder structure (`derive_label`), runs MediaPipe per sampled frame (`--frame-stride`, default 2), builds a 126-dim `[left_63 | right_63]` feature per frame, and resamples/pads every video to a fixed `--target-frames` (default 30) via `fixed_length_sequence` (linear-index resampling if too long, last-frame-padding if too short). Writes `.npy` sequences, `manifest.csv`, `label_map.json`, `dataset_summary.json`.
- **v2 — `tools/prepare_isl_dataset_v2.py`**: same overall shape but ported to the newer `mediapipe.tasks.vision.HandLandmarker` Tasks API (`detect_for_video`, explicit per-frame timestamps tracked across the whole video with a running `global_timestamp_offset_ms`), and adds `ensure_model_file` to auto-download `hand_landmarker.task` from Google's CDN if missing. Defaults are hardcoded to the author's machine (`Path.home() / "Downloads" / "archive"`, `Path.home() / "Downloads" / "processed_isl"`).
- **Padding/resampling strategy is identical in both**: exact-length videos pass through; short videos repeat their last frame to fill 30 slots (this biases the model toward "freeze-frame" endings); long videos are uniformly subsampled via `np.linspace` index rounding (can silently skip the most motion-informative frames if a sign's key transition is brief).
- **No augmentation, normalization, or hand-presence filtering happens during dataset prep itself** — all frames are kept even when `valid_frames == 0` would have meant skipping (videos with zero detected landmarks across the whole clip are explicitly excluded: `if sequence is None or valid_frames == 0: ... status=skipped_no_landmarks`), but partially-empty videos (some all-zero frames mixed with valid ones) are kept verbatim, meaning the model trains on frames with literal all-zero hand vectors interspersed with real signal.
- **Label derivation is filename/folder-based and fragile**: `derive_label` walks up the path looking for the first folder name that isn't in `CONTAINER_NAMES` (`archive`, `sample videos`, `video_dataset`) — any unexpected folder nesting silently produces wrong labels with no validation step.
- **No train/test separation at the dataset-prep stage** — that split happens later, per-training-script, directly from the manifest. This means the same raw video could end up resampled differently across pipeline runs (overwrite-protected by default — `seq_path.exists() and not args.overwrite` — so re-running without `--overwrite` silently keeps stale sequences even if the source video changed).

---

## 6. MediaPipe Integration

Two separate MediaPipe integrations exist for two separate purposes, and they are not the same library/runtime:

1. **Browser-side (production path)** — `@mediapipe/tasks-vision@0.10.34` (npm), used only in `app/page.tsx`:
   - `FilesetResolver.forVisionTasks(WASM_URL)` where `WASM_URL` is a **CDN URL** (`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm`) — note `@latest`, not a pinned version, despite the npm package itself being pinned to `0.10.34`. This is a version-drift risk: the WASM runtime can silently update independently of the pinned JS API surface.
   - `HandLandmarker.createFromOptions(...)` loads its model from `MODEL_URL = "/models/hand_landmarker.task"`, served from `public/models/` (self-hosted, 7.5 MB).
   - Config: `runningMode: "VIDEO"`, `numHands: 2`, `minHandDetectionConfidence/minHandPresenceConfidence/minTrackingConfidence: 0.5`.
   - Detection runs inside a `requestAnimationFrame` loop gated on `video.currentTime !== lastVideoTimeRef.current` (avoids re-processing the same frame), calling `landmarker.detectForVideo(video, performance.now())` synchronously on the main thread — no Web Worker, so detection competes with React rendering and canvas drawing every frame.
2. **Python offline (dataset-prep path)** — the `mediapipe` PyPI package, used only in `tools/prepare_isl_dataset.py` (legacy `solutions.hands` API) and `tools/prepare_isl_dataset_v2.py` (Tasks API, same `hand_landmarker.task` model file, duplicated at both `public/models/hand_landmarker.task` and `tools/models/hand_landmarker.task` — identical 7.5 MB file checked in twice).

There is no shared abstraction between the two: landmark→vector conversion (`landmarks_to_vector`/`frame_to_feature`) is hand-reimplemented three times — once in Python v1, once in Python v2, once in TypeScript (`lib/islRecognizer.ts: landmarksToVector`/`buildFrameFeature`). All three must agree on left/right ordering, zero-padding for missing hands, and 63-float-per-hand layout, or training/inference will silently diverge. They currently do agree, but nothing enforces it (no shared constants file, no schema version stamped into the model or manifest).

---

## 7. Recognition Pipeline

End-to-end per-frame flow in the browser (`app/page.tsx` + `lib/islRecognizer.ts`):

1. `requestAnimationFrame` loop calls `handLandmarker.detectForVideo(...)` → raw `{landmarks, handedness}`.
2. `analyzeHands(result, handHistoriesRef.current, options)` (`lib/islRecognizer.ts`) runs **two parallel, independently-evolving classification strategies simultaneously**:
   - **Heuristic fallback** (`classifyFallback`): pure geometric finger-extension counting (`isFingerExtended` compares tip vs. PIP joint y-coordinate) → labels like "Fist", "Open palm", "Point", "Two/Three fingers", "Hand detected". This always runs, immediately, with no network call.
   - **Server-backed model prediction**: every frame's 126-dim feature vector is pushed into a rolling `frameBuffer` (module-level array, capped at `FRAME_SEQUENCE_LENGTH=30`); once full, `dispatchPrediction` POSTs the 30-frame window to the FastAPI server, throttled by `MIN_INFERENCE_GAP_MS=280` and skipped entirely for 3s after any failure (`serverOfflineUntil`). The buffer is a **sliding window, not segmented per-sign** — it fires a prediction every ~280ms regardless of whether a sign gesture has actually started or ended, so the model is constantly asked to classify arbitrary 1-second windows of motion, including transitions between signs.
3. `classifyHand(hand)` (used only for the canvas label overlay in `drawResults`, not for the committed transcript) prefers the server's `stablePrediction`, then `lastLivePrediction`, then falls back to the geometric heuristic — meaning **the on-screen skeleton label and the transcript-driving label can disagree**, since `buildTextFromPrimaryHand` in `page.tsx` reads from `primaryHand.stableLabel` (computed inside `analyzeHands`) while the canvas overlay independently re-derives its own per-hand label via `classifyHand` each render.
4. Stability logic is duplicated, not shared, between two layers:
   - `lib/islRecognizer.ts: deriveStablePrediction()` groups the last 10 server predictions (`predictionHistory`, TTL 1800ms) by label, requires `count >= STABLE_REPEAT_THRESHOLD (3)` and `avgConfidence >= 0.74` to call something "stable".
   - `app/page.tsx`'s inline `SentenceEngine` then applies **its own separate** stability gate (`STABLE_THRESHOLD=4` consecutive identical frames + `COMMIT_HOLD_MS=350`) before committing a word to the transcript. A label has to pass both gates independently, with different thresholds (3 vs 4, different windows), to become transcript text — there's no single source of truth for "is this prediction stable."
5. Confidence thresholds are scattered across the file at different magic numbers: `LIVE_CONFIDENCE_THRESHOLD=0.70`, `STABLE_AVG_CONFIDENCE_THRESHOLD=0.74`, `MIN_RAW_CONFIDENCE=0.55`, `MIN_ACTIVE_CONFIDENCE=0.82` (this last one declared in `page.tsx` but never actually read anywhere in the file's logic — only passed into `analyzeHands`'s `options` object, where `RecognitionOptions.minActiveConfidence` is accepted but never referenced in `lib/islRecognizer.ts`'s body. **Dead parameter.**).
6. Module-level mutable state in `lib/islRecognizer.ts` (`frameBuffer`, `lastLivePrediction`, `stablePrediction`, `predictionHistory`, `inferenceInFlight`, `lastInferenceAt`, `serverOfflineUntil`, `handHistories`) is **global to the module**, not component-scoped. If `Home()` ever unmounted/remounted (e.g., React Strict Mode double-invoke, or a future multi-instance use), this state would leak/persist incorrectly across instances since it's not reset by any cleanup effect — `cleanupCameraResources` resets `handHistoriesRef.current` (a `page.tsx`-local ref) but never calls anything in `islRecognizer.ts` to clear `frameBuffer`/`predictionHistory`/etc.

---

## 8. Transcript Generation Pipeline

Two byte-for-byte-similar but independently maintained `SentenceEngine` implementations exist:

- `lib/sentenceEngine.ts` (exported class, 322 lines) — **not imported by any file** (confirmed via repo-wide search; only self-reference).
- `app/page.tsx` lines 144–347 — an inline, private class also called `SentenceEngine`, actually used by the app.

They are *not* identical:
| Behavior | `lib/sentenceEngine.ts` | `app/page.tsx` inline |
|---|---|---|
| Noise input while pending | Still attempts `maybeFinalizeOnPause` and returns early | Skips noise entirely, no finalize check on noise frames |
| Punctuation tokenizing (`parseTranscript`) | Splits multi-char punctuation runs (`[.!?]+`) into individual tokens | Only matches a single trailing punctuation char (`[.!?]`) |
| ALL-CAPS short tokens (`formatWordForSentence`) | Special-cased: `/^[A-Z0-9]+$/` ≤5 chars passed through uppercase (e.g. preserves acronyms/fingerspelled letters) | No such case — everything is lowercased except sentence-initial capitalization |
| `getFinalizedSentence()` accessor | Present | Not present (not needed; page.tsx reads `result.finalizedSentence` directly) |

Whichever engineer maintains `page.tsx` going forward is editing logic that has silently forked from the "canonical" library version, and any bugfix made in one will not propagate to the other. This is the single clearest piece of technical debt in the repo.

**Pipeline as actually used** (`app/page.tsx`):
1. Each detection frame, `buildTextFromPrimaryHand` extracts a label string (`primaryHand.stableLabel` or `stableGesture` fallback), filtering out noise placeholders (`"—"`, `"Searching..."`, `"Searching."`, `"No hand detected"`).
2. `processLiveText(candidate, false)` feeds it into the engine's `processPrediction`, which tracks a `pendingWord` + repetition count; once the same word has appeared ≥`STABLE_THRESHOLD` (4) consecutive non-forced calls **and** has been pending for ≥`COMMIT_HOLD_MS` (350ms), it's committed.
3. Commit applies: noise check, repeat-suppression (won't re-commit the identical word within `REPEAT_SUPPRESSION_MS`=900ms of its last commit — this exists specifically to stop a held sign from spamming the same word every tick), sentence-start capitalization (`capitalizeWord`, with a special case making "i" → "I"), then pushes the formatted token onto `transcriptTokens`.
4. A separate `setInterval` (every `PREDICTION_TICK_MS=250ms`, only while `autoCommitEnabled`) calls `engine.tick(...)`, which checks `maybeFinalizeOnPause`: if no valid (non-noise) prediction has been seen for `PAUSE_FINALIZE_MS` (2000ms) and there's pending transcript content, it auto-appends a period and marks the sentence "finalized" (triggering an auto-speak if voice is on).
5. Manual overrides: "Commit Stable Label" button forces an immediate commit (`forceCommit: true`) bypassing the stability/hold gates entirely; "Backspace" pops the last token; "Clear" wipes everything.
6. Transcript persistence: `transcript` state is round-tripped through `localStorage` under key `samvaad:session:v4` together with `speechEnabled`/`autoCommitEnabled`, restored on mount via a `useEffect` that also calls `processingEngineRef.current.loadTranscript(parsed.transcript)` to resync the engine's internal token array with the restored text — re-parsing previously-formatted text back into tokens (lossy round-trip risk: e.g. multi-word phrases like "good morning" become two separate tokens on reload, indistinguishable from two separately-signed words).

---

## 9. Speech Synthesis Pipeline

Entirely client-side, using the Web Speech API (`window.speechSynthesis`) — no server-side TTS, no audio files, no offline fallback:

- `speakText(text, replaceQueue)`: trims and validates text (rejects empty/"—"), checks `window.speechSynthesis` exists (sets a user-facing error message if not — i.e., Safari/older-browser users without SpeechSynthesis support get a visible error rather than silent failure), then either:
  - `replaceQueue=true` (used for finalized sentences and the manual "Speak Transcript" button): clears the queue, calls `speechSynthesis.cancel()`, sets a single-item queue.
  - `replaceQueue=false` (used for per-word auto-speak): de-dupes against `speechLastTextRef.current` only when the queue is currently empty, then pushes onto `speechQueueRef.current`.
- `drainSpeechQueue` is a self-recursive callback (`utterance.onend`/`utterance.onerror` both call `drainSpeechQueue()` again) that processes one queued string at a time, guarded by `speechBusyRef`. Utterance config is fixed: `rate=0.95`, `pitch=1`, `lang="en-IN"` — **hardcoded to Indian English**, no language selection UI, no voice picker, and no handling for the case where no `en-IN` voice is installed on the user's OS/browser (Web Speech API will silently substitute a default voice rather than error, so this fails open/silently rather than failing loudly).
- Per-word speech and per-sentence (finalized) speech are both routed through the same queue/cooldown logic, but a `SPEAK_COOLDOWN_MS=900` constant is declared at module scope and **never referenced anywhere in the speech functions** — dead constant, redundant with `REPEAT_SUPPRESSION_MS` which actually gates repeats at the transcript-commit layer, not the speech layer.
- Cleanup: `cleanupCameraResources` and `resetSession` both call `speechSynthesis.cancel()` and reset the queue/busy refs, so stopping the camera or resetting reliably stops in-flight speech — this part is handled correctly.
- No queueing priority between "live word" speech and "finalized sentence" speech beyond `replaceQueue` semantics — if a word is mid-utterance when a sentence finalizes, `speechSynthesis.cancel()` will cut it off (acceptable for a live captioning tool, but not documented as intentional behavior anywhere).

---

## 10. State Management Flow

No external state library (no Redux/Zustand/Jotai/Context API usage) — everything lives in one component's closures:

- **React state** (`useState`, 16 variables): UI-facing — `cameraStatus`, `modelStatus`, `errorMessage`, `rawGesture`, `stableGesture`, `latestHandedness`, `gestureConfidence`, `handsDetected`, `currentHands`, `primaryHand`, `speechEnabled`, `autoCommitEnabled`, `transcript`, `logoBroken`, `copyFeedback`, `isHydrated`.
- **Refs** (`useRef`, not triggering re-render): hold mutable runtime objects that must survive renders without re-triggering them — `videoRef`, `canvasRef`, `streamRef`, `rafRef`, `handLandmarkerRef`, `handLandmarkerPromiseRef`, `cameraActiveRef`, `lastVideoTimeRef`, `speechEnabledRef` (a manual ref-mirror of `speechEnabled` state, kept in sync via a dedicated `useEffect` — done because the `requestAnimationFrame` loop closure needs the *current* value without retriggering the detection loop's effect), `speechQueueRef`, `speechBusyRef`, `speechLastTextRef`, `processingEngineRef` (holds the `SentenceEngine` instance itself), `handHistoriesRef`.
- **Module-level mutable state** outside React entirely, inside `lib/islRecognizer.ts` (see §7): `frameBuffer`, `lastLivePrediction`, `stablePrediction`, `predictionHistory`, `inferenceInFlight`, `lastInferenceAt`, `serverOfflineUntil`, `handHistories` (a `Map`, separate from but overlapping in purpose with `page.tsx`'s `handHistoriesRef`).
- **Persisted state**: `localStorage["samvaad:session:v4"]` mirrors `{speechEnabled, autoCommitEnabled, transcript}` only — camera/model status and live recognition state are intentionally not persisted (correct choice, since they're meaningless across reloads).
- **Hydration gate**: `isHydrated` blocks first paint of the real UI until the `localStorage` read effect completes, rendering a minimal "Loading..." placeholder — this avoids SSR/CSR text mismatches but means the app always flashes a loading screen even when there's nothing to restore.
- **Data flow per frame**: `detectForVideo` → `analyzeHands` (mutates module-level recognizer state, returns a snapshot) → 7 `setX` calls in `updateRecognition` (7 separate state updates per detected frame, each potentially triggering its own re-render pass since they're not batched into one state object — React 19 auto-batches synchronous updates within the same event handler, which this is, so in practice they do batch into one render, but the code does not rely on or document that, e.g. no `useReducer` consolidating these into one atomic update) → `drawResults` (direct canvas mutation, outside React) → `processLiveText` (mutates the `SentenceEngine` instance ref) → `applyEngineResult` (one more `setTranscript` + possible speech side-effect).
- There is no single "frame result" object threaded through the system — recognition snapshot, drawing, and transcript processing are three separate passes over conceptually the same detection result, each recomputing parts of it independently (e.g. `classifyHand` is called once inside `analyzeHands`'s heuristic and again, redundantly, inside `drawResults`'s per-hand loop for the overlay label).

---

## 11. User Workflow

1. User opens the page → sees a static "Loading..." placeholder for one tick while `localStorage` is read, then the full layout (header, live camera panel, recognition panel, transcript panel, status panel).
2. User clicks **Start** → browser camera permission prompt → on grant, `getUserMedia({video: {facingMode: "user", width: 1280, height: 720}})` starts; video element gets the stream; MediaPipe `HandLandmarker` loads (status badge shows "Model: loading" → "ready"); detection loop starts.
3. User performs a sign in front of the camera. The skeleton overlay (green dots/lines + handedness•label tag) renders live on canvas. The "Recognition" panel shows raw/stable/handedness/confidence/hands-detected, continuously updating.
4. If "Auto Build" is on (default) and the recognized label stabilizes long enough, it's appended to the transcript automatically; if the user pauses signing for 2s, the engine auto-finalizes the sentence with a period.
5. If "Voice" is on (default off), each committed word and/or finalized sentence is spoken aloud via the Web Speech API in `en-IN`.
6. User can manually: force-commit the current stable label, speak the whole transcript on demand, copy transcript to clipboard, download transcript as a `.txt` file, backspace the last word, clear the transcript, or fully reset the session (clears transcript + turns off voice/auto-build flags + stops any speech).
7. User clicks **Stop** → camera, detection loop, and any in-flight speech are torn down; UI resets to idle placeholders.
8. On reload, `speechEnabled`, `autoCommitEnabled`, and the prior `transcript` text are restored from `localStorage` (camera does not auto-restart — user must click Start again).

There is no onboarding, no tutorial, no example sign reference, no indication anywhere in the UI of which 61 words/phrases the model actually recognizes — a first-time user has no way to discover the model's closed vocabulary except by trial and error or by reading `trained_model/label_encoder.json` directly.

---

## 12. Current Weaknesses

- **Validation accuracy of 100.00% across all three trained models (61-class, 4-class, 26-class) is not a credible result for this task and strongly suggests data leakage or near-duplicate train/val samples**, most likely because: (a) the per-class split (`split_all_samples_per_class`/`split_indices_per_class`) splits *already-extracted, fixed-length sequences*, and if multiple sequences were derived from the same source video (or very similar takes recorded back-to-back by the same signer under the same lighting/background), train and val samples for a class can be near-identical; (b) frame-stride sampling (every 2nd frame) plus `np.linspace` resampling to a fixed 30 frames can make two windows from the same recording session nearly indistinguishable; (c) no signer-disjoint or session-disjoint split is performed anywhere in the pipeline. Real-world accuracy is very unlikely to match these numbers, and there is currently no held-out, truly independent test set to know the real error rate.
- **The actually-served model (`trained_model/`) was never updated with the explicit augmentation strategy proven out in `trained_model_4class/`** — the 4-class experiment's own code comment says to point `ai_server.py` at the new model, and this was never done.
- **Two parallel `SentenceEngine` implementations** (`lib/sentenceEngine.ts`, unused; inline copy in `page.tsx`, used) have already diverged in tokenizing and formatting behavior (§8) — a textbook case of copy-paste drift with no single source of truth.
- **`lib/modelAdapter.ts` is fully dead code**: a placeholder client-side "model" that hardcodes `"Hello"`/`"Stop"`/`"Gesture"` based on simple finger-extension geometry, explicitly commented as a temporary stand-in ("Replace with tfjs prediction later"), superseded by the real server-backed pipeline in `islRecognizer.ts`, but never deleted. It is imported by nothing.
- **`@tensorflow/tfjs` is a declared dependency that is never imported anywhere in the app** — the actual architecture uses a Python FastAPI server, not in-browser TFJS inference, despite the export tooling (`export_tfjs_model.py`) and the npm dependency both existing for that purpose. This is partially-built, abandoned-in-place infrastructure.
- **`tools/train_alnum_image_model.py` is an empty file**, yet its output artifacts (`trained_alnum_model/*`) remain in the repo, fully unreproducible and unintegrated.
- **No shared contract between Python and TypeScript** for the feature vector format (126 floats, hand ordering, frame count) — currently kept in sync by convention/memory only, verified solely by both sides happening to agree at runtime.
- **`MIN_ACTIVE_CONFIDENCE` is a dead parameter** — declared, passed into `analyzeHands` options, never read inside `lib/islRecognizer.ts`.
- **`SPEAK_COOLDOWN_MS` is a dead constant** in `page.tsx`, declared and never referenced.
- **Global console patching** (`console.error`/`warn`/`info` overridden for the component's mounted lifetime) is a blunt, page-wide side effect to suppress a handful of known-noisy MediaPipe log lines, applied indiscriminately to all console output during that time.
- **Recognition correctness depends on two independently-tuned stability gates** (`lib/islRecognizer.ts`'s server-prediction stability vs. `page.tsx`'s transcript-commit stability) with different thresholds and no shared concept of "this prediction is trustworthy."
- **Duplicate 7.5 MB `hand_landmarker.task` binary** checked in at both `public/models/` and `tools/models/`.
- **`public.zip`** (8.4 MB) is a stale, committed-looking archive snapshot of an earlier project layout sitting in the repo root with no apparent purpose for the running app.
- No automated tests of any kind (unit, integration, or e2e) for either the TypeScript or Python code.
- The `AGENTS.md` file claims breaking changes versus this Next.js version's training-data conventions and directs implementers to consult `node_modules/next/dist/docs/` before writing code — a useful guardrail for future code changes, but it is also a reminder that this Next.js major version (16.2.3) is newer than common tooling familiarity, raising the risk of subtly incorrect API usage already present in the code that wouldn't be caught by habit alone.

---

## 13. Scalability Issues

- **Single in-process Python model server with no concurrency story**: `ai_server.py` loads one TensorFlow model into one process; FastAPI's default (no explicit `uvicorn --workers` config exists anywhere) means a single worker handling all `/predict` calls. TensorFlow inference under load will serialize, and there is no batching, no async model call (`model.predict` is a synchronous, blocking call inside an `async def predict` route, which will block the event loop for the duration of each prediction).
- **No horizontal scaling path**: no containerization, no load balancer config, no statelessness guarantee documented (the model is loaded once at import time per process; scaling to N processes means N redundant model loads, fine for this model's small size but undocumented as a deployment pattern).
- **Hardcoded `localhost:8000` default for the AI server URL**: scaling beyond one developer's machine requires every deployment to set `NEXT_PUBLIC_AI_SERVER_URL`, and there is no service discovery, health-check–driven failover, or multi-instance routing — just a single fetch target with a 3-second cooldown on failure (`serverOfflineUntil`).
- **Per-client polling load**: each connected browser tab independently runs its own `requestAnimationFrame` detection loop and independently POSTs to `/predict` roughly every 280ms while a hand is visible — there is no batching across concurrent users, so server load scales linearly with concurrent active users with no smoothing.
- **Closed, fixed 61-word vocabulary**: adding new signs requires re-collecting video, re-running the entire offline dataset-prep + training pipeline by hand, and manually redeploying a new `.keras` file — there is no incremental learning, no class-addition path, no versioned model registry. Given the suspected leakage issue (§12), it's also unclear how the model would generalize to new signers at all.
- **All app/recognition state lives in one client-side component** with module-level mutable globals in `lib/islRecognizer.ts` — this works for exactly one active session per browser tab but has no concept of multiple simultaneous users on a shared page, multi-device sync, or server-persisted transcripts (everything is `localStorage`-only, per-browser, per-device).
- **No CDN/cache strategy beyond Next.js defaults** for the 7.5 MB `hand_landmarker.task` model file served from `public/` — fine for a small audience, but no cache headers, versioning, or CDN offload strategy is configured in `next.config.ts` (which is otherwise empty/default).
- **Frame-by-frame inference on the main thread**: detection (`detectForVideo`) runs synchronously in the animation frame callback with no Web Worker — as resolution/complexity grows (e.g., more landmarks, more hands, higher frame rate), this directly competes with UI responsiveness; there's no built-in degradation path (no frame-skipping under load beyond the existing "skip if same video timestamp" check).

---

## 14. Accessibility Issues

This is positioned as an assistive communication tool for Deaf/Mute users and hearing communication partners, which makes accessibility gaps especially significant:

- **No ARIA labeling or live-region announcements anywhere in `page.tsx`**: the transcript text, recognition labels, and status badges are plain `<p>`/`<span>` elements with no `aria-live`, `role="status"`, or `role="log"` — a screen-reader user relying on assistive tech (e.g., a hearing-impaired user who is also low-vision, or a sighted communication partner using a screen reader) gets no announcement when the live transcript or recognized word updates. For a tool whose entire purpose is real-time communication, this is a significant gap for any user not visually tracking the screen continuously.
- **Icon-only/ambiguous controls lack textual alternative state descriptions**: buttons like "Voice" and "Auto Build" use `aria-pressed` (good — this is correctly implemented) but have no `aria-label` clarifying what "Voice"/"Auto Build" mean beyond the visible text, and no `title`/tooltip for first-time users.
- **No keyboard-only workflow verification**: all controls are real `<button>` elements (good baseline), but there is no visible focus-style override beyond browser defaults, no documented tab order rationale, and no skip-link given the page's dense control layout.
- **No captioning of non-text status purely via color/badge text**: status pills ("Camera: live", "Model: ready") are text-based (good, not color-only), but error states (`errorMessage`) only appear inline near the camera box and are not also announced via `aria-live="assertive"` — a non-sighted or screen-distracted user could miss a camera-blocked error entirely.
- **Hardcoded English-only UI and `en-IN` speech**: no language selection for either the interface text or the synthesized speech voice, despite Indian Sign Language usage spanning many spoken/written-language communities (Hindi, regional languages) who would benefit from transcript/speech localization. The README/AGENTS files don't mention multi-language support as a goal, but for an ISL tool specifically, this is a notable omission given the linguistic diversity of its likely user base.
- **Reliance on camera + visual hand-tracking with no alternative input modality**: there is no fallback text-entry, no alternative confirmation method (e.g., switch-access or keyboard-driven sign selection) for users who cannot reliably perform signs in front of a webcam (e.g., due to motor impairment, poor lighting, or camera unavailability) — the tool is unusable without a functioning camera and a hand fully in frame.
- **No font-size/contrast user controls**: the dark theme (`bg-[#050816]`) and fixed text sizes are not user-adjustable; for low-vision users this could be a barrier, though base contrast (white/emerald text on dark background) is reasonably high by default.
- **No indication of model vocabulary or sign accuracy confidence in plain language**: confidence is shown as a raw percentage ("Confidence: 92%"), which assumes numeracy and doesn't communicate uncertainty in an accessible way (e.g., no "low confidence — try signing again" guidance).
- **Speech synthesis failure is surfaced as a generic error message string** (`"Speech synthesis is not supported in this browser."`) rather than offering a fallback (e.g., enlarging the on-screen transcript, or vibrating/flashing) for the Deaf users this is partly built for, who by definition may not need or use the speech feature themselves but rely on it solely for their hearing conversation partner — there's no UI cue distinguishing "this is for the other person" from "this is for you," which is a UX/communication-design gap as much as a technical accessibility one.

---

## 15. Deployment Readiness

**Not deployment-ready as a single product today.** Concretely:

- **No deployment configuration exists anywhere**: no `Dockerfile`, no `docker-compose.yml`, no CI/CD workflow files (no `.github/workflows`), no `vercel.json`, no `Procfile`, no IaC of any kind.
- **Two independently-deployed processes with no orchestration**: the Next.js frontend can deploy to Vercel/any Node host trivially (it's a stock Next.js app), but the FastAPI model server has zero deployment story — it must be hosted somewhere reachable over HTTPS, with CORS already wide-open (`allow_origins=["*"]`, fine functionally but not hardened), and the frontend must be configured via `NEXT_PUBLIC_AI_SERVER_URL` to find it. Nothing in the repo documents this required two-service topology.
- **No Python dependency manifest** (`requirements.txt`/`pyproject.toml`) — anyone trying to deploy `ai_server.py` has to reverse-engineer the dependency list from imports (`fastapi`, `pydantic`, `tensorflow`, `numpy`, plus an ASGI server like `uvicorn` which is never even imported/referenced, meaning even the *run command* is undocumented).
- **No environment variable documentation** (no `.env.example`): the only configurable knob (`NEXT_PUBLIC_AI_SERVER_URL`) is discoverable only by reading `lib/islRecognizer.ts` source.
- **Hardcoded localhost defaults** mean the app works out of the box only when both processes run on the same machine during development; production behavior (cross-origin HTTPS server, mixed-content concerns if the frontend is HTTPS and the AI server isn't) is entirely unaddressed.
- **MediaPipe WASM runtime pulled from a `@latest`-pinned CDN URL at runtime** — a production deploy has an external runtime dependency on jsDelivr availability and on that CDN serving a WASM build compatible with the pinned `0.10.34` JS package; this is a available-but-fragile pattern, not pinned/self-hosted.
- **No model versioning or rollback strategy**: `trained_model/samvaad_isl_model.keras` is the only model `ai_server.py` will ever load; deploying a retrained model means manually replacing this file with no staged rollout, no canary, no rollback artifact retained automatically (though `trained_model_4class`/`trained_alnum_model` happen to exist as prior artifacts, that's incidental, not a designed versioning system).
- **No health checks, readiness probes, or monitoring/alerting** for the AI server — if the model fails to load at import time, the process won't start at all (no graceful degraded mode), and there's nothing to detect or alert if it crashes mid-traffic.
- **No automated tests gating any change** — there is nothing in CI (because there is no CI) to catch a regression in either the TypeScript transcript logic or the Python inference contract before it reaches users.
- **Given the suspected model evaluation leakage (§12)**, the model's real-world readiness is itself unverified — the reported "100% accuracy" cannot currently be trusted as a pre-deployment quality gate.
- **Static assets are reasonably deployment-safe** in isolation (icons, logo, the hand-landmarker task file all live correctly under `public/` and would serve fine from Vercel/any static host), but the repo also still carries unrelated clutter (`public.zip`, duplicate `tools/models/hand_landmarker.task`, dead `lib/sentenceEngine.ts` and `lib/modelAdapter.ts`, an empty Python training script with orphaned artifacts) that should be resolved before treating the repository as a clean deployment source.

---

## Summary

SAMVAAD currently exists as a working **local prototype**: a polished single-page Next.js UI with real MediaPipe hand tracking, a real (if unverified) LSTM sign-classification model served by a separate FastAPI process, working transcript-building and text-to-speech behavior, and session persistence via `localStorage`. It is not yet a deployable product: the two runtime processes have no orchestration or hosting story, the headline model accuracy (100% across three separate models) is not credible without a leakage-free evaluation, there is real dead code and duplicated logic (`lib/sentenceEngine.ts`, `lib/modelAdapter.ts`, unused `@tensorflow/tfjs`, an empty training script with orphaned outputs) that increases maintenance risk, and accessibility/internationalization for its core Deaf/Mute-communication use case is largely unaddressed beyond basic ARIA-pressed states and high-contrast styling.
