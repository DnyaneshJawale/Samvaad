# SAMVAAD — Product Redesign for a Deployable Assistive Communication Product

Status: design document only. No code in this document, and no code in the repository was changed to produce it.
Grounded in `PROJECT_AUDIT.md` (current-state findings) — this document is the "what it should become."

---

## 0. Reframing the Product

The current prototype is framed as "ISL → Text and Speech": one direction, one user. That framing undersells — and undersizes — what this needs to be.

**The real unit of use is a conversation between two people, only one of whom signs.** A Deaf or Mute person and a hearing person who doesn't know ISL are standing at a pharmacy counter, a bank, a school admissions desk, a clinic reception. They have one device between them. SAMVAAD's job is to carry meaning in both directions across that gap, in real time, in a way both people trust enough to rely on for something that matters (a prescription, a form, a diagnosis).

That reframing drives every decision below:

- **Transcript-first**: the conversation record — not the camera feed — is the product. The camera is an input mechanism; the transcript is what both people actually read, point at, and rely on.
- **Two channels, not one**: Sign → Text/Speech (existing) **and** Speech → Text (new, for the hearing partner to talk back into something the Deaf user can read). Without the second channel this is a broadcast tool, not a conversation tool.
- **Non-scrollable**: a device handed back and forth between two people across a counter cannot ask either person to scroll to find the current state of the conversation. The *page* never scrolls; only a self-managing, auto-scrolling conversation log scrolls internally, the same way every chat app already behaves.
- **Stability and trust are the actual feature.** A clinic will not adopt a tool that silently mis-hears a dosage. Every "smart" path needs a manual, always-available fallback, and every uncertain output needs to say so in plain language.

---

## 1. Design Principles (mapped to your focus areas)

| Focus area | Principle | Where it shows up |
|---|---|---|
| Transcript-first | The transcript panel is visually dominant and never collapses to zero; camera/mic docks shrink first | §4 Wireframe, §5 Components |
| Speech behavior | Two independent speech systems (TTS for signer, STT for partner), each interruptible, each with visible state | §7 Speech Flow |
| Accessibility | Live regions, plain-language confidence, adjustable text size, full keyboard path, captioned everything | §8 Accessibility Spec |
| Non-scrollable layout | Viewport-locked shell; only the conversation log and modal sheets scroll internally | §4 Wireframe |
| Real-world workflow | "Conversation Mode" with a hand-off/flip affordance, designed around a shared single device at a counter | §2 User Journey |
| Multi-language | UI chrome, TTS voice, STT language, and transcript display language are four independently-set options | §9 Multi-Language Plan |
| Stability | Tiered degradation: full AI → heuristic-only → manual text, never a dead end | §10 Stability & Trust |
| User trust | Vocabulary is discoverable up front; confidence is plain-language; privacy of the camera feed is stated, not assumed | §10 Stability & Trust |

---

## 2. User Journey

**Personas**
- **Signer** — Deaf or Mute primary user, fluent in ISL, holds/owns the device.
- **Partner** — hearing person with no ISL knowledge, encountered in the moment (clerk, doctor, teacher, stranger).

**Scenario: pharmacy counter, first-time partner, no prior relationship.**

1. **Setup (Signer, before the conversation, possibly days earlier)** — Signer opens the app once at home, picks interface language and a TTS voice/language for the partner's ear, optionally skims the "What can I sign" reference. This is a one-time setup, not a per-conversation tax.
2. **Initiation (Signer, at the counter)** — Signer opens the app, taps **Start Conversation**. Camera permission is requested with a plain-language reason ("Used to read your hand signs. Video is not recorded or stored."). The conversation view appears — empty transcript, large "Ready" state, camera dock active.
3. **Signer turn** — Signer signs. Live captioning shows the in-progress word at the bottom of the transcript (similar to a live-caption "typing" indicator), confidence shown as a plain word ("Clear" / "Keep going" / "Not sure — try again"), not a percentage. When a phrase is stable, it commits into the transcript as a labeled bubble: **"You signed:" — "I need this prescription refilled."** It is simultaneously spoken aloud through the device speaker for the partner.
4. **Hand-off** — Signer taps **Flip for reply** (or the device is simply turned around — the layout itself doesn't require a tap, see §4). The screen reorients: large, high-contrast text, the same transcript, now facing the partner, with a clear **mic button** and a **type instead** option underneath a prompt: *"Please reply here — by voice or by typing."*
5. **Partner turn** — Partner taps the mic and speaks, or types. Live interim captions appear as they talk (so they can self-correct, the way phone dictation does), and on a pause it commits as a labeled bubble: **"They said:" — "Do you have your prescription ID?"** — spoken back to nobody (the partner can already hear themselves), but rendered large enough for the Signer to read when the device flips back.
6. **Continued exchange** — Turns alternate. The transcript is the shared memory of the whole exchange — either party can scroll back *within the transcript panel only* to confirm something said two turns ago, without losing the rest of the layout.
7. **Trust check moments** — If recognition confidence is low or the AI server is unreachable, a banner appears in plain language ("Limited recognition right now — try typing this part") rather than silently guessing. Recognized text remains editable by tapping it, so a misread word can be corrected without restarting the whole exchange.
8. **Close-out** — Signer taps **End & Save**, gets the option to copy, download (PDF/txt), or discard the transcript. Nothing is uploaded or shared automatically; the default is local-only, ephemeral unless explicitly saved.

**Scenario variant: repeat partner (e.g., a regular doctor)** — same flow, but the app remembers the last-used partner language/voice per "saved contact" style shortcut (optional, off by default, never silently shares data).

---

## 3. What Stays the Same vs. What's New

| Capability | Current | Redesigned |
|---|---|---|
| Sign → text | Yes (one engine, works) | Kept, hardened (single source of truth, editable output) |
| Sign → speech | Yes | Kept, language/voice selectable |
| Speech → text (partner replying) | **Does not exist** | **New** — Web Speech Recognition + typed fallback |
| Conversation history as a shared object | Implicit (one flat transcript string) | **New** — structured turn-by-turn log with speaker attribution |
| Vocabulary discoverability | None | **New** — persistent "what can I sign" reference, served from backend |
| Degraded-mode behavior | Silent 3s retry, no user-visible state | **New** — explicit banner + automatic fallback to manual text |
| Device hand-off between two people | Not designed for | **New** — "Conversation Mode" flip/orient affordance |
| Multi-language | English UI, `en-IN` speech only, hardcoded | **New** — independent UI language, TTS language, STT language, transcript display language |

---

## 4. Wireframe

The page itself never scrolls. Three fixed zones (status, stage, controls) fill the viewport (`100dvh`), and only the conversation log (inside the Transcript panel) and modal sheets (Settings, Vocabulary Reference) scroll internally.

### 4.1 Primary device: handheld / tablet, portrait (the realistic "counter" device)

```
┌──────────────────────────────────────────────┐
│ ● Recognition: Ready   🌐 EN/HI   ⚙          │  ← Status bar (fixed, ~6% height)
├──────────────────────────────────────────────┤
│                                                │
│   TRANSCRIPT (dominant — ~60% height)         │
│   ┌────────────────────────────────────────┐ │
│   │ You signed:                             │ │
│   │ "I need this prescription refilled."    │ │  ← committed turn, speaker-labeled
│   │                                          │ │
│   │ They said:                              │ │
│   │ "Do you have your prescription ID?"     │ │  ← committed turn, speaker-labeled
│   │                                          │ │
│   │ [live caption] signing… "yes—"          │ │  ← in-progress, visually distinct
│   │                                          │ │     (internal auto-scroll only)
│   └────────────────────────────────────────┘ │
│                                                │
├──────────────────────────────────────────────┤
│  SIGN CAPTURE DOCK (collapsible, ~20% height) │
│   ┌──────────────┐  Confidence: Clear         │
│   │  camera +     │  Hands: 1                  │
│   │  skeleton     │  [Minimize ▾]               │
│   └──────────────┘                             │
├──────────────────────────────────────────────┤
│  PARTNER REPLY DOCK (~14% height)             │
│   🎤 Hold to speak     ⌨ Type instead          │
├──────────────────────────────────────────────┤
│ [Flip for reply] [Voice ▣] [Vocabulary] [⋯]   │  ← Control bar (fixed, ~10% height)
└──────────────────────────────────────────────┘
```

### 4.2 "Flipped" / partner-facing state (triggered by Flip for reply, or simply detecting the device was turned via orientation+motion heuristics)

```
┌──────────────────────────────────────────────┐
│         Please read, then reply below          │
├──────────────────────────────────────────────┤
│                                                │
│        "Do you have your                      │
│         prescription ID?"                      │   ← last 1–2 turns, MAX font size,
│                                                │       high contrast, no chrome
│        "I need this prescription               │
│         refilled."                              │
│                                                │
├──────────────────────────────────────────────┤
│     🎤  Tap and speak your reply               │
│     ⌨   Or type here                           │
├──────────────────────────────────────────────┤
│              [Flip back to signer]             │
└──────────────────────────────────────────────┘
```

### 4.3 Wide / desktop fallback (not the primary use case, but supported)

```
┌───────────────────────────────────────────────────────────────┐
│ ● Ready   🌐 EN/HI   ⚙                                          │
├───────────────────────────────────┬─────────────────────────────┤
│  TRANSCRIPT (dominant, ~65% width) │  SIGN CAPTURE DOCK           │
│  (same content as portrait)        │  (camera + skeleton)         │
│                                     ├─────────────────────────────┤
│                                     │  PARTNER REPLY DOCK          │
│                                     │  🎤 / ⌨                       │
├───────────────────────────────────┴─────────────────────────────┤
│ [Flip for reply] [Voice ▣] [Vocabulary] [⋯]                      │
└───────────────────────────────────────────────────────────────┘
```

Layout rule that makes this non-scrollable at every breakpoint: **the camera dock is the element allowed to shrink/collapse, never the transcript.** On very small viewports, the camera dock collapses to a small floating thumbnail (picture-in-picture style) rather than forcing the transcript to shrink below a legible minimum.

---

## 5. Component Hierarchy

Replacing the current single 1358-line component with a real tree (names indicate responsibility, not literal file names):

```
AppShell
├─ I18nProvider                      (interface language context)
├─ ConversationProvider              (turn log, session state — see §6)
├─ ErrorBoundary (top-level)
│
├─ StatusBar
│   ├─ ConnectionIndicator           (AI server: online / degraded / offline)
│   ├─ LanguageSwitcher              (interface language only)
│   └─ SettingsButton → SettingsSheet
│
├─ ConversationStage                  (the non-scrolling main area)
│   ├─ TranscriptPanel               (dominant, always rendered, never unmounts)
│   │   ├─ ConversationLog           (internal auto-scroll; renders TurnBubble[])
│   │   │   └─ TurnBubble             (speaker label, text, confidence tag, edit affordance, timestamp)
│   │   └─ LiveCaptionLine           (current in-progress sign or speech, ARIA live region)
│   │
│   ├─ SignCaptureDock                (collapsible; independent ErrorBoundary)
│   │   ├─ CameraPreview
│   │   ├─ HandOverlayCanvas
│   │   ├─ RecognitionStatusChip      (plain-language confidence, not raw %)
│   │   └─ MinimizeToggle
│   │
│   └─ PartnerReplyDock                (independent ErrorBoundary)
│       ├─ SpeechCaptureButton          (push-to-talk, Web Speech Recognition)
│       ├─ TypedReplyInput              (always available, even if mic works)
│       └─ ReplyLanguageHint
│
├─ ControlBar
│   ├─ FlipModeButton                  (toggles ConversationStage orientation/zoom)
│   ├─ VoiceToggle
│   ├─ VocabularyReferenceButton → VocabularyReferenceSheet
│   └─ OverflowMenu (Copy / Export / Clear / End Session)
│
├─ VocabularyReferenceSheet            (modal, scrolls internally; backend-driven list)
├─ SettingsSheet                       (modal; language x3, text size, contrast, voice, privacy info)
└─ OnboardingTour                      (first-run only, dismissible, never re-forced)
```

Key structural change from today: **`SignCaptureDock` and `PartnerReplyDock` each have their own error boundary.** A camera failure or a Speech Recognition API failure degrades *that dock only* — the transcript and the other dock keep working. This directly targets the "stability" requirement: nothing about the camera should be able to take down the conversation record or the partner's ability to reply.

---

## 6. State Flow

Replacing scattered `useState`/module-globals with explicit, named state machines. Described here as states and transitions, not code.

### 6.1 Session state (top-level)
`idle → active → ended`
- `idle`: landing/onboarding only.
- `active`: conversation in progress; this is where almost all UI lives.
- `ended`: summary + export/discard choice, then returns to `idle`.

### 6.2 Connection state (AI server reachability) — drives the trust banner
`connected → degraded → offline → connected`
- `connected`: full model-backed recognition.
- `degraded`: intermittent failures observed (e.g., 2 of last 5 requests failed) — banner shown, recognition continues but flagged lower-trust.
- `offline`: server unreachable beyond a grace period — automatic fallback to local heuristic recognition *or* a clear prompt to use typed/manual input; banner persists until reconnected.
- This state is **derived and owned in one place** (unlike today's `serverOfflineUntil` buried in `lib/islRecognizer.ts` with no UI visibility) and is rendered via `ConnectionIndicator` plus an `aria-live` announcement on every transition.

### 6.3 Capture state (sign channel)
`idle → requesting-permission → live → (paused | error)`
- Independent of Connection state: camera can be `live` while Connection is `degraded`/`offline` — the heuristic fallback still uses live camera frames.

### 6.4 Turn state (conversation orchestration — the genuinely new piece)
`waiting-for-signer ⇄ signer-active ⇄ committing-signer-turn ⇄ waiting-for-partner ⇄ partner-active ⇄ committing-partner-turn ⇄ waiting-for-signer …`
- Transition triggers: hand presence + stable recognition → enters `signer-active`; pause-finalize → `committing-signer-turn` → appends to log → returns to a neutral `waiting` state.
- Partner side: mic press or first keystroke → `partner-active`; mic release + STT-final or Enter/submit on typed input → `committing-partner-turn`.
- This state is **what the Flip affordance reads** — flipping doesn't force a turn, but the UI can suggest it (e.g., gently highlight "Flip for reply" once a signer turn commits).
- Either party can speak/sign out of turn (no hard lock) — this state is advisory for UI emphasis, not an enforced gate, because real conversations interrupt each other.

### 6.5 Recognition confidence state (per in-progress prediction)
`searching → low-confidence → clear → committed`
- Mapped to plain-language labels in the UI ("Searching…", "Not sure — try again", "Clear"), never raw percentages as the primary readout (raw % available as a secondary detail in Settings → Advanced for power users/developers).

### 6.6 Speech states (two independent machines, detailed in §7)
- TTS (signer's output → spoken aloud): `idle → queued → speaking → idle`
- STT (partner's voice → text): `idle → listening → interim → finalizing → idle`

---

## 7. Transcript Flow

The transcript becomes a **structured, ordered log of turns**, not a flat string (today's `transcript: string` + parse/reformat round-trip).

**Turn record concept** (described, not coded): each entry carries — a stable id, a speaker (`signer` or `partner`), the committed text, the recognition/transcription confidence band at commit time, the channel it came from (`sign-recognition` | `speech-recognition` | `typed`), a timestamp, an edited flag, and an optional translated/localized display string (see §9).

**Lifecycle of a signer turn:**
1. Per-frame recognized label → in-progress, rendered only in `LiveCaptionLine` (never in the committed log yet).
2. Stability gate passes (one gate, not two competing ones as today — see §11 frontend changes) → turn is committed: appended to the log, spoken via TTS, `LiveCaptionLine` clears.
3. User may tap any committed bubble to **edit it inline** (new capability — addresses trust: a misread sign shouldn't require restarting). Edited turns are marked (small "edited" tag) so the record stays honest about what the AI actually produced vs. what was corrected.

**Lifecycle of a partner turn:** symmetric — STT interim text shown live (clearly marked "they're typing/speaking…"), finalized text committed as a labeled bubble, also editable by either party (e.g., if STT mis-heard a word).

**Ordering and merge:** turns are appended strictly by commit time, regardless of channel — so a typed reply and a spoken reply interleave correctly with signed turns in one single chronological log. This replaces today's single mutable string with something that can be rendered, exported, and audited as an actual conversation transcript (useful for the Signer's own later reference — e.g., "what exactly did the doctor say").

**Persistence:** turns persist to local storage as a structured list (not a pre-formatted string), so reload-and-restore no longer has to re-parse text back into tokens (today's lossy `loadTranscript(parsed.transcript)` round-trip disappears entirely). Export (copy/download) renders the structured log into plain text or PDF at export time, with speaker labels intact.

**Privacy default:** transcripts are local-only by default; nothing is sent to any server beyond the per-frame landmark vectors and audio needed for live recognition. Saving/exporting is an explicit user action, never automatic.

---

## 8. Speech Flow

Two independent, clearly-separated speech subsystems — today there is only the first one:

### 8.1 Signer → Speech (text-to-speech, existing capability, hardened)
- Trigger: a signer turn commits.
- Queue: one utterance at a time, interruptible by the next commit (kept from today's working `replaceQueue` behavior).
- **Language/voice is a user setting**, not hardcoded `en-IN` — chosen once in Settings, applies until changed.
- Failure behavior: if no TTS voice is available for the chosen language, **say so explicitly** ("Spoken output isn't available in [language] on this device — showing text only") rather than silently substituting a different voice or accent, which is what happens today.
- Visual mirror: whenever speech plays, the corresponding transcript bubble shows a brief "speaking" indicator — so a Signer who can't hear the output still has visual confirmation it played (important: the Signer is often the one who most needs confirmation that the *partner* heard something, since the Signer can't hear it themselves).

### 8.2 Partner → Text (speech-to-text, new capability)
- Trigger: Partner presses-and-holds (or toggles) the mic button in `PartnerReplyDock`.
- Uses the browser's Speech Recognition API where available; interim results stream into `LiveCaptionLine` on the partner-facing view so the partner can see/self-correct while talking (same UX convention as phone dictation, which most partners will already recognize).
- On a natural pause or explicit stop, the result finalizes into a committed turn.
- **Language is independently selectable from the TTS language** — a Signer might want spoken output in Hindi for a partner who speaks Hindi, while that same partner's speech-to-text needs to be configured for the language they're actually speaking, which the system should ask for explicitly the first time (or auto-detect with a confirm-once prompt) rather than assume.
- **Fallback is mandatory, not optional**: `TypedReplyInput` is always visible and usable, regardless of whether Speech Recognition is supported, granted, or working. No conversation can dead-end because a browser lacks an API or a permission was denied.

---

## 9. Backend Changes

Restructuring the model-serving side from "a script someone runs locally" into an actual service:

1. **Containerize and document the runtime.** Ship a Dockerfile and a pinned dependency manifest (`requirements.txt`/`pyproject.toml`) for the inference service — today there is neither, and the run command itself is undocumented.
2. **Add `/health` and `/ready` endpoints**, separate from `/` — readiness should fail loudly if the model didn't load, rather than the process simply never starting with no operator-visible reason.
3. **Re-evaluate the model with a leakage-free split** before trusting any accuracy number: split by source video/recording session (signer-disjoint, session-disjoint), not by post-extraction sequence — the current 100%-across-the-board results (documented in `PROJECT_AUDIT.md` §12) cannot be used as a deployment gate as-is.
4. **Calibrate confidence**, not just report softmax max — plain-language confidence bands in the UI (§6.5) need the underlying number to actually mean something (e.g., temperature scaling or similar), otherwise "Clear" vs "Not sure" is cosmetic.
5. **Serve a `/vocabulary` endpoint** returning the supported sign list with per-language display strings (so the frontend's "What can I sign" reference and the multi-language transcript display, §9.2 below, both read from one server-owned source of truth instead of a hardcoded label file baked into the frontend).
6. **Restrict CORS to known frontend origins** in any non-local deployment — `allow_origins=["*"]` is acceptable for development only.
7. **Add structured logging and minimal metrics** (request count, inference latency, error rate, model version in use) — there is currently no observability at all.
8. **Introduce model versioning**: a registry directory/convention (`models/<task>/<version>/`) with the server reading a configured "active version," so a retrain can be staged and rolled back without manually overwriting the one file currently in use, and so the orphaned `trained_model_4class`/`trained_alnum_model` artifacts either get a real place in this registry or get retired.
9. **Move from per-frame HTTP polling to a streaming connection** (e.g., a persistent connection per active session) for the landmark sequence, reducing the ~280ms-interval HTTP overhead and giving the server backpressure control under load — directly addresses the scalability findings in the audit.
10. **Validate and bound every request** (sequence shape, length, value ranges) before it reaches the model, with clear rejected-request responses — today malformed/oversized input is only caught indirectly by a shape-mismatch error deep in `predict_sequence`.
11. **Speech-to-text for the partner channel** can remain entirely client-side (browser Speech Recognition API) — no new backend component required for that channel, keeping the backend's scope limited to the sign-recognition model it already owns.
12. **Decide, deliberately, on one inference location.** Either commit to server-based inference as the sole path (simpler, recommended primary), or finish the already-half-built TFJS export pipeline specifically as an **offline-degraded-mode fallback model** (smaller/distilled, lower accuracy acceptable) that runs in-browser only when the server is confirmed unreachable — this turns currently-abandoned infrastructure (`export_tfjs_model.py`, the unused `@tensorflow/tfjs` dependency) into a deliberate stability feature instead of dead weight.

---

## 10. Frontend Changes

1. **Decompose the monolith** into the component tree in §5 — independent error boundaries around the camera dock and the reply dock are the single highest-leverage stability change, since today one thrown error inside the detection loop can take down the entire page.
2. **Delete duplicated logic, keep one source of truth**: retire the inline `SentenceEngine` copy and the unused `lib/sentenceEngine.ts`/`lib/modelAdapter.ts` in favor of one maintained transcript/turn engine that both the sign channel and the new speech-to-text channel feed into.
3. **Collapse the two independent stability gates** (today: one in the recognizer, one in the sentence engine, different thresholds) into a single, named "commit confidence" pipeline stage, so there is exactly one place that decides "this is trustworthy enough to become a transcript turn."
4. **Introduce explicit state machines** for Session/Connection/Capture/Turn (§6) instead of 16 loosely related `useState` calls and several module-level mutable globals — this is what makes the Connection-state banner, the Turn-state-driven Flip suggestion, and independent dock error recovery actually implementable cleanly.
5. **Add internationalization** for all UI chrome (buttons, labels, banners, onboarding) via a standard i18n approach, decoupled from the TTS/STT language choices (§9 below) — today there is zero i18n and one hardcoded speech language.
6. **Build the Vocabulary Reference and Settings sheets** as the first-class trust features they are, sourced from the backend `/vocabulary` endpoint (§9.5) rather than left undiscoverable as today.
7. **Make recognized text editable** in the transcript log (new capability, §7) — this is a relatively small frontend change with an outsized trust payoff.
8. **Accessibility is implemented as part of every component**, not bolted on (full spec in §11) — live regions on `LiveCaptionLine` and the Connection banner, visible focus states everywhere, adjustable text size as a real Settings control, and a verified full-keyboard path through Start → sign/Type reply → Flip → End.
9. **Resolve the repo-level cleanup items** identified in the audit as part of this redesign's groundwork: remove the duplicate 7.5 MB `hand_landmarker.task` (keep one canonical copy), remove `public.zip`, and either delete or properly source-control whatever produced `trained_alnum_model` before it's wired into anything.
10. **Document the frame/feature contract once**, in one place both the frontend and the Python services reference by version number (e.g., "landmark-feature-v1: 30×126, left-then-right, zero-padded") — today this contract exists only as matching magic numbers independently typed into both languages.

---

## 11. Accessibility Specification

- **Live regions**: `LiveCaptionLine` and the Connection-state banner are `aria-live="polite"` (or `assertive` for connection loss / camera error specifically); every committed turn bubble is announced once via the log's live region, not on every re-render.
- **Plain-language confidence**, not raw percentages, as the default UI; raw numbers available only in an opt-in "Advanced" settings view.
- **Full keyboard path**: every control in the Control Bar, both docks, and both modal sheets reachable and operable via keyboard alone, with a visible focus ring (not just default browser outline suppressed with nothing replacing it, as is the current risk with custom-styled buttons).
- **Adjustable text size** as a real Settings control (not just relying on browser zoom), since legibility at a distance/across a counter is core to this product's actual use case, not a generic nice-to-have.
- **High-contrast and reduced-motion modes**, the latter disabling non-essential transition animations (e.g., the Flip transition) for users sensitive to motion.
- **Captioned everything, never color-only**: connection/camera/recognition states always carry a text label alongside any color or icon.
- **Speaker attribution is explicit text** ("You signed:" / "They said:"), never relying on bubble color or alignment alone to convey who said what — critical given the device gets physically handed between two people with very different visual contexts (camera angle, glare, etc.).
- **Screen-reader pass required before ship**: a manual VoiceOver/NVDA/TalkBack pass through the full conversation flow (start → sign → flip → reply → end) should be a release gate, not a someday item — there is currently zero ARIA implementation to build from.

---

## 12. Multi-Language Plan

Four independently configurable language settings, deliberately not collapsed into one "app language":

1. **Interface language** — menus, buttons, onboarding, settings copy. Standard i18n, extensible word-list, starting with English + Hindi and structured to add further Indian regional languages without re-architecture.
2. **TTS output language/voice** (signer → partner) — chosen by the Signer for whoever they expect to be talking to; explicit fallback messaging if unavailable on-device (§7.1), never a silent substitution.
3. **STT input language** (partner → text) — chosen or confirmed by the partner at the start of their first reply; independent of the TTS setting, since the partner's spoken language and the Signer's preferred output language need not match.
4. **Transcript display language** — the committed text for sign-recognized turns can show a backend-served localized string per recognized label (from the `/vocabulary` mapping, §9.5), so the same recognized sign can render as "thank you" or "धन्यवाद" depending on who's currently meant to read it, without retraining or changing the underlying recognition model (the sign vocabulary itself is fixed; only its textual rendering is localized).

This separation is what makes a single conversation between, say, an ISL-fluent Signer and a Hindi-speaking partner actually work without forcing one party to use the other's preferred language for everything.

---

## 13. Stability & Trust

**Tiered degradation — there is no failure mode that fully blocks communication:**

```
Full recognition (server connected)
        ↓ server degraded/offline
Heuristic-only / cached fallback recognition (clearly labeled "Limited mode")
        ↓ camera/recognition unusable at all
Manual typed input (always available, on both Sign-side and Partner-side)
```

**Trust mechanisms, concretely:**
- Vocabulary is discoverable before it's needed (onboarding mention + persistent reference button), not discovered only through trial and error.
- Confidence is communicated in plain language, calibrated against a credible (re-evaluated, leakage-free) accuracy baseline.
- Every AI-produced turn is editable, and edits are visibly marked — the transcript never pretends to be more authoritative than it is.
- Privacy posture is stated, not assumed: a persistent, one-tap-away explanation of what is processed locally, what is sent to a server, what is stored, and what an export actually contains.
- Connection state is always visible (§6.2) — silence about server health, as exists today, is itself a trust failure: a user should never wonder *why* recognition suddenly got worse.
- No conversation-ending dead ends: camera failure, mic permission denial, missing TTS voice, or AI server outage each have a defined, visible, immediately-usable fallback rather than a stuck or broken UI.

---

## 14. Suggested Phasing (sequencing only, not a timeline commitment)

1. **Foundation**: decompose the monolith into the component tree (§5), introduce the state machines (§6), delete the duplicated/dead code, document the frame/feature contract — no new user-facing capability yet, but everything after this is safer to build on top of.
2. **Trust & stability baseline**: tiered degradation, plain-language confidence, vocabulary reference, editable transcript turns, accessibility pass (§11).
3. **The second channel**: Partner Reply Dock (STT + typed fallback), structured turn-based transcript (§7), Conversation Mode flip affordance.
4. **Multi-language**: i18n for UI chrome, independent TTS/STT/display language settings, backend `/vocabulary` localization.
5. **Backend hardening for real deployment**: containerization, health checks, model versioning, leakage-free re-evaluation, observability, CORS lockdown, streaming transport.

Each phase is independently shippable and testable — none requires the others to deliver value, which matters given today's starting point has none of this in place.
