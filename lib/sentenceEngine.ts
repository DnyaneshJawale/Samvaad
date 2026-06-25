// lib/sentenceEngine.ts

export type SentenceEngineProcessOptions = {
  now?: number;
  autoCommitEnabled?: boolean;
  forceCommit?: boolean;
};

export type SentenceEngineResult = {
  liveWord: string;
  committedSentence: string;
  shouldSpeak: boolean;
  speechText: string;
  finalizedSentence: string;
  pendingWord: string;
  isFinalized: boolean;
};

type ResolvedSentenceEngineOptions = {
  stableThreshold: number;
  commitHoldMs: number;
  pauseFinalizeMs: number;
  repeatSuppressionMs: number;
  noiseLabels: string[];
};

const DEFAULT_OPTIONS: ResolvedSentenceEngineOptions = {
  stableThreshold: 4,
  commitHoldMs: 350,
  pauseFinalizeMs: 2000,
  repeatSuppressionMs: 900,
  noiseLabels: ["", "—", "Searching...", "Searching.", "No hand detected"],
};

const TERMINAL_PUNCTUATION = new Set([".", "!", "?"]);

function normalizeText(value: string): string {
  return String(value ?? "").trim().replace(/\s+/g, " ");
}

function isTerminalPunctuation(token: string): boolean {
  return TERMINAL_PUNCTUATION.has(token);
}

function isNoiseLabel(value: string, noiseLabels: string[]): boolean {
  const normalized = normalizeText(value).toLowerCase();
  if (!normalized) return true;
  return noiseLabels.some((label) => normalized === label.toLowerCase());
}

function parseTranscript(text: string): string[] {
  const cleaned = normalizeText(text);
  if (!cleaned) return [];

  const tokens: string[] = [];

  for (const part of cleaned.split(/\s+/)) {
    const match = part.match(/^(.+?)([.!?]+)$/);
    if (match) {
      if (match[1]) tokens.push(match[1]);
      tokens.push(...match[2].split(""));
      continue;
    }
    tokens.push(part);
  }

  return tokens.filter(Boolean);
}

function formatTokens(tokens: string[]): string {
  const out: string[] = [];

  for (const token of tokens) {
    if (!token) continue;

    if (isTerminalPunctuation(token)) {
      if (out.length > 0) {
        out[out.length - 1] = `${out[out.length - 1]}${token}`;
      }
      continue;
    }

    out.push(token);
  }

  return out.join(" ").replace(/\s+/g, " ").trim();
}

function capitalizeWord(word: string): string {
  const normalized = normalizeText(word);
  if (!normalized) return "";

  const lower = normalized.toLowerCase();
  if (lower === "i") return "I";

  return lower.charAt(0).toUpperCase() + lower.slice(1);
}

function formatWordForSentence(word: string, sentenceStart: boolean): string {
  const normalized = normalizeText(word);
  if (!normalized) return "";

  if (isTerminalPunctuation(normalized)) return normalized;

  if (/^[A-Z0-9]+$/.test(normalized) && normalized.length <= 5) {
    return normalized;
  }

  const lower = normalized.toLowerCase();
  return sentenceStart ? capitalizeWord(lower) : lower;
}

export class SentenceEngine {
  private readonly options: ResolvedSentenceEngineOptions;
  private transcriptTokens: string[] = [];
  private pendingWord = "";
  private pendingSince = 0;
  private pendingCount = 0;
  private lastCommittedWord = "";
  private lastCommittedAt = 0;
  private lastValidAt = 0;
  private lastFinalizedText = "";

  constructor(options: Partial<ResolvedSentenceEngineOptions> = {}) {
    this.options = {
      ...DEFAULT_OPTIONS,
      ...options,
      noiseLabels: (options.noiseLabels ?? DEFAULT_OPTIONS.noiseLabels).map((label) =>
        normalizeText(label)
      ),
    };
  }

  loadTranscript(text: string): void {
    this.transcriptTokens = parseTranscript(text);
    this.pendingWord = "";
    this.pendingSince = 0;
    this.pendingCount = 0;
    this.lastCommittedWord = this.getLastContentWord().toLowerCase();
    this.lastCommittedAt = 0;
    this.lastValidAt = 0;
    this.lastFinalizedText = this.endsWithTerminalPunctuation()
      ? this.getTranscript()
      : "";
  }

  reset(): void {
    this.transcriptTokens = [];
    this.pendingWord = "";
    this.pendingSince = 0;
    this.pendingCount = 0;
    this.lastCommittedWord = "";
    this.lastCommittedAt = 0;
    this.lastValidAt = 0;
    this.lastFinalizedText = "";
  }

  clear(): void {
    this.reset();
  }

  getTranscript(): string {
    return formatTokens(this.transcriptTokens);
  }

  getFinalizedSentence(): string {
    return this.lastFinalizedText;
  }

  backspace(): string {
    if (!this.transcriptTokens.length) {
      this.pendingWord = "";
      this.pendingSince = 0;
      this.pendingCount = 0;
      return "";
    }

    this.transcriptTokens.pop();

    this.pendingWord = "";
    this.pendingSince = 0;
    this.pendingCount = 0;
    this.lastCommittedWord = this.getLastContentWord().toLowerCase();
    this.lastCommittedAt = 0;
    this.lastFinalizedText = "";

    return this.getTranscript();
  }

  processPrediction(
    rawWord: string,
    options: SentenceEngineProcessOptions = {}
  ): SentenceEngineResult {
    const now = options.now ?? Date.now();
    const autoCommitEnabled = options.autoCommitEnabled ?? true;
    const forceCommit = options.forceCommit ?? false;
    const cleaned = normalizeText(rawWord);

    if (isNoiseLabel(cleaned, this.options.noiseLabels)) {
      const finalizedSentence = this.maybeFinalizeOnPause(now);
      const liveWord = this.pendingWord || this.getLastContentWord() || "—";

      return {
        liveWord,
        committedSentence: this.getTranscript(),
        shouldSpeak: Boolean(finalizedSentence),
        speechText: finalizedSentence,
        finalizedSentence,
        pendingWord: this.pendingWord,
        isFinalized: Boolean(finalizedSentence),
      };
    }

    this.lastValidAt = now;
    this.lastFinalizedText = "";

    if (cleaned === this.pendingWord) {
      this.pendingCount += 1;
    } else {
      this.pendingWord = cleaned;
      this.pendingSince = now;
      this.pendingCount = 1;
    }

    let shouldSpeak = false;
    let speechText = "";

    if (forceCommit || (autoCommitEnabled && this.shouldCommit(now))) {
      const committed = this.commitWord(this.pendingWord, now);
      if (committed) {
        shouldSpeak = true;
        speechText = committed;
      }
    }

    return {
      liveWord: this.pendingWord || this.getLastContentWord() || "—",
      committedSentence: this.getTranscript(),
      shouldSpeak,
      speechText,
      finalizedSentence: "",
      pendingWord: this.pendingWord,
      isFinalized: false,
    };
  }

  private shouldCommit(now: number): boolean {
    if (!this.pendingWord) return false;
    if (this.pendingCount < this.options.stableThreshold) return false;
    return now - this.pendingSince >= this.options.commitHoldMs;
  }

  private commitWord(word: string, now: number): string {
    const cleaned = normalizeText(word);
    if (!cleaned || isNoiseLabel(cleaned, this.options.noiseLabels)) return "";

    const normalized = cleaned.toLowerCase();
    if (
      normalized === this.lastCommittedWord &&
      now - this.lastCommittedAt < this.options.repeatSuppressionMs
    ) {
      return "";
    }

    const formatted = formatWordForSentence(cleaned, this.isSentenceStart());
    if (!formatted) return "";

    this.transcriptTokens.push(formatted);

    this.lastCommittedWord = normalized;
    this.lastCommittedAt = now;
    this.pendingWord = "";
    this.pendingSince = 0;
    this.pendingCount = 0;
    this.lastFinalizedText = "";

    return formatted;
  }

  private maybeFinalizeOnPause(now: number): string {
    if (!this.transcriptTokens.length) return "";
    if (this.lastValidAt === 0) return "";
    if (now - this.lastValidAt < this.options.pauseFinalizeMs) return "";

    const current = this.getTranscript();
    if (!current) return "";
    if (current === this.lastFinalizedText) return "";

    if (!this.endsWithTerminalPunctuation()) {
      this.transcriptTokens.push(".");
    }

    const finalized = this.getTranscript();
    this.lastFinalizedText = finalized;
    this.pendingWord = "";
    this.pendingSince = 0;
    this.pendingCount = 0;

    return finalized;
  }

  private isSentenceStart(): boolean {
    if (this.transcriptTokens.length === 0) return true;
    return isTerminalPunctuation(
      this.transcriptTokens[this.transcriptTokens.length - 1]
    );
  }

  private endsWithTerminalPunctuation(): boolean {
    if (!this.transcriptTokens.length) return false;
    return isTerminalPunctuation(
      this.transcriptTokens[this.transcriptTokens.length - 1]
    );
  }

  private getLastContentWord(): string {
    for (let i = this.transcriptTokens.length - 1; i >= 0; i -= 1) {
      const token = this.transcriptTokens[i];
      if (!isTerminalPunctuation(token)) return token;
    }
    return "";
  }
}