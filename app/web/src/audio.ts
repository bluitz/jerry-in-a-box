// Mic + AudioWorklet wiring.
//
// Owns the AudioContext lifecycle. Posts NoteEvents and ChromaEvents
// out of the worklet to the caller via callbacks.

export type NoteEvent = {
  type: "note";
  midi: number; pc: number; freq: number; confidence: number; rms: number; t: number;
};
export type ChromaEvent = {
  type: "chroma";
  chroma: Float32Array; rms: number; t: number;
};
export type OnsetEvent = {
  type: "onset";
  rms: number; t: number;
};
export type BpmEvent = {
  type: "bpm";
  bpm: number;     // estimated BPM, mapped into [60, 180]
  confidence: number;  // 0..1, from IOI variance
  t: number;
};

export type AudioCallbacks = {
  onNote: (e: NoteEvent) => void;
  onChroma: (e: ChromaEvent) => void;
  onOnset?: (e: OnsetEvent) => void;
  onBpm?: (e: BpmEvent) => void;
  onError: (msg: string) => void;
};

export class AudioPipeline {
  private ctx?: AudioContext;
  private source?: MediaStreamAudioSourceNode;
  private worklet?: AudioWorkletNode;
  private stream?: MediaStream;
  private cb: AudioCallbacks;
  // BPM estimation state
  private onsetTimes: number[] = [];     // recent onset timestamps (seconds)
  private static ONSET_HISTORY = 12;     // keep last N onsets
  private lastBpm = 100;                 // last emitted BPM, for hysteresis

  constructor(cb: AudioCallbacks) {
    this.cb = cb;
  }

  async start(): Promise<void> {
    try {
      // Create the AudioContext only after a user gesture (Safari requirement).
      // The browser will pick a sample rate (usually 44100/48000).
      this.ctx = new (window.AudioContext || (window as any).webkitAudioContext)({
        latencyHint: "interactive",
      });

      // The worklet is plain JS in /public so Vite copies it as-is to the
      // dist root; the browser fetches it from /yin-worklet.js.
      await this.ctx.audioWorklet.addModule("/yin-worklet.js");

      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          // Disable processing that would mess with pitch detection
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl:  false,
          channelCount:     1,
        },
        video: false,
      });

      this.source = this.ctx.createMediaStreamSource(this.stream);
      this.worklet = new AudioWorkletNode(this.ctx, "yin-detector");
      this.worklet.port.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === "note")        this.cb.onNote(m as NoteEvent);
        else if (m.type === "chroma") this.cb.onChroma(m as ChromaEvent);
        else if (m.type === "onset")  this.handleOnset(m as OnsetEvent);
      };
      // Connect: mic -> worklet (worklet doesn't output; it just analyzes).
      // We still need to terminate the graph; route worklet to a muted gain node.
      const sink = this.ctx.createGain();
      sink.gain.value = 0;
      this.source.connect(this.worklet);
      this.worklet.connect(sink);
      sink.connect(this.ctx.destination);
    } catch (err) {
      this.cb.onError(err instanceof Error ? err.message : String(err));
      throw err;
    }
  }

  stop(): void {
    try { this.worklet?.disconnect(); } catch {}
    try { this.source?.disconnect(); } catch {}
    try { this.stream?.getTracks().forEach(t => t.stop()); } catch {}
    try { this.ctx?.close(); } catch {}
    this.worklet = undefined;
    this.source = undefined;
    this.stream = undefined;
    this.ctx = undefined;
  }

  get isActive(): boolean { return !!this.ctx; }
  get sampleRate(): number { return this.ctx?.sampleRate ?? 0; }

  // ---- Onset / BPM tracking ----
  // Onsets typically correspond to strums. For 4/4 strumming patterns the
  // median inter-onset interval (IOI) is usually one beat (1 strum/beat) or
  // half a beat (down-up). We map the median IOI to BPM and fold it into
  // the conventional [60, 180] range by halving/doubling.
  private handleOnset(e: OnsetEvent): void {
    this.cb.onOnset?.(e);
    this.onsetTimes.push(e.t);
    if (this.onsetTimes.length > AudioPipeline.ONSET_HISTORY) {
      this.onsetTimes.shift();
    }
    if (this.onsetTimes.length < 4) return;

    const iois: number[] = [];
    for (let i = 1; i < this.onsetTimes.length; i++) {
      iois.push(this.onsetTimes[i] - this.onsetTimes[i - 1]);
    }
    const sorted = iois.slice().sort((a, b) => a - b);
    const medianIoi = sorted[Math.floor(sorted.length / 2)];
    if (medianIoi <= 0) return;

    let bpm = 60.0 / medianIoi;
    while (bpm > 180) bpm /= 2;
    while (bpm < 60) bpm *= 2;
    if (bpm < 60 || bpm > 180) return; // bail on degenerate values

    // Confidence from IOI dispersion (low spread => high confidence).
    const mean = iois.reduce((a, b) => a + b, 0) / iois.length;
    let varSum = 0;
    for (const v of iois) varSum += (v - mean) * (v - mean);
    const std = Math.sqrt(varSum / iois.length);
    const cov = mean > 0 ? std / mean : 1;     // coefficient of variation
    const confidence = Math.max(0, Math.min(1, 1 - cov));

    // Light hysteresis: smooth toward new estimate.
    this.lastBpm = 0.7 * this.lastBpm + 0.3 * bpm;
    this.cb.onBpm?.({ type: "bpm", bpm: this.lastBpm, confidence, t: e.t });
  }
}

export const PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;
