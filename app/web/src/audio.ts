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

export type AudioCallbacks = {
  onNote: (e: NoteEvent) => void;
  onChroma: (e: ChromaEvent) => void;
  onError: (msg: string) => void;
};

export class AudioPipeline {
  private ctx?: AudioContext;
  private source?: MediaStreamAudioSourceNode;
  private worklet?: AudioWorkletNode;
  private stream?: MediaStream;
  private cb: AudioCallbacks;

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
        if (m.type === "note")   this.cb.onNote(m as NoteEvent);
        else if (m.type === "chroma") this.cb.onChroma(m as ChromaEvent);
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
}

export const PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"] as const;
