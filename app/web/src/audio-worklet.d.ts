// Type declarations for the AudioWorklet global scope.
// (Only needed for files that run inside an AudioWorklet, like yin-worklet.ts.)

declare const sampleRate: number;
declare const currentTime: number;
declare const currentFrame: number;

declare class AudioWorkletProcessor {
  readonly port: MessagePort;
  constructor(options?: AudioWorkletNodeOptions);
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>,
  ): boolean;
}

declare function registerProcessor(
  name: string,
  ctor: new (options?: AudioWorkletNodeOptions) => AudioWorkletProcessor,
): void;
