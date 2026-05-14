// AudioWorklet pitch + chroma detector. Runs on the audio thread.
//
// - Accumulates 2048 samples, hops every 512.
// - YIN (cumulative-mean-normalized difference + parabolic interp) for monophonic pitch.
// - 12-bin chroma from a magnitude FFT every hop.
// - RMS gate, aperiodicity gate, median-of-3 pc smoother.
//
// Posts to main thread:
//   { type:"note",   midi, pc, freq, confidence, rms, t }
//   { type:"chroma", chroma, rms, t }
//
// Plain JS (not TS) so it ships as-is to the browser; AudioWorklet
// modules must be served as a real JS file.

const FRAME = 2048;
const HOP = 512;
const RMS_GATE = 0.01;
const YIN_THRESHOLD = 0.15;
const F_MIN = 70;
const F_MAX = 1500;
const SMOOTH_N = 3;

class YinDetector extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(FRAME);
    this.writeIdx = 0;
    this.samplesSinceLast = 0;
    this.pcHistory = [];
    this.startTime = currentTime;
    this.d = new Float32Array(FRAME / 2);
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const n = ch.length;
    for (let i = 0; i < n; i++) {
      this.buf[this.writeIdx] = ch[i];
      this.writeIdx = (this.writeIdx + 1) % FRAME;
    }
    this.samplesSinceLast += n;
    if (this.samplesSinceLast < HOP) return true;
    this.samplesSinceLast = 0;

    // Build a contiguous oldest->newest frame.
    const frame = new Float32Array(FRAME);
    const start = this.writeIdx;
    for (let i = 0; i < FRAME; i++) {
      frame[i] = this.buf[(start + i) % FRAME];
    }

    // RMS gate.
    let sum = 0;
    for (let i = 0; i < FRAME; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / FRAME);
    if (rms < RMS_GATE) {
      this.pcHistory.length = 0;
      return true;
    }

    const tNow = currentTime - this.startTime;

    // Chroma every hop, regardless of pitch detectability.
    const chroma = computeChroma(frame, sampleRate);
    this.port.postMessage({ type: "chroma", chroma, rms, t: tNow });

    // YIN
    const { freq, aperiodicity } = yin(frame, sampleRate, this.d);
    if (!freq || aperiodicity > YIN_THRESHOLD) return true;
    if (freq < F_MIN || freq > F_MAX) return true;

    const midi = Math.round(69 + 12 * Math.log2(freq / 440));
    const pc = ((midi % 12) + 12) % 12;

    this.pcHistory.push(pc);
    if (this.pcHistory.length > SMOOTH_N) this.pcHistory.shift();
    const pcSmoothed = median(this.pcHistory);

    const confidence = Math.max(0, Math.min(1, 1 - aperiodicity));

    this.port.postMessage({
      type: "note",
      midi, pc: pcSmoothed, freq, confidence, rms, t: tNow,
    });

    return true;
  }
}

function median(arr) {
  if (arr.length === 0) return 0;
  const a = arr.slice().sort((x, y) => x - y);
  return a[Math.floor(a.length / 2)];
}

function yin(buf, sr, d) {
  const N = buf.length;
  const W = N >>> 1;
  for (let tau = 0; tau < W; tau++) {
    let s = 0;
    for (let i = 0; i < W; i++) {
      const diff = buf[i] - buf[i + tau];
      s += diff * diff;
    }
    d[tau] = s;
  }
  d[0] = 1;
  let running = 0;
  for (let tau = 1; tau < W; tau++) {
    running += d[tau];
    d[tau] = running > 0 ? (d[tau] * tau) / running : 1;
  }
  let tau = -1;
  for (let i = 2; i < W - 1; i++) {
    if (d[i] < YIN_THRESHOLD) {
      let j = i;
      while (j + 1 < W - 1 && d[j + 1] < d[j]) j++;
      tau = j;
      break;
    }
  }
  if (tau < 0) {
    let bestI = 2, bestV = d[2];
    for (let i = 3; i < W; i++) if (d[i] < bestV) { bestV = d[i]; bestI = i; }
    tau = bestI;
  }
  let tauRefined = tau;
  if (tau > 0 && tau < W - 1) {
    const s0 = d[tau - 1], s1 = d[tau], s2 = d[tau + 1];
    const denom = 2 * (2 * s1 - s2 - s0);
    if (denom !== 0) tauRefined = tau + (s2 - s0) / denom;
  }
  if (tauRefined <= 0) return { freq: 0, aperiodicity: 1 };
  return { freq: sr / tauRefined, aperiodicity: d[tau] };
}

function computeChroma(buf, sr) {
  const N = buf.length;
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  // Hann window into re[]
  for (let i = 0; i < N; i++) {
    re[i] = buf[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)));
  }
  fft(re, im);
  const chroma = new Float32Array(12);
  const minBin = Math.max(1, Math.floor((F_MIN * N) / sr));
  const maxBin = Math.min(N / 2 - 1, Math.ceil((F_MAX * N) / sr));
  for (let k = minBin; k <= maxBin; k++) {
    const f = (k * sr) / N;
    const midiF = 69 + 12 * Math.log2(f / 440);
    const pc = ((Math.round(midiF) % 12) + 12) % 12;
    chroma[pc] += Math.hypot(re[k], im[k]);
  }
  let sum = 0;
  for (let i = 0; i < 12; i++) sum += chroma[i];
  if (sum > 0) for (let i = 0; i < 12; i++) chroma[i] /= sum;
  return chroma;
}

// Iterative radix-2 Cooley-Tukey FFT (in-place). N must be a power of 2.
function fft(re, im) {
  const N = re.length;
  let j = 0;
  for (let i = 0; i < N - 1; i++) {
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
    let m = N >> 1;
    while (m >= 1 && j >= m) { j -= m; m >>= 1; }
    j += m;
  }
  for (let size = 2; size <= N; size <<= 1) {
    const half = size >> 1;
    const tableStep = (-2 * Math.PI) / size;
    for (let i = 0; i < N; i += size) {
      for (let k = 0; k < half; k++) {
        const angle = tableStep * k;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const ti = i + k;
        const tj = i + k + half;
        const tre = re[tj] * cos - im[tj] * sin;
        const tim = re[tj] * sin + im[tj] * cos;
        re[tj] = re[ti] - tre;
        im[tj] = im[ti] - tim;
        re[ti] += tre;
        im[ti] += tim;
      }
    }
  }
}

registerProcessor("yin-detector", YinDetector);
