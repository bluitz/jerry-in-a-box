# Jerry in a Box — webapp

Web app that listens to a guitar (via the laptop / iPad mic), feeds the
noisy single-pitch stream from a guitar-tuner-style detector into a
Bayesian song matcher, ranks the top-5 candidate songs from the Jerry
Garcia Songbook, and when confident, jumps to the matched page in the
PDF.

The matcher's job is *disambiguation* — the tuner is monophonic and
fires false notes that are mostly chord tones / in-key — so every layer
of the system is built around that constraint.

## Layout

```
app/
  parser/      Build canonical songs.json + page_index.json from the songbook
               CSV and PDF. No lyrics or PDF body text are stored — only
               chord-symbol sequences (factual notation), titles, artist,
               section labels (e.g. "Verse 1"), page numbers, and computed
               numerical features (pc histograms, chord bigrams, key).
  matcher/     Online Bayesian matcher. Bag-of-notes log-likelihood +
               per-song HMM over chord positions, combined log-linearly.
               Pure NumPy, no I/O at update time.
  server/      FastAPI app with REST + WebSocket. Serves the user's local
               PDF for in-browser rendering by pdf.js (no text extraction).
  web/         Vite + TypeScript frontend. AudioWorklet runs YIN pitch
               detection + a 12-bin chroma in the audio thread; the main
               thread streams notes over WS and renders the top-5 + PDF.
  tests/
    unit/        pytest suite (chord parser, matcher behavior, server)
    synthetic/   Trace generator + eval harness
    fixtures/    Generated traces (easy / medium / hard)
```

## Run it locally

```bash
# 1) Backend
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy pymupdf fastapi uvicorn websockets pytest httpx
python -m app.parser.songdb       # build app/data/songs.json
python -m app.parser.page_index   # build app/data/page_index.json

# 2) Frontend
cd app/web && npm install && npm run build && cd ../..

# 3) Serve (dev)
python -m uvicorn app.server.main:app --host 127.0.0.1 --port 8000
# Open http://127.0.0.1:8000  -- tap Start, allow mic, play.
```

For frontend dev with hot reload:
```bash
cd app/web && npm run dev   # serves at :5173 with proxy to :8000
```

## How the matcher works

For each song `s`, we maintain two log-likelihoods updated per observation:

1. **Bag-of-notes**: `log P(o₁..t | s) = Σᵢ wᵢ · log P(pcᵢ | s)`
   where `P(pc | s) = (1−ε)·hₛ(pc) + ε/12` and `hₛ` is the song's
   chord-tone histogram. `wᵢ` is the YIN per-frame confidence.

2. **HMM** over chord positions:
   - states = collapsed chord vocabulary of `s`
   - transitions = the song's bigram (Laplace-smoothed) blended with a
     self-loop so the matcher tolerates dwelling on one chord
   - emission `P(pc | chord)` favors chord tones, then in-key, then random.

Combined log-linearly with weights `α=0.30` / `1−α=0.70` plus a uniform
prior, then softmax over songs.

A **decision** ("we're confident enough to jump to the page") fires only
when all of:
- `n_obs ≥ 24`
- `top1 ≥ 0.92`
- `top1 / top2 ≥ 4`
- sustained for `≥ 10` consecutive updates (~1 sec)

The thresholds were picked by a sweep on the synthetic eval set
(see `app/tests/synthetic/evaluate.py`). They favor *precision* over
*recall*: when the matcher commits, it's right ~95% of the time.

## Evaluation

```bash
python -m app.tests.synthetic.generate --preset easy
python -m app.tests.synthetic.generate --preset medium
python -m app.tests.synthetic.generate --preset hard
python -m app.tests.synthetic.evaluate
```

Current numbers (72 songs × 1 trace per preset):

| dataset | top-1 final | top-5 final | decisions | precision | wrong |
|---------|-------------|-------------|-----------|-----------|-------|
| easy    | 0.81        | 0.94        | 26 / 72   | 84.6%     | 4     |
| medium  | 0.63        | 0.86        | 25 / 72   | 96.0%     | 1     |
| hard    | 0.47        | 0.82        | 14 / 72   | 92.9%     | 1     |

Top-5 hit rate is the metric the user actually sees (the UI shows top-5);
top-1 final is dragged down by legitimate same-key ambiguity (e.g. Box
of Rain vs Loser, Ripple vs Wild Horses) which the synthetic generator
fairly produces.

## Synthetic generator

Models a competent guitarist + a noisy monophonic tuner:

- per-emission probabilities: `p_root`, `p_chord_tone`, `p_bass`,
  `p_in_key`, `p_random`
- player-side noise: `p_chord_skip`, `p_chord_repeat`,
  `p_sustain_bleed`, `p_dropped_first`
- confidence draws are Gaussian with separate `chord` vs `wrong` means
  (so wrong notes show up with lower confidence — the matcher gets to
  use that)
- presets: `easy`, `medium`, `hard` map to coherent knob settings

## Tests

```bash
python -m pytest app/tests/unit/ -v
# 22 tests: chord parser, matcher behavior, server REST + WebSocket
```

### Speaker -> mic round-trip test (manual)

The unit tests cover the file -> matcher path only. To verify the FULL
end-to-end pipeline including the laptop speakers and mic, there is
a manual test that plays a reference recording through the system
speakers, captures it with the default mic, and checks that the
matcher ranks the expected song highly.

```bash
pip install sounddevice  # only needed for this test

# 1) Calibration (5s) -- confirm the mic actually hears the speakers
python -m app.tools.speaker_mic_test --check-only

# 2) Full 35s test, default file is test-audio/Friend of the devil.m4a,
#    expected target is "friend-of-the-devil"
python -m app.tools.speaker_mic_test \
    --duration 35 --rank-threshold 20 \
    --save-wav /tmp/jerry-mic-capture.wav

# Or via pytest (gated behind an env var so it doesn't run in normal CI)
JERRY_RUN_SPEAKER_MIC=1 pytest -s app/tests/manual/
```

Requires macOS (uses `afplay`) and a microphone. The first run will
trigger a system permission prompt — Terminal/iTerm needs Microphone
permission in System Settings > Privacy & Security > Microphone. If
the calibration step reports RMS below ~0.005 the mic isn't picking
up the speakers; raise system output volume and confirm the right
output device is active.

## What's NOT in here

- No song lyrics, prose, or other PDF body text is extracted, stored, or
  displayed via the data path. The PDF is only ever shown to the user by
  pdf.js rendering their own copy of their own file.
- No ML / training: pure DSP + Bayesian inference, all deterministic.
- No external API calls at runtime.
