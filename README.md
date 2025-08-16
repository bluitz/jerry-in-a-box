## Jerry in a Box

An AI‑assisted guitar pedal built on a Raspberry Pi that listens to your playing, recognizes the chords in real time, looks at the last few you played, and figures out which Jerry Garcia/Grateful Dead song you’re likely playing. It then shows the chord sequence on an LED/OLED screen so you can jam without stopping to check a chart.

### Why this exists

I love building end‑to‑end systems that mix hardware, DSP, and AI. This project is a hobby build aimed at showcasing product engineering craft for roles at companies like OpenAI and Anthropic: tight feedback loops, pragmatic ML, delightful UX, and shipping on real hardware.

## What it does

- **Real‑time chord detection**: Streams audio in, performs FFT/chroma analysis, and detects the most likely chord/note with a confidence score.
- **Rolling progression memory**: Keeps track of your last few chords to form a short progression window.
- **Song identification**: Compares the window against a curated Garcia/Grateful Dead progression database and can use an AI assist to choose the best match when multiple songs are plausible.
- **On‑pedal display**: Shows current chord, recent chords, best‑match songs, and the next likely chords on a small LED/OLED screen.
- **Hands‑free questions (optional)**: Ask music‑theory questions via mic ("What scale over this?"), answered with OpenAI.

## How it works (signal flow)

1. Audio In (guitar/DI/mic) → USB audio interface → Raspberry Pi.
2. Streaming DSP: windowing + zero‑padded FFT → peak picking → chroma vector.
3. Chord estimation: template matching (major/minor/7/maj7/power) + consistency smoothing.
4. Progression window: maintain last N chords; compare against song progressions in `jerry_in_a_box/data/songs.json`.
5. AI assist: when there are several high‑confidence candidates, call OpenAI to adjudicate using context about Garcia progressions and your recent chords.
6. Display: render current chord, last N, top matches, and predicted next chords on the pedal’s screen.

## Quickstart

### Prerequisites

- Python 3.10+
- On macOS: `brew install portaudio` (for mic/line‑in), and allow microphone access.
- On Raspberry Pi (Debian/Ubuntu): `sudo apt-get update && sudo apt-get install -y python3-venv libportaudio2 libatlas-base-dev`.
- An audio input (USB interface is ideal) and an OLED/LED display (e.g., SSD1306) if running on pedal hardware.

### Setup

```bash
git clone https://github.com/yourname/jerry-in-a-box.git
cd jerry-in-a-box
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# If needed for your platform
pip install numpy scipy sounddevice

# OpenAI (optional but recommended for AI assist / voice Q&A)
export OPENAI_API_KEY=YOUR_KEY
```

### Run

```bash
# Via console script
jerry-in-a-box

# or explicitly
python -m jerry_in_a_box.main
```

While running, you can also drive it from the keyboard for testing:

- A–G: add chords
- 1–5: sharps (1=A#, 2=C#, 3=D#, 4=F#, 5=G#)
- C: clear progression
- ?: ask a question (voice)
- Q: quit

## Data

- Song progressions are stored in `jerry_in_a_box/data/songs.json`. You can edit or expand this list for more Garcia/Dead tunes.
- Utilities in `scripts/` help parse chord charts and generate structured progressions.

## Hardware build (pedal)

- Raspberry Pi 4/5
- USB audio interface (guitar in, line out)
- 0.96"–1.3" OLED (e.g., SSD1306) or small LED matrix
- Footswitch(es) for mode/scroll
- 3D‑printed or repurposed enclosure

The software runs on macOS or Linux for development; the pedal experience shines on the Pi with the screen and footswitch attached.

## For recruiters (what this demonstrates)

- Hardware + software integration: audio IO, displays, input devices
- Applied DSP: FFT/chroma, confidence scoring, smoothing
- ML/AI product thinking: lightweight local heuristics plus cloud AI assist when ambiguous
- UX under constraints: latency, readability, foot‑controlled flows
- Pragmatic shipping: JSON data sources, CLI + on‑device UI, scripts for content ingestion

## Roadmap / future improvements

- Generalization: make the pedal configurable to detect songs for **any band or genre** by swapping in different chord corpora.
- Local inference: train and distill a **smaller open‑source model** to run on‑device (quantized on ARM) for faster, offline classification.
- Better DSP: polyphonic chord detection, key estimation, tempo/section detection.
- Smarter ranking: combine progression similarity with key/tempo/section cues; learn‑to‑rank with feedback.
- UI polish: section markers (verse/chorus/bridge), scrollable setlists, “next up” hints.
- Latency wins: audio buffering tweaks, SIMD/NEON, smaller FFT windows with multi‑frame smoothing.
- Reliability: offline mode with cached embeddings; graceful degradation without network.

## Notes

- The project currently ships with a local similarity matcher and an optional OpenAI assist. The hardware‑display version runs on a Pi; desktop mode prints the same information in the terminal.
- This is a work in progress—PRs and ideas welcome.

## License

MIT
