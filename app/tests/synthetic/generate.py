"""Synthetic note-stream generator that simulates:

  - a competent guitarist playing through a song's chord sequence,
  - a guitar TUNER (monophonic) listening, producing noisy single-pitch
    observations that are biased toward chord tones, sometimes in-key
    non-chord-tones, and rarely random pcs.

The generator's noise model deliberately mirrors the matcher's emission
model so the matcher *can* in principle do well — but the generator can
also be cranked harder (more random / more in-key non-chord) to test the
matcher's robustness to mis-specification.

A trace is a list of NoteEvent dicts:

    {"t": 0.123, "pitch_class": 7, "midi": 55, "confidence": 0.83,
     "true_chord": "G"}

Plus a header with metadata.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from app.parser.chords import _QUALITY_TONES
from app.matcher.emission import in_key_set


@dataclass
class GenConfig:
    """Knobs that control how realistic / how hard the trace is."""
    # Per-emission probabilities — must roughly sum to ~1 (normalized internally).
    p_root:       float = 0.40
    p_chord_tone: float = 0.30
    p_bass:       float = 0.05
    p_in_key:     float = 0.20
    p_random:     float = 0.05

    # Tempo (BPM) and rhythmic structure
    bpm: float = 100.0
    beats_per_chord: int = 4         # how long the player holds each chord
    emissions_per_beat: int = 2      # tuner re-fires this many times per beat

    # Player-side noise
    p_chord_skip: float = 0.03       # skip the next chord entirely
    p_chord_repeat: float = 0.05     # play the current chord again
    p_sustain_bleed: float = 0.15    # at chord boundary, emit one note from the previous chord
    p_dropped_first: float = 0.05    # drop the first emission of a new chord

    # Confidence model: "right" emissions get a higher base confidence
    conf_chord_mu: float = 0.85
    conf_chord_sigma: float = 0.08
    conf_wrong_mu: float = 0.45
    conf_wrong_sigma: float = 0.20

    # Tempo jitter (fraction)
    tempo_jitter: float = 0.10


@dataclass
class Trace:
    song_id: str
    song_title: str
    config: dict
    seed: int
    events: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# Difficulty presets matching the plan ("easy / medium / hard").
PRESETS = {
    "easy": GenConfig(
        p_root=0.55, p_chord_tone=0.30, p_bass=0.05, p_in_key=0.08, p_random=0.02,
        p_chord_skip=0.01, p_chord_repeat=0.02, p_sustain_bleed=0.05,
        tempo_jitter=0.05,
    ),
    "medium": GenConfig(),  # the defaults — realistic
    "hard": GenConfig(
        p_root=0.30, p_chord_tone=0.25, p_bass=0.05, p_in_key=0.30, p_random=0.10,
        p_chord_skip=0.05, p_chord_repeat=0.07, p_sustain_bleed=0.20,
        tempo_jitter=0.25,
    ),
}


# -------- internals --------

def _chord_tones_for(root_pc: int, quality: str, bass_pc: Optional[int]) -> set[int]:
    offs = _QUALITY_TONES.get(quality, _QUALITY_TONES["maj"])
    s = {(root_pc + o) % 12 for o in offs}
    if bass_pc is not None:
        s.add(bass_pc % 12)
    return s


def _sample_pc(
    chord_tuple: tuple[int, str, Optional[int]],
    in_key: frozenset[int],
    cfg: GenConfig,
    rng: random.Random,
) -> tuple[int, str]:
    """Sample one pitch class given the current chord. Returns (pc, label).

    label is one of {"root", "chord", "bass", "in_key", "random"} so the
    test harness / debug stream can know what kind of emission this was.
    """
    root, qual, bass = chord_tuple
    tones = _chord_tones_for(root, qual, bass)

    # Categorical draw.
    weights = [cfg.p_root, cfg.p_chord_tone, cfg.p_bass, cfg.p_in_key, cfg.p_random]
    s = sum(weights)
    weights = [w / s for w in weights]
    u = rng.random()
    acc = 0.0
    for label, w in zip(("root", "chord", "bass", "in_key", "random"), weights):
        acc += w
        if u <= acc:
            kind = label
            break

    # Resolve to a pc. Handle empty fallbacks gracefully.
    if kind == "root":
        return root, "root"
    if kind == "bass" and bass is not None and bass != root:
        return int(bass) % 12, "bass"
    if kind == "chord":
        non_root = [t for t in tones if t != root and (bass is None or t != bass)]
        if non_root:
            return rng.choice(non_root), "chord"
        return root, "root"
    if kind == "in_key":
        candidates = [pc for pc in in_key if pc not in tones]
        if candidates:
            return rng.choice(candidates), "in_key"
        return rng.randrange(12), "random"
    # "random" or fallthrough
    return rng.randrange(12), "random"


def _confidence_for(label: str, cfg: GenConfig, rng: random.Random) -> float:
    # "right" emissions = root/chord/bass; "wrong" = in_key/random.
    if label in ("root", "chord", "bass"):
        v = rng.gauss(cfg.conf_chord_mu, cfg.conf_chord_sigma)
    else:
        v = rng.gauss(cfg.conf_wrong_mu, cfg.conf_wrong_sigma)
    return float(max(0.05, min(0.99, v)))


def _midi_for(pc: int, prev_midi: int | None, rng: random.Random) -> int:
    """Pick an octave that makes sense on guitar (E2..A5).

    On a real tuner the pc collapses across octaves; the matcher only
    uses pc, but we record midi too so traces look real.
    """
    base = (rng.choice([3, 3, 4, 4, 4, 5]) * 12) + pc
    return base


def generate_trace(
    song: dict,
    seed: int = 0,
    cfg: GenConfig | None = None,
    n_chord_events_max: Optional[int] = None,
) -> Trace:
    """Generate one synthetic trace for a song.

    `song` is a record from songs.json (must have chord_tuples + key_pc + key_mode).
    """
    cfg = cfg or GenConfig()
    rng = random.Random(seed)
    chord_tuples = song["chord_tuples"]
    chord_names = song["chords"]
    key_pc = song["key_pc"]
    key_mode = song["key_mode"]
    in_key = in_key_set(key_pc, key_mode)

    # Walk through the chord sequence with skip/repeat noise.
    seq: list[int] = []  # indices into chord_tuples
    i = 0
    while i < len(chord_tuples):
        seq.append(i)
        u = rng.random()
        if u < cfg.p_chord_skip:
            i += 2
        elif u < cfg.p_chord_skip + cfg.p_chord_repeat:
            pass  # repeat: don't advance
        else:
            i += 1
    if n_chord_events_max:
        seq = seq[:n_chord_events_max]

    events: list[dict] = []
    t = 0.0
    sec_per_beat_nominal = 60.0 / max(1.0, cfg.bpm)
    prev_chord_idx: int | None = None

    for chord_idx in seq:
        ct = chord_tuples[chord_idx]
        chord_tuple = (int(ct[0]), str(ct[1]), None if ct[2] is None else int(ct[2]))
        beats_this_chord = cfg.beats_per_chord
        n_emits_total = beats_this_chord * cfg.emissions_per_beat

        for emit_idx in range(n_emits_total):
            jitter = 1.0 + rng.uniform(-cfg.tempo_jitter, cfg.tempo_jitter)
            dt = sec_per_beat_nominal * jitter / cfg.emissions_per_beat
            t += dt

            # Sustain bleed: at the very first emission of a new chord, we
            # might emit a note from the previous chord instead.
            if (emit_idx == 0
                    and prev_chord_idx is not None
                    and rng.random() < cfg.p_sustain_bleed):
                pct = chord_tuples[prev_chord_idx]
                src = (int(pct[0]), str(pct[1]), None if pct[2] is None else int(pct[2]))
            else:
                src = chord_tuple

            # Drop the first emission of a new chord with some probability.
            if emit_idx == 0 and rng.random() < cfg.p_dropped_first:
                continue

            pc, label = _sample_pc(src, in_key, cfg, rng)
            conf = _confidence_for(label, cfg, rng)
            midi = _midi_for(pc, None, rng)
            events.append({
                "t": round(t, 4),
                "pitch_class": int(pc),
                "midi": int(midi),
                "confidence": round(conf, 3),
                "true_chord": chord_names[chord_idx],
                "true_chord_idx": chord_idx,
                "label": label,
            })

        prev_chord_idx = chord_idx

    return Trace(
        song_id=song["id"],
        song_title=song["title"],
        config=asdict(cfg),
        seed=seed,
        events=events,
    )


def write_dataset(
    songs: list[dict],
    out_dir: Path,
    preset: str = "medium",
    seeds_per_song: int = 1,
    base_seed: int = 0,
) -> list[Path]:
    """Generate one or more traces per song using a preset config."""
    cfg = PRESETS[preset]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for s_i, song in enumerate(songs):
        for k in range(seeds_per_song):
            trace = generate_trace(song, seed=base_seed + s_i * 1000 + k, cfg=cfg)
            p = out_dir / f"{song['id']}__{preset}__s{base_seed + s_i * 1000 + k}.json"
            p.write_text(json.dumps(trace.to_dict()))
            paths.append(p)
    return paths


def main(argv: list[str] | None = None) -> None:
    import argparse

    repo = Path(__file__).resolve().parents[3]
    default_songs = repo / "app" / "data" / "songs.json"
    default_out = repo / "app" / "tests" / "fixtures"

    p = argparse.ArgumentParser()
    p.add_argument("--songs", default=str(default_songs))
    p.add_argument("--out",   default=str(default_out))
    p.add_argument("--preset", default="medium", choices=list(PRESETS.keys()))
    p.add_argument("--seeds-per-song", type=int, default=1)
    p.add_argument("--base-seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="If set, only generate traces for the first N songs")
    args = p.parse_args(argv)

    db = json.loads(Path(args.songs).read_text())
    songs = db["songs"]
    if args.limit:
        songs = songs[: args.limit]
    paths = write_dataset(
        songs,
        Path(args.out) / args.preset,
        preset=args.preset,
        seeds_per_song=args.seeds_per_song,
        base_seed=args.base_seed,
    )
    print(f"Wrote {len(paths)} traces to {Path(args.out) / args.preset}")


if __name__ == "__main__":
    main()
