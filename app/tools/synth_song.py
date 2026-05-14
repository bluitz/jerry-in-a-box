"""Synthesize a clean one-guitar pluck recording from a song's chord
chart. Used to manufacture 'easy case' test inputs (one guitar, no
drums, no vocals, no room noise) so we can validate the matcher
end-to-end without arguing with how a particular live recording
diverges from the songbook.

This is NOT meant to sound like a real guitar; it's a stylized
plucky synthesizer that reliably produces the chord-tone harmonic
content the chord classifier looks for.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
SR = 44100

QUAL_INTERVALS: dict[str, list[int]] = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "m7":   [0, 3, 7, 10],
    "maj7": [0, 4, 7, 11],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus":  [0, 5, 7],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
}


def _chord_sample(root_pc: int, qual: str, dur_s: float,
                  decay: float = 2.0) -> np.ndarray:
    """A single 'pluck' of one chord — sine partials at chord-tone
    frequencies across three octaves, plucky exponential decay."""
    n = int(SR * dur_s)
    out = np.zeros(n, dtype=np.float32)
    intervals = QUAL_INTERVALS.get(qual, QUAL_INTERVALS["maj"])
    t = np.arange(n) / SR
    env = np.exp(-decay * t).astype(np.float32)
    for octv in (3, 4, 5):
        for iv in intervals:
            midi = 12 * octv + root_pc + iv
            f = 440.0 * 2 ** ((midi - 69) / 12.0)
            out += 0.08 * env * np.sin(2 * np.pi * f * t).astype(np.float32)
    return out


def synth_song(song_id: str, *,
               total_duration_s: float = 30.0,
               chord_dur_s: float = 2.5,
               normalize: bool = True) -> np.ndarray:
    """Synthesize ~total_duration_s of audio for the given song id by
    plucking each (collapsed) chord in the song's chart for chord_dur_s,
    looping until we hit the duration."""
    songs_path = REPO / "app" / "data" / "songs.json"
    songs = {s["id"]: s for s in json.loads(songs_path.read_text())["songs"]}
    rec = songs.get(song_id)
    if rec is None:
        raise KeyError(f"Song id not found: {song_id}")

    tuples = rec.get("chord_tuples") or []
    if not tuples:
        raise ValueError(f"Song {song_id} has no chord_tuples")

    # Collapse adjacent same-root chords so a long held G doesn't
    # eat all of our budget.
    collapsed: list[tuple[int, str]] = []
    for tup in tuples:
        root = int(tup[0]) % 12
        qual = str(tup[1]) if len(tup) > 1 else "maj"
        if not collapsed or collapsed[-1][0] != root:
            collapsed.append((root, qual))

    chunks: list[np.ndarray] = []
    elapsed = 0.0
    i = 0
    while elapsed < total_duration_s:
        root, qual = collapsed[i % len(collapsed)]
        chunks.append(_chord_sample(root, qual, chord_dur_s))
        elapsed += chord_dur_s
        i += 1
    audio = np.concatenate(chunks)
    audio = audio[: int(SR * total_duration_s)]

    if normalize:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio * (0.7 / peak)
    return audio


def write_wav(audio: np.ndarray, path: Path) -> None:
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("song_id", help="Song id slug, e.g. friend-of-the-devil")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--chord-dur", type=float, default=2.5)
    args = p.parse_args()

    audio = synth_song(args.song_id,
                       total_duration_s=args.duration,
                       chord_dur_s=args.chord_dur)
    write_wav(audio, args.out)
    print(f"Wrote {args.out} ({len(audio)/SR:.1f}s)")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
