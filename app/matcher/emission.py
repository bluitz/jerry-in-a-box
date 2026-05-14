"""Emission model: P(observed pitch_class | true chord).

This is the core of the "disambiguate the noisy tuner" strategy.

When the chord is C = (root, quality, bass), the tuner is most likely to
emit a chord tone (root/3rd/5th/7th), occasionally an in-key non-chord
tone, and rarely a random pitch class. We capture this as a 12-bin
distribution P(observed_pc | chord, key).

Probabilities (defaults; tunable via :class:`EmissionConfig`):
    p_root           weight pushed onto the root
    p_chord_tone     weight pushed onto non-root chord tones (split equally)
    p_bass           extra weight on the bass note (slash chords)
    p_in_key         weight on in-key non-chord-tones (split equally)
    p_random         weight uniformly across all 12 pcs

Together they sum to ~1 (we re-normalize). The result is a 12-vector.

The same module knows how to build an in-key set from a (key_pc, mode)
pair using major / natural-minor scale degrees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Major / natural-minor scale (semitones from tonic).
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
_NAT_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def in_key_set(key_pc: int, mode: str) -> frozenset[int]:
    scale = _MAJOR_SCALE if mode == "maj" else _NAT_MINOR_SCALE
    return frozenset((key_pc + d) % 12 for d in scale)


@dataclass(frozen=True)
class EmissionConfig:
    p_root:       float = 0.40
    p_chord_tone: float = 0.30
    p_bass:       float = 0.05
    p_in_key:     float = 0.20
    p_random:     float = 0.05

    def normalized(self) -> "EmissionConfig":
        s = self.p_root + self.p_chord_tone + self.p_bass + self.p_in_key + self.p_random
        if s == 0:
            return self
        return EmissionConfig(
            p_root=self.p_root / s,
            p_chord_tone=self.p_chord_tone / s,
            p_bass=self.p_bass / s,
            p_in_key=self.p_in_key / s,
            p_random=self.p_random / s,
        )


def emission_for_chord(
    root_pc: int,
    quality: str,
    bass_pc: int | None,
    chord_tones: frozenset[int],
    in_key: frozenset[int],
    cfg: EmissionConfig = EmissionConfig(),
) -> np.ndarray:
    """Return a length-12 probability vector P(observed_pc | chord)."""

    cfg = cfg.normalized()
    e = np.zeros(12, dtype=np.float64)

    # Random floor across all 12 pcs
    e += cfg.p_random / 12.0

    # In-key (non-chord-tone) — split p_in_key equally over those pcs.
    in_key_non_chord = in_key - chord_tones
    if in_key_non_chord:
        per = cfg.p_in_key / len(in_key_non_chord)
        for pc in in_key_non_chord:
            e[pc] += per
    else:
        # Fold the in-key weight into the random floor.
        e += cfg.p_in_key / 12.0

    # Chord tones (non-root) — split p_chord_tone equally over those pcs.
    non_root_tones = chord_tones - {root_pc}
    if bass_pc is not None and bass_pc != root_pc:
        non_root_tones = non_root_tones - {bass_pc}
    if non_root_tones:
        per = cfg.p_chord_tone / len(non_root_tones)
        for pc in non_root_tones:
            e[pc] += per
    else:
        e += cfg.p_chord_tone / 12.0

    # Root
    e[root_pc] += cfg.p_root

    # Bass
    if bass_pc is not None and bass_pc != root_pc:
        e[bass_pc] += cfg.p_bass
    else:
        # Fold bass weight onto the root if no slash bass.
        e[root_pc] += cfg.p_bass

    # Numerical floor + renormalize (defends against any weight=0 corner).
    e = np.clip(e, 1e-9, None)
    e /= e.sum()
    return e


def build_emission_table(
    chord_tuples: list[tuple[int, str, int | None]],
    chord_tones_list: list[frozenset[int]],
    key_pc: int,
    mode: str,
    cfg: EmissionConfig = EmissionConfig(),
) -> np.ndarray:
    """Build an (N, 12) emission matrix for a song's chord vocabulary."""
    in_key = in_key_set(key_pc, mode)
    n = len(chord_tuples)
    out = np.zeros((n, 12), dtype=np.float64)
    for i, ((root, qual, bass), tones) in enumerate(zip(chord_tuples, chord_tones_list)):
        out[i] = emission_for_chord(root, qual, bass, tones, in_key, cfg)
    return out
