"""Chord-template classifier.

Given a 12-bin chroma vector, return a posterior distribution over the
24 common chord states (12 major + 12 minor). The major/minor templates
are simple binary masks over chord tones (root, third, fifth).

This is the standard chroma -> chord recognizer used in MIR work
(Fujishima 1999) and is the right substrate for distinguishing
chord-driven songs that share the same key.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


# Chord-tone offsets relative to root. We restrict the template set to
# pure triads. Adding 7th-chord templates is counterproductive: e.g.
# Fmaj7 (F,A,C,E) is a superset of Am (A,C,E), so any Am chroma also
# scores well as Fmaj7 and the larger template often wins. Songbook
# chord charts in our DB are mostly triads with occasional sevenths,
# and the SequenceMatcher's chord_similarity collapses min/min7 etc.
# down to high similarity anyway.
QUALITIES = (
    ("maj", (0, 4, 7)),
    ("min", (0, 3, 7)),
)


def _build_templates() -> tuple[list[tuple[int, str]], np.ndarray]:
    """Return (chord_labels, templates[N, 12])."""
    labels: list[tuple[int, str]] = []
    rows: list[np.ndarray] = []
    for root in range(12):
        for q_name, tones in QUALITIES:
            v = np.zeros(12, dtype=np.float64)
            for t in tones:
                v[(root + t) % 12] = 1.0
            v /= v.sum()
            labels.append((root, q_name))
            rows.append(v)
    return labels, np.stack(rows, axis=0)


_LABELS, _TEMPLATES = _build_templates()


def classify_chroma(chroma: np.ndarray, *,
                    sharpen: float = 8.0) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Return (labels, posterior) over chord states.

    `posterior` is a softmax over similarity scores; `sharpen` controls
    how concentrated it is (higher = more peaked).
    """
    c = np.asarray(chroma, dtype=np.float64).reshape(12)
    s = c.sum()
    if s > 0:
        c = c / s
    # Cosine similarity between chroma and each template.
    tnorms = np.linalg.norm(_TEMPLATES, axis=1)
    cnorm = np.linalg.norm(c)
    if cnorm == 0 or tnorms.max() == 0:
        n = _TEMPLATES.shape[0]
        return _LABELS, np.full(n, 1.0 / n)
    sims = (_TEMPLATES @ c) / (tnorms * cnorm + 1e-12)
    # Softmax (sharper => more peaked posterior)
    z = np.exp(sharpen * (sims - sims.max()))
    post = z / z.sum()
    return _LABELS, post


def top_chord(chroma: np.ndarray) -> tuple[int, str, float]:
    """Return (root_pc, quality, prob) for the best chord template."""
    labels, post = classify_chroma(chroma)
    i = int(np.argmax(post))
    root, qual = labels[i]
    return root, qual, float(post[i])


def classify_stream(chromas: Iterable[np.ndarray]) -> list[tuple[int, str, float]]:
    """Convenience: per-frame top-chord classification."""
    return [top_chord(c) for c in chromas]
