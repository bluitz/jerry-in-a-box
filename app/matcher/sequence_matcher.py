"""Chord-sequence matcher.

Given a stream of (root_pc, quality, weight) chord observations from the
chord classifier, this matcher scores each song by the BEST alignment
between the observed chord sequence and the song's chord progression.

This is fundamentally more discriminating than the bag-of-notes + HMM
combo because it directly compares ordered chord identity, which is
what songs uniquely have. Two songs that both use G/C/D/Am have very
different orderings and repetition patterns.

Algorithm: dynamic-programming local-alignment (Smith-Waterman style)
with a similarity score between chord identities. We collapse runs in
the observation stream so a held chord doesn't dominate the alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Quality similarity between observed and template chord (cosine of
# template tones). 1.0 = identical, 0 = orthogonal.
_QUAL_TONES: dict[str, frozenset[int]] = {
    "maj":  frozenset({0, 4, 7}),
    "min":  frozenset({0, 3, 7}),
    "7":    frozenset({0, 4, 7, 10}),
    "m7":   frozenset({0, 3, 7, 10}),
    "maj7": frozenset({0, 4, 7, 11}),
    "dim":  frozenset({0, 3, 6}),
    "aug":  frozenset({0, 4, 8}),
    "sus":  frozenset({0, 5, 7}),
    "sus2": frozenset({0, 2, 7}),
    "sus4": frozenset({0, 5, 7}),
}


def chord_similarity(a: tuple[int, str], b: tuple[int, str]) -> float:
    """Similarity in [0, 1] between two chord identities.

    Score is the size of the intersection of their pc-sets divided by
    max chord size. Plus a small bonus when the roots match.
    """
    ar, aq = a
    br, bq = b
    a_tones = _QUAL_TONES.get(aq, _QUAL_TONES["maj"])
    b_tones = _QUAL_TONES.get(bq, _QUAL_TONES["maj"])
    a_pcs = frozenset((ar + t) % 12 for t in a_tones)
    b_pcs = frozenset((br + t) % 12 for t in b_tones)
    inter = len(a_pcs & b_pcs)
    union = max(len(a_pcs), len(b_pcs))
    score = inter / union
    # Strong bonus for exact identity (root + quality).
    if ar == br and aq == bq:
        score += 0.5
    elif ar == br:
        score += 0.15
    return min(score, 1.0)


def _root_collapse(seq: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Collapse adjacent chords with the same ROOT into one entry.

    Songs often write things like "G G7 G G7" which is musically a single
    held G (the 7 is a passing color); collapsing by root keeps the
    musical fingerprint (which is the *changes*) and stops repetitive
    same-root patterns from spuriously matching one another.
    """
    out: list[tuple[int, str]] = []
    for c in seq:
        if not out or out[-1][0] != c[0]:
            out.append(c)
    return out


@dataclass
class _SongRef:
    sid: str
    title: str
    chord_seq: list[tuple[int, str]]  # ROOT-collapsed progression

    @classmethod
    def from_record(cls, rec: dict) -> "_SongRef":
        seq: list[tuple[int, str]] = []
        for tup in rec.get("chord_tuples") or []:
            if not tup:
                continue
            root = int(tup[0])
            qual = str(tup[1]) if len(tup) > 1 else "maj"
            seq.append((root % 12, qual))
        return cls(sid=rec["id"], title=rec["title"],
                   chord_seq=_root_collapse(seq))


class SequenceMatcher:
    """Stateful sequence matcher. Feed observed (root, quality) chords
    one at a time; ranks all songs by best local alignment of the
    accumulated observation sequence against each song's progression.
    """

    def __init__(self, songs: list[dict]) -> None:
        self.songs = [_SongRef.from_record(s) for s in songs]
        self.titles = {s.sid: s.title for s in self.songs}
        self._obs: list[tuple[int, str]] = []
        # Collapsed observations (drop adjacent duplicates) — alignment
        # against the song's collapsed chord_seq is more meaningful than
        # against the per-frame stream.
        self._collapsed: list[tuple[int, str]] = []

    def add_chord(self, root: int, qual: str) -> None:
        c = (int(root) % 12, str(qual))
        self._obs.append(c)
        # Root-collapse: a held chord (with passing 7ths/sus) is one
        # entry. The musical fingerprint is in the changes.
        if not self._collapsed or self._collapsed[-1][0] != c[0]:
            self._collapsed.append(c)

    def reset(self) -> None:
        self._obs.clear()
        self._collapsed.clear()

    def n_observations(self) -> int:
        return len(self._obs)

    def n_unique_chords(self) -> int:
        return len(self._collapsed)

    def score_all(self) -> list[tuple[str, float]]:
        """Return [(song_id, score)] sorted descending."""
        out: list[tuple[str, float]] = []
        if not self._collapsed:
            n = len(self.songs)
            return [(s.sid, 1.0 / n) for s in self.songs]
        for s in self.songs:
            out.append((s.sid, self._best_alignment(s.chord_seq)))
        out.sort(key=lambda kv: -kv[1])
        return out

    def softmax_post(self, sharpen: float = 12.0) -> list[tuple[str, float]]:
        """Return [(song_id, prob)] — a softmax over scores."""
        scored = self.score_all()
        sids = [sid for sid, _ in scored]
        s = np.array([sc for _, sc in scored], dtype=np.float64)
        z = np.exp(sharpen * (s - s.max()))
        p = z / z.sum()
        return list(zip(sids, p.tolist()))

    def _best_alignment(self, song_seq: list[tuple[int, str]]) -> float:
        """Best alignment score across all 12 transpositions of song_seq.

        Score combines:
          - vocab_overlap: how many of the song's distinct chord families
            appear in the observation (Jaccard).
          - alignment: longest run of equal chords between obs and song,
            weighted by the number of DISTINCT chords appearing in that
            run, so alternating two-chord vamps don't dominate.

        The alignment is searched across all 12 transpositions (capo
        invariance) and we keep the best one. Vocab overlap is computed
        for each transposition too: a song matches better when its full
        chord vocabulary, transposed, is the one we observe.
        """
        if not song_seq:
            return 0.0
        obs = self._collapsed
        if not obs:
            return 0.0
        obs_fams = self._families(obs)
        obs_bigrams = self._bigrams(obs)

        # Native-key (shift=0) gets full credit. Transposed candidates
        # pay a penalty so songs with rich chord vocabularies don't
        # "swallow" everything by finding some shift that matches.
        best = 0.0
        for shift in range(12):
            shifted = [((r + shift) % 12, q) for r, q in song_seq]
            song_fams = self._families(shifted)
            inter = len(obs_fams & song_fams)
            union = len(obs_fams | song_fams) or 1
            if inter < 2:
                continue
            jaccard = inter / union
            song_bigrams = self._bigrams(shifted)
            bigram_inter = len(obs_bigrams & song_bigrams)
            bigram_union = len(obs_bigrams | song_bigrams) or 1
            bigram_jacc = bigram_inter / bigram_union
            align = self._align_one(obs, shifted)
            # Heavy transpose penalty: songs are charted in their
            # natural key and most recordings are in that key. Songs
            # with rich chord vocabularies otherwise spuriously
            # "swallow" any obs by finding a clever transposition.
            transpose_penalty = 0.0 if shift == 0 else -10.0
            score = (align
                     + 4.0 * jaccard
                     + 6.0 * bigram_jacc
                     + transpose_penalty)
            if score > best:
                best = score
        return best

    @staticmethod
    def _bigrams(seq: list[tuple[int, str]]) -> set[tuple[int, int]]:
        """Adjacent (root, root) pairs as a set. Direction-sensitive:
        G->C is different from C->G. We use ROOT only (not family)
        because the chord-template classifier is unreliable about
        major/minor (e.g. classifies Am as A when the melody emphasizes
        the major third). Chord-PAIR bigrams are still very
        discriminating: two songs sharing chord vocabulary differ in
        HOW they move between chords.
        """
        out: set[tuple[int, int]] = set()
        for i in range(len(seq) - 1):
            a = seq[i][0]
            b = seq[i + 1][0]
            if a == b:
                continue
            out.add((a, b))
        return out

    @staticmethod
    def _families(seq: list[tuple[int, str]]) -> set[tuple[int, str]]:
        return {(r, SequenceMatcher._family(q)) for r, q in seq}

    @staticmethod
    def _chord_eq(a: tuple[int, str], b: tuple[int, str]) -> bool:
        """FULL chord equality: same root AND same major/minor family.
        Major and minor third are the most discriminative chord feature
        (D and Dm appear in totally different songs), so we use them.
        7th-extensions etc. are folded back to their triad family.
        """
        if a[0] != b[0]:
            return False
        return SequenceMatcher._family(a[1]) == SequenceMatcher._family(b[1])

    @staticmethod
    def _family(q: str) -> str:
        # Order matters: "maj"/"maj7" must match before bare "m" prefix,
        # and "dim" before "m".
        if q.startswith("maj") or q in ("", "maj"):
            return "maj"
        if q.startswith("dim"):
            return "dim"
        if q.startswith("m") or q.startswith("min"):
            return "min"
        return "maj"

    @staticmethod
    def _align_one(obs: list[tuple[int, str]],
                   song: list[tuple[int, str]]) -> float:
        m, n = len(obs), len(song)
        if m == 0 or n == 0:
            return 0.0
        # Longest run of consecutive chord-similarity anywhere in song.
        # We accept a "near match" (same root, different family) as a
        # half-credit step, so a single Dm classification in the middle
        # of D-Am-D-Am doesn't break the run; but two consecutive
        # mismatches do. Track length AND distinct-chord-count so
        # alternating two-chord vamps don't dominate.
        best_run_score = 0.0
        for i in range(m):
            for j in range(n):
                k = 0
                run_score = 0.0
                seen: set[tuple[int, str]] = set()
                while i + k < m and j + k < n:
                    a = obs[i + k]
                    b = song[j + k]
                    if SequenceMatcher._chord_eq(a, b):
                        seen.add(SequenceMatcher._fam_key(a))
                        run_score += 1.0
                        k += 1
                    elif a[0] == b[0]:  # same root, family mismatch
                        seen.add(SequenceMatcher._fam_key(a))
                        run_score += 0.5
                        k += 1
                    else:
                        break
                if k >= 2 and run_score >= 2.0:
                    distinct = len(seen)
                    # Normalize by obs length so long song progressions
                    # don't unfairly find a "needle in haystack" match
                    # — they have to cover the same FRACTION of obs.
                    rs = (run_score / float(m)) * (float(distinct) ** 1.2)
                    if rs > best_run_score:
                        best_run_score = rs

        # Smith-Waterman with reward for exact match, partial credit
        # for same-root or shared-tone chords, penalty for gaps.
        gap = -0.5
        H = np.zeros((m + 1, n + 1), dtype=np.float64)
        sw_best = 0.0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                sim = chord_similarity(obs[i - 1], song[j - 1])
                step = (sim * 1.5) - 0.4
                H[i, j] = max(
                    0.0,
                    H[i - 1, j - 1] + step,
                    H[i - 1, j] + gap,
                    H[i, j - 1] + gap,
                )
                if H[i, j] > sw_best:
                    sw_best = H[i, j]

        sw_norm = sw_best / max(1.0, m)
        # Run-score is in fractional units (0..~10), SW is similar.
        # Multiply run by ~10 to make it comparable to jaccard*4 which
        # also gives ~3.
        return 10.0 * best_run_score + 0.5 * sw_norm

    @staticmethod
    def _fam_key(c: tuple[int, str]) -> tuple[int, str]:
        return (c[0], SequenceMatcher._family(c[1]))
