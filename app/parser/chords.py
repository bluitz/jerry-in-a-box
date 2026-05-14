"""Canonical chord representation and parser.

A chord is canonicalized into:

    Chord(root_pc, quality, bass_pc)

where:
    root_pc:  pitch class of the root, 0..11 (C=0, C#=1, ..., B=11)
    quality:  one of QUALITIES (e.g. "maj", "min", "7", "maj7", "m7", ...)
    bass_pc:  pitch class of the bass note if a slash chord, else None.
              For E/G# this is 8 (G#).

The parser also returns the set of chord-tone pitch classes implied by the
chord (root, 3rd, 5th, plus 7th/extensions where applicable). The matcher
uses these to define per-chord emission distributions.

This module is deliberately small and dependency-free so the matcher can
run in a pure-Python sandbox or be re-implemented in TS later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Optional


# C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
PITCH_CLASS = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

PC_TO_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Quality -> chord-tone offsets relative to root (semitones).
# Kept compact; the matcher only cares about which pitch classes are
# "in the chord". We include the standard triad/seventh extensions.
_QUALITY_TONES = {
    "maj":   (0, 4, 7),
    "min":   (0, 3, 7),
    "dim":   (0, 3, 6),
    "aug":   (0, 4, 8),
    "sus2":  (0, 2, 7),
    "sus4":  (0, 5, 7),
    "5":     (0, 7),                 # power chord
    "7":     (0, 4, 7, 10),          # dominant 7
    "maj7":  (0, 4, 7, 11),
    "m7":    (0, 3, 7, 10),
    "m7b5":  (0, 3, 6, 10),
    "dim7":  (0, 3, 6, 9),
    "9":     (0, 4, 7, 10, 2),
    "maj9":  (0, 4, 7, 11, 2),
    "m9":    (0, 3, 7, 10, 2),
    "6":     (0, 4, 7, 9),
    "m6":    (0, 3, 7, 9),
    "add9":  (0, 4, 7, 2),
    "7sus4": (0, 5, 7, 10),
}

QUALITIES = tuple(_QUALITY_TONES.keys())


@dataclass(frozen=True)
class Chord:
    root_pc: int
    quality: str
    bass_pc: Optional[int] = None
    # Frozenset of pitch classes (0..11) implied by this chord.
    tones: FrozenSet[int] = field(default_factory=frozenset)

    @property
    def name(self) -> str:
        root = PC_TO_NAME[self.root_pc]
        # Display: "min" -> "m" so we render Gm not Gmin.
        q_map = {"maj": "", "min": "m"}
        q = q_map.get(self.quality, self.quality)
        slash = ""
        if self.bass_pc is not None and self.bass_pc != self.root_pc:
            slash = f"/{PC_TO_NAME[self.bass_pc]}"
        return f"{root}{q}{slash}"

    def to_tuple(self) -> tuple:
        return (self.root_pc, self.quality, self.bass_pc)


# Token shapes we recognise for the QUALITY suffix. Order matters --
# longer, more specific suffixes are tried before shorter ones so that
# "maj7" doesn't get parsed as "maj" + leftover "7".
_QUALITY_REGEX = [
    ("maj7",  r"maj7|M7|Maj7|△7"),
    ("m7b5",  r"m7b5|ø|half-?dim"),
    ("m7",    r"m7|min7"),
    ("dim7",  r"dim7|°7|o7"),
    ("dim",   r"dim|°|o(?!7)"),
    ("aug",   r"aug|\+"),
    ("sus2",  r"sus2"),
    ("sus4",  r"sus4|sus"),
    ("7sus4", r"7sus4|7sus"),
    ("maj9",  r"maj9|M9"),
    ("m9",    r"m9|min9"),
    ("9",     r"9"),
    ("add9",  r"add9"),
    ("m6",    r"m6|min6"),
    ("6",     r"6"),
    ("7",     r"7"),
    ("5",     r"5"),
    ("min",   r"m(?!aj)|min(?!9)"),  # m, min — but not maj or m9 (m9 already matched)
]


def parse_chord(token: str) -> Optional[Chord]:
    """Parse a chord token (e.g. 'G', 'Em7', 'Bb7', 'E/G#', 'F#m7b5').

    Returns None if the token is clearly not a chord (e.g. '|', '/', '%',
    a number, a section heading). Tolerant of common notations seen in
    the songbook CSV: 'Bb7(C)', 'Emmaj7' (treated as Em(maj7)? no -- see
    below), parentheticals, trailing commas, etc.
    """
    if token is None:
        return None

    s = token.strip()
    if not s:
        return None

    # Strip trailing punctuation / annotations
    s = s.rstrip(".,;:")
    # Drop a leading "(" or trailing ")" if either is unbalanced
    s = s.replace("(", "").replace(")", "")
    if not s:
        return None

    # First char must be A-G
    if s[0] not in "ABCDEFG":
        return None

    # Root: A-G with optional accidental
    m = re.match(r"^([A-G])([#b])?", s)
    if not m:
        return None
    root_name = m.group(1) + (m.group(2) or "")
    if root_name not in PITCH_CLASS:
        return None
    root_pc = PITCH_CLASS[root_name]
    rest = s[m.end():]

    # Bass (slash chord)
    bass_pc: Optional[int] = None
    if "/" in rest:
        rest, bass_str = rest.split("/", 1)
        bass_str = bass_str.strip()
        bm = re.match(r"^([A-G])([#b])?", bass_str)
        if bm:
            bass_name = bm.group(1) + (bm.group(2) or "")
            if bass_name in PITCH_CLASS:
                bass_pc = PITCH_CLASS[bass_name]

    # Special-case: "Emmaj7" appears in the source CSV. Treat the literal
    # token "mmaj7" / "mMaj7" as a minor-major-7 -> we collapse to "m7"
    # since we don't model mMaj7 separately; minor + 7th is close enough
    # for a noisy-tuner matcher.
    rest_norm = rest.replace("mmaj7", "m7").replace("mMaj7", "m7")

    quality = "maj"
    rest_lower = rest_norm
    for q, pat in _QUALITY_REGEX:
        rm = re.match(pat, rest_lower)
        if rm:
            quality = q
            break

    # Build chord tones
    offsets = _QUALITY_TONES.get(quality, _QUALITY_TONES["maj"])
    tones = frozenset((root_pc + o) % 12 for o in offsets)
    if bass_pc is not None:
        tones = tones | {bass_pc}

    return Chord(root_pc=root_pc, quality=quality, bass_pc=bass_pc, tones=tones)


# Token regex used by parse_chord_string: a chord starts with A-G,
# optional accidental, then any non-whitespace until the next delimiter.
_CHORD_TOKEN = re.compile(r"[A-G][A-Za-z0-9#bø°△+/()]*")

# Repeat bracket:  (Nx)? ||:  content  :||  (Nx)?
# The `||` without a colon also starts a repeat when it precedes `:||`.
_REPEAT_RE = re.compile(
    r"(?:(\d+)\s*[xX]\s*)?"   # optional count before: "3x"
    r"\|\|:?"                  # begin: "||:" or "||"
    r"\s*(.*?)\s*"             # content (non-greedy)
    r":\|\|"                   # end: ":||"
    r"(?:\s*(\d+)\s*[xX])?",  # optional count after
    re.IGNORECASE | re.DOTALL,
)


def _expand_repeats(s: str) -> str:
    """Expand ||: ... :|| repeat brackets by duplicating their content.

    Examples:
        "|| G / / / | C / / G :||"
          -> "G / / / | C / / G | G / / / | C / / G"

        "||: G / / / | C6 / / / :||  D / / /"
          -> "G / / / | C6 / / / | G / / / | C6 / / / | D / / /"

        "3x ||: G / / / | C / / G :||"
          -> "G / / / | C / / G | G / / / | C / / G | G / / / | C / / G"
    """
    def _repl(m: re.Match) -> str:
        n_before, content, n_after = m.group(1), m.group(2), m.group(3)
        n = int(n_before or n_after or 2)
        content = content.strip().strip("|").strip()
        return " | ".join([content] * n)

    return _REPEAT_RE.sub(_repl, s)


def parse_chord_string(s: str, beats_per_bar: int = 4) -> list[Chord]:
    """Parse a free-form chord string (a measure-style line from the CSV).

    The songbook's measure notation uses:
        | -- bar line
        / -- one beat of "repeat last chord"
        % -- repeat the previous bar in full
        ||: ... :|| -- repeat bracket (play content N times, default 2)

    Returned list is a flat sequence of chords (one entry per beat). The
    bigram statistics the matcher uses are derived from chord *changes*,
    so per-beat resolution is fine.
    """
    s = _expand_repeats(s)
    tokens = re.findall(r"[A-G][A-Za-z0-9#bø°△+/()]*|\||%|/", s)
    progression: list[Chord] = []
    last_bar: list[Chord] = []
    current_bar: list[Chord] = []
    last: Optional[Chord] = None

    def flush_bar():
        nonlocal current_bar, last_bar
        if current_bar:
            last_bar = list(current_bar)
            current_bar = []

    for tok in tokens:
        if tok == "|":
            flush_bar()
            continue
        if tok == "/":
            if last is not None:
                progression.append(last)
                current_bar.append(last)
            continue
        if tok == "%":
            # Repeat the previous full bar (or, if no previous bar yet,
            # repeat the last chord beats_per_bar times).
            if last_bar:
                progression.extend(last_bar)
                current_bar.extend(last_bar)
            elif last is not None:
                for _ in range(beats_per_bar):
                    progression.append(last)
                    current_bar.append(last)
            continue
        c = parse_chord(tok)
        if c is None:
            continue
        progression.append(c)
        current_bar.append(c)
        last = c

    return progression
