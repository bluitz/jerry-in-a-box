"""Extract chord progressions directly from the Jerry Garcia Songbook PDF.

We do NOT extract or store any lyric/prose lines — only chord-bearing
lines (those containing the bar character `|` and at least two chord-
like tokens) and section-header lines (Verse / Chorus / Bridge / Solo /
Intro / Outro / Tag / Break / Coda...).

Output shape mirrors what songdb.py expects via _read_csv_sections:

    {title: [(section_name, chord_string), ...]}

This is then merged with sources/jerry_song_book.csv inside songdb.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


# A page in the songbook starts with "Page <N>" on its own line and is
# immediately followed by the song title on the next non-empty line.
PAGE_MARKER = re.compile(r"^\s*Page\s+(\d+)\s*$")

# A chord token: roots (A-G), optional accidental, optional quality and
# extensions, optional bass after a slash. We don't need this regex to be
# musically perfect; it just needs to discriminate chord-bearing lines
# from lyrics.
CHORD_TOK = re.compile(
    r"\b("
    r"[A-G][#b]?"                         # root
    r"(?:maj|min|m|dim|aug|sus|add)?"     # quality
    r"(?:[0-9]+)?"                        # extension
    r"(?:/[A-G][#b]?)?"                   # bass
    r")\b"
)

# Common section names (case-insensitive). Followed by an optional
# digit/qualifier ("Verse 2", "Chorus 1a", "Solo Break").
SECTION_TOK = re.compile(
    r"^\s*(?:"
    r"intro|outro|verse|chorus|refrain|bridge|tag|break|coda|"
    r"solo|riff|interlude|pre[- ]?chorus|hook|turnaround|ending|"
    r"vamp|jam|instrumental|head"
    r")\b",
    re.IGNORECASE,
)


def _is_chord_line(line: str) -> bool:
    if "|" not in line:
        return False
    toks = CHORD_TOK.findall(line)
    return len(toks) >= 2


def _is_section_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Section headers are short and don't contain bars (those are chord lines).
    if "|" in s:
        return False
    if len(s) > 40:
        return False
    return SECTION_TOK.match(s) is not None


def _normalize_title(s: str) -> str:
    """Strip common decorations from titles: "Page 93\nFriend of the Devil"
    sometimes has trailing whitespace, weird unicode, parenthetical key
    annotations, etc."""
    s = s.strip()
    # Strip trailing arrows/keys like "(Grateful Dead Play in B – Orignal is in C)"
    s = re.sub(r"\s*\([^)]+\)\s*$", "", s).strip()
    return s


def _looks_like_title(s: str) -> bool:
    """Reject lines that obviously aren't song titles.

    Real song titles in this PDF: short, often title-cased, contain no
    bar separators, and are not full English sentences. Lyric lines
    typically end with a period/comma and contain >5 words; song titles
    are usually 1-7 words and don't end with sentence punctuation.
    """
    if not s or len(s) > 60 or len(s) < 2:
        return False
    if "|" in s:
        return False
    if SECTION_TOK.match(s):
        return False
    if s.count("-") > 3:
        return False
    # Reject likely lyric lines: end with sentence punctuation OR
    # contain commas/periods inside (most titles don't), OR have lots
    # of words.
    if s.rstrip().endswith((".", ",", "!", "?", ";", ":")):
        return False
    if "," in s:
        return False
    n_words = len(s.split())
    if n_words > 8:
        return False
    # Reject lines that are mostly lowercase (titles are usually Title
    # Case or ALL CAPS).
    letters = [c for c in s if c.isalpha()]
    if letters:
        upper_frac = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_frac < 0.10:
            return False
    return True


def extract_chord_data(pdf_path: Path,
                       skip_first_pages: int = 4) -> dict[str, list[tuple[str, str]]]:
    """Walk the PDF and emit, per song title, a list of (section_name,
    chord_string) tuples — same shape that songdb._read_csv_sections
    produces from the CSV."""
    out: dict[str, list[tuple[str, str]]] = {}
    doc = fitz.open(str(pdf_path))

    current_title: str | None = None
    current_section = "Intro"
    # Buffer chord-line strings under (title, section) keys. We
    # concatenate them at the end so a chord run that spans multiple
    # printed lines stays as one progression.
    buf: dict[tuple[str, str], list[str]] = {}

    for p in range(skip_first_pages, doc.page_count):
        lines = doc[p].get_text().split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Page-marker -> next non-empty line that looks like a title.
            m = PAGE_MARKER.match(stripped)
            if m:
                title = None
                # Look further (up to 12 lines) since some pages have
                # blank lines or stray section headers between the page
                # number and the actual title.
                for j in range(i + 1, min(i + 12, len(lines))):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    if _is_chord_line(cand):
                        continue
                    if _looks_like_title(cand):
                        title = _normalize_title(cand)
                        break
                if title:
                    current_title = title
                    current_section = "Intro"
                i += 1
                continue

            # Section header
            if current_title and _is_section_header(stripped):
                current_section = stripped
                i += 1
                continue

            # Chord line: append to (title, section) buffer.
            if current_title and _is_chord_line(stripped):
                key = (current_title, current_section)
                buf.setdefault(key, []).append(stripped)

            i += 1

    # Coalesce buffered chord lines into one chord-string per section.
    for (title, section), chord_lines in buf.items():
        # Join with a single bar between lines so the chord parser sees
        # them as continuous bars.
        joined = " | ".join(chord_lines)
        out.setdefault(title, []).append((section, joined))

    return out


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    pdf = repo / "jerry-garcia-song-book-ver-9-online.pdf"
    data = extract_chord_data(pdf)
    print(f"Extracted chord data for {len(data)} songs.")
    for title in sorted(data.keys())[:5]:
        secs = data[title]
        n = sum(c.count("|") for _, c in secs)
        print(f"  {title}: {len(secs)} sections, ~{n} bars")


if __name__ == "__main__":
    main()
