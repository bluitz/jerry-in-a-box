"""Extract chord progressions directly from the Jerry Garcia Songbook PDF.

We do NOT extract or store any lyric/prose lines — only chord-bearing
lines (those containing the bar character `|` and at least two chord-
like tokens) and section-header lines (Verse / Chorus / Bridge / Solo /
Intro / Outro / Tag / Break / Coda...).

The songbook prints "Page N" before each song's first page. We use that
as a hard reset point and look for the song's title in the next few
lines. Many songs in this book inline the title with the first chord
progression (e.g. "Far From Me  Intro  | D7 / / / | ..."), so we
also handle title-on-chord-line splitting.

Output shape mirrors what songdb.py expects via _read_csv_sections:

    {title: [(section_name, chord_string), ...]}
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF


# A page in the songbook starts with "Page <N>" on its own line.
PAGE_MARKER = re.compile(r"^\s*Page\s+(\d+)\s*$")

# A chord token: roots (A-G), optional accidental, optional quality and
# extensions, optional bass after a slash.
CHORD_TOK = re.compile(
    r"\b("
    r"[A-G][#b]?"                         # root
    r"(?:maj|min|m|dim|aug|sus|add)?"     # quality
    r"(?:[0-9]+)?"                        # extension
    r"(?:/[A-G][#b]?)?"                   # bass
    r")"
    r"(?=\b|[A-G])"                       # boundary OR next-chord-start
)

# Section names that appear as their own line (Verse / Chorus / etc.)
# OR inline before a chord progression on the same line as the title.
SECTION_NAMES = (
    "intro", "outro", "verse", "chorus", "refrain", "bridge", "tag",
    "break", "coda", "solo", "riff", "interlude", "pre-chorus",
    "prechorus", "hook", "turnaround", "ending", "vamp", "jam",
    "instrumental", "head", "lead",
)
SECTION_TOK = re.compile(
    r"^\s*(?:" + "|".join(re.escape(n) for n in SECTION_NAMES) + r")\b",
    re.IGNORECASE,
)
# Anywhere-in-line section keyword (used to split title from chord
# progression on a single line). We require a word boundary BEFORE so
# we don't match e.g. "Intro" inside a longer English word.
SECTION_INLINE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(n) for n in SECTION_NAMES) + r")\b",
    re.IGNORECASE,
)


def _is_tab_line(line: str) -> bool:
    """Return True for guitar TAB notation lines.

    TAB lines look like "e|-1----1----1----0-0-|---" or "||-Bob Intro--"
    and contain bars + chord-like tokens, so they fool a naive chord
    detector. We skip them outright.
    """
    s = line.strip()
    if not s:
        return False
    # Standard 6-string TAB: starts with note name + bar.
    if re.match(r"^[eEBGDAg]\s*\|", s):
        return True
    # Continuation TAB starts with multiple bars and dashes.
    if re.match(r"^\|\|?-", s):
        return True
    # TAB heavy in dashes / digits (heuristic).
    if s.count("-") >= 8 and "|" in s:
        # If the line is mostly dashes/digits/bars, it's TAB.
        non_tab = sum(1 for c in s if c not in "-|/0123456789hpsv^\\bBcCxX#~ \t")
        if non_tab / max(1, len(s)) < 0.10:
            return True
    return False


# Tokens that may appear on a chord line WITHOUT being chord names, e.g.
# "Intro = C F Am C", "Verse 1 G C D", "2 x A D | %", "Chorus: D G".
_CHORD_LINE_ALLOW = {
    "intro", "outro", "verse", "chorus", "refrain", "bridge", "tag",
    "break", "coda", "solo", "riff", "interlude", "prechorus",
    "pre-chorus", "hook", "turnaround", "ending", "vamp", "jam",
    "instrumental", "head", "lead", "bobs",
    "x", "rpt", "repeat",
    "=", ":", "%", "*", "-", ".", "/", "//", "///",
    "||", "||:", "|", ":||", ":||:",
}


def _is_chord_line(line: str) -> bool:
    """Two formats are accepted:

    1. Bar-form: contains `|` and at least two chord tokens.
       e.g. "|| : G / / / | C / G / | G / D / |"

    2. Spaced chord-positions form (chord names laid out above the
       lyrics, no bars). e.g. "C   F   Am   C" or "Intro = C F Am C".
       Detected when virtually every token on the line is a chord
       token, a punctuation mark, or one of a small allowlist
       (section names, "x", "=", "%", numerics).

    TAB lines and lines that look like English (>1 unrecognised word)
    are rejected.
    """
    s = line.strip()
    if not s or _is_tab_line(s):
        return False

    chord_tokens = CHORD_TOK.findall(s)
    has_bar = "|" in s
    if has_bar and len(chord_tokens) >= 2:
        return True
    # Single-chord-on-its-own-line form: chord names laid out above
    # lyrics, one per line. The line must be just one short chord-like
    # token (with optional whitespace).
    if (len(chord_tokens) == 1 and not has_bar
            and len(s) <= 8
            and re.fullmatch(r"\s*[A-G][#b]?[a-z0-9/#b]*\s*", s)):
        return True
    if len(chord_tokens) < 2:
        return False

    # Spaced-chord-positions form: tokenize, count non-chord tokens.
    # If everything except chord tokens is in the allowlist, it's a
    # chord line. We allow at most ONE unrecognised token (e.g. a stray
    # "to" or "the" if it happens to be on the chord-line row).
    chord_set = set(chord_tokens)
    unrecognised = 0
    for tok in re.findall(r"\S+", s):
        # Strip surrounding punctuation that we treat as separators.
        core = tok.strip(",.;:()[]")
        if not core:
            continue
        if core in chord_set:
            continue
        # Allowlist (case-insensitive).
        if core.lower() in _CHORD_LINE_ALLOW:
            continue
        # Numeric repeat counts ("2", "3x", "4").
        if re.fullmatch(r"\d+x?", core, re.IGNORECASE):
            continue
        # Pure punctuation/repeat-bar/percent.
        if re.fullmatch(r"[|/:%=\-.,*]+", core):
            continue
        # If core itself is just chord-token-ish (catch CHORD_TOK
        # variants the regex didn't match because of slash bass).
        if re.fullmatch(r"[A-G][#b]?[a-z0-9/]*", core):
            continue
        unrecognised += 1
    return unrecognised <= 1


def _strip_subtitle(s: str) -> str:
    """Strip noise decorations from a title string.

    Handles parentheticals (only the LONG ones — short parens like "(in
    F)" or "(Ballad of)" are kept, since they discriminate between
    distinct entries in the songbook), em/en-dashes (but NOT regular
    hyphens — those appear in legit titles like "1 - 4 - 5"), the
    Unicode arrow `\\uf0e0` (font glyph for medley arrows), explicit
    "as performed by..." attributions, embedded quoted asides like
    'Title "Slogan" Blues', and double-spaced suffix annotations.
    """
    s = s.strip()
    # Strip ONLY long parentheticals (>=18 chars of content). Short
    # parens like "(in F)" or "(Ballad of)" disambiguate two distinct
    # entries (Brokedown Palace in F vs in G, Casey Jones vs Casey
    # Jones (Ballad of)) and must be preserved.
    s = re.sub(r"\s*\([^)]{18,}\)\s*", " ", s)
    # Strip embedded quoted asides + everything after them.
    s = re.split(r'\s+["“][^"”]*["”]', s)[0]
    # Strip after a long run of spaces (≥4) — used in this PDF to
    # separate title from inline metadata like "6/8 time where / = ///".
    s = re.split(r"\s{4,}", s)[0]
    # Strip after em/en-dash if it introduces an "as performed" or
    # "Original is in..." style annotation. We keep dashes that
    # introduce structural subtitles (e.g. "Terrapin Station – At A
    # Siding") because those discriminate distinct entries.
    s = re.split(r"\s+[\u2013\u2014–—]\s+(?:Grateful|original|originally|"
                 r"Performed|Played|Live|in\s+[A-G][#b]?\s*$)",
                 s, flags=re.IGNORECASE)[0]
    # Strip "as performed by ..." style suffixes.
    s = re.split(r"\s+as\s+performed\b", s, flags=re.IGNORECASE)[0]
    # Strip the medley-arrow glyph and everything after it (e.g.
    # "Baba O'Riley → Tomorrow Never Knows" — the arrow is a hint that
    # the line is a medley header that's longer than a title).
    s = re.split(r"\s*[\uf0e0→\u2192]\s*", s)[0]
    return s.strip()


def _normalize_title(s: str) -> str:
    s = _strip_subtitle(s)
    # Strip trailing period if it looks like a sentence end.
    s = s.rstrip(" .,;:")
    return s


def _looks_like_title(s: str) -> bool:
    """Return True if s plausibly is a song title (after subtitle-stripping).

    Real titles in this PDF tend to be 1-10 words, mostly Title Case,
    not ending with sentence punctuation, and not containing chord
    bars. Many do contain commas ("Oh, the Wind and the Rain"), so we
    no longer reject those.
    """
    s = _strip_subtitle(s)
    if not s or len(s) > 80 or len(s) < 2:
        return False
    if "|" in s:
        return False
    if SECTION_TOK.match(s):
        return False
    if _is_tab_line(s):
        return False
    # Reject lines that are basically tab-fret labels or pure numbers.
    if re.fullmatch(r"\d+", s):
        return False
    if s.count("-") > 4:
        return False
    # Reject likely lyric lines: end with sentence punctuation OR
    # have lots of words.
    if s.endswith((".", "!", "?", ";", ":")):
        return False
    n_words = len(s.split())
    if n_words > 12:
        return False
    # Lowercase-only multi-word lines are almost always lyrics
    # (titles in this book are Title Case or ALL CAPS).
    letters = [c for c in s if c.isalpha()]
    if letters and n_words >= 2:
        upper_frac = sum(1 for c in letters if c.isupper()) / len(letters)
        if upper_frac < 0.05:
            return False
    return True


def _split_title_and_chord(line: str) -> tuple[str | None, str | None]:
    """If a line begins with a song title and continues with a chord
    progression on the same physical PDF line, split it.

    Examples this handles:
        "Far From Me     Intro  | D7  /  /  / | D7 / / G D |"
        "Lost Sailor    Intro 4 x||:  F#m7 / / / |"
        "Pretty Peggy-O ||: A / / /|D / / / |"
        "Lucky ol' Sun  | C / / / |    %    |"
        "Tangled Up In Blue 2 x ||:  A  /  G  / |"

    Returns (title, chord_progression) or (None, None) if no plausible
    title can be carved off the front.
    """
    bar_idx = line.find("|")
    if bar_idx < 0:
        return None, None
    # First section-keyword position (e.g. "Intro" within the line).
    sec = SECTION_INLINE.search(line)
    sec_idx = sec.start() if sec else len(line)
    cut = min(bar_idx, sec_idx)
    if cut <= 0:
        return None, None
    title_part = line[:cut].strip()
    rest = line[cut:].strip()
    # Title must look like a title; rest must look like chord content
    # (have at least one bar and one chord token).
    if not _looks_like_title(title_part):
        return None, None
    if "|" not in rest or not CHORD_TOK.search(rest):
        return None, None
    # Title must not be a single chord token (e.g. just "G").
    if CHORD_TOK.fullmatch(title_part) is not None:
        return None, None
    return _normalize_title(title_part), rest


def extract_chord_data(pdf_path: Path,
                       skip_first_pages: int = 4
                       ) -> dict[str, list[tuple[str, str]]]:
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

            # Page-marker -> hunt for the title in the next 14 lines.
            m = PAGE_MARKER.match(stripped)
            if m:
                page_no = int(m.group(1))
                title: str | None = None
                first_chord_on_title_line: str | None = None

                for j in range(i + 1, min(i + 14, len(lines))):
                    cand = lines[j].strip()
                    if not cand:
                        continue
                    if _is_tab_line(cand):
                        # TAB-continuation — keep looking, don't take
                        # this line and don't burn through the budget.
                        continue
                    # Try a clean title line first.
                    if _looks_like_title(cand):
                        title = _normalize_title(cand)
                        break
                    # Try the title-+-chord-on-one-line split.
                    t, rest = _split_title_and_chord(cand)
                    if t:
                        title = t
                        first_chord_on_title_line = rest
                        break
                    # Pure chord line — ignore, keep looking.
                    if _is_chord_line(cand):
                        continue
                    # Anything else (lyric, etc.) — keep looking.
                    continue

                # If we found a title, this is a new song — reset
                # current_title and current_section, and register the
                # title even if no chord lines arrive (TAB- or
                # lyrics-only entries still belong in the roster).
                #
                # If we did NOT find a title, two cases apply:
                #   (a) the page is explicitly blank ("blank page"
                #       text) — leave current_title alone, do nothing.
                #   (b) the page is a continuation page (TAB,
                #       lyrics, or lead sheet of the previous song —
                #       e.g. Chimes of Freedom intro on Page 48 after
                #       Page 47 blank). Leave current_title alone and
                #       let chord lines from this page extend the
                #       previous song's progression.
                if title:
                    current_title = title
                    current_section = "Intro"
                    buf.setdefault((current_title, current_section), [])
                    if first_chord_on_title_line:
                        buf[(current_title, current_section)].append(
                            first_chord_on_title_line)
                # else: keep current_title as-is.
                i += 1
                continue

            # Section header.
            if current_title and _is_section_header_line(stripped):
                current_section = stripped
                i += 1
                continue

            # Chord line: append to (title, section) buffer.
            if current_title and _is_chord_line(stripped):
                key = (current_title, current_section)
                buf.setdefault(key, []).append(stripped)

            i += 1

    # Coalesce buffered chord lines into one chord-string per section.
    # Empty sections are kept so songs with no extractable chords
    # (lyrics-only or TAB-only entries) still appear in the roster.
    for (title, section), chord_lines in buf.items():
        joined = " | ".join(chord_lines)
        out.setdefault(title, []).append((section, joined))

    return out


def _is_section_header_line(line: str) -> bool:
    s = line.strip()
    if not s or "|" in s or len(s) > 40:
        return False
    return SECTION_TOK.match(s) is not None


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
