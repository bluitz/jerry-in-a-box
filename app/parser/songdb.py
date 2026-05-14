"""Song database compiler.

Reads:
    sources/jerry_song_book.csv          (raw measure strings, per section)
    jerry-in-a-box/data/songs.json       (legacy DB; some songs have nice
                                          per-section "chords" arrays already)

Writes:
    app/data/songs.json                  (canonical: chord sequences + features)

Each song record looks like:

    {
      "id":             "ripple",
      "title":          "Ripple",
      "artist":         "Grateful Dead",
      "key_pc":         7,                   # estimated key (pitch class)
      "key_mode":       "maj"|"min",
      "chords":         ["G", "C", "G", "D", ...],     # canonical names
      "chord_tuples":   [[7,"maj",null], ...],         # for the matcher
      "pc_histogram":   [12 floats summing to 1.0],    # bag-of-notes prior
      "bigram":         [[12*Q, 12*Q]],                # transition matrix
                                                      # over distinct chord types
      "chord_vocab":    ["G","C","D",...],             # ordered, indexes bigram
      "sections":       [
          { "name": "Verse 1", "chords": [...] },
          ...
      ]
    }

Lyrics are intentionally NEVER stored. The PDF page index (page_index.json)
maps title -> page number(s) so the frontend can jump to the rendered page.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

from app.parser.chords import (
    Chord,
    parse_chord,
    parse_chord_string,
    PC_TO_NAME,
)
from app.parser.pdf_extract import extract_chord_data


REPO = Path(__file__).resolve().parents[2]
CSV_PATH = REPO / "sources" / "jerry_song_book.csv"
LEGACY_JSON = REPO / "jerry-in-a-box" / "data" / "songs.json"
PDF_PATH = REPO / "jerry-garcia-song-book-ver-9-online.pdf"
OUT_PATH = REPO / "app" / "data" / "songs.json"


def _slugify(title: str) -> str:
    s = re.sub(r"[^\w\s-]", "", title).strip().lower()
    s = re.sub(r"[-\s]+", "-", s)
    return s


# Krumhansl-Kessler key profiles (relative weights for tonic-relative pcs).
# Used for a cheap key estimate from the song's pitch-class histogram.
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def _estimate_key(pc_hist: list[float]) -> tuple[int, str]:
    """Return (root_pc, mode) maximizing dot-product with KS profile."""
    best = (0, "maj", -1.0)
    for tonic in range(12):
        rot = pc_hist[-tonic % 12:] + pc_hist[: -tonic % 12] if tonic else pc_hist
        # Rotated so that index 0 corresponds to tonic.
        rot = [pc_hist[(tonic + i) % 12] for i in range(12)]
        maj = sum(a * b for a, b in zip(rot, _KS_MAJOR))
        minr = sum(a * b for a, b in zip(rot, _KS_MINOR))
        if maj > best[2]:
            best = (tonic, "maj", maj)
        if minr > best[2]:
            best = (tonic, "min", minr)
    return best[0], best[1]


def _pc_histogram(chords: list[Chord]) -> list[float]:
    """Histogram of chord-tone pitch classes across all chord events.

    Each chord contributes uniformly across its tones (so a triad puts 1/3
    weight on each tone, a 7th chord 1/4, etc.). Then we normalize.
    """
    hist = [0.0] * 12
    if not chords:
        return hist
    for c in chords:
        if not c.tones:
            continue
        w = 1.0 / len(c.tones)
        for t in c.tones:
            hist[t] += w
    s = sum(hist)
    if s > 0:
        hist = [x / s for x in hist]
    return hist


def _build_bigram(chords: list[Chord]) -> tuple[list[str], list[list[float]]]:
    """Build a chord-bigram transition matrix using deduplicated chord runs.

    "Chord runs" = collapse consecutive identical chords down to one. This
    is the structure that matters for matching (the *changes*), and it
    makes the matcher robust to per-beat resolution choices.
    """
    if not chords:
        return [], []

    # Collapse runs.
    runs: list[Chord] = []
    for c in chords:
        if not runs or runs[-1].to_tuple() != c.to_tuple():
            runs.append(c)

    # Build vocab.
    vocab_order: list[str] = []
    vocab_idx: dict[str, int] = {}
    for c in runs:
        n = c.name
        if n not in vocab_idx:
            vocab_idx[n] = len(vocab_order)
            vocab_order.append(n)
    n = len(vocab_order)

    # Counts with Laplace smoothing (alpha=0.5) so unseen transitions
    # have nonzero probability — important because the synthetic generator
    # injects skips/repeats.
    alpha = 0.5
    counts = [[alpha] * n for _ in range(n)]
    for a, b in zip(runs, runs[1:]):
        counts[vocab_idx[a.name]][vocab_idx[b.name]] += 1.0

    # Row-normalize.
    bigram: list[list[float]] = []
    for row in counts:
        s = sum(row)
        bigram.append([x / s for x in row] if s else [1.0 / n] * n)
    return vocab_order, bigram


def _flatten_legacy_sections(legacy_song: dict) -> list[Chord]:
    """Pull a flat chord list out of a legacy songs.json record.

    Prefer per-section "chords" arrays (already expanded by an older parser).
    Fall back to the top-level "progression" array.
    """
    sections = legacy_song.get("sections") or {}
    chords: list[Chord] = []
    if isinstance(sections, dict) and sections:
        for sec_data in sections.values():
            if isinstance(sec_data, dict) and "chords" in sec_data:
                for s in sec_data["chords"]:
                    c = parse_chord(s)
                    if c is not None:
                        chords.append(c)
    if chords:
        return chords
    # Fallback: top-level progression
    for s in legacy_song.get("progression") or []:
        c = parse_chord(s)
        if c is not None:
            chords.append(c)
    return chords


def _read_csv_sections(csv_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Group CSV rows by title -> list of (section_name, chord_string)."""
    out: dict[str, list[tuple[str, str]]] = {}
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            title, section, chords_str = row[0].strip(), row[1].strip(), row[2].strip()
            if not title:
                continue
            out.setdefault(title, []).append((section, chords_str))
    return out


def compile_song_db() -> dict:
    """Compile the canonical song DB. Returns the dict that gets written
    to ``app/data/songs.json``."""

    # Load legacy JSON keyed by lowercase title.
    legacy_records: dict[str, dict] = {}
    if LEGACY_JSON.exists():
        with LEGACY_JSON.open() as f:
            for rec in json.load(f):
                legacy_records[rec["title"].strip().lower()] = rec

    csv_records = _read_csv_sections(CSV_PATH)

    # Pull chord progressions directly from the PDF for any song the CSV
    # didn't cover (the CSV only has ~75 songs; the book has ~250).
    pdf_records: dict[str, list[tuple[str, str]]] = {}
    if PDF_PATH.exists():
        try:
            pdf_records = extract_chord_data(PDF_PATH)
        except Exception as e:
            print(f"warning: PDF extraction failed: {e}")

    titles = sorted(
        set(legacy_records.keys())
        | {t.lower() for t in csv_records.keys()}
        | {t.lower() for t in pdf_records.keys()}
    )

    # Build a title -> canonical-display-title map (prefer the casing used
    # in the legacy JSON when available; else the CSV).
    canon_title: dict[str, str] = {}
    for t_lower in titles:
        if t_lower in legacy_records:
            canon_title[t_lower] = legacy_records[t_lower]["title"]
            continue
        for csv_title in csv_records.keys():
            if csv_title.lower() == t_lower:
                canon_title[t_lower] = csv_title
                break
        if t_lower in canon_title:
            continue
        for pdf_title in pdf_records.keys():
            if pdf_title.lower() == t_lower:
                canon_title[t_lower] = pdf_title
                break

    songs_out: list[dict] = []
    for t_lower in titles:
        title = canon_title[t_lower]

        # Prefer CSV sections (richer); fall back to legacy.
        csv_secs = csv_records.get(title) or csv_records.get(title.lower()) or []
        # Title-casing in CSV may not match exactly: try a relaxed match.
        if not csv_secs:
            for k, v in csv_records.items():
                if k.strip().lower() == t_lower:
                    csv_secs = v
                    break

        sections_out: list[dict] = []
        all_chords: list[Chord] = []
        source_used = ""
        if csv_secs:
            for section_name, raw in csv_secs:
                ch = parse_chord_string(raw)
                if not ch:
                    continue
                sections_out.append({
                    "name": section_name,
                    "chords": [c.name for c in ch],
                })
                all_chords.extend(ch)
            if all_chords:
                source_used = "csv"

        if not all_chords:
            # Fall back to PDF-extracted progressions.
            pdf_secs = []
            for k, v in pdf_records.items():
                if k.lower() == t_lower:
                    pdf_secs = v
                    break
            for section_name, raw in pdf_secs:
                ch = parse_chord_string(raw)
                if not ch:
                    continue
                sections_out.append({
                    "name": section_name,
                    "chords": [c.name for c in ch],
                })
                all_chords.extend(ch)
            if all_chords:
                source_used = "pdf"

        if not all_chords and t_lower in legacy_records:
            all_chords = _flatten_legacy_sections(legacy_records[t_lower])
            if all_chords:
                sections_out = [{"name": "All", "chords": [c.name for c in all_chords]}]
                source_used = "legacy"

        if not all_chords:
            continue

        artist = ""
        if t_lower in legacy_records:
            artist = legacy_records[t_lower].get("artist", "")
        if not artist:
            artist = "Grateful Dead"

        pc_hist = _pc_histogram(all_chords)
        key_pc, key_mode = _estimate_key(pc_hist)
        vocab, bigram = _build_bigram(all_chords)

        # Collapse runs for chord_seq + chord_tuples (one entry per chord change).
        run_chords: list[Chord] = []
        for c in all_chords:
            if not run_chords or run_chords[-1].to_tuple() != c.to_tuple():
                run_chords.append(c)

        songs_out.append({
            "id":           _slugify(title),
            "title":        title,
            "artist":       artist,
            "source":       source_used,
            "key_pc":       key_pc,
            "key_mode":     key_mode,
            "key_name":     PC_TO_NAME[key_pc] + ("" if key_mode == "maj" else "m"),
            "chords":       [c.name for c in run_chords],
            "chord_tuples": [[c.root_pc, c.quality, c.bass_pc] for c in run_chords],
            "pc_histogram": pc_hist,
            "chord_vocab":  vocab,
            "bigram":       bigram,
            "n_chord_events": len(all_chords),
            "n_unique_chords": len(set(c.name for c in all_chords)),
            "sections":     sections_out,
        })

    db = {
        "schema_version": 1,
        "songs":          songs_out,
        "n_songs":        len(songs_out),
    }
    return db


def _quality_index(quality: str) -> int:
    from app.parser.chords import QUALITIES
    try:
        return QUALITIES.index(quality)
    except ValueError:
        return 0


def main() -> None:
    db = compile_song_db()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(db, f, indent=1)
    print(f"Wrote {OUT_PATH}: {db['n_songs']} songs")


if __name__ == "__main__":
    main()
