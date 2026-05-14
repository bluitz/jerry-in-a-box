"""Audit script: prints a sanity table for the compiled song DB.

Shows: title, key, # chord events, unique chord count, first 8 chord changes.
This is intentionally chord-only — no lyrics or other PDF content.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.parser.songdb import OUT_PATH


def main(limit: int | None = None) -> None:
    db = json.loads(OUT_PATH.read_text())
    songs = db["songs"]
    if limit:
        songs = songs[:limit]
    print(f"{'TITLE':<35} {'KEY':<5} {'#EV':>5} {'#UQ':>4}  FIRST CHORDS")
    print("-" * 100)
    for s in songs:
        first8 = " ".join(s["chords"][:8])
        print(
            f"{s['title'][:34]:<35} "
            f"{s['key_name']:<5} "
            f"{s['n_chord_events']:>5} "
            f"{s['n_unique_chords']:>4}  "
            f"{first8}"
        )
    print(f"\n{len(songs)} songs")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(n)
