"""Build a title -> page-number index for the songbook PDF.

We use PyMuPDF (fitz). The script reads the PDF only to identify which
PAGE NUMBER each song title appears on. The output JSON contains:

    { "song-id": [page_num, ...], ... }

No lyric text, body text, or other PDF content is extracted or stored.
The PDF itself is rendered to the user via pdf.js in the browser; this
index just tells the frontend which page to jump to.

Title detection strategy:
  1) If the PDF has built-in bookmarks / outline / TOC, use them — fast
     and unambiguous.
  2) Otherwise, scan each page for short text spans that fuzzy-match a
     known song title from songs.json, preferring large-font spans near
     the top of the page. We don't keep any matched text — only page
     numbers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz  # PyMuPDF


REPO = Path(__file__).resolve().parents[2]
PDF_PATH = REPO / "jerry-garcia-song-book-ver-9-online.pdf"
SONGS_JSON = REPO / "app" / "data" / "songs.json"
OUT_PATH = REPO / "app" / "data" / "page_index.json"


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)  # strip parentheticals like "(in F)"
    s = re.sub(r"^\s*\d+\s*[.)\-]\s*", "", s)  # strip "1. " or "47) " prefixes
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _from_outline(doc: fitz.Document, titles: dict[str, str]) -> dict[str, list[int]]:
    """Try to populate from the PDF outline. Returns {} if no outline.

    Some PDFs (including ours) ship a TOC where every entry points to the
    same TOC page. Detect and reject this case.
    """
    toc = doc.get_toc(simple=True) or []
    if not toc:
        return {}
    # Reject the TOC if the vast majority of entries point at the same
    # page (a degenerate index where every link goes to the TOC itself).
    from collections import Counter
    page_counts = Counter(entry[2] for entry in toc)
    most_common_pg, most_common_n = page_counts.most_common(1)[0]
    if most_common_n / len(toc) > 0.5:
        return {}

    out: dict[str, list[int]] = {}
    norm_titles = {_normalize(v): k for k, v in titles.items()}
    for level, title, page in toc:
        nt = _normalize(title)
        if not nt:
            continue
        sid = norm_titles.get(nt)
        if sid is None:
            for k, v in norm_titles.items():
                if k == nt or k in nt or nt in k:
                    sid = v
                    break
        if sid is not None:
            out.setdefault(sid, []).append(int(page))
    for sid in out:
        out[sid] = sorted(set(out[sid]))
    return out


def _from_page_scan(doc: fitz.Document, titles: dict[str, str]) -> dict[str, list[int]]:
    """Per-page scan: only inspect short, large-font, top-of-page spans
    and check whether any of them fuzzy-matches a known title.

    We never STORE the matched text. We only record (song_id -> page).
    """
    norm_titles = {sid: _normalize(t) for sid, t in titles.items()}
    out: dict[str, list[int]] = {}

    for page_num, page in enumerate(doc, start=1):
        # Skip cover (1) and TOC (2). The TOC is unreliable in this PDF.
        if page_num <= 2:
            continue
        try:
            data = page.get_text("dict")
        except Exception:
            continue
        page_height = page.rect.height
        candidates: list[tuple[float, str]] = []
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                # Combine spans on the same line
                text = " ".join(span.get("text", "") for span in line.get("spans", []))
                text = text.strip()
                if not text or len(text) > 80:
                    continue
                # Top of page only — song titles are at the top
                y = line["bbox"][1]
                if y > 0.15 * page_height:
                    continue
                # Largest font on the line; song titles are size >= 16
                max_size = max((s.get("size", 0) for s in line.get("spans", [])), default=0)
                if max_size < 15:
                    continue
                candidates.append((max_size, text))

        if not candidates:
            continue
        # Try the largest-font candidate first.
        candidates.sort(key=lambda x: -x[0])
        for _size, text in candidates[:5]:
            nt = _normalize(text)
            if not nt:
                continue
            for sid, ntitle in norm_titles.items():
                if ntitle == nt or (len(ntitle) > 4 and ntitle in nt) or (len(nt) > 4 and nt in ntitle):
                    out.setdefault(sid, []).append(page_num)
                    break

    for sid in out:
        out[sid] = sorted(set(out[sid]))
    return out


def build_page_index() -> dict[str, list[int]]:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
    if not SONGS_JSON.exists():
        raise FileNotFoundError(f"songs.json not found at {SONGS_JSON}; run songdb first")

    songs = json.loads(SONGS_JSON.read_text())["songs"]
    titles = {s["id"]: s["title"] for s in songs}

    with fitz.open(PDF_PATH) as doc:
        # The PDF's outline points many entries at the TOC pages themselves;
        # the page-scan is more reliable for this particular PDF. Always
        # prefer the page-scan and only fall back to the outline if the
        # scan can't find anything.
        out = _from_page_scan(doc, titles)
        if not out:
            out = _from_outline(doc, titles)
        # Keep only the FIRST page hit per song (the title page).
        first_pages = {sid: pages[0] for sid, pages in out.items() if pages}
    return {sid: [pg] for sid, pg in first_pages.items()}


def main() -> None:
    idx = build_page_index()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(idx, indent=1))
    print(f"Wrote {OUT_PATH}: {len(idx)} songs indexed")


if __name__ == "__main__":
    main()
