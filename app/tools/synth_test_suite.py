"""Batch synth-test suite: synthesize audio for every song with chord data,
run each through the file-direct matcher pipeline, and report per-song
pass/fail with overall accuracy metrics.

Usage:
  python -m app.tools.synth_test_suite
  python -m app.tools.synth_test_suite --threshold 3 --failures-only
  python -m app.tools.synth_test_suite --jobs 4 --report results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _test_one(song_id: str, duration: float, chord_dur: float,
              bin_seconds: float) -> dict:
    """Synthesize + detect + rank a single song.  Returns a result dict.

    Isolated as a top-level function so it can be pickled for
    multiprocessing.
    """
    from app.tools.synth_song import synth_song
    from app.tools.replay_audio import detect_chords
    from app.matcher.sequence_matcher import SequenceMatcher

    songs_path = REPO / "app" / "data" / "songs.json"
    songs = json.loads(songs_path.read_text())["songs"]

    t0 = time.monotonic()
    try:
        audio = synth_song(song_id, total_duration_s=duration,
                           chord_dur_s=chord_dur)
    except (KeyError, ValueError) as e:
        return {"song_id": song_id, "error": str(e), "rank": None,
                "elapsed": 0.0}

    segs = detect_chords(audio, 44100, bin_seconds=bin_seconds)
    matcher = SequenceMatcher(songs)
    for seg in segs:
        matcher.add_chord(seg["root"], seg["quality"])
    ranked = matcher.softmax_post()

    ids = [sid for sid, _ in ranked]
    rank = (ids.index(song_id) + 1) if song_id in ids else None
    top1_id = ids[0] if ids else None
    top1_title = matcher.titles.get(top1_id, "?") if top1_id else "?"
    elapsed = time.monotonic() - t0

    return {
        "song_id": song_id,
        "rank": rank,
        "top1_id": top1_id,
        "top1_title": top1_title,
        "n_chords": len(segs),
        "elapsed": round(elapsed, 2),
        "error": None,
    }


def run_suite(duration: float = 20.0, chord_dur: float = 2.0,
              bin_seconds: float = 2.0, threshold: int = 1,
              jobs: int = 1, failures_only: bool = False,
              report_path: Path | None = None) -> dict:
    songs_path = REPO / "app" / "data" / "songs.json"
    all_songs = json.loads(songs_path.read_text())["songs"]
    testable = [s for s in all_songs if s.get("chord_tuples")]
    n = len(testable)
    ids = [s["id"] for s in testable]
    titles = {s["id"]: s["title"] for s in testable}

    print(f"Testing {n} songs (threshold=rank<={threshold}, "
          f"duration={duration}s, chord_dur={chord_dur}s, jobs={jobs})",
          file=sys.stderr)

    results: list[dict] = []
    t_start = time.monotonic()

    if jobs <= 1:
        for i, sid in enumerate(ids):
            r = _test_one(sid, duration, chord_dur, bin_seconds)
            r["title"] = titles[sid]
            results.append(r)
            _print_line(i + 1, n, r, threshold, failures_only)
    else:
        futures = {}
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for i, sid in enumerate(ids):
                f = pool.submit(_test_one, sid, duration, chord_dur,
                                bin_seconds)
                futures[f] = (i, sid)
            done_order = 0
            for f in as_completed(futures):
                done_order += 1
                idx, sid = futures[f]
                r = f.result()
                r["title"] = titles[sid]
                results.append(r)
                _print_line(done_order, n, r, threshold, failures_only)

    wall = time.monotonic() - t_start

    results.sort(key=lambda r: r["song_id"])
    summary = _summarize(results, threshold, wall)
    _print_summary(summary)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(
            {"summary": summary, "per_song": results}, indent=2))
        print(f"\nReport written to {report_path}", file=sys.stderr)

    return summary


def _print_line(i: int, n: int, r: dict, threshold: int,
                failures_only: bool) -> None:
    rank = r["rank"]
    passed = rank is not None and rank <= threshold
    if failures_only and passed and r["error"] is None:
        return
    tag = "PASS" if passed else ("ERR " if r["error"] else "FAIL")
    rank_s = f"rank={rank}" if rank else "rank=?"
    extra = ""
    if not passed and r.get("top1_id") and r["top1_id"] != r["song_id"]:
        extra = f"  top1={r['top1_title']!r}"
    print(f"[{i:3d}/{n}] {tag}  {r['song_id']:<42} {rank_s:<8}"
          f"  ({r['elapsed']:.1f}s){extra}", file=sys.stderr)


def _summarize(results: list[dict], threshold: int,
               wall: float) -> dict:
    n = len(results)
    errors = [r for r in results if r["error"]]
    valid = [r for r in results if not r["error"]]
    nv = len(valid)

    def count_at(k: int) -> int:
        return sum(1 for r in valid if r["rank"] is not None and r["rank"] <= k)

    top1 = count_at(1)
    top3 = count_at(3)
    top5 = count_at(5)
    top10 = count_at(10)
    at_threshold = count_at(threshold)

    failures = sorted(
        [r for r in valid if r["rank"] is None or r["rank"] > threshold],
        key=lambda r: (r["rank"] or 9999),
    )

    return {
        "n_songs": n,
        "n_testable": nv,
        "n_errors": len(errors),
        "threshold": threshold,
        "pass": at_threshold,
        "pass_pct": round(at_threshold / max(1, nv) * 100, 1),
        "top1": top1,
        "top1_pct": round(top1 / max(1, nv) * 100, 1),
        "top3": top3,
        "top3_pct": round(top3 / max(1, nv) * 100, 1),
        "top5": top5,
        "top5_pct": round(top5 / max(1, nv) * 100, 1),
        "top10": top10,
        "top10_pct": round(top10 / max(1, nv) * 100, 1),
        "wall_seconds": round(wall, 1),
        "failures": [
            {"song_id": r["song_id"], "title": r["title"],
             "rank": r["rank"], "confused_with": r.get("top1_title")}
            for r in failures[:50]
        ],
        "errors": [{"song_id": r["song_id"], "error": r["error"]}
                   for r in errors],
    }


def _print_summary(s: dict) -> None:
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  {s['n_testable']} songs tested  "
          f"({s['n_errors']} errors)  "
          f"[{s['wall_seconds']}s wall]", file=sys.stderr)
    print(f"  top1:  {s['top1']}/{s['n_testable']}  ({s['top1_pct']}%)",
          file=sys.stderr)
    print(f"  top3:  {s['top3']}/{s['n_testable']}  ({s['top3_pct']}%)",
          file=sys.stderr)
    print(f"  top5:  {s['top5']}/{s['n_testable']}  ({s['top5_pct']}%)",
          file=sys.stderr)
    print(f"  top10: {s['top10']}/{s['n_testable']}  ({s['top10_pct']}%)",
          file=sys.stderr)
    if s["failures"]:
        print(f"\n  {len(s['failures'])} failures (rank>{s['threshold']}):",
              file=sys.stderr)
        for f in s["failures"][:30]:
            print(f"    {f['song_id']:<42} rank={f['rank']:<4}  "
                  f"confused_with={f['confused_with']!r}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Run synth detection test for all songs")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Seconds of synth audio per song (default 20)")
    p.add_argument("--chord-dur", type=float, default=2.0,
                   help="Seconds per chord pluck (default 2.0)")
    p.add_argument("--bin-seconds", type=float, default=2.0,
                   help="Detection bin width (default 2.0)")
    p.add_argument("--threshold", type=int, default=1,
                   help="Pass if rank <= this (default 1)")
    p.add_argument("--jobs", type=int, default=1,
                   help="Parallel workers (default 1)")
    p.add_argument("--failures-only", action="store_true",
                   help="Only print failures")
    p.add_argument("--report", type=Path, default=None,
                   help="Write JSON report to this path")
    args = p.parse_args()

    summary = run_suite(
        duration=args.duration,
        chord_dur=args.chord_dur,
        bin_seconds=args.bin_seconds,
        threshold=args.threshold,
        jobs=args.jobs,
        failures_only=args.failures_only,
        report_path=args.report,
    )
    return 0 if summary["pass"] == summary["n_testable"] else 1


if __name__ == "__main__":
    sys.exit(main())
