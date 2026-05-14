"""Evaluation harness: run a directory of synthetic traces through the
matcher and produce summary metrics.

Metrics reported:
  - top1_correct@N  for N in {4, 8, 16, 32, 64, full}: fraction of traces
    where the top-1 song after N observations is the source song.
  - decision_correct: of traces where the matcher reached a decision,
    fraction that decided on the source song.
  - decision_wrong:   fraction that decided on the wrong song.
  - decision_none:    fraction that never decided.
  - mean_obs_to_decision: mean #obs when first decision occurred.

Also prints a top-confusion table for the misses.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.matcher import Matcher, MatcherConfig, NoteEvent


def evaluate_trace(
    matcher: Matcher,
    trace: dict[str, Any],
    checkpoints: tuple[int, ...] = (4, 8, 16, 32, 64),
) -> dict[str, Any]:
    matcher.reset()
    events = trace["events"]
    target_id = trace["song_id"]

    # top-1 predictions at each checkpoint
    cps: dict[int, str] = {}
    cps_top5: dict[int, list[str]] = {}
    first_decision_at: int | None = None
    first_decision_id: str | None = None

    for i, ev in enumerate(events, start=1):
        u = matcher.update(NoteEvent(
            pitch_class=int(ev["pitch_class"]),
            confidence=float(ev["confidence"]),
            t=float(ev.get("t", 0.0)),
        ))
        if i in checkpoints:
            cps[i] = u.top[0].song_id if u.top else "<none>"
            cps_top5[i] = [t.song_id for t in u.top]
        if u.decided and first_decision_at is None:
            first_decision_at = i
            first_decision_id = u.decided_song_id

    final = matcher._compute_top()
    cps["final"] = final.top[0].song_id if final.top else "<none>"
    cps_top5["final"] = [t.song_id for t in final.top]
    return {
        "song_id": target_id,
        "song_title": trace.get("song_title", target_id),
        "n_events": len(events),
        "checkpoints": cps,
        "checkpoints_top5": cps_top5,
        "final_top5": [(t.song_id, t.prob) for t in final.top],
        "first_decision_at": first_decision_at,
        "first_decision_id": first_decision_id,
    }


def evaluate_dir(
    matcher: Matcher,
    dir_path: Path,
    checkpoints: tuple[int, ...] = (4, 8, 16, 32, 64),
) -> dict[str, Any]:
    files = sorted(dir_path.glob("*.json"))
    per_trace = []
    for f in files:
        trace = json.loads(f.read_text())
        per_trace.append(evaluate_trace(matcher, trace, checkpoints))

    n = len(per_trace)
    if n == 0:
        return {"n": 0}

    # Per-checkpoint accuracy (top-1 and top-5)
    top1_at: dict[str, float] = {}
    top5_at: dict[str, float] = {}
    for cp in list(checkpoints) + ["final"]:
        correct1 = sum(1 for r in per_trace if r["checkpoints"].get(cp) == r["song_id"])
        correct5 = sum(1 for r in per_trace
                       if r["song_id"] in r.get("checkpoints_top5", {}).get(cp, []))
        top1_at[str(cp)] = correct1 / n
        top5_at[str(cp)] = correct5 / n

    decided = [r for r in per_trace if r["first_decision_at"] is not None]
    correct_decisions = [r for r in decided if r["first_decision_id"] == r["song_id"]]
    wrong_decisions   = [r for r in decided if r["first_decision_id"] != r["song_id"]]

    mean_obs = (
        sum(r["first_decision_at"] for r in correct_decisions)
        / max(1, len(correct_decisions))
    )

    # Confusion: target -> wrong-final-top1
    confusion: dict[str, Counter] = defaultdict(Counter)
    for r in per_trace:
        if r["checkpoints"]["final"] != r["song_id"]:
            confusion[r["song_id"]][r["checkpoints"]["final"]] += 1

    return {
        "n": n,
        "top1_at": top1_at,
        "top5_at": top5_at,
        "decision": {
            "n_decided":          len(decided),
            "n_correct":          len(correct_decisions),
            "n_wrong":            len(wrong_decisions),
            "n_undecided":        n - len(decided),
            "frac_decided":       len(decided) / n,
            "frac_correct_of_decisions": (len(correct_decisions) / len(decided)) if decided else 0.0,
            "frac_wrong":         len(wrong_decisions) / n,
            "mean_obs_to_correct_decision": mean_obs if correct_decisions else None,
        },
        "wrong_finals": [
            (r["song_title"], r["checkpoints"]["final"])
            for r in per_trace
            if r["checkpoints"]["final"] != r["song_id"]
        ][:20],
        "wrong_decisions": [
            (r["song_title"], r["first_decision_id"], r["first_decision_at"])
            for r in wrong_decisions
        ][:20],
    }


def fmt_report(name: str, summary: dict[str, Any]) -> str:
    lines = [f"\n=== {name} ({summary['n']} traces) ==="]
    for cp in summary["top1_at"]:
        t1 = summary["top1_at"][cp]
        t5 = summary["top5_at"].get(cp, 0.0)
        lines.append(f"  @{cp:<6}  top1={t1:.3f}  top5={t5:.3f}")
    d = summary["decision"]
    lines.append("")
    lines.append(f"  decisions:    {d['n_decided']}/{summary['n']} "
                 f"(correct={d['n_correct']} wrong={d['n_wrong']} undecided={d['n_undecided']})")
    lines.append(f"  frac_correct_of_decisions: {d['frac_correct_of_decisions']:.3f}")
    lines.append(f"  frac_wrong_decisions:      {d['frac_wrong']:.3f}")
    if d["mean_obs_to_correct_decision"] is not None:
        lines.append(f"  mean obs to correct decision: {d['mean_obs_to_correct_decision']:.1f}")
    if summary.get("wrong_decisions"):
        lines.append("\n  wrong decisions (first 10):")
        for tgt, got, at in summary["wrong_decisions"][:10]:
            lines.append(f"    {tgt!r:<35} -> {got!r}  @ {at}")
    return "\n".join(lines)


def main(argv=None) -> None:
    repo = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser()
    p.add_argument("--songs", default=str(repo / "app" / "data" / "songs.json"))
    p.add_argument("--page-index", default=str(repo / "app" / "data" / "page_index.json"))
    p.add_argument("--fixtures", default=str(repo / "app" / "tests" / "fixtures"))
    p.add_argument("--presets", nargs="+", default=["easy", "medium", "hard"])
    p.add_argument("--checkpoints", nargs="+", type=int, default=[4, 8, 16, 32, 64])
    p.add_argument("--bag-weight", type=float, default=None)
    p.add_argument("--hmm-weight", type=float, default=None)
    p.add_argument("--decision-min-prob", type=float, default=None)
    p.add_argument("--decision-min-ratio", type=float, default=None)
    p.add_argument("--decision-sustain", type=int, default=None)
    p.add_argument("--decision-min-obs", type=int, default=None)
    p.add_argument("--bag-epsilon", type=float, default=None)
    args = p.parse_args(argv)

    cfg_kwargs = {}
    for k in ("bag_weight", "hmm_weight", "decision_min_prob",
              "decision_min_ratio", "decision_sustain",
              "decision_min_obs", "bag_epsilon"):
        v = getattr(args, k.replace("_", "_"))
        if v is not None:
            cfg_kwargs[k] = v
    config = MatcherConfig(**cfg_kwargs) if cfg_kwargs else MatcherConfig()
    print(f"Matcher config: {config}")

    matcher = Matcher.from_paths(args.songs, args.page_index, config=config)

    overall = {}
    for preset in args.presets:
        d = Path(args.fixtures) / preset
        if not d.exists():
            print(f"(skipping {preset}: {d} not found)")
            continue
        s = evaluate_dir(matcher, d, tuple(args.checkpoints))
        overall[preset] = s
        print(fmt_report(preset, s))

    return overall


if __name__ == "__main__":
    main()
