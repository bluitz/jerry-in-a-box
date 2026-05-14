"""Speaker-to-mic end-to-end test.

Plays a reference audio file through the system speakers while
recording from the default microphone. The captured audio is then
fed through the same chord-classifier + sequence-matcher pipeline
that the live web app uses, and the rank of a target song over time
is reported.

This is the only way to validate the FULL end-to-end matching
quality, including room acoustics, mic frequency response, and
playback-speaker colouration. The replay_audio.py tool (file -> pipeline)
gives a clean upper bound; this test gives the realistic floor.

Requirements:
  - macOS (uses `afplay` for playback). Linux/Windows would need a
    different player.
  - sounddevice + scipy for audio capture and WAV writing.
  - Microphone permissions granted to the running terminal.

Usage:
  python -m app.tools.speaker_mic_test \
      --audio test-audio/"Friend of the devil.m4a" \
      --target friend-of-the-devil \
      --duration 35 \
      --rank-threshold 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]


def _ensure_macos_player() -> str:
    afplay = shutil.which("afplay")
    if not afplay:
        sys.stderr.write(
            "ERROR: afplay not found. This test currently only runs on macOS.\n"
        )
        sys.exit(2)
    return afplay


def _capture_mic(duration_s: float, sample_rate: int = 44100) -> np.ndarray:
    """Record `duration_s` seconds of mono audio from the default input
    device. Returns float32 in [-1, 1].

    We capture at the same 44.1 kHz the rest of the pipeline expects,
    in mono (microphones we care about are mono — MacBook builtin and
    iPhone). The first call will prompt for microphone access.
    """
    import sounddevice as sd

    n_frames = int(duration_s * sample_rate)
    data = sd.rec(
        n_frames,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocking=False,
    )
    sd.wait()
    return data.reshape(-1).astype(np.float32)


def _play_audio(afplay: str, audio_path: Path,
                started: threading.Event,
                stop: threading.Event) -> None:
    """Play the audio file via afplay. Sets `started` when playback
    has actually begun, and exits as soon as either the file ends or
    `stop` is set."""
    proc = subprocess.Popen(
        [afplay, str(audio_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started.set()
    while proc.poll() is None:
        if stop.wait(timeout=0.2):
            proc.terminate()
            break
    proc.wait()


def _save_wav(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Write audio (-1..1 float) as 16-bit PCM WAV for debugging."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def run_test(audio_path: Path, target_id: str,
             duration_s: float = 35.0,
             rank_threshold: int = 10,
             save_wav_to: Path | None = None,
             verbose: bool = True) -> dict:
    """Play `audio_path` over speakers, record `duration_s` seconds
    from the mic, run the captured audio through the matcher, and
    return a result dict.

    Returns:
        {
          "passed": bool,                # target rank <= rank_threshold
          "final_rank": int | None,
          "final_prob": float,
          "first_in_top_n_at_s": float | None,
          "timeline": [...],             # per-report-step rankings
          "captured_wav_path": Path | None,
          "final_top": [(sid, title, prob)],
        }
    """
    afplay = _ensure_macos_player()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    sample_rate = 44100

    if verbose:
        print(f"\n=== Speaker-to-mic test ===", file=sys.stderr)
        print(f"  source : {audio_path}", file=sys.stderr)
        print(f"  target : {target_id}", file=sys.stderr)
        print(f"  capture: {duration_s}s @ {sample_rate} Hz", file=sys.stderr)
        print(f"  Make sure the speakers are on and audible to the mic.",
              file=sys.stderr)
        print(f"  Recording starts in 2s...", file=sys.stderr)
        time.sleep(2)

    # Start playback thread; record is synchronous in the main thread.
    started = threading.Event()
    stop = threading.Event()
    play_thread = threading.Thread(
        target=_play_audio,
        args=(afplay, audio_path, started, stop),
        daemon=True,
    )
    play_thread.start()
    # Give afplay a moment to start before the mic recording begins.
    started.wait(timeout=2.0)
    if verbose:
        print(f"  Playback started; recording for {duration_s}s...",
              file=sys.stderr)

    captured = _capture_mic(duration_s, sample_rate=sample_rate)
    stop.set()
    play_thread.join(timeout=2.0)

    rms = float(np.sqrt(np.mean(captured ** 2)))
    peak = float(np.max(np.abs(captured)))
    if verbose:
        print(f"  Captured {captured.size / sample_rate:.1f}s "
              f"(RMS={rms:.4f}, peak={peak:.3f})",
              file=sys.stderr)

    if save_wav_to is not None:
        _save_wav(captured, sample_rate, save_wav_to)
        if verbose:
            print(f"  Saved captured audio to {save_wav_to}",
                  file=sys.stderr)

    # Sanity-check the capture before we even bother running the
    # matcher. Speech at conversational volume measures ~RMS 0.05-0.15
    # in our setup; anything under 0.005 means the mic didn't actually
    # hear the speakers (volume off, wrong device, mic muted, or terminal
    # lacking microphone permission).
    MIN_USABLE_RMS = 0.005
    capture_too_quiet = rms < MIN_USABLE_RMS

    # Run captured audio through the matcher pipeline. We do this by
    # writing to a temp WAV and re-loading via librosa so the path is
    # identical to what replay_audio.py exercises.
    from app.tools.replay_audio import detect_chords
    from app.matcher.sequence_matcher import SequenceMatcher

    chord_segs = detect_chords(captured, sample_rate, bin_seconds=2.0)
    if verbose:
        print(f"  Detected {len(chord_segs)} chord segments:", file=sys.stderr)
        PC = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        for seg in chord_segs:
            qmark = "" if seg["quality"] == "maj" else seg["quality"]
            qmark = qmark.replace("min", "m")
            print(f"    {seg['t_start']:5.1f}-{seg['t_end']:5.1f}s "
                  f"{PC[seg['root']]}{qmark}  conf={seg['conf']:.2f}",
                  file=sys.stderr)

    songs = json.loads((REPO / "app" / "data" / "songs.json").read_text())["songs"]
    matcher = SequenceMatcher(songs)
    titles = matcher.titles

    timeline: list[dict] = []
    first_in_top = None
    for seg in chord_segs:
        matcher.add_chord(seg["root"], seg["quality"])
        ranked = matcher.softmax_post()
        ids = [sid for sid, _ in ranked]
        try:
            rank = ids.index(target_id) + 1
        except ValueError:
            rank = None
        prob = dict(ranked).get(target_id, 0.0)
        timeline.append({
            "t": seg["t_end"],
            "rank": rank,
            "prob": prob,
            "top5": [(sid, titles[sid], p) for sid, p in ranked[:5]],
        })
        if (first_in_top is None and rank is not None
                and rank <= rank_threshold):
            first_in_top = seg["t_end"]

    # Final ranking
    final_ranked = matcher.softmax_post()
    final_ids = [sid for sid, _ in final_ranked]
    final_rank = (final_ids.index(target_id) + 1
                  if target_id in final_ids else None)
    final_prob = dict(final_ranked).get(target_id, 0.0)

    if verbose:
        print(f"\n  Final top 10:", file=sys.stderr)
        for i, (sid, p) in enumerate(final_ranked[:10]):
            mark = " <-- target" if sid == target_id else ""
            print(f"    {i+1:2d}. {titles[sid]:<40} {p*100:5.1f}%{mark}",
                  file=sys.stderr)
        print(f"\n  Target '{target_id}' rank over time:", file=sys.stderr)
        for tl in timeline:
            r = f"{tl['rank']}" if tl['rank'] else ">all"
            print(f"    t={tl['t']:5.1f}s  rank={r:>4s}  prob={tl['prob']*100:5.1f}%",
                  file=sys.stderr)
        print(f"\n  First time in top {rank_threshold}: "
              f"{first_in_top!r}", file=sys.stderr)

    passed = (final_rank is not None and final_rank <= rank_threshold
              and not capture_too_quiet)

    if verbose and capture_too_quiet:
        print(
            f"\n  WARNING: capture RMS {rms:.4f} < {MIN_USABLE_RMS} — the "
            f"mic likely did not hear the speakers. Check that the "
            f"system output volume is up, the correct output device "
            f"is selected, and that Terminal/iTerm has Microphone "
            f"permission (System Settings > Privacy & Security > "
            f"Microphone).",
            file=sys.stderr,
        )

    return {
        "passed": passed,
        "final_rank": final_rank,
        "final_prob": final_prob,
        "first_in_top_n_at_s": first_in_top,
        "rank_threshold": rank_threshold,
        "capture_rms": rms,
        "capture_peak": peak,
        "capture_too_quiet": capture_too_quiet,
        "timeline": timeline,
        "captured_wav_path": save_wav_to,
        "final_top": [(sid, titles[sid], p) for sid, p in final_ranked[:20]],
    }


def _run_trials(args) -> int:
    """Run N speaker->mic trials and report pass rate."""
    results: list[dict] = []
    print(f"\n=== Running {args.trials} trials ({args.duration}s each) ===",
          file=sys.stderr)
    print(f"  source: {args.audio}", file=sys.stderr)
    print(f"  target: {args.target}", file=sys.stderr)
    print(f"  pass:   target rank <= {args.rank_threshold}\n",
          file=sys.stderr)

    for i in range(1, args.trials + 1):
        print(f"--- Trial {i}/{args.trials} ---", file=sys.stderr)
        try:
            r = run_test(
                audio_path=args.audio,
                target_id=args.target,
                duration_s=args.duration,
                rank_threshold=args.rank_threshold,
                save_wav_to=None,
                verbose=False,
            )
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            results.append({"trial": i, "error": str(e), "passed": False})
            continue
        # Compact per-trial line.
        rms = r["capture_rms"]
        rank = r["final_rank"]
        prob = r["final_prob"] * 100
        top1 = r["final_top"][0][1] if r["final_top"] else "-"
        verdict = "PASS" if r["passed"] else (
            "QUIET" if r["capture_too_quiet"] else "FAIL")
        print(f"  Trial {i}: {verdict}  rank={rank}  prob={prob:.1f}%  "
              f"rms={rms:.4f}  top1={top1!r}",
              file=sys.stderr)
        results.append({
            "trial": i,
            "passed": r["passed"],
            "rank": r["final_rank"],
            "prob": r["final_prob"],
            "rms": r["capture_rms"],
            "too_quiet": r["capture_too_quiet"],
            "top1": top1,
        })

    n_pass = sum(1 for r in results if r.get("passed"))
    n_quiet = sum(1 for r in results if r.get("too_quiet"))

    print(f"\n=== {n_pass}/{args.trials} trials passed "
          f"(rank <= {args.rank_threshold}) ===", file=sys.stderr)
    if n_quiet:
        print(f"  {n_quiet} trial(s) had inaudible capture; raise volume "
              f"or check Mic permission.", file=sys.stderr)

    print(json.dumps({
        "trials": args.trials,
        "passed": n_pass,
        "too_quiet": n_quiet,
        "rank_threshold": args.rank_threshold,
        "per_trial": results,
    }, indent=2))
    return 0 if n_pass == args.trials else 1


def _check_only(audio_path: Path, sample_rate: int = 44100) -> int:
    """Calibration mode: play the file for 5 seconds, record from the
    mic, and report capture levels. Useful for verifying speaker
    volume / mic permission before running the full 35-second test."""
    afplay = _ensure_macos_player()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    print("=== Mic level check (5s) ===", file=sys.stderr)
    print(f"  source : {audio_path}", file=sys.stderr)
    print("  Make sure system output volume is up.", file=sys.stderr)
    started = threading.Event()
    stop = threading.Event()
    t = threading.Thread(
        target=_play_audio, args=(afplay, audio_path, started, stop),
        daemon=True,
    )
    t.start()
    started.wait(timeout=2.0)
    captured = _capture_mic(5.0, sample_rate=sample_rate)
    stop.set()
    t.join(timeout=2.0)
    rms = float(np.sqrt(np.mean(captured ** 2)))
    peak = float(np.max(np.abs(captured)))
    print(f"  RMS  = {rms:.4f}", file=sys.stderr)
    print(f"  Peak = {peak:.3f}", file=sys.stderr)
    if rms < 0.005:
        print("  Result: TOO QUIET. Raise system volume, confirm the "
              "right output is playing, and verify Terminal has "
              "Microphone permission.", file=sys.stderr)
        return 1
    if rms < 0.02:
        print("  Result: marginal. Should work but louder is better.",
              file=sys.stderr)
    else:
        print("  Result: good capture level.", file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path,
                   default=REPO / "test-audio" / "Friend of the devil.m4a")
    p.add_argument("--target", default="friend-of-the-devil",
                   help="Expected song id (slug from songs.json).")
    p.add_argument("--duration", type=float, default=35.0,
                   help="Recording duration in seconds.")
    p.add_argument("--rank-threshold", type=int, default=10,
                   help="Test passes if target ranks <= this.")
    p.add_argument("--save-wav", type=Path, default=None,
                   help="Optional path to save the captured audio.")
    p.add_argument("--check-only", action="store_true",
                   help="Just verify mic can hear speakers (5s check).")
    p.add_argument("--trials", type=int, default=1,
                   help="Run this many independent trials and "
                        "report pass rate.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    if args.check_only:
        return _check_only(args.audio)

    if args.trials > 1:
        return _run_trials(args)

    result = run_test(
        audio_path=args.audio,
        target_id=args.target,
        duration_s=args.duration,
        rank_threshold=args.rank_threshold,
        save_wav_to=args.save_wav,
        verbose=not args.quiet,
    )

    print(json.dumps({
        "passed": result["passed"],
        "final_rank": result["final_rank"],
        "final_prob": result["final_prob"],
        "first_in_top_n_at_s": result["first_in_top_n_at_s"],
        "capture_rms": result["capture_rms"],
        "capture_peak": result["capture_peak"],
        "capture_too_quiet": result["capture_too_quiet"],
    }, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
