"""Headless replay harness: feed an audio file through the same pipeline
the live web app uses (YIN-equivalent pitch detection -> ChordSegmenter ->
Matcher) and report the top-N candidate songs over time.

This is the iteration tool for tuning the matcher against real-world audio
without going through the speaker -> mic round trip.

Pipeline parity with the browser:
  1. Decode audio to mono float32 at 44.1 kHz.
  2. Hop every 512 samples; FRAME = 2048 samples.
  3. RMS gate -> onset detector (same thresholds as the AudioWorklet) ->
     YIN pitch detection -> 12-bin chroma per hop.
  4. BPM estimated from inter-onset intervals.
  5. Notes/onsets/BPM fed into ChordSegmenter; consolidated NoteEvents
     sent to Matcher.

Usage:
  python -m app.tools.replay_audio path/to/audio.m4a \
      --report-every 1.0 \
      --top 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from app.matcher import (
    ChordSegmenter, Matcher, MatcherConfig, NoteEvent, SegmenterConfig,
)
from app.matcher.chord_classifier import top_chord
from app.matcher.sequence_matcher import SequenceMatcher


# ---- DSP constants (must match app/web/public/yin-worklet.js) ----
SR_TARGET = 44100
FRAME = 2048
HOP = 512
RMS_GATE = 0.01
YIN_THRESHOLD = 0.15
F_MIN = 70.0
F_MAX = 1500.0
SMOOTH_N = 3

# Onset detection
ONSET_RATIO = 1.6
ONSET_FLOOR = 0.025
ONSET_MIN_GAP = 0.12
ONSET_BASELINE_TC = 0.5


def yin_pitch(frame: np.ndarray, sr: int) -> tuple[float, float]:
    """Port of yin() from yin-worklet.js. Returns (freq_hz, aperiodicity)."""
    N = frame.size
    W = N // 2
    d = np.zeros(W, dtype=np.float64)
    # Difference function
    for tau in range(W):
        diff = frame[: W] - frame[tau : tau + W]
        d[tau] = np.dot(diff, diff)
    d[0] = 1.0
    # CMND
    running = 0.0
    for tau in range(1, W):
        running += d[tau]
        d[tau] = (d[tau] * tau) / running if running > 0 else 1.0
    # First minimum below threshold
    tau = -1
    for i in range(2, W - 1):
        if d[i] < YIN_THRESHOLD:
            j = i
            while j + 1 < W - 1 and d[j + 1] < d[j]:
                j += 1
            tau = j
            break
    if tau < 0:
        idx = int(np.argmin(d[2:W])) + 2
        tau = idx
    # Parabolic interpolation
    tau_refined = float(tau)
    if 0 < tau < W - 1:
        s0, s1, s2 = d[tau - 1], d[tau], d[tau + 1]
        denom = 2 * (2 * s1 - s2 - s0)
        if denom != 0:
            tau_refined = tau + (s2 - s0) / denom
    if tau_refined <= 0:
        return 0.0, 1.0
    return sr / tau_refined, float(d[tau])


def chroma_from_frame(frame: np.ndarray, sr: int) -> np.ndarray:
    """12-bin chroma from a Hann-windowed magnitude FFT."""
    N = frame.size
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    spec = np.abs(np.fft.rfft(frame * win))
    chroma = np.zeros(12, dtype=np.float64)
    min_bin = max(1, int(np.floor(F_MIN * N / sr)))
    max_bin = min(N // 2 - 1, int(np.ceil(F_MAX * N / sr)))
    for k in range(min_bin, max_bin + 1):
        f = k * sr / N
        if f <= 0:
            continue
        midi_f = 69 + 12 * np.log2(f / 440.0)
        pc = int(round(midi_f)) % 12
        chroma[pc] += spec[k]
    s = chroma.sum()
    if s > 0:
        chroma /= s
    return chroma


def detect_pitches(audio: np.ndarray, sr: int, *,
                   use_chroma: bool = True,
                   chroma_top_k: int = 3) -> Iterable[dict]:
    """Generator yielding {t, type, ...} events: 'note', 'onset'.

    Two modes:
      use_chroma=False: YIN monophonic pitch (only useful for clean,
        single-note input like a guitar tuner).
      use_chroma=True (default): emit the top-K chroma pcs per hop as
        separate note events, weighted by chroma magnitude. This is
        appropriate for polyphonic audio (recordings, full guitar
        chords, etc.) and is what the live web app should use too.

    Onset detection still runs on RMS in both modes so BPM estimation
    works.
    """
    pc_history: list[int] = []
    rms_baseline = 0.0
    last_onset_t = -1.0
    above_threshold = False
    n_hops = (audio.size - FRAME) // HOP + 1
    for h in range(n_hops):
        start = h * HOP
        frame = audio[start : start + FRAME]
        if frame.size < FRAME:
            break
        rms = float(np.sqrt(np.mean(frame * frame)))
        t_now = (start + FRAME) / sr
        if rms < RMS_GATE:
            pc_history.clear()
            continue

        dt = HOP / sr
        alpha = 1.0 - np.exp(-dt / ONSET_BASELINE_TC)
        if rms_baseline == 0.0:
            rms_baseline = rms
        baseline = rms_baseline
        is_peak = rms > ONSET_FLOOR and rms > baseline * ONSET_RATIO
        if (is_peak and not above_threshold
                and (last_onset_t < 0 or t_now - last_onset_t >= ONSET_MIN_GAP)):
            last_onset_t = t_now
            yield {"type": "onset", "t": t_now, "rms": rms}
        above_threshold = is_peak
        rms_baseline = baseline + alpha * (rms - baseline)

        if use_chroma:
            chroma = chroma_from_frame(frame, sr)
            order = np.argsort(-chroma)
            top = order[:chroma_top_k]
            top_sum = float(chroma[top].sum())
            if top_sum <= 0:
                continue
            for pc in top:
                w = float(chroma[pc]) / top_sum
                if w <= 0:
                    continue
                yield {"type": "note", "t": t_now, "pc": int(pc),
                       "confidence": w, "src": "chroma"}
        else:
            freq, aperiodicity = yin_pitch(frame, sr)
            if not freq or aperiodicity > YIN_THRESHOLD:
                continue
            if freq < F_MIN or freq > F_MAX:
                continue
            midi = int(round(69 + 12 * np.log2(freq / 440.0)))
            pc = midi % 12
            pc_history.append(pc)
            if len(pc_history) > SMOOTH_N:
                pc_history.pop(0)
            sorted_pcs = sorted(pc_history)
            pc_smoothed = sorted_pcs[len(sorted_pcs) // 2]
            confidence = max(0.0, min(1.0, 1.0 - aperiodicity))
            yield {"type": "note", "t": t_now, "pc": pc_smoothed,
                   "confidence": confidence, "freq": freq, "src": "yin"}


def estimate_bpm_track(onsets: list[float]) -> list[tuple[float, float]]:
    """Return a list of (t, bpm) estimates, one per onset after the 4th.

    Same logic as audio.ts handleOnset.
    """
    out: list[tuple[float, float]] = []
    history_size = 12
    last_bpm = 100.0
    onsets_window: list[float] = []
    for t in onsets:
        onsets_window.append(t)
        if len(onsets_window) > history_size:
            onsets_window.pop(0)
        if len(onsets_window) < 4:
            continue
        iois = [onsets_window[i] - onsets_window[i - 1]
                for i in range(1, len(onsets_window))]
        sorted_ioi = sorted(iois)
        median_ioi = sorted_ioi[len(sorted_ioi) // 2]
        if median_ioi <= 0:
            continue
        bpm = 60.0 / median_ioi
        while bpm > 180:
            bpm /= 2
        while bpm < 60:
            bpm *= 2
        if bpm < 60 or bpm > 180:
            continue
        last_bpm = 0.7 * last_bpm + 0.3 * bpm
        out.append((t, last_bpm))
    return out


def detect_chords(audio: np.ndarray, sr: int, *,
                  bin_seconds: float = 1.0,
                  min_frames_per_bin: int = 5) -> list[dict]:
    """Per-bin chord classification using librosa's CQT chroma.

    Strategy: compute log-frequency (constant-Q) chroma — much more
    accurate than FFT-bin chroma at separating pitch classes for
    polyphonic guitar — then aggregate per `bin_seconds` window and
    classify the averaged chroma to a (root, quality) chord.

    Returns a list of {t_start, t_end, root, quality, conf}.
    """
    # Constant-Q chroma. n_fft and hop_length tuned for ~23ms hops at
    # 44.1 kHz. norm=None lets us aggregate by addition (energy-summing)
    # before normalising in the classifier.
    chroma_cqt = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("C2"),
        n_chroma=12, n_octaves=5, bins_per_octave=36,
    )
    # chroma_cqt shape: (12, n_frames). Aggregate over `bin_seconds`.
    frames_per_bin = max(1, int(round(bin_seconds * sr / HOP)))
    n_frames = chroma_cqt.shape[1]
    bin_chords: list[tuple[float, int, str, float]] = []
    for b_start in range(0, n_frames, frames_per_bin):
        b_end = min(b_start + frames_per_bin, n_frames)
        if b_end - b_start < min_frames_per_bin:
            continue
        avg = chroma_cqt[:, b_start:b_end].mean(axis=1)
        root, qual, p = top_chord(avg)
        t_center = (b_start + (b_end - b_start) / 2.0) * HOP / sr
        bin_chords.append((t_center, root, qual, float(p)))

    # Smooth two kinds of artifacts:
    # (a) Single-bin QUALITY flickers within a held root (D, Dm, D):
    #     promote the middle to the neighbors' quality.
    # (b) Single-bin TRANSITION-ZONE noise: a low-confidence bin whose
    #     neighbors are both different higher-confidence chords. This
    #     bin is almost always a chord-change crossfade (e.g. Dm
    #     classified between D and Am because the audio mid-transition
    #     contains both D and F notes). Replace it with the more
    #     similar neighbor (we just pick the previous neighbor — the
    #     classifier output for "this 2-sec window" is essentially
    #     uninformative when below 0.30 confidence).
    if len(bin_chords) >= 3:
        smoothed = list(bin_chords)
        for i in range(1, len(bin_chords) - 1):
            tc, r, q, p = bin_chords[i]
            _, rp, qp, pp = bin_chords[i - 1]
            _, rn, qn, pn = bin_chords[i + 1]
            if rp == r == rn and qp == qn and qp != q:
                smoothed[i] = (tc, r, qp, p)
            elif (p < 0.30 and rp != r and rn != r
                  and pp >= 0.30 and pn >= 0.30):
                smoothed[i] = (tc, rp, qp, pp)
        bin_chords = smoothed

    # Collapse adjacent identical bins into segments.
    segs: list[dict] = []
    if not bin_chords:
        return segs
    cur_root, cur_qual = bin_chords[0][1], bin_chords[0][2]
    seg_start = bin_chords[0][0] - bin_seconds / 2
    confs: list[float] = [bin_chords[0][3]]
    for t_center, root, qual, p in bin_chords[1:]:
        if root == cur_root and qual == cur_qual:
            confs.append(p)
            continue
        seg_end = t_center - bin_seconds / 2
        segs.append({
            "t_start": seg_start, "t_end": seg_end,
            "root": cur_root, "quality": cur_qual,
            "conf": float(np.mean(confs)),
        })
        cur_root, cur_qual = root, qual
        seg_start = seg_end
        confs = [p]
    segs.append({
        "t_start": seg_start, "t_end": bin_chords[-1][0] + bin_seconds / 2,
        "root": cur_root, "quality": cur_qual,
        "conf": float(np.mean(confs)),
    })
    return segs


def replay(audio_path: Path, *,
           report_every: float = 1.0,
           top_n: int = 10,
           matcher_config: MatcherConfig | None = None,
           segmenter_config: SegmenterConfig | None = None,
           use_chroma: bool = True,
           chroma_top_k: int = 3,
           use_sequence: bool = False,
           verbose: bool = False) -> dict:
    """Run the audio file end-to-end through the pipeline.

    Returns a summary dict with timeline of top-K and (if applicable) the
    decision time.
    """
    REPO = Path(__file__).resolve().parents[2]
    songs = json.loads((REPO / "app" / "data" / "songs.json").read_text())["songs"]
    page_index_path = REPO / "app" / "data" / "page_index.json"
    page_index = (json.loads(page_index_path.read_text())
                  if page_index_path.exists() else {})

    matcher = Matcher(songs, page_index, matcher_config or MatcherConfig())
    segmenter = ChordSegmenter(segmenter_config or SegmenterConfig())
    seq_matcher = SequenceMatcher(songs) if use_sequence else None
    PC = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

    print(f"Loading {audio_path} ...", file=sys.stderr)
    audio, sr = librosa.load(str(audio_path), sr=SR_TARGET, mono=True)
    duration = audio.size / sr
    print(f"  {duration:.1f}s @ {sr} Hz", file=sys.stderr)

    if seq_matcher is not None:
        # Sequence path: classify chords from chroma (smoothed), feed to
        # SequenceMatcher, report top-N over time. 2s bins are more
        # robust against melodic notes than 1s bins.
        chord_segs = detect_chords(audio, sr, bin_seconds=2.0)
        print(f"  detected {len(chord_segs)} chord segments",
              file=sys.stderr)
        if verbose:
            for seg in chord_segs[:30]:
                qmark = "" if seg["quality"] == "maj" else seg["quality"]
                qmark = qmark.replace("min", "m")
                print(f"    {seg['t_start']:5.1f}-{seg['t_end']:5.1f}s "
                      f"{PC[seg['root']]}{qmark}  conf={seg['conf']:.2f}",
                      file=sys.stderr)

        timeline: list[dict] = []
        decision_time: float | None = None
        decision_song: str | None = None
        next_report = report_every

        for seg in chord_segs:
            seq_matcher.add_chord(seg["root"], seg["quality"])
            if seg["t_end"] >= next_report:
                ranked = seq_matcher.softmax_post()
                top = [(sid, p) for sid, p in ranked[:top_n]]
                timeline.append({
                    "t": seg["t_end"],
                    "n_obs": seq_matcher.n_observations(),
                    "n_unique": seq_matcher.n_unique_chords(),
                    "elapsed_seconds": seg["t_end"],
                    "bpm": 0.0,
                    "decided": False,
                    "decided_song_id": None,
                    "top": [(sid, round(p, 4)) for sid, p in top],
                })
                if verbose:
                    print(f"\nt={seg['t_end']:.1f}s  unique_chords={seq_matcher.n_unique_chords()}",
                          file=sys.stderr)
                    for i, (sid, p) in enumerate(top[:5]):
                        print(f"  {i + 1}. {seq_matcher.titles[sid]:<40} {p * 100:5.1f}%",
                              file=sys.stderr)
                next_report += report_every

        ranked = seq_matcher.softmax_post()
        return {
            "duration_s": duration,
            "decision_time_s": decision_time,
            "decision_song_id": decision_song,
            "final_top": [(sid, seq_matcher.titles[sid], round(p, 4))
                          for sid, p in ranked[:top_n]],
            "timeline": timeline,
        }

    # First pass: collect pitches and onsets so we can derive a BPM track.
    events = list(detect_pitches(audio, sr,
                                 use_chroma=use_chroma,
                                 chroma_top_k=chroma_top_k))
    onsets = [e["t"] for e in events if e["type"] == "onset"]
    bpm_track = estimate_bpm_track(onsets)
    bpm_idx = 0
    print(f"  {len(events)} events  ({len(onsets)} onsets)", file=sys.stderr)
    if bpm_track:
        bpms = [b for _, b in bpm_track]
        print(f"  BPM range: {min(bpms):.0f}-{max(bpms):.0f} "
              f"(median {sorted(bpms)[len(bpms) // 2]:.0f})", file=sys.stderr)

    timeline: list[dict] = []
    decision_time: float | None = None
    decision_song: str | None = None
    next_report = report_every

    for ev in events:
        # Update BPM in the segmenter as the BPM track advances.
        while bpm_idx < len(bpm_track) and bpm_track[bpm_idx][0] <= ev["t"]:
            segmenter.set_bpm(bpm_track[bpm_idx][1])
            bpm_idx += 1

        if ev["type"] != "note":
            continue
        ne = NoteEvent(
            pitch_class=int(ev["pc"]),
            confidence=float(ev["confidence"]),
            t=float(ev["t"]),
        )
        consolidated = segmenter.add(ne)
        if consolidated is None:
            continue
        update = matcher.update(consolidated)
        # Periodic report
        if ev["t"] >= next_report:
            top = update.top[:top_n]
            timeline.append({
                "t": ev["t"],
                "n_obs": update.n_obs,
                "elapsed_seconds": update.elapsed_seconds,
                "bpm": segmenter.bpm,
                "decided": update.decided,
                "decided_song_id": update.decided_song_id,
                "top": [(t.song_id, round(t.prob, 4)) for t in top],
            })
            if verbose:
                print(f"\nt={ev['t']:.1f}s  obs={update.n_obs}  bpm={segmenter.bpm:.0f}",
                      file=sys.stderr)
                for i, t in enumerate(top[:5]):
                    print(f"  {i + 1}. {t.title:<40} {t.prob * 100:5.1f}%",
                          file=sys.stderr)
            next_report += report_every
        if update.decided and decision_time is None:
            decision_time = update.elapsed_seconds
            decision_song = update.decided_song_id

    return {
        "duration_s": duration,
        "decision_time_s": decision_time,
        "decision_song_id": decision_song,
        "final_top": [(t.song_id, t.title, round(t.prob, 4))
                      for t in update.top[:top_n]],
        "timeline": timeline,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--report-every", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--target-song", type=str, default=None,
                        help="If set, print whether this id appears in top-N.")
    parser.add_argument("--mode", choices=["chroma", "yin", "sequence"],
                        default="sequence",
                        help="Detection mode. 'sequence' = chord-template "
                             "classifier + sequence alignment (best for "
                             "polyphonic recordings).")
    parser.add_argument("--chroma-top-k", type=int, default=3,
                        help="In chroma mode, how many top pcs to emit per hop.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    out = replay(args.audio, report_every=args.report_every,
                 top_n=args.top, verbose=args.verbose,
                 use_chroma=(args.mode != "yin"),
                 chroma_top_k=args.chroma_top_k,
                 use_sequence=(args.mode == "sequence"))

    print()
    print(f"Audio duration: {out['duration_s']:.1f}s")
    if out["decision_time_s"] is not None:
        print(f"DECIDED at t={out['decision_time_s']:.1f}s -> {out['decision_song_id']}")
    else:
        print("No decision reached.")

    print(f"\nFinal top {args.top}:")
    for i, (sid, title, prob) in enumerate(out["final_top"]):
        marker = ""
        if args.target_song and sid == args.target_song:
            marker = "  <- target"
        print(f"  {i + 1:2d}. {title:<42} {prob * 100:6.2f}%{marker}")

    if args.target_song:
        # Find rank at each timeline point.
        print(f"\nRank of '{args.target_song}' over time:")
        for row in out["timeline"]:
            rank = next((i + 1 for i, (sid, _) in enumerate(row["top"])
                         if sid == args.target_song), None)
            prob = next((p for sid, p in row["top"]
                         if sid == args.target_song), 0.0)
            r = f"{rank}" if rank else f">{len(row['top'])}"
            print(f"  t={row['t']:5.1f}s  rank={r:>4}  prob={prob * 100:5.2f}%  "
                  f"obs={row['n_obs']}  bpm={row['bpm']:.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
