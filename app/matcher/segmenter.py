"""Chord segmentation: collapse a fast stream of YIN-frame note events
into one consolidated observation per (estimated) chord segment.

Real strumming changes chord at most every half-bar (2 beats in 4/4),
typically every full bar (4 beats). YIN, in contrast, emits a pitch
estimate every ~12 ms, so without aggregation a held chord generates
hundreds of HMM transitions instead of one self-loop.

The segmenter buffers note events until at least `min_segment_seconds`
of wall time has elapsed since the segment started AND the dominant pc
has changed (or until a hard ceiling is reached for a held chord).
On segment close, it emits ONE consolidated `NoteEvent`:
  - pc      = confidence-weighted mode of the buffered pcs
  - confidence = mean confidence in the segment, clipped to [0, 1]
  - t       = time of the last raw event in the segment
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.matcher.matcher import NoteEvent


@dataclass
class SegmenterConfig:
    # How many beats per segment (musical resolution of chord changes).
    # 2 = half-bar in 4/4. The matcher's HMM will see one transition per
    # segment.
    min_beats: float = 2.0
    # Hard ceiling: even a held chord re-emits at this rate so the matcher
    # keeps getting evidence. 4 beats = 1 bar of 4/4.
    max_beats: float = 4.0
    # If BPM is unknown / hasn't been estimated yet, fall back to this.
    default_bpm: float = 100.0
    # Clamp the BPM range we'll honor.
    bpm_min: float = 60.0
    bpm_max: float = 180.0


class ChordSegmenter:
    """Stateful per-session segmenter. Feed every raw note event in;
    receives 0 or 1 consolidated `NoteEvent` back per call.
    """

    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.config = config or SegmenterConfig()
        self._bpm: float = self.config.default_bpm
        self._buf_pc: list[int] = []
        self._buf_w: list[float] = []
        self._buf_t: list[float] = []
        self._segment_start_t: float | None = None
        self._last_emitted_pc: int | None = None

    # ---- BPM control ----
    def set_bpm(self, bpm: float) -> None:
        cfg = self.config
        if bpm <= 0:
            return
        if bpm < cfg.bpm_min or bpm > cfg.bpm_max:
            return
        self._bpm = float(bpm)

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def min_segment_seconds(self) -> float:
        return (60.0 / self._bpm) * self.config.min_beats

    @property
    def max_segment_seconds(self) -> float:
        return (60.0 / self._bpm) * self.config.max_beats

    # ---- Stream interface ----
    def add(self, ev: NoteEvent) -> NoteEvent | None:
        """Buffer the event. Return a consolidated event when a segment
        closes, otherwise None.
        """
        if ev.confidence <= 0:
            return None

        if self._segment_start_t is None:
            self._segment_start_t = ev.t

        self._buf_pc.append(int(ev.pitch_class) % 12)
        self._buf_w.append(max(0.0, min(1.0, float(ev.confidence))))
        self._buf_t.append(float(ev.t))

        elapsed = ev.t - self._segment_start_t
        # Determine the current dominant pc in the buffer.
        dom_pc = self._weighted_mode()

        # Close a segment when:
        #   (a) at least min_segment_seconds has elapsed AND the dominant pc
        #       has changed compared to the last emitted segment, OR
        #   (b) max_segment_seconds has elapsed (held-chord ceiling).
        change_close = (elapsed >= self.min_segment_seconds
                        and dom_pc != self._last_emitted_pc)
        ceiling_close = elapsed >= self.max_segment_seconds

        if change_close or ceiling_close:
            return self._flush()
        return None

    def reset(self) -> None:
        self._buf_pc.clear()
        self._buf_w.clear()
        self._buf_t.clear()
        self._segment_start_t = None
        self._last_emitted_pc = None

    # ---- internals ----
    def _weighted_mode(self) -> int:
        if not self._buf_pc:
            return -1
        scores: Counter[int] = Counter()
        for pc, w in zip(self._buf_pc, self._buf_w):
            scores[pc] += w
        return max(scores.items(), key=lambda kv: kv[1])[0]

    def _flush(self) -> NoteEvent | None:
        if not self._buf_pc:
            self._segment_start_t = None
            return None
        dom_pc = self._weighted_mode()
        mean_conf = sum(self._buf_w) / max(1, len(self._buf_w))
        last_t = self._buf_t[-1]
        # Reset segment state for the next chord.
        self._buf_pc.clear()
        self._buf_w.clear()
        self._buf_t.clear()
        self._segment_start_t = last_t
        self._last_emitted_pc = dom_pc
        return NoteEvent(
            pitch_class=dom_pc,
            confidence=min(1.0, mean_conf),
            t=last_t,
        )
