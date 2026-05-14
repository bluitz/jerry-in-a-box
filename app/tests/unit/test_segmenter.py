"""Tests for app.matcher.segmenter.ChordSegmenter."""

from __future__ import annotations

from app.matcher import ChordSegmenter, NoteEvent, SegmenterConfig


def test_held_chord_emits_at_most_one_per_max_segment():
    """A held chord at high frame rate should emit at most one observation
    per max_segment ceiling — NOT per frame.
    """
    seg = ChordSegmenter(SegmenterConfig(default_bpm=120, min_beats=2, max_beats=4))
    # at 120 BPM: min_seg=1.0s, max_seg=2.0s
    # The first segment fires at min_seg (to register the first chord),
    # subsequent self-loops fire at max_seg.
    emitted = []
    for i in range(500):  # 5.0 seconds at 100 fps
        out = seg.add(NoteEvent(pitch_class=7, confidence=1.0, t=i * 0.01))
        if out:
            emitted.append(out)
    # Expected: ~3 emissions over 5s (one at 1s, then self-loops every 2s).
    # Definitely fewer than 10 (which would mean we're emitting every frame).
    assert 1 <= len(emitted) <= 5, f"got {len(emitted)} emissions"
    for e in emitted:
        assert e.pitch_class == 7


def test_chord_change_after_min_segment():
    """A genuine chord change after min_segment_seconds should emit."""
    seg = ChordSegmenter(SegmenterConfig(default_bpm=120, min_beats=2, max_beats=4))
    # at 120 BPM: min_seg=1.0s
    # 1.2s of G then 1.2s of C
    emitted = []
    t = 0.0
    for _ in range(120):  # 1.2s of G
        out = seg.add(NoteEvent(pitch_class=7, confidence=1.0, t=t))
        if out:
            emitted.append(out)
        t += 0.01
    for _ in range(120):  # 1.2s of C
        out = seg.add(NoteEvent(pitch_class=0, confidence=1.0, t=t))
        if out:
            emitted.append(out)
        t += 0.01
    # We expect 1 emission for the G->C boundary.
    assert len(emitted) >= 1
    # The first emitted segment should be G (the chord that ended).
    assert emitted[0].pitch_class == 7


def test_brief_noise_does_not_split_a_chord():
    """Diatonic noise (a few B frames during a G chord) should NOT cause
    multiple emissions — the dominant pc stays G.
    """
    seg = ChordSegmenter(SegmenterConfig(default_bpm=120, min_beats=2, max_beats=4))
    emitted = []
    t = 0.0
    # 3s of G with 5% noise (B, diatonic) — at 100 fps.
    for i in range(300):
        pc = 11 if i % 20 == 0 else 7
        out = seg.add(NoteEvent(pitch_class=pc, confidence=1.0, t=t))
        if out:
            emitted.append(out)
        t += 0.01
    # Expect 1 emission at t=1.0s (first chord) and 1 self-loop at ~3.0s.
    # All emissions should report G (pc=7) — never the noise pc.
    assert 1 <= len(emitted) <= 3
    for e in emitted:
        assert e.pitch_class == 7, f"noise leaked into segmenter: {e}"


def test_set_bpm_updates_segment_length():
    seg = ChordSegmenter(SegmenterConfig(default_bpm=60, min_beats=2, max_beats=4))
    # at 60 BPM: min_seg=2.0s
    assert abs(seg.min_segment_seconds - 2.0) < 1e-6
    seg.set_bpm(120)
    # at 120 BPM: min_seg=1.0s
    assert abs(seg.min_segment_seconds - 1.0) < 1e-6
    # Out-of-range values are ignored.
    seg.set_bpm(20)
    assert abs(seg.min_segment_seconds - 1.0) < 1e-6
    seg.set_bpm(500)
    assert abs(seg.min_segment_seconds - 1.0) < 1e-6


def test_reset_clears_buffer():
    seg = ChordSegmenter(SegmenterConfig(default_bpm=120))
    seg.add(NoteEvent(pitch_class=7, confidence=1.0, t=0.0))
    seg.add(NoteEvent(pitch_class=7, confidence=1.0, t=0.5))
    seg.reset()
    out = seg.add(NoteEvent(pitch_class=0, confidence=1.0, t=0.6))
    assert out is None  # fresh segment, nothing to emit yet


def test_low_confidence_frames_ignored():
    seg = ChordSegmenter(SegmenterConfig(default_bpm=120))
    out = seg.add(NoteEvent(pitch_class=7, confidence=0.0, t=0.0))
    assert out is None
    # The buffer should still be empty.
    assert seg._segment_start_t is None
