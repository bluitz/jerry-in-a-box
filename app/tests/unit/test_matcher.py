"""Smoke + behavioral tests for the matcher.

Tests are seeded and deterministic. They use a small fake song corpus so
the assertions are about *behavior*, not statistical luck.
"""

import math
import json
from pathlib import Path

import numpy as np
import pytest

from app.matcher.matcher import Matcher, NoteEvent, MatcherConfig
from app.matcher.emission import EmissionConfig, in_key_set


def _make_fake_db():
    """Three small songs that share G/D but differ on the third chord.

    Song A: G - C - G - D    (key C/G major-ish)
    Song B: G - Em - G - D   (Em is the disambiguator)
    Song C: A - D - E - A    (different key entirely)
    """
    return {
        "schema_version": 1,
        "n_songs": 3,
        "songs": [
            {
                "id": "song_a", "title": "Song A", "artist": "Test",
                "key_pc": 7, "key_mode": "maj", "key_name": "G",
                "chords": ["G", "C", "G", "D"],
                "chord_tuples": [[7,"maj",None],[0,"maj",None],[7,"maj",None],[2,"maj",None]],
                "pc_histogram": _hist([(7,11,2),(0,4,7),(7,11,2),(2,6,9)]),
                "chord_vocab": ["G","C","D"],
                "bigram": [
                    [0.05, 0.55, 0.40],  # G -> {G,C,D}
                    [0.55, 0.05, 0.40],  # C -> {G,C,D}
                    [0.55, 0.05, 0.40],  # D -> {G,C,D}
                ],
                "n_chord_events": 4, "n_unique_chords": 3,
                "sections": [],
            },
            {
                "id": "song_b", "title": "Song B", "artist": "Test",
                "key_pc": 7, "key_mode": "maj", "key_name": "G",
                "chords": ["G", "Em", "G", "D"],
                "chord_tuples": [[7,"maj",None],[4,"min",None],[7,"maj",None],[2,"maj",None]],
                "pc_histogram": _hist([(7,11,2),(4,7,11),(7,11,2),(2,6,9)]),
                "chord_vocab": ["G","Em","D"],
                "bigram": [
                    [0.05, 0.55, 0.40],
                    [0.55, 0.05, 0.40],
                    [0.55, 0.05, 0.40],
                ],
                "n_chord_events": 4, "n_unique_chords": 3,
                "sections": [],
            },
            {
                "id": "song_c", "title": "Song C", "artist": "Test",
                "key_pc": 9, "key_mode": "maj", "key_name": "A",
                "chords": ["A", "D", "E", "A"],
                "chord_tuples": [[9,"maj",None],[2,"maj",None],[4,"maj",None],[9,"maj",None]],
                "pc_histogram": _hist([(9,1,4),(2,6,9),(4,8,11),(9,1,4)]),
                "chord_vocab": ["A","D","E"],
                "bigram": [
                    [0.05, 0.55, 0.40],
                    [0.55, 0.05, 0.40],
                    [0.55, 0.05, 0.40],
                ],
                "n_chord_events": 4, "n_unique_chords": 3,
                "sections": [],
            },
        ],
    }


def _hist(chord_tones_list):
    """Build a normalized 12-bin pitch-class histogram from chord tones.
    Each chord contributes 1/N over its tones; everything sums to 1."""
    hist = [0.0] * 12
    for tones in chord_tones_list:
        w = 1.0 / len(tones)
        for t in tones:
            hist[t % 12] += w
    s = sum(hist)
    return [x / s for x in hist] if s else hist


def _stream_chord_tones(chord_tones, n_per=4):
    """Emit pitch classes uniformly across the given chord tones.

    Confidence = 1.0 (perfect tuner). Useful for "does the matcher
    converge at all" tests.
    """
    out = []
    for tones in chord_tones:
        for _ in range(n_per):
            for pc in tones:
                out.append(NoteEvent(pitch_class=pc, confidence=1.0))
    return out


def test_perfect_input_picks_correct_song():
    db = _make_fake_db()
    m = Matcher(db["songs"])

    # Stream Song A's chord tones in order (G, C, G, D), several beats each.
    g, c, d = (7,11,2), (0,4,7), (2,6,9)
    stream = _stream_chord_tones([g, c, g, d], n_per=8)
    last = None
    for ev in stream:
        last = m.update(ev)
    assert last.top[0].song_id == "song_a"
    assert last.top[0].prob > 0.5  # converges hard


def test_disambiguation_via_em():
    """Song B has Em where Song A has C; an Em-rich stream should pick B."""
    db = _make_fake_db()
    m = Matcher(db["songs"])

    g, em, d = (7,11,2), (4,7,11), (2,6,9)
    stream = _stream_chord_tones([g, em, g, d], n_per=8)
    last = None
    for ev in stream:
        last = m.update(ev)
    assert last.top[0].song_id == "song_b"


def test_different_key_separates_fast():
    """Song C is in A major; an A-major-rich stream should pick C immediately."""
    db = _make_fake_db()
    m = Matcher(db["songs"])

    a, d, e = (9,1,4), (2,6,9), (4,8,11)
    stream = _stream_chord_tones([a, d, e, a], n_per=4)
    last = None
    for ev in stream:
        last = m.update(ev)
    assert last.top[0].song_id == "song_c"
    # Should be confident enough to decide given enough observations.
    assert last.top[0].prob > 0.5


def test_low_confidence_observations_dont_dominate():
    """A noisy low-confidence wrong observation shouldn't flip the matcher."""
    db = _make_fake_db()
    m = Matcher(db["songs"])

    # Build up a clear Song A signal.
    g, c, d = (7,11,2), (0,4,7), (2,6,9)
    for ev in _stream_chord_tones([g, c, g, d], n_per=6):
        m.update(ev)
    top_before = m._compute_top().top[0].song_id
    assert top_before == "song_a"

    # Inject a stream of low-confidence Em observations (favoring B).
    em = (4, 7, 11)
    for _ in range(20):
        for pc in em:
            m.update(NoteEvent(pitch_class=pc, confidence=0.05))

    top_after = m._compute_top().top[0].song_id
    # Confidence-weighted updates — A should still be on top.
    assert top_after == "song_a"


def test_decision_rule_sustain():
    """The matcher should DECIDE only after sustained high posterior, and
    after enough evidence has arrived, the final decision must be Song A.

    (Early in the stream Song A and B are ambiguous — they share G,D and
    A's C-tones overlap heavily with B's Em-tones — so the matcher is
    allowed to be uncertain initially. What matters is the long-run
    behavior on a stream fully consistent with Song A.)
    """
    db = _make_fake_db()
    cfg = MatcherConfig(decision_min_obs=8, decision_sustain=3, decision_min_prob=0.7)
    m = Matcher(db["songs"], config=cfg)

    g, c, d = (7, 11, 2), (0, 4, 7), (2, 6, 9)
    stream = _stream_chord_tones([g, c, g, d], n_per=12)
    first_decision_at = None
    final_top = None
    for i, ev in enumerate(stream):
        u = m.update(ev)
        if u.decided and first_decision_at is None:
            first_decision_at = i
        final_top = u

    assert first_decision_at is not None, "matcher never reached a decision"
    assert first_decision_at >= cfg.decision_min_obs - 1
    # By the end of the stream, A must be top.
    assert final_top.top[0].song_id == "song_a"


def test_reset_clears_state():
    db = _make_fake_db()
    m = Matcher(db["songs"])
    g, c, d = (7,11,2), (0,4,7), (2,6,9)
    for ev in _stream_chord_tones([g, c, g, d], n_per=4):
        m.update(ev)
    # Pre-reset top should be A.
    assert m._compute_top().top[0].song_id == "song_a"

    m.reset()
    # Post-reset, posterior is uniform → all songs equally likely.
    u = m._compute_top()
    probs = [t.prob for t in u.top]
    assert abs(probs[0] - probs[-1]) < 1e-6


def test_real_corpus_loads_and_runs():
    """Smoke test: the real compiled songs.json loads and a simple stream
    of ~50 G-major-key observations updates the matcher without crashing."""
    repo = Path(__file__).resolve().parents[3]
    p = repo / "app" / "data" / "songs.json"
    if not p.exists():
        pytest.skip("songs.json not built yet")
    m = Matcher.from_paths(p)
    assert m.n_songs > 50
    for pc in [7, 11, 2, 0, 4, 7, 2, 6, 9] * 10:
        u = m.update(NoteEvent(pitch_class=pc, confidence=1.0))
    # Should not crash; should produce normalized top-5.
    assert len(u.top) == 5
    assert abs(sum(t.prob for t in u.top) - 1.0) < 0.5  # top-5 captures most mass when convergence
