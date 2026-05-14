"""Unit tests for the chord parser."""

from app.parser.chords import (
    Chord,
    parse_chord,
    parse_chord_string,
    PITCH_CLASS,
)


def test_basic_majors():
    assert parse_chord("G").to_tuple() == (7, "maj", None)
    assert parse_chord("C").to_tuple() == (0, "maj", None)
    assert parse_chord("F#").to_tuple() == (6, "maj", None)
    assert parse_chord("Bb").to_tuple() == (10, "maj", None)


def test_minors_and_sevenths():
    assert parse_chord("Em").to_tuple() == (4, "min", None)
    assert parse_chord("Am").to_tuple() == (9, "min", None)
    assert parse_chord("D7").to_tuple() == (2, "7", None)
    assert parse_chord("G7").to_tuple() == (7, "7", None)
    assert parse_chord("Cmaj7").to_tuple() == (0, "maj7", None)
    assert parse_chord("Em7").to_tuple() == (4, "m7", None)
    assert parse_chord("F#m7b5").to_tuple() == (6, "m7b5", None)


def test_slash_chords():
    c = parse_chord("E/G#")
    assert c.root_pc == PITCH_CLASS["E"]
    assert c.bass_pc == PITCH_CLASS["G#"]
    assert c.quality == "maj"
    # Tones include bass even though it's already a chord tone here.
    assert PITCH_CLASS["G#"] in c.tones

    c = parse_chord("C/B")
    assert c.root_pc == 0
    assert c.bass_pc == 11
    # B is not in C major triad; bass adds it to tones set.
    assert 11 in c.tones


def test_parens_and_trailing_punct():
    # CSV has things like "Bb7(C)". The "(C)" should be ignored as junk.
    c = parse_chord("Bb7(C)")
    assert c.root_pc == 10
    assert c.quality == "7"
    # Trailing punctuation
    assert parse_chord("G,").to_tuple() == (7, "maj", None)
    assert parse_chord("D7.").to_tuple() == (2, "7", None)


def test_emmaj7_collapses():
    # Source CSV has "Emmaj7"; we collapse mMaj7 -> m7 (close enough for matcher).
    assert parse_chord("Emmaj7").to_tuple() == (4, "m7", None)


def test_non_chord_tokens():
    assert parse_chord("|") is None
    assert parse_chord("%") is None
    assert parse_chord("/") is None
    assert parse_chord("") is None
    assert parse_chord("Verse") is None  # starts with V, not A-G
    # H is not a note in this notation
    assert parse_chord("H7") is None


def test_chord_tones():
    g = parse_chord("G")
    # G major: G, B, D = 7, 11, 2
    assert g.tones == frozenset({7, 11, 2})

    em = parse_chord("Em")
    # E minor: E, G, B = 4, 7, 11
    assert em.tones == frozenset({4, 7, 11})

    d7 = parse_chord("D7")
    # D7: D, F#, A, C = 2, 6, 9, 0
    assert d7.tones == frozenset({2, 6, 9, 0})


def test_parse_chord_string_basic():
    # Pattern from CSV: "| G / / / | C / / / | G / / / | D / / / |"
    prog = parse_chord_string("| G / / / | C / / / | G / / / | D / / / |")
    names = [c.name for c in prog]
    assert names == ["G", "G", "G", "G",
                     "C", "C", "C", "C",
                     "G", "G", "G", "G",
                     "D", "D", "D", "D"]


def test_parse_chord_string_percent_repeats_bar():
    # "%" repeats the previous full bar (4 beats in 4/4)
    prog = parse_chord_string("| C / / / | % | F / / / |")
    names = [c.name for c in prog]
    assert names == ["C", "C", "C", "C",  # bar 1
                     "C", "C", "C", "C",  # bar 2 = % = repeat bar 1
                     "F", "F", "F", "F"]  # bar 3


def test_parse_chord_string_with_real_csv_line():
    # Real line from sources/jerry_song_book.csv (Candyman, Verse 1)
    line = "| C / / / | % |Gm / / / |F / / / | % | % | % |G / / / | % |"
    prog = parse_chord_string(line)
    # 9 bars of 4 beats = 36 chord slots.
    assert len(prog) == 36
    # First bar all C, third bar all Gm, last bar (% of G bar) all G
    assert prog[0].name == "C"
    assert prog[8].name == "Gm"
    assert prog[-1].name == "G"
    # Bars 5/6/7 are %% of the F bar, so all F
    assert prog[16:28] == [prog[12]] * 12  # F repeated
