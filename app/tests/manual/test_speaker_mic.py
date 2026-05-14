"""Manual end-to-end speaker -> mic -> matcher test.

Runs the actual physical speaker-and-mic round trip. Requires:
  - a quiet room
  - speakers loud enough for the mic to pick up
  - macOS (afplay)
  - Microphone permission for the running terminal

Skipped by default. Enable with:
    JERRY_RUN_SPEAKER_MIC=1 pytest -s app/tests/manual

The test plays test-audio/Friend of the devil.m4a for 35 seconds and
verifies the matcher ranks "friend-of-the-devil" within the top 20.
We use top-20 (not top-1) because the songbook has many songs that
share Friend of the Devil's chord vocabulary (G, C, D, Am) and a 35s
sample doesn't always produce enough disambiguating chord changes.
This test is a smoke check that the pipeline runs end-to-end and
identifies the song as plausibly correct, not that it pinpoints it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
AUDIO = REPO / "test-audio" / "Friend of the devil.m4a"


@pytest.mark.skipif(
    os.environ.get("JERRY_RUN_SPEAKER_MIC") != "1",
    reason="Manual hardware test; set JERRY_RUN_SPEAKER_MIC=1 to enable.",
)
@pytest.mark.skipif(
    not AUDIO.exists(),
    reason=f"Reference audio not present at {AUDIO}.",
)
def test_speaker_mic_friend_of_the_devil():
    from app.tools.speaker_mic_test import run_test

    result = run_test(
        audio_path=AUDIO,
        target_id="friend-of-the-devil",
        duration_s=35.0,
        rank_threshold=20,
        verbose=True,
    )

    assert result["final_rank"] is not None, (
        "friend-of-the-devil missing from song database"
    )

    if result["capture_too_quiet"]:
        pytest.skip(
            f"Mic capture too quiet (RMS={result['capture_rms']:.4f}). "
            f"Raise speaker volume and confirm Terminal has Microphone "
            f"permission, then re-run."
        )

    assert result["passed"], (
        f"friend-of-the-devil ranked {result['final_rank']} "
        f"(need <= {result['rank_threshold']}). "
        f"Top 5 was: "
        + ", ".join(f"{t}({p*100:.1f}%)"
                    for _, t, p in result["final_top"][:5])
    )
