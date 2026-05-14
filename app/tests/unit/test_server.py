"""Smoke tests for the FastAPI server: health endpoint, REST data, and WS round-trip."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server.main import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    assert j["n_songs"] > 50


def test_songs_payload(client):
    r = client.get("/api/songs")
    assert r.status_code == 200
    j = r.json()
    assert j["n"] == len(j["songs"])
    s0 = j["songs"][0]
    # Frontend-facing payload — must NOT contain raw histograms/bigrams.
    assert "title" in s0 and "id" in s0
    assert "pc_histogram" not in s0
    assert "bigram" not in s0


def test_pdf_served(client):
    r = client.get("/api/songbook.pdf")
    if r.status_code == 404:
        pytest.skip("PDF not present in this environment")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/pdf")


def test_ws_basic_round_trip(client):
    """Open WS, send a note, expect the server to respond with the right
    message types. The matcher's actual convergence is covered by the
    matcher unit tests."""
    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["n_songs"] > 50

        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"

        # Send notes with real timestamps so the chord segmenter actually
        # closes segments (it buffers until the segment duration elapses).
        # At default 100 BPM, min segment ≈ 1.2s; we use t-spacing of 0.1s
        # so a burst of 24 frames spans 2.4s and produces ≥ 1 segment.
        chords = [7, 11, 2, 0, 4, 7, 9, 2, 6, 9, 0, 7]
        t = 0.0
        for pc in chords * 4:  # 48 frames over 4.8s
            ws.send_json({"type": "note", "pc": pc, "confidence": 0.9, "t": t})
            t += 0.1
        ws.send_json({"type": "ping"})

        # Drain: read messages until we see the pong sentinel.
        n_updates = 0
        last_top = None
        for _ in range(200):
            msg = ws.receive_json()
            t = msg.get("type")
            if t == "update":
                n_updates += 1
                last_top = msg["top"]
            elif t == "pong":
                break
            elif t == "error":
                pytest.fail(f"server error: {msg}")
        assert last_top is not None, "no update received"
        assert len(last_top) >= 5
        for item in last_top:
            assert {"id", "title", "prob", "page"} <= set(item.keys())


def test_ws_reset(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # ready
        ws.send_json({"type": "reset"})
        msg = ws.receive_json()
        assert msg["type"] == "update"
        # After reset, top-5 should be uniform-ish
        probs = [t["prob"] for t in msg["top"]]
        assert max(probs) - min(probs) < 1e-3
