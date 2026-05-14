"""FastAPI server for the Jerry-in-a-box web app.

Endpoints:
  GET  /api/songs           - the canonical songs.json (titles + pages only fields needed by UI)
  GET  /api/page_index      - title -> page mapping
  GET  /api/songbook.pdf    - serves the user's local PDF for pdf.js rendering
  GET  /                    - serves the built frontend (if present)
  WS   /ws                  - per-session matcher.

WebSocket protocol (JSON messages):

  client -> server
    {"type": "note", "pc": 0..11, "confidence": 0..1, "t": float}
    {"type": "reset"}
    {"type": "config", "config": {...}}            # update matcher config
    {"type": "ping"}

  server -> client
    {"type": "ready", "n_songs": int}
    {"type": "update",
     "top":   [{"id","title","prob","page"}, ...],   # top-5
     "decided": bool,
     "decided_song_id": str | null,
     "n_obs": int,
     "entropy": float}
    {"type": "pong"}
    {"type": "error", "message": str}

The matcher itself is built fresh per WebSocket connection (per session).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.matcher import Matcher, MatcherConfig, NoteEvent


REPO = Path(__file__).resolve().parents[2]
SONGS_JSON     = REPO / "app" / "data" / "songs.json"
PAGE_INDEX     = REPO / "app" / "data" / "page_index.json"
PDF_PATH       = REPO / "jerry-garcia-song-book-ver-9-online.pdf"
WEB_DIST       = REPO / "app" / "web" / "dist"


def _load_db() -> tuple[dict, dict]:
    songs = json.loads(SONGS_JSON.read_text())
    page_index = json.loads(PAGE_INDEX.read_text()) if PAGE_INDEX.exists() else {}
    return songs, page_index


def create_app() -> FastAPI:
    app = FastAPI(title="Jerry in a Box", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         # local dev
        allow_methods=["*"],
        allow_headers=["*"],
    )

    songs_db, page_index = _load_db()
    # Slim down the songs payload sent to the browser. The frontend only
    # needs id/title/artist/key/page; it does NOT need histograms/bigrams.
    titles_payload = [
        {
            "id":        s["id"],
            "title":     s["title"],
            "artist":    s.get("artist", ""),
            "key":       s.get("key_name", ""),
            "page":      page_index.get(s["id"], [None])[0],
            "n_chord_events": s.get("n_chord_events", 0),
            "chord_vocab":    s.get("chord_vocab", []),
        }
        for s in songs_db["songs"]
    ]

    @app.get("/api/songs")
    def get_songs():
        return JSONResponse({"songs": titles_payload, "n": len(titles_payload)})

    @app.get("/api/page_index")
    def get_page_index():
        return JSONResponse(page_index)

    @app.get("/api/songbook.pdf")
    def get_pdf():
        if not PDF_PATH.exists():
            return JSONResponse({"error": "PDF not found"}, status_code=404)
        # Note: this serves the user's own local PDF file as-is for in-browser
        # rendering by pdf.js. No text extraction happens server-side.
        return FileResponse(
            PDF_PATH,
            media_type="application/pdf",
            filename=PDF_PATH.name,
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/api/health")
    def health():
        return {
            "ok":      True,
            "n_songs": len(titles_payload),
            "pdf":     PDF_PATH.exists(),
        }

    @app.websocket("/ws")
    async def ws(ws: WebSocket):
        await ws.accept()
        # Per-session matcher.
        matcher = Matcher(songs_db["songs"], page_index, MatcherConfig())
        await ws.send_json({"type": "ready", "n_songs": matcher.n_songs})

        last_send = 0.0
        SEND_HZ = 10.0  # cap server -> client updates to ~10/sec

        try:
            while True:
                msg = await ws.receive_text()
                try:
                    obj = json.loads(msg)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "bad json"})
                    continue

                kind = obj.get("type")
                if kind == "note":
                    try:
                        pc = int(obj["pc"]) % 12
                        conf = float(obj.get("confidence", 1.0))
                        t = float(obj.get("t", 0.0))
                    except (KeyError, ValueError, TypeError):
                        await ws.send_json({"type": "error", "message": "bad note event"})
                        continue
                    update = matcher.update(NoteEvent(pitch_class=pc, confidence=conf, t=t))
                    now = time.monotonic()
                    # Always send on a decision; otherwise rate-limit.
                    if update.decided or (now - last_send) >= 1.0 / SEND_HZ:
                        last_send = now
                        await ws.send_json(_update_to_dict(update))
                elif kind == "reset":
                    matcher.reset()
                    update = matcher._compute_top()
                    await ws.send_json(_update_to_dict(update))
                elif kind == "ping":
                    await ws.send_json({"type": "pong"})
                else:
                    await ws.send_json({"type": "error", "message": f"unknown type: {kind}"})

        except WebSocketDisconnect:
            return
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "message": str(e)})
            except Exception:
                pass

    # Serve the built frontend if present.
    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:
        @app.get("/")
        def root():
            return JSONResponse({
                "ok": True,
                "note": "frontend not built yet — visit the dev server",
            })

    return app


def _update_to_dict(update) -> dict:
    return {
        "type": "update",
        "top": [
            {"id": t.song_id, "title": t.title, "prob": t.prob, "page": t.page}
            for t in update.top
        ],
        "decided": bool(update.decided),
        "decided_song_id": update.decided_song_id,
        "n_obs": int(update.n_obs),
        "entropy": float(update.entropy),
    }


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server.main:app", host="127.0.0.1", port=8000, reload=False)
