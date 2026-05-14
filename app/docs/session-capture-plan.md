# Session Capture & Labeling Plan

Plan for capturing every user session and prompting for a ground-truth song
label, so we can replay real sessions to tune the matcher and (later)
retrain the acoustic model.

## Goals

- Every session that gets at least N seconds of audio is captured.
- After **Stop Listening** or after a decision, a small labeling dialog asks
  the user "which song was that?" with a searchable list.
- Labeled sessions are stored on disk in a stable, machine-readable format
  suitable for replay and tuning.
- Nothing leaves the user's machine; the user can disable capture or delete
  any session.

## What we capture per session

Two layers, both optional but the first is on by default:

1. **Event stream** (small, ~tens of KB per minute):
   - YIN frames sent over WS: `{t, pc, confidence}`
   - Onset events: `{t, rms}`
   - Estimated BPM events: `{t, bpm, confidence}`
   - Server-side derived events: chord segments emitted, matcher updates
     (top-1, top-1 prob, entropy), decision events, resets
2. **Raw mic audio** (heavy, ~5 MB/min at 16-bit 44.1 kHz mono — opt-in):
   - Saved as a PCM/WAV file alongside the events for full re-analysis later.

Plus per-session metadata: app version, matcher config, segmenter config,
sample rate, audio device label (when available), session start/stop wall
clock, label.

## Storage layout

```
app/data/sessions/
  index.jsonl                     # one line per session for fast listing
  YYYY-MM-DD/
    <session-id>/
      events.jsonl                # append-only event log
      meta.json                   # configs, versions, timestamps
      label.json                  # null until user labels; updated atomically
      audio.wav                   # only if "Save audio" is on
```

`session-id` = `YYYYMMDD-HHMMSS-<6-char-rand>`. Date-bucketed so the
directory scales.

`index.jsonl` line:

```json
{
  "id": "...",
  "started": "...",
  "ended": "...",
  "duration_s": 47.3,
  "n_events": 1421,
  "label": {"song_id": "friend-of-the-devil", "source": "user_confirmed"},
  "matcher_decision": "friend-of-the-devil"
}
```

## WS protocol additions

Client → server:

- `{"type": "label", "session_id": "...", "song_id": "...",
    "source": "user_confirmed" | "user_overrode" | "not_a_song" | "skipped",
    "notes": "..."}`
- `{"type": "end_session"}` — sent before WS close so the server can
  finalize cleanly

Server → client:

- `{"type": "session_started", "session_id": "..."}` — after `ready`, so
  the UI knows the id
- `{"type": "prompt_label", "session_id": "...",
    "predicted_song_id": "..." | null,
    "reason": "decision" | "stop" | "reset"}`
  fired when the matcher decides, or when the client sends `end_session`
  with at least N seconds of audio.
- `{"type": "label_ack", "session_id": "..."}`

## Backend changes (`app/server/`)

1. New module `app/server/recorder.py`:
   - `SessionRecorder(session_dir, save_audio=False)` with
     `append_event(d)`, `set_label(d)`, `finalize()`.
   - `events.jsonl` is appended line-by-line so a crash mid-session still
     leaves recoverable data.
   - Atomic writes for `label.json` and `meta.json`.
2. `main.py` changes:
   - Per-WS `recorder = SessionRecorder(...)` constructed after `ready`,
     send `session_started`.
   - Hook every incoming and derived event into `recorder.append_event(...)`.
   - On a matcher decision, send `prompt_label` (non-blocking — user can
     answer at their leisure).
   - On `end_session` or WS disconnect, `finalize()` and update
     `index.jsonl`.
   - Handle `label` messages → `recorder.set_label(...)` → `label_ack`.
3. New REST endpoints:
   - `GET /api/sessions?limit=50` — list from `index.jsonl`
   - `GET /api/sessions/{id}` — events + meta + label
   - `DELETE /api/sessions/{id}` — privacy
   - `GET /api/sessions/stats` — total count, labeled count, songs covered

## Frontend changes (`app/web/src/`)

1. New `LabelDialog` component (plain TS, no framework):
   - Triggered when `prompt_label` arrives, OR when **Stop listening** is
     clicked and the session has at least N seconds of audio.
   - Searchable song list (already have `/api/songs`).
   - Pre-selects `predicted_song_id` if present.
   - Buttons: **Confirm**, **Pick a different song**, **Not a song from
     the book**, **Skip / don't save label**.
   - Optional notes field.
2. Stop listening flow:
   - Send `end_session` over WS, then `close()`.
   - Wait for the next `prompt_label` and show the dialog before tearing
     down.
3. Settings panel (small, collapsed by default):
   - Toggle: **Capture sessions** (default on)
   - Toggle: **Also save raw audio** (default off, with a one-line note
     about disk usage)
   - Link: **View saved sessions** → simple list view
   - Button: **Delete all sessions** with confirm

## Optional raw-audio capture

When the audio toggle is on:

- Browser side: connect a `MediaRecorder` (or capture PCM via the same
  source node) and on stop, upload as `multipart/form-data` to
  `POST /api/sessions/{id}/audio`.
- Server writes to `audio.wav`. Size budget: one hour is ~300 MB at
  44.1 kHz / 16-bit; fine on a laptop.
- Toggle is sticky in `localStorage`; default off.

## Migration / replay

- Add `app/tests/synthetic/replay_session.py` that takes a `session_id`,
  replays its events through `Matcher` (and optionally a different
  `MatcherConfig` / `SegmenterConfig`), and reports decision vs. label.
- Extend the existing eval harness to use labeled real sessions as an
  additional test set alongside the synthetic dataset. This is the actual
  "training" loop: tune knobs, replay, measure accuracy.

## Privacy and ergonomics

- All storage is local; no upload code paths.
- Sessions can be deleted individually or all at once from the UI.
- A footer in the UI shows "N sessions captured · M labeled".
- `app/data/sessions/` is added to `.gitignore`.

## Suggested phasing

1. **Phase 1 — capture + label (no audio).** `SessionRecorder`,
   `events.jsonl`, label dialog, `prompt_label` on decision and stop.
   Smallest useful slice; unblocks data collection right away.
2. **Phase 2 — viewer + replay.** `/api/sessions`, simple session list
   view, `replay_session.py`. Lets you actually use the data.
3. **Phase 3 — tuning loop.** Wire labeled sessions into the eval
   harness; auto-tune `MatcherConfig` thresholds against the real
   dataset.
4. **Phase 4 — raw audio (opt-in).** `MediaRecorder`, upload endpoint,
   `audio.wav`. Defer until there's a use for it (e.g. retraining the
   chord recognizer).

## Open questions

1. **Minimum session length to prompt for a label.** Default 10 s? Below
   that we just discard, since labeling 2-second blips creates more
   noise than signal.
2. **What counts as "the session" when the user hits Reset?** Proposed:
   Reset closes and saves the current session, then starts a new one.
   This matches the matcher's actual state lifecycle and gives cleanly-
   separated examples.
3. **Modal vs. non-blocking label dialog.** Modal forces an answer;
   non-blocking is friendlier but produces more "skipped" labels.
4. **Save raw audio in Phase 1 or strictly Phase 4?** Capturing audio
   from day one (even unused) means the labeled corpus grows in
   parallel and is ready when retraining is.
