// Top-level wiring for Jerry-in-a-box web app.

import { AudioPipeline, PC_NAMES, type ChromaEvent, type NoteEvent } from "./audio";
import { MatcherClient, type ServerUpdate } from "./ws";
import { PdfViewer } from "./pdf";

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;

const statusEl   = $<HTMLDivElement>("#status");
const startBtn   = $<HTMLButtonElement>("#start");
const resetBtn   = $<HTMLButtonElement>("#reset");
const metaEl     = $<HTMLSpanElement>("#meta");
const nowNoteEl  = $<HTMLDivElement>("#now-note");
const nowDetail  = $<HTMLDivElement>("#now-detail");
const chromaEl   = $<HTMLDivElement>("#chroma-bars");
const top5El     = $<HTMLOListElement>("#top5");

const listeningSec = $<HTMLElement>("#listening");
const songbookSec  = $<HTMLElement>("#songbook");
const backBtn      = $<HTMLButtonElement>("#back");
const pageTitle    = $<HTMLSpanElement>("#page-title");
const prevPageBtn  = $<HTMLButtonElement>("#prev-page");
const nextPageBtn  = $<HTMLButtonElement>("#next-page");
const pageInfoEl   = $<HTMLSpanElement>("#page-info");
const pdfContainer = $<HTMLDivElement>("#pdf-container");

let pipeline: AudioPipeline | undefined;
let client: MatcherClient | undefined;
let pdf: PdfViewer | undefined;

let lastUpdate: ServerUpdate | undefined;
let nNoteSent = 0;

function setStatus(text: string, kind: "idle" | "listening" | "error" | "decided" = "idle") {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

// ---- Render: chroma bars (12) ----
const chromaBars: HTMLDivElement[] = [];
for (let i = 0; i < 12; i++) {
  const bar = document.createElement("div");
  bar.className = "bar";
  const fill = document.createElement("div");
  fill.className = "fill";
  fill.style.height = "0%";
  bar.appendChild(fill);
  const name = document.createElement("div");
  name.className = "name";
  name.textContent = PC_NAMES[i];
  bar.appendChild(name);
  chromaEl.appendChild(bar);
  chromaBars.push(fill);
}

function drawChroma(c: Float32Array): void {
  let max = 0;
  for (let i = 0; i < 12; i++) if (c[i] > max) max = c[i];
  for (let i = 0; i < 12; i++) {
    const h = max > 0 ? (c[i] / max) * 100 : 0;
    chromaBars[i].style.height = `${h.toFixed(1)}%`;
  }
}

// ---- Render: now note ----
function drawNote(e: NoteEvent): void {
  nowNoteEl.textContent = PC_NAMES[e.pc];
  nowDetail.textContent = `${e.freq.toFixed(1)} Hz · midi ${e.midi} · conf ${(e.confidence * 100).toFixed(0)}%`;
}

// ---- Render: top-5 ----
function drawTop5(u: ServerUpdate): void {
  top5El.innerHTML = "";
  for (let i = 0; i < u.top.length; i++) {
    const t = u.top[i];
    const li = document.createElement("li");
    if (u.decided && u.decided_song_id === t.id) li.classList.add("decided");
    const rank = document.createElement("div");
    rank.className = "rank";
    rank.textContent = String(i + 1);
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = t.title;
    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.className = "fill";
    fill.style.width = `${(t.prob * 100).toFixed(1)}%`;
    bar.appendChild(fill);
    const pct = document.createElement("div");
    pct.className = "pct";
    pct.textContent = `${(t.prob * 100).toFixed(1)}%`;
    li.append(rank, title, bar, pct);
    top5El.appendChild(li);
  }
}

// ---- PDF view ----
async function showSongPage(songTitle: string, pageNum: number): Promise<void> {
  if (!pdf) {
    pdf = new PdfViewer(pdfContainer);
    await pdf.load("/api/songbook.pdf");
  }
  await pdf.show(pageNum);
  pageTitle.textContent = songTitle;
  pageInfoEl.textContent = `${pdf.page} / ${pdf.numPages}`;
  listeningSec.hidden = true;
  songbookSec.hidden = false;
}

backBtn.addEventListener("click", () => {
  songbookSec.hidden = true;
  listeningSec.hidden = false;
});
prevPageBtn.addEventListener("click", async () => {
  if (!pdf) return;
  await pdf.show(pdf.page - 1);
  pageInfoEl.textContent = `${pdf.page} / ${pdf.numPages}`;
});
nextPageBtn.addEventListener("click", async () => {
  if (!pdf) return;
  await pdf.show(pdf.page + 1);
  pageInfoEl.textContent = `${pdf.page} / ${pdf.numPages}`;
});

// ---- Server messages ----
function onServerMessage(m: any): void {
  if (m.type === "ready") {
    metaEl.textContent = `${m.n_songs} songs loaded`;
  } else if (m.type === "update") {
    lastUpdate = m as ServerUpdate;
    drawTop5(lastUpdate);
    if (lastUpdate.decided && lastUpdate.decided_song_id) {
      const top = lastUpdate.top.find(t => t.id === lastUpdate!.decided_song_id);
      if (top && top.page) {
        setStatus(`decided: ${top.title}`, "decided");
        showSongPage(top.title, top.page).catch(err => {
          console.error(err);
          setStatus("PDF error", "error");
        });
      }
    }
  } else if (m.type === "error") {
    console.warn("server error:", m.message);
  }
}

// ---- Start / Reset ----
startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  setStatus("connecting…", "idle");
  client = new MatcherClient({
    onMessage: onServerMessage,
    onStatus: (s) => {
      if (s === "open")    setStatus("listening", "listening");
      if (s === "closed")  setStatus("disconnected", "error");
      if (s === "error")   setStatus("connection error", "error");
    },
  });
  client.connect();

  pipeline = new AudioPipeline({
    onNote: (e: NoteEvent) => {
      drawNote(e);
      client?.sendNote(e.pc, e.confidence, e.t);
      nNoteSent++;
      if (nNoteSent % 25 === 0 && lastUpdate) {
        metaEl.textContent = `obs ${lastUpdate.n_obs} · entropy ${lastUpdate.entropy.toFixed(2)}`;
      }
    },
    onChroma: (e: ChromaEvent) => drawChroma(e.chroma),
    onError: (msg) => setStatus(`mic error: ${msg}`, "error"),
  });

  try {
    await pipeline.start();
    setStatus(`listening @ ${pipeline.sampleRate} Hz`, "listening");
    resetBtn.disabled = false;
  } catch {
    startBtn.disabled = false;
  }
});

resetBtn.addEventListener("click", () => {
  client?.reset();
  nNoteSent = 0;
  metaEl.textContent = "reset";
});

// Health check on load
fetch("/api/health")
  .then(r => r.json())
  .then(j => { metaEl.textContent = `${j.n_songs} songs loaded`; })
  .catch(() => { metaEl.textContent = "(server offline)"; });
