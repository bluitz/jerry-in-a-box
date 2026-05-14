// pdf.js wrapper. Renders pages of the songbook into a canvas.
//
// The PDF is fetched from /api/songbook.pdf -- the user's own copy, served
// by the FastAPI backend. We never extract its text; we only render pages.

import * as pdfjsLib from "pdfjs-dist";
// Vite-friendly worker import: bundles the worker as an asset and gets a URL.
import workerSrc from "pdfjs-dist/build/pdf.worker.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

export class PdfViewer {
  private doc?: any;
  private container: HTMLElement;
  private currentPage = 1;
  private currentDpr = 1;

  constructor(container: HTMLElement) {
    this.container = container;
  }

  async load(url: string): Promise<void> {
    const task = pdfjsLib.getDocument(url);
    this.doc = await task.promise;
  }

  async show(pageNum: number): Promise<void> {
    if (!this.doc) throw new Error("PDF not loaded");
    if (pageNum < 1) pageNum = 1;
    if (pageNum > this.doc.numPages) pageNum = this.doc.numPages;
    this.currentPage = pageNum;
    const page = await this.doc.getPage(pageNum);

    const dpr = window.devicePixelRatio || 1;
    this.currentDpr = dpr;
    // Fall back to 800px if the container is hidden / zero-width (shouldn't
    // happen now that showSongPage reveals the section first, but defensive).
    const containerW = this.container.clientWidth || 800;
    const cssWidth = Math.min(containerW - 24, 900);
    const unscaled = page.getViewport({ scale: 1 });
    const scale = (cssWidth / unscaled.width) * dpr;
    const viewport = page.getViewport({ scale });

    this.container.innerHTML = "";
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", { alpha: false });
    if (!ctx) return;
    canvas.width  = Math.ceil(viewport.width);
    canvas.height = Math.ceil(viewport.height);
    canvas.style.width  = `${cssWidth}px`;
    canvas.style.height = `${cssWidth * (unscaled.height / unscaled.width)}px`;
    this.container.appendChild(canvas);
    await page.render({ canvasContext: ctx, viewport }).promise;
  }

  get page(): number { return this.currentPage; }
  get numPages(): number { return this.doc?.numPages ?? 0; }
}
