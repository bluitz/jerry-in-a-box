// Thin WebSocket wrapper for the matcher session.

export type ServerUpdate = {
  type: "update";
  top: { id: string; title: string; prob: number; page: number | null }[];
  decided: boolean;
  decided_song_id: string | null;
  n_obs: number;
  entropy: number;
  elapsed_seconds: number;
};

export type ServerReady = { type: "ready"; n_songs: number };
export type ServerPong  = { type: "pong" };
export type ServerError = { type: "error"; message: string };
export type ServerMessage = ServerUpdate | ServerReady | ServerPong | ServerError;

export class MatcherClient {
  private ws?: WebSocket;
  private url: string;
  private onMessage: (m: ServerMessage) => void;
  private onStatus: (s: "connecting" | "open" | "closed" | "error") => void;

  constructor(opts: {
    url?: string;
    onMessage: (m: ServerMessage) => void;
    onStatus: (s: "connecting" | "open" | "closed" | "error") => void;
  }) {
    // Same-origin /ws — works behind the Vite proxy in dev and the static
    // mount in prod.
    if (opts.url) {
      this.url = opts.url;
    } else {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      this.url = `${proto}://${location.host}/ws`;
    }
    this.onMessage = opts.onMessage;
    this.onStatus = opts.onStatus;
  }

  connect(): void {
    this.onStatus("connecting");
    this.ws = new WebSocket(this.url);
    this.ws.onopen    = () => this.onStatus("open");
    this.ws.onclose   = () => this.onStatus("closed");
    this.ws.onerror   = () => this.onStatus("error");
    this.ws.onmessage = (ev) => {
      try {
        const m = JSON.parse(ev.data) as ServerMessage;
        this.onMessage(m);
      } catch {/* ignore */}
    };
  }

  sendNote(pc: number, confidence: number, t: number): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "note", pc, confidence, t }));
  }

  reset(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "reset" }));
  }

  ping(): void {
    if (this.ws?.readyState !== WebSocket.OPEN) return;
    this.ws.send(JSON.stringify({ type: "ping" }));
  }

  close(): void {
    try { this.ws?.close(); } catch {}
  }
}
