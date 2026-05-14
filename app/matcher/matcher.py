"""Bayesian song matcher.

Two likelihood components per song:

  1) Bag-of-notes:  log P(o_1..t | s) = sum_i log P(pc_i | s)
                    where P(pc | s) = (1-eps_h) * h_s(pc) + eps_h / 12
                    and h_s is the song's chord-tone histogram.

  2) HMM over chord positions:
       states     = collapsed chord vocabulary of song s
       transition = bigram matrix from songdb (with self-loops added so
                    the matcher tolerates dwelling on one chord)
       emission   = P(pc | chord) from emission.py

  We track log alpha_t(state) per song and on each observation do a
  forward step. log P(o_1..t | s) = logsumexp(alpha_t).

The combined posterior is:

  log P(s | o) ~ a*bag_log + (1-a)*hmm_log + log_prior(s)

then softmax-normalized across songs. Per-observation weight w_i is the
detector confidence, so low-confidence frames don't move the posterior
much. The decision rule consumes a sliding history of top-1 prob and the
top-1/top-2 ratio.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from app.matcher.emission import EmissionConfig, build_emission_table
from app.parser.chords import _QUALITY_TONES


# -------- public dataclasses --------

@dataclass
class NoteEvent:
    """A single observation from the pitch detector.

    pitch_class: 0..11 (C=0..B=11)
    confidence:  0..1, used as weight
    t:           wall-clock time in seconds (optional, used for plotting)
    """
    pitch_class: int
    confidence: float = 1.0
    t: float = 0.0


@dataclass
class TopK:
    song_id: str
    title: str
    prob: float
    page: Optional[int] = None


@dataclass
class MatchUpdate:
    top: list[TopK]
    decided: bool
    decided_song_id: Optional[str]
    n_obs: int
    entropy: float
    elapsed_seconds: float = 0.0


@dataclass
class MatcherConfig:
    """Defaults are the operating point picked by the eval sweep on the
    medium synthetic dataset (precision-favoring: <5% wrong decisions,
    accept more undecided cases as the trade-off)."""
    # Likelihood mixing — chord ORDER (HMM) does the work of disambiguating
    # same-key songs; the bag is a fast tiebreaker for far-key songs.
    bag_weight: float = 0.30
    hmm_weight: float = 0.70
    # Emission noise floor on the bag-of-notes side
    bag_epsilon: float = 0.08
    # HMM transition smoothing: probability of self-loop added to bigram
    hmm_self_loop: float = 0.55
    # Decision rule. NOTE: with the chord segmenter in place, each
    # observation corresponds to ~1 chord (one half- to full-bar). 30s of
    # music yields ~15-30 observations at 60-180 BPM, so the obs gate is
    # set conservatively below that and the time gate (decision_min_seconds)
    # is the dominant constraint.
    decision_min_obs: int = 12
    decision_min_seconds: float = 30.0  # wall-clock time before any decision
    decision_min_prob: float = 0.55     # top-1 posterior probability
    decision_min_ratio: float = 2.0     # top1/top2 probability ratio
    decision_sustain: int = 4           # observations meeting criteria (within window)
    decision_sustain_window: int = 6    # window size for sustain check
    # How many songs to return in each update
    top_k: int = 20
    # Sliding window — drop oldest evidence for songs that fall too far behind
    log_floor: float = -100.0


# -------- internal song state --------

@dataclass
class _Song:
    sid: str
    title: str
    page: Optional[int]
    pc_hist_log: np.ndarray         # (12,) log of (1-eps)*h + eps/12
    bigram_log: np.ndarray          # (V, V)
    emission: np.ndarray            # (V, 12)
    emission_log: np.ndarray        # (V, 12)
    alpha_log: np.ndarray = field(init=False)  # (V,) running forward state

    def init_alpha(self):
        v = self.bigram_log.shape[0]
        self.alpha_log = np.full(v, -math.log(v))


# -------- matcher --------

def _softmax_log(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax in log space."""
    m = x.max()
    return np.exp(x - m) / np.exp(x - m).sum()


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


class Matcher:
    """Online matcher across a song corpus.

    Usage:
        m = Matcher.from_songs_json(path)
        for ev in stream:
            update = m.update(ev)
            if update.decided:
                ...

    The matcher is fully deterministic given the same observation stream;
    no internal randomness. This makes the eval harness reproducible.
    """

    def __init__(self, songs: list[dict], page_index: dict[str, list[int]] | None = None,
                 config: MatcherConfig | None = None,
                 emission_cfg: EmissionConfig | None = None):
        self.config = config or MatcherConfig()
        self.emission_cfg = emission_cfg or EmissionConfig()
        self._page_index = page_index or {}

        self.songs: list[_Song] = []
        for s in songs:
            song = self._compile_song(s)
            song.init_alpha()
            self.songs.append(song)

        n = len(self.songs)
        # Uniform prior over songs (log).
        self._log_prior = np.full(n, -math.log(n))
        # Running sum of bag-of-notes log-likelihood per song.
        self._bag_log_sum = np.zeros(n, dtype=np.float64)
        # Running HMM log-likelihood per song:
        # logZ_s = logsumexp(alpha_s) after the most recent observation.
        self._hmm_log_evid = np.zeros(n, dtype=np.float64)
        self._n_obs = 0
        self._sustain_window: list[bool] = []   # sliding window of pass/fail
        self._last_decision: Optional[str] = None
        self._first_obs_t: Optional[float] = None  # t of first confident observation

    @classmethod
    def from_paths(cls,
                   songs_path: str | Path,
                   page_index_path: str | Path | None = None,
                   config: MatcherConfig | None = None) -> "Matcher":
        with open(songs_path) as f:
            db = json.load(f)
        page_index = {}
        if page_index_path and Path(page_index_path).exists():
            with open(page_index_path) as f:
                page_index = json.load(f)
        return cls(db["songs"], page_index, config=config)

    def _compile_song(self, s: dict) -> _Song:
        cfg = self.config
        # Bag-of-notes log emissions.
        h = np.array(s["pc_histogram"], dtype=np.float64)
        if h.sum() == 0:
            h = np.full(12, 1 / 12.0)
        else:
            h = h / h.sum()
        eps = cfg.bag_epsilon
        h_smoothed = (1 - eps) * h + eps / 12.0
        pc_hist_log = np.log(np.clip(h_smoothed, 1e-12, None))

        # HMM bigram (over chord_vocab order). Reconstruct chord tuples in vocab order.
        vocab = s["chord_vocab"]
        if not vocab:
            # Degenerate; fake a single-state HMM that emits uniformly.
            bigram = np.array([[1.0]])
            emission = np.full((1, 12), 1 / 12.0)
        else:
            bigram = np.array(s["bigram"], dtype=np.float64)
            # Add a self-loop component so the HMM can sit on one chord.
            sl = cfg.hmm_self_loop
            n = bigram.shape[0]
            bigram = (1 - sl) * bigram + sl * np.eye(n)
            # Renormalize rows.
            bigram = bigram / bigram.sum(axis=1, keepdims=True)
            # Build emission table from chord names in vocab.
            tuples, tones = self._chord_vocab_to_tuples(vocab, s["chord_tuples"], s["chords"])
            emission = build_emission_table(
                tuples, tones,
                s["key_pc"], s["key_mode"],
                self.emission_cfg,
            )

        bigram_log = np.log(np.clip(bigram, 1e-12, None))
        emission_log = np.log(np.clip(emission, 1e-12, None))

        return _Song(
            sid=s["id"],
            title=s["title"],
            page=(self._page_index.get(s["id"], [None])[0] if self._page_index else None),
            pc_hist_log=pc_hist_log,
            bigram_log=bigram_log,
            emission=emission,
            emission_log=emission_log,
        )

    def _chord_vocab_to_tuples(
        self,
        vocab: list[str],
        run_tuples: list[list],
        run_names: list[str],
    ) -> tuple[list[tuple[int, str, int | None]], list[frozenset[int]]]:
        """Map vocab[i] (chord name) -> (root,qual,bass) and chord-tone set.

        We use the run_tuples / run_names arrays (parallel) as a lookup:
        each name appearing in vocab will appear at least once in run_names.
        """
        name_to_tuple: dict[str, tuple[int, str, int | None]] = {}
        for name, tup in zip(run_names, run_tuples):
            if name not in name_to_tuple:
                name_to_tuple[name] = (int(tup[0]), str(tup[1]), tup[2] if tup[2] is None else int(tup[2]))

        tuples: list[tuple[int, str, int | None]] = []
        tones: list[frozenset[int]] = []
        for name in vocab:
            tup = name_to_tuple.get(name, (0, "maj", None))
            offsets = _QUALITY_TONES.get(tup[1], _QUALITY_TONES["maj"])
            t = set((tup[0] + o) % 12 for o in offsets)
            if tup[2] is not None:
                t.add(tup[2])
            tuples.append(tup)
            tones.append(frozenset(t))
        return tuples, tones

    def update(self, ev: NoteEvent) -> MatchUpdate:
        pc = int(ev.pitch_class) % 12
        w = max(0.0, min(1.0, float(ev.confidence)))

        if w > 0:
            self._n_obs += 1
            if self._first_obs_t is None:
                self._first_obs_t = ev.t
            # Bag-of-notes update: weighted log-likelihood
            for i, s in enumerate(self.songs):
                self._bag_log_sum[i] += w * s.pc_hist_log[pc]

            # HMM forward step (one per song). For each song:
            #   alpha'_j = (logsumexp_i (alpha_i + log_T_ij)) + log_E_jc
            # Multiply emission log by w (the per-frame confidence weight).
            for i, s in enumerate(self.songs):
                # Shape: (V, V) = alpha[:,None] + bigram_log
                a = s.alpha_log[:, None] + s.bigram_log  # broadcast to (V,V)
                m = a.max(axis=0)  # logsumexp over states i
                trans = m + np.log(np.exp(a - m[None, :]).sum(axis=0))
                new_alpha = trans + w * s.emission_log[:, pc]
                # Numerical floor
                new_alpha = np.clip(new_alpha, self.config.log_floor, None)
                s.alpha_log = new_alpha
                # Evidence: log P(o_1..t | s) = logsumexp(alpha)
                am = new_alpha.max()
                self._hmm_log_evid[i] = am + math.log(float(np.exp(new_alpha - am).sum()))

        return self._compute_top(ev.t)

    def _compute_top(self, ev_t: float = 0.0) -> MatchUpdate:
        cfg = self.config
        # Combine the two likelihoods log-linearly + uniform prior.
        score = (cfg.bag_weight * self._bag_log_sum
                 + cfg.hmm_weight * self._hmm_log_evid
                 + self._log_prior)
        post = _softmax_log(score)
        order = np.argsort(-post)
        k = min(cfg.top_k, len(self.songs))
        top = [
            TopK(
                song_id=self.songs[i].sid,
                title=self.songs[i].title,
                prob=float(post[i]),
                page=self.songs[i].page,
            )
            for i in order[:k]
        ]
        ent = _entropy(post)

        # Decision logic
        elapsed = (ev_t - self._first_obs_t) if self._first_obs_t is not None else 0.0
        decided_id: Optional[str] = None
        decided_now = False

        time_ok = (self._n_obs >= cfg.decision_min_obs
                   and elapsed >= cfg.decision_min_seconds)
        quality_ok = (len(top) >= 2
                      and top[0].prob >= cfg.decision_min_prob
                      and top[1].prob > 0
                      and (top[0].prob / top[1].prob) >= cfg.decision_min_ratio)
        candidate = top[0].song_id if (time_ok and len(top) > 0) else None

        # If the top song changes, reset the window entirely.
        if candidate != self._last_decision:
            self._sustain_window = []
            self._last_decision = candidate

        if time_ok:
            self._sustain_window.append(quality_ok)
            if len(self._sustain_window) > cfg.decision_sustain_window:
                self._sustain_window.pop(0)
            passes = sum(self._sustain_window)
            if (len(self._sustain_window) >= cfg.decision_sustain_window
                    and passes >= cfg.decision_sustain
                    and candidate is not None):
                decided_id = candidate
                decided_now = True

        return MatchUpdate(
            top=top,
            decided=decided_now,
            decided_song_id=decided_id,
            n_obs=self._n_obs,
            entropy=ent,
            elapsed_seconds=elapsed,
        )

    def reset(self) -> None:
        self._bag_log_sum[:] = 0
        self._hmm_log_evid[:] = 0
        for s in self.songs:
            s.init_alpha()
        self._n_obs = 0
        self._sustain_window = []
        self._last_decision = None
        self._first_obs_t = None

    @property
    def n_songs(self) -> int:
        return len(self.songs)
