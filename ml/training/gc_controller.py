"""Phase 5 - learned GC-trigger controller (tabular Q-learning / bandit).

The controller decides, at every "open segment full" point, whether to run a
greedy GC pass now or wait, from the contextual state

    [free_segment_ratio, recent_write_rate, recent_waf]

discretised exactly as the C++ ``GcTriggerController::discretize`` (3x3x3 = 27
states).  It is trained here against a compact Python mirror of the single-stream
greedy LFS engine, then the greedy Q-table is exported as
``data/predictions/gc_policy_<name>.csv`` (``state_id,q_wait,q_trigger``) for the
C++ simulator's ``--learned-trigger --gc-policy-table`` path.  Reward is the
negative count of valid blocks migrated in the interval since the last decision
(a proxy for -WAF increment), so the controller learns to time GC to minimise
copy-forward work.  The fixed-watermark baseline still runs whenever
``gc.learned_trigger: false``.
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import pandas as pd

from ml.common.config import load_config, get, repo_path, seed as cfg_seed

N_STATES = 27
ACTION_WAIT, ACTION_TRIGGER = 0, 1


def _bucket3(v: float, lo: float, hi: float) -> int:
    return 0 if v < lo else (1 if v < hi else 2)


def discretize(free_ratio: float, write_rate: float, waf: float) -> int:
    a = _bucket3(free_ratio, 0.10, 0.30)
    b = _bucket3(write_rate, 0.33, 0.66)
    c = _bucket3(waf - 1.0, 0.05, 0.15)
    return a * 9 + b * 3 + c


class GcEnv:
    """Minimal single-stream greedy LFS mirroring the C++ accounting closely
    enough to train GC-trigger timing."""

    def __init__(self, lbas: np.ndarray, total_segments: int, blocks_per_segment: int,
                 trigger_window: int, fixed_watermark: int = 2):
        self.lbas = lbas
        self.S = total_segments
        self.B = blocks_per_segment
        self.win = max(1, trigger_window)
        self.floor = max(2, fixed_watermark)  # safety reserve (mirrors C++)

    def reset(self):
        self.seg_lba = [np.full(self.B, -1, np.int64) for _ in range(self.S)]
        self.valid = np.zeros(self.S, np.int64)
        self.invalid = np.zeros(self.S, np.int64)
        self.nextw = np.zeros(self.S, np.int64)
        self.is_open = np.zeros(self.S, bool)
        self.free = list(range(self.S))
        self.l2p: dict[int, tuple[int, int]] = {}
        self.head = self.free.pop(0)
        self.is_open[self.head] = True
        self.logical = 0
        self.migrated = 0
        self.migrated_at_mark = 0
        self.writes_at_mark = 0
        self.nwrites = 0

    # -- mechanics -------------------------------------------------
    def _roll_head(self) -> bool:
        """Close the full head and open a fresh one from the free pool.
        Returns False if the pool is exhausted (caller must have reserved)."""
        self.is_open[self.head] = False
        if not self.free:
            return False
        self.head = self.free.pop(0)
        self.is_open[self.head] = True
        return True

    def _invalidate(self, lba: int) -> None:
        old = self.l2p.get(lba)
        if old is not None:
            os_, op_ = old
            if self.seg_lba[os_][op_] == lba:
                self.seg_lba[os_][op_] = -1
                self.valid[os_] -= 1
                self.invalid[os_] += 1

    def _write(self, lba: int) -> None:
        """Place a block into the current head; never recurses into GC."""
        if self.nextw[self.head] >= self.B and not self._roll_head():
            raise RuntimeError("GcEnv out of space (reserve violated)")
        self._invalidate(lba)
        pos = int(self.nextw[self.head])
        self.seg_lba[self.head][pos] = lba
        self.valid[self.head] += 1
        self.nextw[self.head] += 1
        self.l2p[lba] = (self.head, pos)

    def _greedy_gc(self) -> bool:
        cand = [i for i in range(self.S)
                if not self.is_open[i]
                and not (self.nextw[i] == 0 and self.valid[i] == 0 and self.invalid[i] == 0)]
        if not cand:
            return False
        victim = min(cand, key=lambda i: (self.valid[i], -self.invalid[i]))
        for pos in range(int(self.nextw[victim])):
            lba = int(self.seg_lba[victim][pos])
            if lba >= 0 and self.l2p.get(lba) == (victim, pos):
                self._write(lba)
                self.migrated += 1
        self.seg_lba[victim][:] = -1
        self.valid[victim] = self.invalid[victim] = self.nextw[victim] = 0
        self.free.append(victim)
        return True

    def state(self) -> tuple[float, float, float]:
        free_ratio = len(self.free) / self.S
        write_rate = min(1.0, (self.nwrites - self.writes_at_mark) / self.win)
        waf = 1.0 + (self.migrated / self.logical) if self.logical else 1.0
        return free_ratio, write_rate, waf

    def step_until_decision(self, idx: int) -> int:
        """Advance writes from ``idx`` until the head fills; return the next
        write index (a decision point)."""
        i = idx
        while i < len(self.lbas):
            if self.nextw[self.head] >= self.B:
                return i
            self._write(int(self.lbas[i]))
            self.logical += 1
            self.nwrites += 1
            i += 1
        return i

    def _gc_topup(self) -> None:
        if len(self.free) < 1:
            return
        self._greedy_gc()
        guard = 0
        while len(self.free) < self.floor + 1 and guard < 2 * self.S:
            before = len(self.free)
            if not self._greedy_gc():
                break
            guard = guard + 1 if len(self.free) <= before else 0

    def apply_action(self, action: int) -> float:
        if action == ACTION_TRIGGER or len(self.free) <= self.floor:
            self._gc_topup()
        if not self._roll_head():
            self._gc_topup()
            if not self._roll_head():
                raise RuntimeError("GcEnv out of space")
        reward = -(self.migrated - self.migrated_at_mark)
        self.migrated_at_mark = self.migrated
        self.writes_at_mark = self.nwrites
        return float(reward)


def train_q(env: GcEnv, episodes: int, epsilon: float, alpha: float, gamma: float,
            rng: np.random.Generator) -> np.ndarray:
    Q = np.zeros((N_STATES, 2), np.float64)
    for ep in range(episodes):
        env.reset()
        eps = epsilon * (1.0 - ep / max(1, episodes)) + 0.01
        i = env.step_until_decision(0)
        s = discretize(*env.state())
        total_r = 0.0
        while i < len(env.lbas):
            a = int(rng.integers(0, 2)) if rng.random() < eps else int(np.argmax(Q[s]))
            r = env.apply_action(a)
            i = env.step_until_decision(i)
            s2 = discretize(*env.state())
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])
            s, total_r = s2, total_r + r
        if (ep + 1) % max(1, episodes // 5) == 0 or ep == 0:
            print(f"  [gc_controller] episode {ep + 1:>3}/{episodes}  "
                  f"return={total_r:.0f}  migrated={env.migrated}  eps={eps:.3f}")
    return Q


def evaluate(env: GcEnv, policy) -> dict:
    env.reset()
    i = env.step_until_decision(0)
    while i < len(env.lbas):
        s = discretize(*env.state())
        a = policy(s, env)
        env.apply_action(a)
        i = env.step_until_decision(i)
    waf = 1.0 + env.migrated / env.logical if env.logical else 1.0
    return {"waf": round(waf, 5), "migrated_blocks": int(env.migrated)}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="SmartGC Phase 5 learned GC-trigger trainer")
    ap.add_argument("--trace", required=True, help="normalized trace CSV")
    ap.add_argument("--name", default=None)
    ap.add_argument("--total-segments", type=int, default=int(get(cfg, "simulator.total_segments", 32)))
    ap.add_argument("--blocks-per-segment", type=int, default=int(get(cfg, "simulator.blocks_per_segment", 64)))
    ap.add_argument("--episodes", type=int, default=int(get(cfg, "gc.controller_episodes", 40)))
    ap.add_argument("--max-events", type=int, default=None,
                    help="train the controller on the first N write events only")
    args = ap.parse_args()

    rng = np.random.default_rng(cfg_seed(cfg))
    stem = os.path.splitext(os.path.basename(args.trace))[0]
    name = args.name or stem

    df = pd.read_csv(args.trace)
    df.columns = [c.strip().lower() for c in df.columns]
    lbas = df.loc[df["operation"] == "W", "lba"].to_numpy(np.int64)
    if args.max_events:
        lbas = lbas[: int(args.max_events)]

    env = GcEnv(lbas, args.total_segments, args.blocks_per_segment,
                int(get(cfg, "gc.trigger_state_window_events", 500)),
                fixed_watermark=int(get(cfg, "simulator.gc_free_segments_threshold", 2)))
    Q = train_q(
        env, args.episodes,
        epsilon=float(get(cfg, "gc.controller_epsilon", 0.15)),
        alpha=float(get(cfg, "gc.controller_alpha", 0.30)),
        gamma=float(get(cfg, "gc.controller_gamma", 0.90)),
        rng=rng,
    )

    fixed_wm = int(get(cfg, "simulator.gc_free_segments_threshold", 2))
    res_fixed = evaluate(env, lambda s, e: ACTION_TRIGGER if len(e.free) <= fixed_wm else ACTION_WAIT)
    res_learned = evaluate(env, lambda s, e: int(np.argmax(Q[s])))
    print(f"[gc_controller] python-env fixed  : {res_fixed}")
    print(f"[gc_controller] python-env learned: {res_learned}")

    out = repo_path("data", "predictions", f"gc_policy_{name}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["state_id", "q_wait", "q_trigger"])
        written = 0
        for sid in range(N_STATES):
            if abs(Q[sid, 0]) + abs(Q[sid, 1]) == 0.0:
                continue  # unvisited -> let the C++ adaptive fallback handle it
            w.writerow([sid, f"{Q[sid, 0]:.6f}", f"{Q[sid, 1]:.6f}"])
            written += 1
    print(f"[gc_controller] wrote Q-table ({written}/{N_STATES} visited states) -> {out}")


if __name__ == "__main__":
    main()
