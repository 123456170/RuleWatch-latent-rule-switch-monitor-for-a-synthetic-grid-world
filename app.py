import os
import io
import json
import time
import math
import hashlib
import itertools
import tempfile
import base64
import tracemalloc
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw


# ----------------------------------------------------------------------
# Constants and synthetic rule library
# ----------------------------------------------------------------------

VERSION = "0.1.0"

GRID = 12
N_AGENTS = 3
N_OBJECTS = 4
N_TERRAIN = 4
BLOCKED = 3
N_RULES = 5
UNSEEN_RULE_INTERNAL = 5
TRACE_LEN = 300
N_DIR = 5
N_ACTIONS = N_AGENTS * N_DIR
OBS_ERROR = 0.05

DIRS = [
    (0, 0),    # 0 stay
    (0, -1),   # 1 north
    (0, 1),    # 2 south
    (-1, 0),   # 3 west
    (1, 0),    # 4 east
]

TERRAIN_NAMES = {0: "open", 1: "rough", 2: "slippery", 3: "blocked"}
TERRAIN_COLORS = {
    0: (144, 238, 144),
    1: (205, 133, 63),
    2: (135, 206, 235),
    3: (80, 80, 80),
}

AGENT_COLORS = [(220, 20, 60), (34, 139, 34), (0, 0, 205)]
OBJECT_COLORS = {0: (255, 140, 0), 1: (128, 0, 128)}

RULE_COLORS = {
    1: (31, 119, 180),
    2: (255, 127, 14),
    3: (44, 160, 44),
    4: (214, 39, 40),
    5: (148, 103, 189),
}


@dataclass(frozen=True)
class RuleSet:
    name: str
    move_probs: Tuple[float, float, float, float]
    push_probs: Tuple[float, float]
    terrain_map: Tuple[int, int, int, int]


RULE_LIBRARY = [
    RuleSet("R1", (0.95, 0.75, 0.55, 0.00), (0.90, 0.50), (0, 1, 2, 3)),
    RuleSet("R2", (0.90, 0.85, 0.45, 0.00), (0.70, 0.35), (0, 0, 2, 3)),
    RuleSet("R3", (0.80, 0.60, 0.75, 0.00), (0.95, 0.65), (1, 1, 2, 3)),
    RuleSet("R4", (0.85, 0.50, 0.65, 0.00), (0.55, 0.25), (0, 2, 1, 3)),
    RuleSet("R5", (0.70, 0.90, 0.80, 0.00), (0.80, 0.80), (2, 1, 0, 3)),
]

UNSEEN_RULESET = RuleSet("R6-unseen", (0.50, 0.50, 0.50, 0.00), (0.10, 0.10), (0, 0, 0, 3))


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def inside(x: int, y: int) -> bool:
    return 0 <= x < GRID and 0 <= y < GRID


def block_idx(x: int, y: int) -> int:
    return (x // 4) * 3 + (y // 4)


def get_rule(idx: int) -> RuleSet:
    if idx < len(RULE_LIBRARY):
        return RULE_LIBRARY[idx]
    return UNSEEN_RULESET


def blend(c1: Tuple[int, int, int], c2: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    alpha = max(0.0, min(1.0, float(alpha)))
    return tuple(int(c1[i] * (1.0 - alpha) + c2[i] * alpha) for i in range(3))


def safe_median(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=float)))


def logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> Any:
    a = np.asarray(a, dtype=float)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    if axis is None:
        if keepdims:
            return out
        return float(out.reshape(-1)[0])
    if keepdims:
        return out
    return np.squeeze(out, axis=axis)


def dist_entropy(counter: Counter, labels: range) -> float:
    arr = np.array([counter.get(i, 0) for i in labels], dtype=float)
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ----------------------------------------------------------------------
# Environment simulator
# ----------------------------------------------------------------------

def make_terrain(seed: int) -> np.ndarray:
    rng = np.random.default_rng((seed * 2654435761) % (2 ** 32))
    terrain = rng.integers(0, 3, size=(GRID, GRID), dtype=np.int8)
    blocked = rng.random((GRID, GRID)) < 0.07
    terrain[blocked] = BLOCKED
    if np.sum(terrain != BLOCKED) < 60:
        terrain[terrain == BLOCKED] = 0
    return terrain.astype(int)


def initial_state(seed: int, rng: np.random.Generator) -> Dict[str, Any]:
    terrain = make_terrain(seed)
    free = [(x, y) for y in range(GRID) for x in range(GRID) if terrain[y, x] != BLOCKED]
    rng.shuffle(free)
    agents = [free[i] for i in range(N_AGENTS)]
    objects = [free[N_AGENTS + i] for i in range(N_OBJECTS)]
    masses = [int(rng.integers(0, 2)) for _ in range(N_OBJECTS)]
    return {
        "terrain": terrain,
        "agents": agents,
        "objects": objects,
        "masses": masses,
    }


def make_hash(state: Dict[str, Any], missing: List[bool]) -> Tuple[int, ...]:
    vals: List[int] = []
    for (x, y) in state["agents"]:
        vals.append(block_idx(x, y))
        vals.append(int(state["terrain"][y, x]))
    for i, (x, y) in enumerate(state["objects"]):
        if missing[i]:
            vals.append(9)
        else:
            vals.append(block_idx(x, y))
    return tuple(vals)


def observe_state(state: Dict[str, Any], rng: np.random.Generator):
    missing = [bool(rng.random() < OBS_ERROR) for _ in range(N_OBJECTS)]
    objects_obs = []
    for i, pos in enumerate(state["objects"]):
        objects_obs.append(None if missing[i] else pos)
    h = make_hash(state, missing)
    return h, objects_obs, missing


def step_state(state: Dict[str, Any], rule_idx: int, action: int, rng: np.random.Generator) -> Dict[str, Any]:
    rule = get_rule(rule_idx)
    terrain = state["terrain"].copy()
    agents = list(state["agents"])
    objects = list(state["objects"])
    masses = state["masses"]

    aidx = action // N_DIR
    dir_idx = action % N_DIR
    moved = False

    if dir_idx != 0:
        dx, dy = DIRS[dir_idx]
        x, y = agents[aidx]
        nx, ny = x + dx, y + dy

        if inside(nx, ny) and terrain[ny, nx] != BLOCKED:
            other_agent = any(agents[j] == (nx, ny) for j in range(N_AGENTS) if j != aidx)
            if not other_agent:
                obj_at = None
                for oi, op in enumerate(objects):
                    if op == (nx, ny):
                        obj_at = oi
                        break

                if obj_at is None:
                    p = rule.move_probs[int(terrain[ny, nx])]
                    if rng.random() < p:
                        agents[aidx] = (nx, ny)
                        moved = True
                else:
                    p = rule.push_probs[masses[obj_at]]
                    if rng.random() < p:
                        onx, onny = nx + dx, ny + dy
                        if inside(onx, onny) and terrain[onny, onx] != BLOCKED:
                            obj_conflict = any(objects[j] == (onx, onny) for j in range(N_OBJECTS) if j != obj_at)
                            agent_conflict = any(agents[j] == (onx, onny) for j in range(N_AGENTS) if j != aidx)
                            if not obj_conflict and not agent_conflict:
                                objects[obj_at] = (onx, onny)
                                agents[aidx] = (nx, ny)
                                moved = True

    if moved:
        tx, ty = agents[aidx]
        old_t = int(terrain[ty, tx])
        if old_t != BLOCKED:
            terrain[ty, tx] = rule.terrain_map[old_t]

    return {
        "terrain": terrain,
        "agents": agents,
        "objects": objects,
        "masses": masses,
    }


def choose_action(state: Dict[str, Any], rng: np.random.Generator, goal: Tuple[int, int]) -> int:
    if rng.random() < 0.20:
        aidx = int(rng.integers(N_AGENTS))
        ax, ay = state["agents"][aidx]
        gx, gy = goal
        dx = gx - ax
        dy = gy - ay
        cand = []
        if abs(dx) >= abs(dy):
            if dx > 0:
                cand.append(4)
            elif dx < 0:
                cand.append(3)
        if abs(dy) > abs(dx):
            if dy > 0:
                cand.append(2)
            elif dy < 0:
                cand.append(1)
        if not cand:
            cand = [0, 1, 2, 3, 4]
        dir_idx = cand[int(rng.integers(len(cand)))]
        return aidx * N_DIR + dir_idx
    return int(rng.integers(N_ACTIONS))


def sample_change_points(rng: np.random.Generator) -> List[int]:
    n = int(rng.integers(2, 5))
    while True:
        cps = sorted(int(v) for v in rng.choice(np.arange(40, 261), size=n, replace=False))
        if all(cps[i + 1] - cps[i] >= 30 for i in range(len(cps) - 1)):
            return cps


def build_rule_sequence(rng: np.random.Generator, length: int, cps: List[int], start: int) -> np.ndarray:
    seq = np.zeros(length, dtype=int)
    if not cps:
        seq[:] = start
        return seq
    seq[: cps[0]] = start
    prev = start
    for i, c in enumerate(cps):
        choices = [r for r in range(N_RULES) if r != prev]
        nxt = choices[int(rng.integers(len(choices)))]
        end = cps[i + 1] if i + 1 < len(cps) else length
        seq[c:end] = nxt
        prev = nxt
    return seq


@dataclass
class Trace:
    seed: int
    split: str
    actions: np.ndarray
    prev_hashes: List[Tuple[int, ...]]
    next_hashes: List[Tuple[int, ...]]
    true_rules: np.ndarray
    change_points: List[int]
    frames: List[Dict[str, Any]] = field(default_factory=list)
    rule_seq: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def make_frame(state: Dict[str, Any], objects_obs: List[Optional[Tuple[int, int]]], true_rule: int) -> Dict[str, Any]:
    return {
        "terrain": state["terrain"].copy(),
        "agents": list(state["agents"]),
        "objects_obs": list(objects_obs),
        "masses": list(state["masses"]),
        "true_rule": int(true_rule),
    }


def generate_trace(
    seed: int,
    split: str = "train",
    full: bool = False,
    force_rule: Optional[int] = None,
    no_change: bool = False,
    start_override: Optional[int] = None,
) -> Trace:
    rng = np.random.default_rng(seed)
    state = initial_state(seed, rng)
    free = [(x, y) for y in range(GRID) for x in range(GRID) if state["terrain"][y, x] != BLOCKED]
    goal = free[int(rng.integers(len(free)))]

    if force_rule is not None:
        rule_seq = np.full(TRACE_LEN, force_rule, dtype=int)
        cps: List[int] = []
    elif no_change:
        start = start_override if start_override is not None else seed % N_RULES
        rule_seq = np.full(TRACE_LEN, start, dtype=int)
        cps = []
    else:
        start = start_override if start_override is not None else seed % N_RULES
        cps = sample_change_points(rng)
        rule_seq = build_rule_sequence(rng, TRACE_LEN, cps, start)

    M = TRACE_LEN - 1
    actions = np.zeros(M, dtype=int)
    prev_hashes: List[Tuple[int, ...]] = []
    next_hashes: List[Tuple[int, ...]] = []
    true_rules = rule_seq[:M].astype(int) + 1
    frames: List[Dict[str, Any]] = []

    h, objects_obs, _ = observe_state(state, rng)
    if full:
        frames.append(make_frame(state, objects_obs, int(rule_seq[0]) + 1))

    for t in range(M):
        action = choose_action(state, rng, goal)
        rule_int = int(rule_seq[t])
        next_state = step_state(state, rule_int, action, rng)
        h_next, objects_obs_next, _ = observe_state(next_state, rng)

        actions[t] = action
        prev_hashes.append(h)
        next_hashes.append(h_next)

        if full:
            frame_rule = int(rule_seq[t + 1]) + 1 if t + 1 < TRACE_LEN else int(rule_seq[-1]) + 1
            frames.append(make_frame(next_state, objects_obs_next, frame_rule))

        state = next_state
        h = h_next
        objects_obs = objects_obs_next

    return Trace(
        seed=seed,
        split=split,
        actions=actions,
        prev_hashes=prev_hashes,
        next_hashes=next_hashes,
        true_rules=true_rules,
        change_points=cps,
        frames=frames,
        rule_seq=rule_seq,
    )


SPLIT_OFFSETS = {
    "train": 0,
    "val": 100000,
    "test": 200000,
    "guard": 300000,
    "unseen": 400000,
    "demo": 900000,
}


def generate_dataset(
    n: int,
    split: str,
    seed_base: int,
    no_change: bool = False,
    unseen: bool = False,
    full: bool = False,
) -> List[Trace]:
    traces = []
    for i in range(n):
        seed = int(seed_base + SPLIT_OFFSETS.get(split, 0) + i * 1009)
        force = UNSEEN_RULE_INTERNAL if unseen else None
        tr = generate_trace(
            seed=seed,
            split=split,
            full=full,
            force_rule=force,
            no_change=no_change,
            start_override=i % N_RULES,
        )
        traces.append(tr)
    return traces


# ----------------------------------------------------------------------
# Validation and corrupted fixtures
# ----------------------------------------------------------------------

def validate_trace(tr: Trace) -> List[str]:
    errors = set()
    if len(tr.actions) != TRACE_LEN - 1:
        errors.add("timestamp_gap")
    if tr.actions.size and (int(tr.actions.min()) < 0 or int(tr.actions.max()) >= N_ACTIONS):
        errors.add("invalid_action")
    for cp in tr.change_points:
        if cp < 40 or cp > 260:
            errors.add("change_point_range")
    if len(tr.change_points) > 1:
        cps = np.array(tr.change_points, dtype=int)
        if int(np.min(np.diff(cps))) < 30:
            errors.add("change_point_gap")
    for f in tr.frames:
        for x, y in f["agents"]:
            if not inside(x, y):
                errors.add("off_grid")
        for pos in f["objects_obs"]:
            if pos is not None:
                x, y = pos
                if not inside(x, y):
                    errors.add("off_grid")
                elif int(f["terrain"][y, x]) == BLOCKED:
                    errors.add("object_on_blocked")
    return sorted(errors)


def validate_record(rec: Dict[str, Any]) -> List[str]:
    errors = set()
    for pos in rec.get("agent_positions", []) + rec.get("object_positions", []):
        x, y = pos
        if not inside(x, y):
            errors.add("off_grid")
    terrain = rec.get("terrain")
    if terrain is not None:
        for x, y in rec.get("object_positions", []):
            if inside(x, y) and int(terrain[y, x]) == BLOCKED:
                errors.add("object_on_blocked")
    for a in rec.get("actions", []):
        if a < 0 or a >= N_ACTIONS:
            errors.add("invalid_action")
    cps = rec.get("change_points", [])
    for cp in cps:
        if cp < 40 or cp > 260:
            errors.add("change_point_range")
    if len(cps) > 1:
        arr = np.array(cps, dtype=int)
        if int(np.min(np.diff(arr))) < 30:
            errors.add("change_point_gap")
    ts = rec.get("timestamps", [])
    for i in range(len(ts) - 1):
        if ts[i + 1] - ts[i] != 1:
            errors.add("timestamp_gap")
    return sorted(errors)


def corrupted_fixtures() -> List[Dict[str, Any]]:
    base_terrain = np.zeros((GRID, GRID), dtype=int)
    base_terrain[1, 1] = BLOCKED
    return [
        {
            "name": "off_grid",
            "expected": ["off_grid"],
            "record": {
                "agent_positions": [(12, 0)],
                "object_positions": [(2, 2)],
                "actions": [0],
                "change_points": [100],
                "timestamps": [0, 1],
                "terrain": base_terrain.copy(),
            },
        },
        {
            "name": "invalid_action",
            "expected": ["invalid_action"],
            "record": {
                "agent_positions": [(0, 0)],
                "object_positions": [(2, 2)],
                "actions": [99],
                "change_points": [100],
                "timestamps": [0, 1],
                "terrain": base_terrain.copy(),
            },
        },
        {
            "name": "object_on_blocked",
            "expected": ["object_on_blocked"],
            "record": {
                "agent_positions": [(0, 0)],
                "object_positions": [(1, 1)],
                "actions": [0],
                "change_points": [100],
                "timestamps": [0, 1],
                "terrain": base_terrain.copy(),
            },
        },
        {
            "name": "change_point_range",
            "expected": ["change_point_range"],
            "record": {
                "agent_positions": [(0, 0)],
                "object_positions": [(2, 2)],
                "actions": [0],
                "change_points": [30],
                "timestamps": [0, 1],
                "terrain": base_terrain.copy(),
            },
        },
        {
            "name": "change_point_gap",
            "expected": ["change_point_gap"],
            "record": {
                "agent_positions": [(0, 0)],
                "object_positions": [(2, 2)],
                "actions": [0],
                "change_points": [40, 50],
                "timestamps": [0, 1],
                "terrain": base_terrain.copy(),
            },
        },
        {
            "name": "timestamp_gap",
            "expected": ["timestamp_gap"],
            "record": {
                "agent_positions": [(0, 0)],
                "object_positions": [(2, 2)],
                "actions": [0],
                "change_points": [100],
                "timestamps": [0, 2, 4],
                "terrain": base_terrain.copy(),
            },
        },
    ]


def rule_aspect_diffs() -> List[Dict[str, Any]]:
    rows = []
    for i in range(N_RULES):
        for j in range(i + 1, N_RULES):
            a, b = RULE_LIBRARY[i], RULE_LIBRARY[j]
            diffs = 0
            diffs += int(a.move_probs != b.move_probs)
            diffs += int(a.push_probs != b.push_probs)
            diffs += int(a.terrain_map != b.terrain_map)
            rows.append({"rule_a": a.name, "rule_b": b.name, "differing_aspects": diffs})
    return rows


def rule_frequency(traces: List[Trace], max_rule: int = N_RULES) -> np.ndarray:
    if not traces:
        return np.zeros(max_rule)
    arr = np.concatenate([tr.true_rules for tr in traces]) if traces else np.array([], dtype=int)
    counts = np.bincount(arr, minlength=max_rule + 2)[: max_rule + 1][1:]
    total = counts.sum()
    if total == 0:
        return np.zeros(max_rule)
    return counts.astype(float) / float(total)


# ----------------------------------------------------------------------
# Baseline: fixed dynamics + CUSUM
# ----------------------------------------------------------------------

class FixedDynamics:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.vocab_size = 1
        self.counts: Dict[Tuple[Tuple[int, ...], int], Dict[Tuple[int, ...], float]] = {}
        self.totals: Dict[Tuple[Tuple[int, ...], int], float] = {}

    def fit(self, traces: List[Trace]) -> None:
        vocab = set()
        for tr in traces:
            vocab.update(tr.next_hashes)
        self.vocab_size = max(1, len(vocab) + 1)

        tmp_counts: Dict[Tuple[Tuple[int, ...], int], Counter] = defaultdict(Counter)
        tmp_totals: Dict[Tuple[Tuple[int, ...], int], float] = defaultdict(float)

        for tr in traces:
            for p, a, n in zip(tr.prev_hashes, tr.actions, tr.next_hashes):
                key = (p, int(a))
                tmp_counts[key][n] += 1
                tmp_totals[key] += 1.0

        self.counts = {k: dict(v) for k, v in tmp_counts.items()}
        self.totals = {k: float(v) for k, v in tmp_totals.items()}

    def log_prob(self, prev_hash: Tuple[int, ...], action: int, next_hash: Tuple[int, ...]) -> float:
        key = (prev_hash, int(action))
        total = self.totals.get(key, 0.0)
        cnt = self.counts.get(key, {}).get(next_hash, 0.0)
        return math.log((cnt + self.alpha) / (total + self.alpha * self.vocab_size))

    def nll_sequence(self, tr: Trace) -> np.ndarray:
        out = []
        for p, a, n in zip(tr.prev_hashes, tr.actions, tr.next_hashes):
            out.append(-self.log_prob(p, int(a), n))
        return np.asarray(out, dtype=float)

    def top_next(self, prev_hash: Tuple[int, ...], action: int) -> Optional[Tuple[int, ...]]:
        key = (prev_hash, int(action))
        c = self.counts.get(key)
        if not c:
            return None
        return max(c.items(), key=lambda kv: kv[1])[0]


def cusum_flags(nll: np.ndarray, k: float, h: float, mu: float, sigma: float) -> List[int]:
    s = 0.0
    flags = []
    for i, x in enumerate(nll):
        z = (float(x) - mu) / (sigma + 1e-9)
        s = max(0.0, s + z - k)
        if s > h:
            flags.append(int(i))
            s = 0.0
    return flags


def cp_detection_stats(
    flags_list: List[List[int]],
    cps_list: List[List[int]],
    tol: int = 5,
    near_tol: int = 10,
    late_tol: int = 20,
) -> Dict[str, Any]:
    tp = fp = fn = 0
    fp_near = 0
    late_miss = 0
    delays: List[int] = []
    matched_pairs: List[Tuple[int, int]] = []

    for flags, cps in zip(flags_list, cps_list):
        flags = sorted(int(f) for f in flags)
        cps = [int(c) for c in cps]
        used = [False] * len(cps)

        for f in flags:
            matched = None
            for i, cp in enumerate(cps):
                if not used[i] and abs(f - cp) <= tol:
                    matched = i
                    break
            if matched is not None:
                used[matched] = True
                tp += 1
                d = f - cps[matched]
                delays.append(d)
                matched_pairs.append((cps[matched], f))
            else:
                fp += 1
                if any(abs(f - cp) <= near_tol for cp in cps):
                    fp_near += 1

        for i, cp in enumerate(cps):
            if not used[i]:
                fn += 1
                nearest = min((abs(f - cp) for f in flags), default=10 ** 9)
                if tol < nearest <= late_tol:
                    late_miss += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "delays": delays,
        "fp_near": int(fp_near),
        "late_miss": int(late_miss),
        "matched_pairs": matched_pairs,
    }


def bootstrap_ci(
    flags_list: List[List[int]],
    cps_list: List[List[int]],
    n_boot: int = 200,
    seed: int = 0,
    tol: int = 5,
) -> Tuple[float, float, float]:
    N = len(flags_list)
    if N == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        fl = [flags_list[i] for i in idx]
        cl = [cps_list[i] for i in idx]
        vals.append(cp_detection_stats(fl, cl, tol=tol)["f1"])
    return (
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
        float(np.mean(vals)),
    )


def tune_cusum(baseline: FixedDynamics, train_traces: List[Trace], val_traces: List[Trace]) -> Dict[str, Any]:
    train_arrays = [baseline.nll_sequence(tr) for tr in train_traces]
    if train_arrays and any(len(a) > 0 for a in train_arrays):
        nll_train = np.concatenate([a for a in train_arrays if len(a) > 0])
    else:
        nll_train = np.array([0.0])
    mu = float(np.mean(nll_train))
    sigma = float(np.std(nll_train)) + 1e-6

    val_arrays = [baseline.nll_sequence(tr) for tr in val_traces]
    val_cps = [tr.change_points for tr in val_traces]

    best_f1 = -1.0
    best_k, best_h = 0.5, 8.0
    for k in [0.1, 0.3, 0.5, 1.0]:
        for h in [2.0, 4.0, 6.0, 8.0, 12.0]:
            flags_list = [cusum_flags(a, k, h, mu, sigma) for a in val_arrays]
            f1 = cp_detection_stats(flags_list, val_cps)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_k, best_h = k, h

    return {
        "k": float(best_k),
        "h": float(best_h),
        "mu": mu,
        "sigma": sigma,
        "val_f1": float(max(0.0, best_f1)),
    }


# ----------------------------------------------------------------------
# Latent switching rule model
# ----------------------------------------------------------------------

def sticky_matrix(K: int, p_stay: float) -> np.ndarray:
    if K <= 1:
        return np.ones((K, K), dtype=float)
    A = np.full((K, K), (1.0 - p_stay) / (K - 1), dtype=float)
    np.fill_diagonal(A, p_stay)
    return A


def forward_backward(logL: np.ndarray, A: np.ndarray, pi: np.ndarray):
    M, K = logL.shape
    logA = np.log(np.clip(A, 1e-12, None))
    logpi = np.log(np.clip(pi, 1e-12, None))

    log_alpha = np.zeros((M, K), dtype=float)
    log_alpha[0] = logpi + logL[0]

    for t in range(1, M):
        for k in range(K):
            log_alpha[t, k] = logL[t, k] + logsumexp(log_alpha[t - 1, :] + logA[:, k])

    ll = logsumexp(log_alpha[-1])

    log_beta = np.zeros((M, K), dtype=float)
    for t in range(M - 2, -1, -1):
        for j in range(K):
            log_beta[t, j] = logsumexp(logA[j, :] + logL[t + 1, :] + log_beta[t + 1, :])

    log_gamma = log_alpha + log_beta
    log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma)

    xi = np.zeros((max(0, M - 1), K, K), dtype=float)
    for t in range(M - 1):
        arr = log_alpha[t][:, None] + logA + logL[t + 1][None, :] + log_beta[t + 1][None, :]
        norm = logsumexp(arr)
        xi[t] = np.exp(arr - norm)

    return gamma, xi, float(ll)


class LatentRuleModel:
    def __init__(
        self,
        K: int = N_RULES,
        p_stay: float = 0.98,
        alpha_dyn: float = 0.05,
        alpha_rule: float = 0.5,
    ):
        self.K = K
        self.p_stay = p_stay
        self.alpha_dyn = alpha_dyn
        self.alpha_rule = alpha_rule
        self.vocab_size = 1
        self.dyn_counts: Dict[Tuple[int, Tuple[int, ...], int], Dict[Tuple[int, ...], float]] = {}
        self.dyn_totals: Dict[Tuple[int, Tuple[int, ...], int], float] = {}
        self.A = sticky_matrix(K, p_stay)
        self.init_prior = np.ones(K, dtype=float) / K
        self.training_likelihoods: List[float] = []

    def build_vocab(self, traces: List[Trace]) -> None:
        vocab = set()
        for tr in traces:
            vocab.update(tr.next_hashes)
        self.vocab_size = max(1, len(vocab) + 1)

    def log_prob_dyn(self, rule: int, prev_hash: Tuple[int, ...], action: int, next_hash: Tuple[int, ...]) -> float:
        key = (int(rule), prev_hash, int(action))
        total = self.dyn_totals.get(key, 0.0)
        cnt = self.dyn_counts.get(key, {}).get(next_hash, 0.0)
        return math.log((cnt + self.alpha_dyn) / (total + self.alpha_dyn * self.vocab_size))

    def top_next(self, rule: int, prev_hash: Tuple[int, ...], action: int) -> Optional[Tuple[int, ...]]:
        key = (int(rule), prev_hash, int(action))
        c = self.dyn_counts.get(key)
        if not c:
            return None
        return max(c.items(), key=lambda kv: kv[1])[0]

    def _update_A(self, counts: np.ndarray) -> np.ndarray:
        prior = sticky_matrix(self.K, self.p_stay)
        c = np.asarray(counts, dtype=float) + self.alpha_rule * prior
        rs = c.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        return c / rs

    def _update_init(self, counts: np.ndarray) -> np.ndarray:
        c = np.asarray(counts, dtype=float) + self.alpha_rule / self.K
        s = c.sum()
        if s <= 0:
            return np.ones(self.K, dtype=float) / self.K
        return c / s

    def initialize_from_baseline(
        self,
        baseline: FixedDynamics,
        traces: List[Trace],
        mu: float,
        sigma: float,
        k: float = 0.5,
        h: float = 8.0,
    ) -> None:
        dyn_counts: Dict[Tuple[int, Tuple[int, ...], int], Counter] = defaultdict(Counter)
        dyn_totals: Dict[Tuple[int, Tuple[int, ...], int], float] = defaultdict(float)
        rule_counts = np.zeros((self.K, self.K), dtype=float)
        init_counts = np.zeros(self.K, dtype=float)

        for tr in traces:
            M = len(tr.actions)
            if M == 0:
                continue
            nll = baseline.nll_sequence(tr)
            flags = set(cusum_flags(nll, k, h, mu, sigma))
            labels = np.zeros(M, dtype=int)
            cur = 0
            for t in range(M):
                if t in flags:
                    cur = (cur + 1) % self.K
                labels[t] = cur

            init_counts[labels[0]] += 1.0
            for t in range(M - 1):
                rule_counts[labels[t], labels[t + 1]] += 1.0

            for t, (p, a, n, lab) in enumerate(zip(tr.prev_hashes, tr.actions, tr.next_hashes, labels)):
                key = (int(lab), p, int(a))
                dyn_counts[key][n] += 1.0
                dyn_totals[key] += 1.0

        self.dyn_counts = {k2: dict(v) for k2, v in dyn_counts.items()}
        self.dyn_totals = {k2: float(v) for k2, v in dyn_totals.items()}
        self.A = self._update_A(rule_counts)
        self.init_prior = self._update_init(init_counts)

    def em_fit(self, traces: List[Trace], iterations: int = 20) -> List[float]:
        prev_ll: Optional[float] = None
        for _ in range(max(0, iterations)):
            dyn_counts: Dict[Tuple[int, Tuple[int, ...], int], Counter] = defaultdict(Counter)
            dyn_totals: Dict[Tuple[int, Tuple[int, ...], int], float] = defaultdict(float)
            rule_counts = np.zeros((self.K, self.K), dtype=float)
            init_counts = np.zeros(self.K, dtype=float)
            total_ll = 0.0

            for tr in traces:
                M = len(tr.actions)
                if M == 0:
                    continue
                logL = np.zeros((M, self.K), dtype=float)
                for t in range(M):
                    p = tr.prev_hashes[t]
                    a = int(tr.actions[t])
                    n = tr.next_hashes[t]
                    for k in range(self.K):
                        logL[t, k] = self.log_prob_dyn(k, p, a, n)

                gamma, xi, ll = forward_backward(logL, self.A, self.init_prior)
                total_ll += ll
                init_counts += gamma[0]
                if M > 1:
                    rule_counts += xi.sum(axis=0)

                for t in range(M):
                    p = tr.prev_hashes[t]
                    a = int(tr.actions[t])
                    n = tr.next_hashes[t]
                    ws = gamma[t]
                    for k, w in enumerate(ws):
                        if w > 1e-8:
                            key = (int(k), p, a)
                            dyn_counts[key][n] += float(w)
                            dyn_totals[key] += float(w)

            if prev_ll is not None and total_ll < prev_ll - 1e-4:
                break

            self.dyn_counts = {k2: dict(v) for k2, v in dyn_counts.items()}
            self.dyn_totals = {k2: float(v) for k2, v in dyn_totals.items()}
            self.A = self._update_A(rule_counts)
            self.init_prior = self._update_init(init_counts)
            self.training_likelihoods.append(float(total_ll))
            prev_ll = float(total_ll)

        return self.training_likelihoods

    def online_filter(self, tr: Trace):
        M = len(tr.actions)
        probs = np.zeros((M, self.K), dtype=float)
        mp = np.zeros(M, dtype=int)
        maxp = np.zeros(M, dtype=float)
        if M == 0:
            return probs, mp, maxp

        log_alpha = np.log(np.clip(self.init_prior, 1e-12, None))
        logA = np.log(np.clip(self.A, 1e-12, None))

        for t in range(M):
            p = tr.prev_hashes[t]
            a = int(tr.actions[t])
            n = tr.next_hashes[t]
            logL = np.array([self.log_prob_dyn(k, p, a, n) for k in range(self.K)], dtype=float)
            log_pred = logsumexp(log_alpha[:, None] + logA, axis=0) + logL
            norm = logsumexp(log_pred)
            log_alpha = log_pred - norm
            posterior = np.exp(log_alpha)
            probs[t] = posterior
            mp[t] = int(np.argmax(posterior))
            maxp[t] = float(posterior.max())

        return probs, mp, maxp


def detect_latent(mp: np.ndarray, maxp: np.ndarray, prob_thresh: float, hold: int) -> Tuple[List[int], List[int]]:
    flags: List[int] = []
    estimates: List[int] = []
    M = len(mp)
    if M < 2:
        return flags, estimates

    last_alert = -10 ** 9
    hold = max(1, int(hold))

    for t in range(1, M):
        if int(mp[t]) != int(mp[t - 1]) and float(maxp[t]) >= prob_thresh:
            end = t + hold
            if end > M:
                continue
            if all(int(mp[i]) == int(mp[t]) for i in range(t, end)):
                alert = end - 1
                if alert - last_alert >= hold:
                    flags.append(int(alert))
                    estimates.append(int(t))
                    last_alert = alert
    return flags, estimates


def align_labels(traces: List[Trace], model: LatentRuleModel) -> Tuple[Tuple[int, ...], float]:
    mp_all: List[int] = []
    true_all: List[int] = []
    for tr in traces:
        _, mp, _ = model.online_filter(tr)
        mp_all.extend(int(x) for x in mp)
        true_all.extend(int(x) for x in tr.true_rules)

    if not mp_all:
        return tuple(range(1, N_RULES + 1)), 0.0

    best_perm = tuple(range(1, N_RULES + 1))
    best_acc = -1.0
    for perm in itertools.permutations(range(1, N_RULES + 1)):
        acc = float(np.mean([perm[m] == t for m, t in zip(mp_all, true_all)]))
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    return best_perm, best_acc


def tune_latent_detector(traces: List[Trace], model: LatentRuleModel, perm: Tuple[int, ...]) -> Dict[str, Any]:
    if not traces:
        return {"prob": 0.9, "hold": 1, "val_f1": 0.0}

    cache = []
    for tr in traces:
        _, mp, maxp = model.online_filter(tr)
        cache.append((mp, maxp, tr.change_points))

    best_f1 = -1.0
    best = {"prob": 0.9, "hold": 1, "val_f1": 0.0}
    for prob in [0.50, 0.65, 0.80, 0.90, 0.95]:
        for hold in [1, 2, 3, 5]:
            flags_list = []
            for mp, maxp, _ in cache:
                flags, _ = detect_latent(mp, maxp, prob, hold)
                flags_list.append(flags)
            cps_list = [cps for _, _, cps in cache]
            f1 = cp_detection_stats(flags_list, cps_list)["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best = {"prob": float(prob), "hold": int(hold), "val_f1": float(f1)}
    if best_f1 < 0:
        best = {"prob": 0.9, "hold": 1, "val_f1": 0.0}
    return best


def simple_eval(model: LatentRuleModel, perm: Tuple[int, ...], params: Dict[str, Any], traces: List[Trace]) -> Dict[str, float]:
    acc_num = 0
    acc_den = 0
    flags_list = []
    cps_list = []
    for tr in traces:
        _, mp, maxp = model.online_filter(tr)
        aligned = np.array([perm[m] for m in mp], dtype=int) if len(mp) else np.array([], dtype=int)
        acc_num += int(np.sum(aligned == tr.true_rules))
        acc_den += int(len(tr.true_rules))
        flags, _ = detect_latent(mp, maxp, params["prob"], params["hold"])
        flags_list.append(flags)
        cps_list.append(tr.change_points)

    stats = cp_detection_stats(flags_list, cps_list)
    return {
        "accuracy": float(acc_num / acc_den) if acc_den else 0.0,
        "f1": float(stats["f1"]),
        "median_delay": safe_median(stats["delays"]),
    }


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_all(
    model: LatentRuleModel,
    perm: Tuple[int, ...],
    lat_params: Dict[str, Any],
    test_traces: List[Trace],
    guard_traces: List[Trace],
    unseen_traces: List[Trace],
    baseline: FixedDynamics,
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    inverse_perm = {true_rule: mod for mod, true_rule in enumerate(perm)}

    acc_num = 0
    acc_den = 0
    raw_acc_num = 0
    flags_list: List[List[int]] = []
    cps_list: List[List[int]] = []
    all_cps: List[int] = []
    all_flags: List[int] = []
    entropies: List[float] = []
    overall_assignment = Counter()
    rule_model_dist: Dict[int, Counter] = defaultdict(Counter)
    worst_traces = []

    h4_emp: Dict[int, Dict[Tuple[Tuple[int, ...], int], Counter]] = defaultdict(lambda: defaultdict(Counter))
    h4_learn: Dict[Tuple[int, Tuple[Tuple[int, ...], int]], Optional[Tuple[int, ...]]] = {}

    for tr in test_traces:
        probs, mp, maxp = model.online_filter(tr)
        aligned = np.array([perm[m] for m in mp], dtype=int) if len(mp) else np.array([], dtype=int)

        acc_num += int(np.sum(aligned == tr.true_rules))
        acc_den += int(len(tr.true_rules))
        raw_acc_num += int(np.sum((mp + 1) == tr.true_rules))

        flags, _ = detect_latent(mp, maxp, lat_params["prob"], lat_params["hold"])
        flags_list.append(flags)
        cps_list.append(tr.change_points)
        all_flags.extend(flags)
        all_cps.extend(tr.change_points)

        if len(aligned):
            counts = np.bincount(aligned, minlength=N_RULES + 2)[1:N_RULES + 1].astype(float)
            p = counts / max(1.0, counts.sum())
            p = p[p > 0]
            entropies.append(float(-np.sum(p * np.log2(p))))
            overall_assignment.update(int(x) for x in aligned)

        for t in range(len(tr.actions)):
            true_rule = int(tr.true_rules[t])
            mod = int(inverse_perm[true_rule])
            key = (tr.prev_hashes[t], int(tr.actions[t]))
            h4_emp[true_rule][key][tr.next_hashes[t]] += 1
            rule_model_dist[true_rule][int(mp[t])] += 1
            h4_learn[(true_rule, key)] = model.top_next(mod, tr.prev_hashes[t], int(tr.actions[t]))

        tr_stats = cp_detection_stats([flags], [tr.change_points])
        trace_acc = float(np.mean(aligned == tr.true_rules)) if len(aligned) else 0.0
        tr_ent = dist_entropy(Counter(int(x) for x in aligned), range(1, N_RULES + 1)) if len(aligned) else 0.0

        if tr_stats["fn"] > 0 and tr_stats["late_miss"] > 0:
            cat = "late_detection"
        elif tr_stats["fp"] > 0 and tr_stats["fp_near"] > 0:
            cat = "false_alarm_near_change"
        elif tr_ent > 1.5:
            cat = "rule_split"
        else:
            cat = "other"

        worst_traces.append({
            "seed": int(tr.seed),
            "accuracy": trace_acc,
            "missed": int(tr_stats["fn"]),
            "false_alarms": int(tr_stats["fp"]),
            "category": cat,
        })

    latent_acc = float(acc_num / acc_den) if acc_den else 0.0
    raw_acc = float(raw_acc_num / acc_den) if acc_den else 0.0
    latent_stats = cp_detection_stats(flags_list, cps_list)
    ci_low, ci_high, ci_mean = bootstrap_ci(flags_list, cps_list, n_boot=200, seed=17)

    h4_results: Dict[int, float] = {}
    for rule in range(1, N_RULES + 1):
        pairs = h4_emp.get(rule, {})
        total = len(pairs)
        match = 0
        for key, counter in pairs.items():
            emp_top = counter.most_common(1)[0][0]
            learn_top = h4_learn.get((rule, key))
            if learn_top == emp_top:
                match += 1
        h4_results[rule] = float(match / total) if total else 0.0

    guard_flags: List[int] = []
    guard_steps = 0
    base_guard_flags: List[int] = []
    for tr in guard_traces:
        _, mp, maxp = model.online_filter(tr)
        flags, _ = detect_latent(mp, maxp, lat_params["prob"], lat_params["hold"])
        guard_flags.extend(flags)
        guard_steps += int(len(tr.actions))

        nll = baseline.nll_sequence(tr)
        bflags = cusum_flags(nll, base_params["k"], base_params["h"], base_params["mu"], base_params["sigma"])
        base_guard_flags.extend(bflags)

    latent_far = float(len(guard_flags) / guard_steps * 100.0) if guard_steps else 0.0
    baseline_far = float(len(base_guard_flags) / guard_steps * 100.0) if guard_steps else 0.0

    unseen_domination = 0.0
    unseen_flags: List[int] = []
    unseen_steps = 0
    unseen_shares = Counter()
    unseen_entropies: List[float] = []
    for tr in unseen_traces:
        _, mp, maxp = model.online_filter(tr)
        aligned = np.array([perm[m] for m in mp], dtype=int) if len(mp) else np.array([], dtype=int)
        flags, _ = detect_latent(mp, maxp, lat_params["prob"], lat_params["hold"])
        unseen_flags.extend(flags)
        unseen_steps += int(len(tr.actions))
        if len(aligned):
            counts = np.bincount(aligned, minlength=N_RULES + 2)[1:N_RULES + 1].astype(float)
            share = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
            unseen_domination = max(unseen_domination, share)
            unseen_shares.update(int(x) for x in aligned)
            p = counts / max(1.0, counts.sum())
            p = p[p > 0]
            unseen_entropies.append(float(-np.sum(p * np.log2(p))))

    base_flags_list: List[List[int]] = []
    for tr in test_traces:
        nll = baseline.nll_sequence(tr)
        bflags = cusum_flags(nll, base_params["k"], base_params["h"], base_params["mu"], base_params["sigma"])
        base_flags_list.append(bflags)
    baseline_stats = cp_detection_stats(base_flags_list, cps_list)

    if test_traces:
        all_true = np.concatenate([tr.true_rules for tr in test_traces])
        counts = np.bincount(all_true, minlength=N_RULES + 2)[1:N_RULES + 1]
        baseline_rule_acc = float(counts.max() / counts.sum()) if counts.sum() > 0 else 0.0
    else:
        baseline_rule_acc = 0.0

    rule_split_count = sum(
        1 for rule, dist in rule_model_dist.items()
        if dist_entropy(dist, range(N_RULES)) > 1.0
    )
    flags_near_start = sum(1 for f in all_flags if f < 40)

    worst_sorted = sorted(worst_traces, key=lambda x: (x["accuracy"], -x["missed"]))[:10]

    collapse_entropy = float(np.mean(entropies)) if entropies else 0.0
    collapse_share = float(max(overall_assignment.values()) / sum(overall_assignment.values())) if overall_assignment else 0.0

    return {
        "latent_rule_accuracy": latent_acc,
        "latent_raw_accuracy_no_align": raw_acc,
        "baseline_trivial_rule_accuracy": baseline_rule_acc,
        "latent_cp_f1": float(latent_stats["f1"]),
        "latent_precision": float(latent_stats["precision"]),
        "latent_recall": float(latent_stats["recall"]),
        "latent_tp": int(latent_stats["tp"]),
        "latent_fp": int(latent_stats["fp"]),
        "latent_fn": int(latent_stats["fn"]),
        "latent_ci_low": float(ci_low),
        "latent_ci_high": float(ci_high),
        "latent_ci_mean": float(ci_mean),
        "latent_median_delay": safe_median(latent_stats["delays"]),
        "latent_mean_abs_delay": float(np.mean(np.abs(latent_stats["delays"]))) if latent_stats["delays"] else float("nan"),
        "latent_delays": latent_stats["delays"],
        "baseline_cp_f1": float(baseline_stats["f1"]),
        "baseline_precision": float(baseline_stats["precision"]),
        "baseline_recall": float(baseline_stats["recall"]),
        "baseline_median_delay": safe_median(baseline_stats["delays"]),
        "h4_per_rule": h4_results,
        "guard_false_alarms_per_100": latent_far,
        "baseline_guard_false_alarms_per_100": baseline_far,
        "entropy_mean": collapse_entropy,
        "entropy_min": float(np.min(entropies)) if entropies else 0.0,
        "max_rule_share": collapse_share,
        "unseen_dominance": float(unseen_domination),
        "unseen_flags_per_100": float(len(unseen_flags) / unseen_steps * 100.0) if unseen_steps else 0.0,
        "unseen_entropy_mean": float(np.mean(unseen_entropies)) if unseen_entropies else 0.0,
        "unseen_shares": dict(unseen_shares),
        "failure": {
            "collapse": bool(collapse_entropy < 0.2 or collapse_share > 0.9),
            "rule_split_count": int(rule_split_count),
            "unseen_absorption": bool(unseen_domination > 0.9),
            "flags_near_start": int(flags_near_start),
            "late_miss": int(latent_stats["late_miss"]),
            "fp_near_change": int(latent_stats["fp_near"]),
        },
        "worst_traces": worst_sorted,
        "all_cps": all_cps,
        "all_flags": all_flags,
    }


# ----------------------------------------------------------------------
# Visualization and demo clip generation
# ----------------------------------------------------------------------

def draw_frame(
    frame: Dict[str, Any],
    step: int,
    aligned_mp: np.ndarray,
    maxp: np.ndarray,
    change_points: List[int],
    flags: List[int],
    width: int = 760,
    cell: int = 34,
) -> Image.Image:
    grid_w = GRID * cell
    margin_x = max(12, cell // 2)
    margin_y = max(24, cell)
    bar_h = max(18, int(cell * 0.65))
    band_top = margin_y + grid_w + 12
    footer_h = 72

    min_width = grid_w + 2 * margin_x + 24
    if width < min_width:
        width = min_width

    height = band_top + bar_h + 48 + footer_h

    # FFmpeg/libx264/yuv420p requires even dimensions.
    if height % 2:
        height += 1
    if width % 2:
        width += 1

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)

    d.text((8, 6), "RuleWatch — synthetic grid-world demo (not a real environment)", fill=(0, 0, 0))

    terrain = frame["terrain"]
    for y in range(GRID):
        for x in range(GRID):
            x0 = margin_x + x * cell
            y0 = margin_y + y * cell
            x1 = x0 + cell - 1
            y1 = y0 + cell - 1
            color = TERRAIN_COLORS.get(int(terrain[y, x]), (200, 200, 200))
            d.rectangle([x0, y0, x1, y1], fill=color, outline=(230, 230, 230))

    missing_count = 0
    obj_r = max(3, int(cell * 0.25))
    for oi, pos in enumerate(frame["objects_obs"]):
        if pos is None:
            missing_count += 1
            continue
        x, y = pos
        cx = margin_x + x * cell + cell // 2
        cy = margin_y + y * cell + cell // 2
        mass = int(frame["masses"][oi]) if oi < len(frame["masses"]) else 0
        color = OBJECT_COLORS.get(mass, (0, 0, 0))
        d.ellipse([cx - obj_r, cy - obj_r, cx + obj_r, cy + obj_r], fill=color, outline=(255, 255, 255))

    agent_r = max(4, int(cell * 0.32))
    for ai, (x, y) in enumerate(frame["agents"]):
        cx = margin_x + x * cell + cell // 2
        cy = margin_y + y * cell + cell // 2
        color = AGENT_COLORS[ai % len(AGENT_COLORS)]
        d.ellipse([cx - agent_r, cy - agent_r, cx + agent_r, cy + agent_r], fill=color, outline=(255, 255, 255))

    band_w = grid_w
    for t in range(TRACE_LEN):
        x0 = margin_x + int(t * band_w / TRACE_LEN)
        x1 = margin_x + int((t + 1) * band_w / TRACE_LEN)
        if t > step or len(aligned_mp) == 0:
            color = (225, 225, 225)
        else:
            idx = min(t, len(aligned_mp) - 1)
            rule = int(aligned_mp[idx])
            prob = float(maxp[idx]) if idx < len(maxp) else 0.0
            base = RULE_COLORS.get(rule, (200, 200, 200))
            color = blend(base, (245, 245, 245), 1.0 - max(0.0, min(1.0, prob)))
        d.rectangle([x0, band_top, x1, band_top + bar_h], fill=color)

    line_w = max(1, cell // 16)
    for cp in change_points:
        if cp <= step:
            x = margin_x + int(cp * band_w / TRACE_LEN)
            d.line([(x, band_top - 6), (x, band_top + bar_h + 6)], fill=(220, 20, 60), width=line_w)

    tri_s = max(4, cell // 5)
    for fl in flags:
        if fl <= step:
            x = margin_x + int(fl * band_w / TRACE_LEN)
            y0 = band_top + bar_h + 10
            d.polygon([(x - tri_s, y0 + tri_s), (x + tri_s, y0 + tri_s), (x, y0)], fill=(255, 165, 0))

    if len(aligned_mp) > 0:
        idx = min(step, len(aligned_mp) - 1)
        current_rule = int(aligned_mp[idx])
        current_prob = float(maxp[idx]) if idx < len(maxp) else 0.0
    else:
        current_rule = 0
        current_prob = 0.0

    info1 = f"Step {step}/{TRACE_LEN - 1} | inferred R{current_rule} | p={current_prob:.2f}"
    info2 = f"true R{frame['true_rule']} | missing objects: {missing_count} | red=true CP, orange=flag"

    d.text((8, band_top + bar_h + 28), info1, fill=(0, 0, 0))
    d.text((8, band_top + bar_h + 46), info2, fill=(80, 80, 80))

    return img


def generate_demo_clip(
    demo: Trace,
    aligned_mp: np.ndarray,
    maxp: np.ndarray,
    flags: List[int],
    fps: int = 15,
) -> Tuple[Optional[bytes], str, str, Optional[str]]:
    n_frames = min(TRACE_LEN, len(demo.frames))
    if n_frames == 0:
        return None, "rulewatch_demo.mp4", "video/mp4", "No demo frames available."

    mp4_error = None

    # Primary attempt: MP4 via imageio-ffmpeg.
    try:
        import imageio.v2 as imageio

        with tempfile.TemporaryDirectory(prefix="rulewatch_") as td:
            path = os.path.join(td, "rulewatch_demo.mp4")
            writer = imageio.get_writer(
                path,
                format="FFMPEG",
                mode="I",
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                output_params=["-movflags", "+faststart"],
            )
            try:
                for step in range(n_frames):
                    im = draw_frame(
                        demo.frames[step],
                        step,
                        aligned_mp,
                        maxp,
                        demo.change_points,
                        flags,
                        width=760,
                        cell=34,
                    )
                    writer.append_data(np.asarray(im))
            finally:
                try:
                    writer.close()
                except Exception:
                    pass

            with open(path, "rb") as f:
                data = f.read()

        if data and len(data) > 1000:
            return data, "rulewatch_demo.mp4", "video/mp4", None

        mp4_error = "MP4 output was too small or empty."
    except Exception as e:
        mp4_error = repr(e)

    # Fallback: animated GIF via Pillow, smaller and sampled to stay memory-friendly.
    try:
        stride = 2
        duration_ms = int((1000.0 / float(fps)) * stride)
        frames = []
        for step in range(0, n_frames, stride):
            im = draw_frame(
                demo.frames[step],
                step,
                aligned_mp,
                maxp,
                demo.change_points,
                flags,
                width=520,
                cell=20,
            )
            frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))

        if not frames:
            return None, "rulewatch_demo.gif", "image/gif", mp4_error or "No GIF frames."

        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        data = buf.getvalue()
        if data:
            return data, "rulewatch_demo.gif", "image/gif", mp4_error

        return None, "rulewatch_demo.gif", "image/gif", mp4_error or "GIF output empty."
    except Exception as e:
        return None, "rulewatch_demo.gif", "image/gif", f"MP4 error: {mp4_error}; GIF error: {e!r}"


def render_clip_html(clip_bytes: bytes, mime: str):
    b64 = base64.b64encode(clip_bytes).decode("ascii")
    if mime.startswith("video"):
        html = (
            f'<video controls autoplay muted loop playsinline '
            f'style="width:100%; max-height:560px; border-radius:10px; background:#000;">'
            f'<source src="data:{mime};base64,{b64}" type="{mime}">'
            f'Your browser does not support embedded video.'
            f'</video>'
        )
    else:
        html = (
            f'<img src="data:{mime};base64,{b64}" '
            f'style="width:100%; max-height:560px; object-fit:contain; border-radius:10px;" '
            f'alt="RuleWatch demo clip">'
        )
    st.markdown(html, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------

def run_pipeline(
    seed: int,
    train_n: int,
    val_n: int,
    test_n: int,
    guard_n: int,
    unseen_n: int,
    em_iters: int,
    run_ablation: bool,
) -> Dict[str, Any]:
    t_start = time.time()
    seed = int(seed)

    train_traces = generate_dataset(train_n, "train", seed * 1000)
    val_traces = generate_dataset(val_n, "val", seed * 1000)
    test_traces = generate_dataset(test_n, "test", seed * 1000)
    guard_traces = generate_dataset(guard_n, "guard", seed * 1000, no_change=True)
    unseen_traces = generate_dataset(unseen_n, "unseen", seed * 1000, unseen=True)
    demo = generate_trace(seed * 1000 + 987654, "demo", full=True)

    baseline = FixedDynamics(alpha=0.05)
    baseline.fit(train_traces)
    base_params = tune_cusum(baseline, train_traces, val_traces)

    model = LatentRuleModel(K=N_RULES, p_stay=0.98, alpha_dyn=0.05, alpha_rule=0.5)
    model.build_vocab(train_traces)
    model.initialize_from_baseline(
        baseline,
        train_traces,
        mu=base_params["mu"],
        sigma=base_params["sigma"],
        k=0.5,
        h=8.0,
    )
    likelihoods = model.em_fit(train_traces, iterations=em_iters)

    perm, val_align_acc = align_labels(val_traces, model)
    lat_params = tune_latent_detector(val_traces, model, perm)

    tracemalloc.start()
    t0 = time.perf_counter()
    _ = model.online_filter(demo)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    latency_ms = float(dt / max(1, len(demo.actions)) * 1000.0)
    memory_mb = float(peak / 1e6)

    metrics = evaluate_all(
        model=model,
        perm=perm,
        lat_params=lat_params,
        test_traces=test_traces,
        guard_traces=guard_traces,
        unseen_traces=unseen_traces,
        baseline=baseline,
        base_params=base_params,
    )

    _, demo_mp, demo_maxp = model.online_filter(demo)
    demo_aligned = np.array([perm[m] for m in demo_mp], dtype=int) if len(demo_mp) else np.array([], dtype=int)
    demo_flags, _ = detect_latent(demo_mp, demo_maxp, lat_params["prob"], lat_params["hold"])

    clip_bytes, clip_filename, clip_mime, clip_warning = generate_demo_clip(
        demo=demo,
        aligned_mp=demo_aligned,
        maxp=demo_maxp,
        flags=demo_flags,
        fps=15,
    )

    ablations = {
        "no_label_alignment_accuracy": float(metrics["latent_raw_accuracy_no_align"]),
        "gru_comparison": "Not run in CPU quick demo; reserved for optional secondary comparison.",
    }
    if run_ablation and train_n <= 120:
        model_ns = LatentRuleModel(K=N_RULES, p_stay=0.50, alpha_dyn=0.05, alpha_rule=0.5)
        model_ns.build_vocab(train_traces)
        model_ns.initialize_from_baseline(
            baseline,
            train_traces,
            mu=base_params["mu"],
            sigma=base_params["sigma"],
            k=0.5,
            h=8.0,
        )
        model_ns.em_fit(train_traces, iterations=max(3, em_iters // 2))
        perm_ns, _ = align_labels(val_traces, model_ns)
        params_ns = tune_latent_detector(val_traces, model_ns, perm_ns)
        ablations["no_sticky"] = simple_eval(model_ns, perm_ns, params_ns, test_traces)

    sample_validation_errors = 0
    for tr in (train_traces[:10] + val_traces[:10] + test_traces[:10]):
        sample_validation_errors += len(validate_trace(tr))

    fixtures = corrupted_fixtures()
    fixture_results = []
    for fx in fixtures:
        caught = validate_record(fx["record"])
        fixture_results.append({
            "fixture": fx["name"],
            "expected": ", ".join(fx["expected"]),
            "caught": ", ".join(caught) if caught else "NONE",
            "ok": set(caught) == set(fx["expected"]),
        })

    balance = {
        "train": rule_frequency(train_traces).tolist(),
        "val": rule_frequency(val_traces).tolist(),
        "test": rule_frequency(test_traces).tolist(),
        "guard": rule_frequency(guard_traces).tolist(),
    }

    manifest = {
        "project": "RuleWatch synthetic latent rule-switch monitor",
        "version": VERSION,
        "seed": seed,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit_hash": os.environ.get("GIT_COMMIT", "unknown"),
        "data_rights": "All traces generated by this project's code; released under an open licence (MIT recommended).",
        "splits": {
            "train": len(train_traces),
            "val": len(val_traces),
            "test": len(test_traces),
            "guard_no_change": len(guard_traces),
            "unseen_sixth_rule": len(unseen_traces),
        },
        "environment": {
            "grid": GRID,
            "agents": N_AGENTS,
            "objects": N_OBJECTS,
            "terrain_types": N_TERRAIN,
            "rule_sets": [r.name for r in RULE_LIBRARY],
            "unseen_rule": UNSEEN_RULESET.name,
        },
        "sample_validation_errors": sample_validation_errors,
    }

    hash_payload = {
        k: v
        for k, v in metrics.items()
        if k not in {"latent_delays", "all_cps", "all_flags", "worst_traces", "unseen_shares"}
    }
    metrics_hash = hashlib.sha256(json.dumps(hash_payload, sort_keys=True, default=str).encode()).hexdigest()[:16]

    runtime = float(time.time() - t_start)

    return {
        "metrics": metrics,
        "manifest": manifest,
        "metrics_hash": metrics_hash,
        "balance": balance,
        "fixtures": fixture_results,
        "rule_diffs": rule_aspect_diffs(),
        "likelihoods": likelihoods,
        "detector_params": lat_params,
        "baseline_params": base_params,
        "alignment_perm": list(perm),
        "validation_alignment_accuracy": float(val_align_acc),
        "ablations": ablations,
        "latency_ms_per_step": latency_ms,
        "memory_peak_mb": memory_mb,
        "runtime_sec": runtime,
        "demo": {
            "seed": demo.seed,
            "frames": demo.frames,
            "aligned_mp": demo_aligned,
            "maxp": demo_maxp,
            "flags": demo_flags,
            "change_points": demo.change_points,
        },
        "clip_bytes": clip_bytes,
        "clip_filename": clip_filename,
        "clip_mime": clip_mime,
        "clip_warning": clip_warning,
    }


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------

def sidebar_params():
    st.sidebar.title("RuleWatch controls")
    st.sidebar.markdown("Synthetic research prototype. No API keys. CPU-only.")

    mode = st.sidebar.selectbox("Mode", ["Quick demo", "Research-scale (slow)"], index=0)
    seed = int(st.sidebar.number_input("Base seed", value=7, step=1, format="%d"))

    if mode == "Quick demo":
        train_n = int(st.sidebar.number_input("Train traces", value=40, step=10, format="%d"))
        val_n = int(st.sidebar.number_input("Validation traces", value=20, step=5, format="%d"))
        test_n = int(st.sidebar.number_input("Test traces", value=30, step=5, format="%d"))
        guard_n = int(st.sidebar.number_input("No-change guardrail traces", value=20, step=5, format="%d"))
        unseen_n = int(st.sidebar.number_input("Unseen sixth-rule traces", value=20, step=5, format="%d"))
        em_iters = int(st.sidebar.number_input("EM iterations", value=6, step=1, format="%d"))
        run_ablation = st.sidebar.checkbox("Run small ablations", value=True)
    else:
        st.sidebar.warning("Research-scale may take several minutes or longer on a laptop CPU.")
        train_n = int(st.sidebar.number_input("Train traces", value=1000, step=50, format="%d"))
        val_n = int(st.sidebar.number_input("Validation traces", value=150, step=10, format="%d"))
        test_n = int(st.sidebar.number_input("Test traces", value=200, step=10, format="%d"))
        guard_n = int(st.sidebar.number_input("No-change guardrail traces", value=50, step=10, format="%d"))
        unseen_n = int(st.sidebar.number_input("Unseen sixth-rule traces", value=50, step=10, format="%d"))
        em_iters = int(st.sidebar.number_input("EM iterations", value=50, step=5, format="%d"))
        run_ablation = st.sidebar.checkbox("Run small ablations", value=False)

    rebuild = st.sidebar.button("Rebuild pipeline")
    reset_demo = st.sidebar.button("Restart live demo")

    if reset_demo:
        st.session_state["auto_play_done"] = False
        st.session_state["live_step"] = 0
        rerun_app()

    return {
        "seed": seed,
        "train_n": train_n,
        "val_n": val_n,
        "test_n": test_n,
        "guard_n": guard_n,
        "unseen_n": unseen_n,
        "em_iters": em_iters,
        "run_ablation": run_ablation,
        "rebuild": rebuild,
    }


def render_demo_tab(pipe: Dict[str, Any]):
    st.subheader("Working demo")
    st.caption("The environment is simulated. Ground-truth labels are shown only for evaluation and are not provided to the learned model.")

    clip_bytes = pipe.get("clip_bytes")
    if clip_bytes:
        st.success("Generated demo clip is ready below.")
        render_clip_html(clip_bytes, pipe["clip_mime"])
        st.download_button(
            "Download demo clip",
            data=clip_bytes,
            file_name=pipe["clip_filename"],
            mime=pipe["clip_mime"],
        )
        if pipe.get("clip_warning"):
            st.info(
                "MP4 generation was unavailable or failed, so the app used an animated GIF fallback. "
                f"Details: {pipe['clip_warning']}"
            )
    else:
        st.error(f"Demo clip generation failed: {pipe.get('clip_warning', 'unknown error')}")

    demo = pipe["demo"]
    frames = demo["frames"]
    aligned_mp = demo["aligned_mp"]
    maxp = demo["maxp"]
    flags = demo["flags"]
    cps = demo["change_points"]

    st.subheader("Live step-through")
    if not st.session_state.get("auto_play_done", False):
        img_slot = st.empty()
        cap_slot = st.empty()
        for step in range(len(frames)):
            im = draw_frame(frames[step], step, aligned_mp, maxp, cps, flags)
            img_slot.image(im, use_container_width=True)
            cap_slot.caption(f"Auto-playing step {step + 1}/{len(frames)}")
            time.sleep(0.012)
        st.session_state["auto_play_done"] = True
        st.session_state["live_step"] = max(0, len(frames) - 1)
        cap_slot.empty()

    st.markdown("### Interactive controls")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("Reset"):
        st.session_state["live_step"] = 0
        rerun_app()
    if c2.button("Step +1"):
        st.session_state["live_step"] = min(len(frames) - 1, int(st.session_state.get("live_step", 0)) + 1)
        rerun_app()
    if c3.button("Step +10"):
        st.session_state["live_step"] = min(len(frames) - 1, int(st.session_state.get("live_step", 0)) + 10)
        rerun_app()

    default_step = int(st.session_state.get("live_step", max(0, len(frames) - 1)))
    step = c4.slider("Jump to step", min_value=0, max_value=max(0, len(frames) - 1), value=default_step)
    st.session_state["live_step"] = step

    im = draw_frame(frames[step], step, aligned_mp, maxp, cps, flags)
    st.image(im, use_container_width=True)


def render_results_tab(pipe: Dict[str, Any]):
    m = pipe["metrics"]
    st.subheader("Primary outcomes")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rule accuracy", f"{m['latent_rule_accuracy']:.3f}")
    c2.metric("Baseline trivial rule acc.", f"{m['baseline_trivial_rule_accuracy']:.3f}")
    c3.metric("Change-point F1", f"{m['latent_cp_f1']:.3f}")
    c4.metric("Median delay", f"{m['latent_median_delay']:.2f}")

    st.markdown("### Change-point detection")
    st.write(
        {
            "F1": m["latent_cp_f1"],
            "bootstrap 95% CI": [m["latent_ci_low"], m["latent_ci_high"]],
            "precision": m["latent_precision"],
            "recall": m["latent_recall"],
            "TP": m["latent_tp"],
            "FP": m["latent_fp"],
            "FN": m["latent_fn"],
            "baseline F1": m["baseline_cp_f1"],
            "baseline median delay": m["baseline_median_delay"],
        }
    )

    st.markdown("### Guardrails")
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("False alarms / 100 steps", f"{m['guard_false_alarms_per_100']:.2f}")
    g2.metric("Baseline false alarms / 100", f"{m['baseline_guard_false_alarms_per_100']:.2f}")
    g3.metric("Mean assignment entropy", f"{m['entropy_mean']:.2f}")
    g4.metric("Max rule share", f"{m['max_rule_share']:.2f}")

    st.markdown("### H4: per-rule transition-table recovery")
    h4_df = pd.DataFrame(
        [{"rule_set": f"R{r}", "top1_agreement": v} for r, v in m["h4_per_rule"].items()]
    )
    st.dataframe(h4_df, use_container_width=True)
    below = [f"R{r}" for r, v in m["h4_per_rule"].items() if v < 0.90]
    if below:
        st.warning("Rule sets below 90% top-1 empirical agreement: " + ", ".join(below))
    else:
        st.success("All rule sets meet the 90% top-1 empirical agreement target in this run.")

    st.markdown("### Sixth rule set / unfamiliar dynamics")
    st.write(
        {
            "max dominance on unseen traces": m["unseen_dominance"],
            "unseen flags per 100 steps": m["unseen_flags_per_100"],
            "unseen mean entropy": m["unseen_entropy_mean"],
            "unseen assignment shares": m["unseen_shares"],
        }
    )

    st.markdown("### Latency and memory")
    st.write(
        {
            "online latency ms/step": pipe["latency_ms_per_step"],
            "inference peak memory MB": pipe["memory_peak_mb"],
            "pipeline runtime sec": pipe["runtime_sec"],
        }
    )

    st.markdown("### Error analysis")
    worst = pd.DataFrame(m["worst_traces"])
    st.dataframe(worst, use_container_width=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    delays = m["latent_delays"]
    if delays:
        axes[0].hist(delays, bins=15, color="#1f77b4")
        axes[0].axvline(0, color="red", linestyle="--")
        axes[0].set_title("Detection delay distribution")
        axes[0].set_xlabel("flag step - true change step")
    else:
        axes[0].text(0.5, 0.5, "No detections", ha="center", va="center")
        axes[0].set_axis_off()

    if m["all_cps"] or m["all_flags"]:
        if m["all_cps"]:
            axes[1].scatter(m["all_cps"], [0] * len(m["all_cps"]), color="lightgray", label="true CP")
        if m["all_flags"]:
            axes[1].scatter(m["all_flags"], [1] * len(m["all_flags"]), color="#d62728", s=18, label="flags")
        axes[1].set_xlim(0, TRACE_LEN)
        axes[1].set_ylim(-0.5, 1.5)
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["true CP", "flags"])
        axes[1].set_title("Flags and true change points")
        axes[1].set_xlabel("step")
        axes[1].legend(fontsize=7)
    else:
        axes[1].text(0.5, 0.5, "No events", ha="center", va="center")
        axes[1].set_axis_off()

    rules = list(m["h4_per_rule"].keys())
    vals = [m["h4_per_rule"][r] for r in rules]
    axes[2].bar([f"R{r}" for r in rules], vals, color="#2ca02c")
    axes[2].axhline(0.90, color="red", linestyle="--", label="0.90 target")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("H4 top-1 table agreement")
    axes[2].legend(fontsize=7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("### Failure-mode report")
    st.json(m["failure"])
    st.caption("Failure modes are reported, not suppressed. Flags are not removed to improve scores.")


def render_hypotheses_tab(pipe: Dict[str, Any]):
    m = pipe["metrics"]
    st.subheader("Hypothesis outcomes")

    h1_margin = m["latent_rule_accuracy"] - m["baseline_trivial_rule_accuracy"]
    h1_status = "Preliminary supported" if h1_margin >= 0.10 else "Not supported (preliminary)"
    h1_status += " — single seed; 4/5-seed acceptance requires multi-seed run"

    h2_status = "Supported" if m["latent_cp_f1"] >= 0.70 else "Refuted"
    h3_status = "Supported" if (not math.isnan(m["latent_median_delay"]) and m["latent_median_delay"] <= 4.0) else "Refuted or inconclusive"
    below = [f"R{r}" for r, v in m["h4_per_rule"].items() if v < 0.90]
    h4_status = "Supported" if not below else "Refuted"

    rows = [
        {
            "hypothesis": "H1 rule identification",
            "metric": f"latent={m['latent_rule_accuracy']:.3f}, baseline={m['baseline_trivial_rule_accuracy']:.3f}, margin={h1_margin:.3f}",
            "condition": ">= baseline + 0.10 on 4 of 5 seeds",
            "status": h1_status,
        },
        {
            "hypothesis": "H2 change-point detection",
            "metric": f"F1={m['latent_cp_f1']:.3f}, CI=[{m['latent_ci_low']:.3f}, {m['latent_ci_high']:.3f}]",
            "condition": "F1 >= 0.70, threshold chosen on validation only",
            "status": h2_status,
        },
        {
            "hypothesis": "H3 detection delay",
            "metric": f"median delay={m['latent_median_delay']:.2f}",
            "condition": "median delay <= 4 steps",
            "status": h3_status,
        },
        {
            "hypothesis": "H4 rule recovery",
            "metric": "per-rule top-1 agreement: " + ", ".join(f"R{r}={v:.2f}" for r, v in m["h4_per_rule"].items()),
            "condition": "each rule set >= 0.90",
            "status": h4_status + ("; below: " + ", ".join(below) if below else ""),
        },
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown("### Ablations and negative results")
    st.json(pipe["ablations"])
    st.caption("The baseline was tuned under the same validation budget. Negative results are shown in the same tables.")


def render_data_tab(pipe: Dict[str, Any]):
    st.subheader("Data, splits, and validation")

    st.markdown("### Rule-step frequency by split")
    bal = pipe["balance"]
    df = pd.DataFrame(
        {
            "split": list(bal.keys()),
            "R1": [bal[s][0] for s in bal],
            "R2": [bal[s][1] for s in bal],
            "R3": [bal[s][2] for s in bal],
            "R4": [bal[s][3] for s in bal],
            "R5": [bal[s][4] for s in bal],
        }
    )
    st.dataframe(df, use_container_width=True)
    st.caption("Generated traces are synthetic. Rule labels are withheld from the learned model and used only by the evaluator.")

    st.markdown("### Rule-library distinguishability check")
    st.dataframe(pd.DataFrame(pipe["rule_diffs"]), use_container_width=True)

    st.markdown("### Validator corrupted fixtures")
    fx = pd.DataFrame(pipe["fixtures"])
    st.dataframe(fx, use_container_width=True)

    st.markdown("### Data manifest")
    st.json(pipe["manifest"])
    st.download_button(
        "Download data manifest JSON",
        data=json.dumps(pipe["manifest"], indent=2).encode(),
        file_name="data_manifest.json",
        mime="application/json",
    )


def render_docs_tab():
    st.subheader("One-page research brief")
    st.markdown(
        """
        **Problem.** A learned model of a discrete grid-world can become stale when hidden transition rules change.
        We ask whether a latent-variable switching model, trained only on unlabelled interaction traces, can recover
        the active rule set and detect unannounced rule changes more accurately and with less delay than a fixed-dynamics
        baseline with a CUSUM detector on prediction error.

        **Design.** The environment is a 12x12 synthetic grid with up to three agents, four terrain types, pushable
        objects, and five designed rule sets. Traces contain 2--4 unannounced change points. The model sees only
        observations and actions, not rule labels. The latent model uses K=5 sticky switching dynamics with per-rule
        transition tables and exact discrete filtering/EM. The baseline uses one fixed transition table and CUSUM.

        **Decision improved.** Whether to keep using a learned environment model or trigger retraining after a detected
        change. Missed changes leave a stale model in use; false alarms waste retraining effort. Both costs are reported.

        **Limitation.** The rule library is a designed test set. Results do not show that the method finds real-world
        regime changes.
        """
    )

    st.subheader("Glossary")
    st.markdown(
        """
        - **Rule set:** A named collection of movement costs, push success probabilities, and terrain transition effects.
        - **Latent state / latent rule:** The hidden rule identity at a step, inferred from observations and actions.
        - **Change point:** The step at which the environment switches from one rule set to another.
        - **Detection delay:** The number of steps between the true change point and the monitor flag.
        - **Identifiability:** Whether the model can assign distinct latent rules to distinct true rule sets rather than
          collapsing to one rule or splitting one true rule into several arbitrary labels.
        """
    )

    st.subheader("User stories")
    st.markdown(
        """
        - As a researcher, I want to see whether an unsupervised latent model notices that the world has changed.
        - As an engineer, I want a monitor that flags when a learned environment model may no longer be valid.
        - As a reviewer, I want false alarms and misses reported separately, not hidden behind one accuracy number.
        - As a maintainer, I want deterministic synthetic data generation, tests, and a one-command Streamlit demo.
        """
    )

    st.subheader("Risk register")
    risk = pd.DataFrame(
        [
            {
                "risk": "Misleading claim",
                "description": "Users may interpret synthetic rule-change detection as evidence for real-world regime-shift detection.",
                "mitigation": "Every screen states the environment is simulated. The report prohibits operational use without separate validation.",
            },
            {
                "risk": "Label permutation illusion",
                "description": "Latent labels are arbitrary and can appear accurate only after alignment.",
                "mitigation": "Alignment is frozen on validation data and raw unaligned accuracy is reported.",
            },
            {
                "risk": "False confidence from small synthetic library",
                "description": "Five designed rule sets are easier to separate than real dynamics.",
                "mitigation": "Report per-rule recovery, unseen sixth-rule behaviour, entropy guardrail, and failure modes.",
            },
            {
                "risk": "Detector threshold overfitting",
                "description": "Tuning on test data would inflate F1.",
                "mitigation": "Threshold and hold length are chosen on validation data only.",
            },
        ]
    )
    st.dataframe(risk, use_container_width=True)
    st.warning(
        "Explicit limitation: the rule library is a designed test set. These results do not show that the method "
        "finds real-world regime changes."
    )


def render_repro_tab(pipe: Dict[str, Any]):
    st.subheader("Reproducibility")
    st.markdown(
        f"""
        - Base seed: **{pipe['manifest']['seed']}**
        - Generator version: **{pipe['manifest']['version']}**
        - Commit hash recorded in manifest: **{pipe['manifest']['commit_hash']}**
        - Metrics hash: **{pipe['metrics_hash']}**
        - Runtime: **{pipe['runtime_sec']:.2f} s**
        """
    )
    st.markdown("One-command launch:")
    st.code("streamlit run app.py", language="bash")
    st.caption("No API keys are used. All data are generated locally by the app.")

    if pipe["likelihoods"]:
        st.markdown("### EM training likelihood")
        st.line_chart(pd.DataFrame({"log_likelihood": pipe["likelihoods"]}))
    else:
        st.info("No EM likelihood trace available.")

    st.markdown("### Component contracts (summary)")
    st.json(
        {
            "simulator": {
                "inputs": "seed, rule index, action",
                "outputs": "next true state",
                "errors": "invalid action or out-of-grid attempt handled as failed move",
                "tests": "deterministic given seed; rule differences checked"
            },
            "trace_generator": {
                "inputs": "seed, split, change-point policy",
                "outputs": "observations, actions, hidden true labels for evaluator only",
                "errors": "validator catches range/gap/timestamp/action/off-grid issues",
                "tests": "change points respect range and minimum gap"
            },
            "baseline": {
                "inputs": "unlabelled transitions",
                "outputs": "fixed transition table, NLL sequence, CUSUM flags",
                "errors": "unseen state-action smoothed by Dirichlet prior",
                "tests": "flags one synthetic change within a sensible window"
            },
            "latent_model": {
                "inputs": "unlabelled transitions",
                "outputs": "per-rule transition tables, online rule posterior, flags",
                "errors": "collapse/split/absorption reported, not suppressed",
                "tests": "single-rule burn-in filter; EM likelihood monitored"
            },
            "evaluator": {
                "inputs": "test traces with hidden labels",
                "outputs": "H1-H4 metrics, guardrails, latency, memory, failure analysis",
                "errors": "missing detections counted as misses, not delayed hits",
                "tests": "bootstrap CI and validation-only threshold selection"
            },
        }
    )


def main():
    st.set_page_config(page_title="RuleWatch", layout="wide", initial_sidebar_state="expanded")

    params = sidebar_params()

    if "pipe" not in st.session_state or params["rebuild"]:
        with st.spinner("Generating synthetic traces, fitting baseline and latent switching model, rendering demo clip..."):
            st.session_state["pipe"] = run_pipeline(
                seed=params["seed"],
                train_n=params["train_n"],
                val_n=params["val_n"],
                test_n=params["test_n"],
                guard_n=params["guard_n"],
                unseen_n=params["unseen_n"],
                em_iters=params["em_iters"],
                run_ablation=params["run_ablation"],
            )
        st.session_state["auto_play_done"] = False
        st.session_state["live_step"] = 0

    pipe = st.session_state["pipe"]

    st.title("RuleWatch: latent rule-switch monitor for a synthetic grid world")
    st.caption(
        "CPU-only, open-source, no API keys. The environment is simulated. Results do not demonstrate real-world "
        "regime-change detection."
    )

    tabs = st.tabs(["Live demo", "Results", "Hypotheses", "Data & validation", "Brief & risks", "Reproducibility"])
    with tabs[0]:
        render_demo_tab(pipe)
    with tabs[1]:
        render_results_tab(pipe)
    with tabs[2]:
        render_hypotheses_tab(pipe)
    with tabs[3]:
        render_data_tab(pipe)
    with tabs[4]:
        render_docs_tab()
    with tabs[5]:
        render_repro_tab(pipe)


if __name__ == "__main__":
    main()