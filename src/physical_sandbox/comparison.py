"""The gaps between two motions of one machine on a site: the numbers a researcher puts in a table.

Both motions are laid on a common clock (step ``step_s``, from 0 to the longer duration; the shorter
motion holds its last pose) for the same-time gaps. The tool-tip path shape is compared without time by
dynamic time warping (DTW) of the two paths, each sampled on its own clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .motion import Motion
from .scene import Scene

_STAGE3 = "奥行きと対象物は段 3"  # depth and target objects arrive in stage 3
_UNKNOWN = ["clearance", "reach", "world"]


@dataclass
class Comparison:
    labels: dict
    step_s: float
    tool_deviation_m: dict  # {max, mean, at_s}: distance between the tool tips at the same time
    shape_deviation_m: float  # mean tool-tip distance along the DTW path of the two tool paths
    joints: dict  # {name: {mean_rad, max_rad, at_s}}
    duration_s: dict  # {a, b, diff} with diff = b - a
    jerk: dict  # {a, b}: RMS of the tool tip's third derivative, m/s^3 (NaN under 4 samples; null in to_dict)
    grasps: list  # [{a_s, b_s, tool_gap_m}]
    grasp_counts: dict  # {a, b}: grasps found in each motion (unpaired ones are not in ``grasps``)
    series: dict  # {t, tool_gap_m} for plotting
    unknown: list = field(default_factory=lambda: list(_UNKNOWN))
    unknown_reasons: dict = field(default_factory=lambda: {k: _STAGE3 for k in _UNKNOWN})
    grip_threshold: float | None = None  # the gripper value a grasp closes through (None: no gripper joint, or
    # it hardly moved, so none was used; ``unknown_reasons["grasps"]`` says which)

    def to_dict(self) -> dict:
        """Plain JSON types only (a NaN jerk becomes null)."""
        return {
            "labels": dict(self.labels),
            "step_s": float(self.step_s),
            "tool_deviation_m": {k: float(v) for k, v in self.tool_deviation_m.items()},
            "shape_deviation_m": float(self.shape_deviation_m),
            "joints": {n: {k: float(v) for k, v in d.items()} for n, d in self.joints.items()},
            "duration_s": {k: float(v) for k, v in self.duration_s.items()},
            "jerk": {k: (float(v) if math.isfinite(v) else None) for k, v in self.jerk.items()},
            "grasps": [{k: float(v) for k, v in g.items()} for g in self.grasps],
            "grasp_counts": {k: int(v) for k, v in self.grasp_counts.items()},
            "grip_threshold": None if self.grip_threshold is None else float(self.grip_threshold),
            "series": {"t": [float(x) for x in self.series["t"]],
                       "tool_gap_m": [float(x) for x in self.series["tool_gap_m"]]},
            "unknown": list(self.unknown),
            "unknown_reasons": dict(self.unknown_reasons),
        }


def _duration(m: Motion) -> float:
    return float(m.t[-1] - m.t[0])


def _grid(duration: float, step_s: float, *, cover: bool) -> np.ndarray:
    """Times 0, step, 2 step, ...; ``cover`` makes the last one reach ``duration`` (else it stays within)."""
    n = (math.ceil(duration / step_s - 1e-9) if cover else math.floor(duration / step_s + 1e-9)) + 1
    return np.arange(max(n, 1)) * step_s


def _sampled(m: Motion, times: np.ndarray) -> Motion:
    q = np.array([m.at(t) for t in times]).reshape(len(times), -1)
    return Motion(machine=m.machine, joint_names=list(m.joint_names), t=np.asarray(times, float), q=q)


def _tips(scene: Scene, m: Motion, times) -> np.ndarray:
    """Tool tips (N, 3) in site coordinates of ``m`` at ``times`` (same transform as ``Scene.tool_path``)."""
    return scene.tool_path(_sampled(m, np.asarray(times, float)))


# DTW keeps an (N + 1) x (2 * band + 3) matrix of float64 costs and int32 lengths (12 bytes a cell).
# 30 million cells is about 360 MB: a 5 minute pair at 30 fps (9000 samples each, default band) needs 16 million.
_MAX_DTW_CELLS = 30_000_000


def _dtw_mean(pa: np.ndarray, pb: np.ndarray, band: int | None = None) -> float:
    """Total Euclidean cost along the optimal DTW path divided by the number of matched pairs.

    The path is kept inside a Sakoe-Chiba band ``|i - j| <= band`` (``band`` is raised to ``|N - M|`` so the
    end points can meet; the default is ``max(|N - M|, 10 % of max(N, M), 10)`` samples). A path that would
    leave the band (for example a long wait in the middle of one motion only) is not found, and the result is
    then at least the unbanded value. Memory is the band only, never N x M: when the band matrix would exceed
    ``_MAX_DTW_CELLS`` cells a ValueError names the sizes (use a larger ``step_s``).
    """
    n, m = len(pa), len(pb)
    if band is None:
        band = max(abs(n - m), math.ceil(0.1 * max(n, m)), 10)
    w = int(max(band, abs(n - m)))
    cells = (n + 1) * (2 * w + 3)
    if cells > _MAX_DTW_CELLS:
        raise ValueError(f"DTW of {n} x {m} samples with a band of {w} needs {cells} cells, over the limit "
                         f"{_MAX_DTW_CELLS}; compare with a larger step_s")
    acc = np.full((n + 1, 2 * w + 3), np.inf)
    length = np.zeros((n + 1, 2 * w + 3), np.int32)
    # cell (i, j) lives at acc[i, j - i + w + 1]; the two spare columns stay inf (outside the band)
    acc[0, w + 1] = 0.0
    for k in range(2, n + m + 1):  # anti-diagonals i + j = k: each cell depends only on the two before it
        lo = max(1, k - m, -((w - k) // 2))  # ceil((k - w) / 2)
        hi = min(n, k - 1, (k + w) // 2)
        if lo > hi:
            continue
        i = np.arange(lo, hi + 1)
        j = k - i
        c = j - i + w + 1
        cand = np.stack([acc[i - 1, c], acc[i - 1, c + 1], acc[i, c - 1]])
        lens = np.stack([length[i - 1, c], length[i - 1, c + 1], length[i, c - 1]])
        pick = cand.argmin(axis=0)
        cols = np.arange(len(i))
        acc[i, c] = np.linalg.norm(pa[i - 1] - pb[j - 1], axis=1) + cand[pick, cols]
        length[i, c] = lens[pick, cols] + 1
    end = m - n + w + 1
    return float(acc[n, end] / length[n, end])


def _jerk(tips: np.ndarray, step_s: float) -> float:
    """RMS of the third finite difference of the tool tip / step_s^3 (m/s^3); NaN under 4 samples."""
    if len(tips) < 4:
        return float("nan")
    d3 = np.diff(tips, n=3, axis=0) / step_s ** 3
    return float(np.sqrt((d3 ** 2).sum(axis=1).mean()))


def _closings(m: Motion, joint: int, threshold: float, band: float) -> list:
    """Times at which the joint closes through ``threshold``: it falls from above ``threshold + band`` to below
    ``threshold - band``, so a wobble about the threshold (sensor noise, a dithering servo) is not several
    closings. The time is that of its first fall through the threshold itself, linear between samples."""
    v, t = m.q[:, joint], m.t
    out, open_, crossed = [], False, None
    for i in range(len(v)):
        if v[i] > threshold + band:
            open_, crossed = True, None
            continue
        if not open_:
            continue
        if crossed is None and v[i] <= threshold < v[i - 1]:
            crossed = float(t[i - 1] + (t[i] - t[i - 1]) * (v[i - 1] - threshold) / (v[i - 1] - v[i]))
        if v[i] < threshold - band:
            out.append(float(t[i]) if crossed is None else crossed)
            open_, crossed = False, None
    return out


# A gripper whose values span less than this fraction of its joint's limits has not opened or closed: the
# middle of such a range is noise, and crossings of it are not grasps.
MIN_GRIP_FRACTION = 0.05
# The closing's hysteresis band, a fraction of the range the gripper went through (each side of the threshold).
GRIP_BAND_FRACTION = 0.1


def _gripper_span(scene: Scene, machine_id: str) -> float | None:
    """The width of the gripper joint's limits in the placed machine, or None when it is not known."""
    try:
        machine = scene.machines[machine_id].machine
        lo, hi = machine.limits[list(machine.joint_names).index("gripper")]
    except (KeyError, ValueError, AttributeError, IndexError):
        return None
    span = float(hi) - float(lo)
    return span if math.isfinite(span) and span > 0 else None


def _find_grasps(scene: Scene, motions: list, grip_threshold: float | None) -> tuple:
    """The grasp rule: ``(threshold, [closing times of each motion], note)`` for motions of one machine.

    A grasp is the gripper joint closing through the threshold. The default threshold is the middle of the range
    the gripper actually went through in all the motions, not of its joint limits: a gripper that only ever opens
    a little (as in many teleoperated datasets) would never cross the latter. But a gripper that hardly moved at
    all has no such middle: its wobble is noise, and no grasp is counted. ``note`` is why nothing can be said
    about grasps (no gripper, it hardly moved, or none closed), else None.
    """
    names = list(motions[0].joint_names)
    none = [[] for _ in motions]
    if "gripper" not in names:
        return grip_threshold, none, "グリッパーの関節（gripper）がありません"  # the machine has no joint named gripper
    gj = names.index("gripper")
    both = np.concatenate([m.q[:, gj] for m in motions])
    lo, hi = float(both.min()), float(both.max())
    span = _gripper_span(scene, motions[0].machine)
    if grip_threshold is None and span is not None and hi - lo < MIN_GRIP_FRACTION * span:
        return None, none, (f"グリッパーがほとんど動いていません（開閉の幅 {hi - lo:.3g} rad、関節の範囲 "
                            f"{span:.3g} rad の {100 * (hi - lo) / span:.0f}%）")
    if grip_threshold is None:
        grip_threshold = (lo + hi) / 2
    elif not math.isfinite(grip_threshold):
        raise ValueError(f"grip_threshold must be a finite number, got {grip_threshold}")
    grip_threshold = float(grip_threshold)
    band = max(0.0, min(GRIP_BAND_FRACTION * (hi - lo), grip_threshold - lo, hi - grip_threshold))
    closings = [_closings(m, gj, grip_threshold, band) for m in motions]
    note = None
    if not any(closings):  # a bare 0 would read as a fact; say none was found, at which threshold
        note = f"つかむ瞬間が見つかりませんでした（しきい値 {grip_threshold:.3g}）"
    return grip_threshold, closings, note


def grasps_of(motion: Motion, scene: Scene, *, grip_threshold: float | None = None) -> dict:
    """The grasps in one motion by the comparison's rule: ``{"count", "times", "threshold", "note"}``.

    ``times`` are the seconds at which the gripper closes, ``threshold`` the gripper value used (None when no
    grasp could be looked for) and ``note`` says why none was (empty when the rule ran and found some).
    """
    threshold, (times,), note = _find_grasps(scene, [motion], grip_threshold)
    return {"count": len(times), "times": times, "threshold": threshold, "note": note or ""}


def compare(scene: Scene, a: Motion, b: Motion, *, step_s: float = 1 / 30,
            grip_threshold: float | None = None) -> Comparison:
    """Numbers for the gaps between motions ``a`` and ``b`` of the same machine placed in ``scene``.

    Everything is measured on the ``step_s`` grid (each motion linearly resampled), so jerk is the jerk of
    that resampled tool tip: a coarse ``step_s`` smooths it, and a motion with fewer than 4 grid samples
    has no jerk (NaN). The path shape uses a banded DTW (see ``_dtw_mean``), which raises ValueError for
    motions too long for the memory limit.
    """
    if not step_s > 0:
        raise ValueError(f"step_s must be positive, got {step_s}")
    if a.machine != b.machine:
        raise ValueError(f"the motions drive different machines: {a.machine!r} and {b.machine!r}")
    if list(a.joint_names) != list(b.joint_names):
        raise ValueError(f"the motions have different joints: {list(a.joint_names)} and {list(b.joint_names)}")
    names = list(a.joint_names)
    dur_a, dur_b = _duration(a), _duration(b)

    # same-time gaps on the common clock
    t = _grid(max(dur_a, dur_b), step_s, cover=True)
    gap = np.linalg.norm(_tips(scene, a, t) - _tips(scene, b, t), axis=1)
    k = int(gap.argmax())
    tool_dev = {"max": float(gap[k]), "mean": float(gap.mean()), "at_s": float(t[k])}
    dq = np.abs(np.array([a.at(x) - b.at(x) for x in t]).reshape(len(t), -1))
    joints = {}
    for j, name in enumerate(names):
        kj = int(dq[:, j].argmax())
        joints[name] = {"mean_rad": float(dq[:, j].mean()), "max_rad": float(dq[kj, j]), "at_s": float(t[kj])}

    # path shape without time, and smoothness, each on the motion's own clock
    shape = _dtw_mean(_tips(scene, a, _grid(dur_a, step_s, cover=True)),
                      _tips(scene, b, _grid(dur_b, step_s, cover=True)))
    jerk = {"a": _jerk(_tips(scene, a, _grid(dur_a, step_s, cover=False)), step_s),
            "b": _jerk(_tips(scene, b, _grid(dur_b, step_s, cover=False)), step_s)}

    # grasps: the gripper joint closing through the threshold, paired in order
    unknown, reasons = list(_UNKNOWN), {k_: _STAGE3 for k_ in _UNKNOWN}
    grip_threshold, (ca, cb), note = _find_grasps(scene, [a, b], grip_threshold)
    counts = {"a": len(ca), "b": len(cb)}
    grasps = []
    for ta, tb in zip(ca, cb):
        gap_g = np.linalg.norm(_tips(scene, a, [ta])[0] - _tips(scene, b, [tb])[0])
        grasps.append({"a_s": ta, "b_s": tb, "tool_gap_m": float(gap_g)})
    if note is not None:
        unknown.append("grasps")
        reasons["grasps"] = note

    return Comparison(
        labels={"a": a.label, "b": b.label}, step_s=float(step_s), tool_deviation_m=tool_dev,
        shape_deviation_m=shape, joints=joints,
        duration_s={"a": dur_a, "b": dur_b, "diff": dur_b - dur_a},
        jerk=jerk, grasps=grasps, grasp_counts=counts,
        series={"t": [float(x) for x in t], "tool_gap_m": [float(x) for x in gap]},
        unknown=unknown, unknown_reasons=reasons, grip_threshold=grip_threshold)
