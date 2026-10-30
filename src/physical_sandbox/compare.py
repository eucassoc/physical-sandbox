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
    jerk: dict  # {a, b}: RMS of the tool tip's third derivative, m/s^3
    grasps: list  # [{a_s, b_s, tool_gap_m}]
    grasp_counts: dict  # {a, b}: grasps found in each motion (unpaired ones are not in ``grasps``)
    series: dict  # {t, tool_gap_m} for plotting
    unknown: list = field(default_factory=lambda: list(_UNKNOWN))
    unknown_reasons: dict = field(default_factory=lambda: {k: _STAGE3 for k in _UNKNOWN})

    def to_dict(self) -> dict:
        """Plain JSON types only."""
        return {
            "labels": dict(self.labels),
            "step_s": float(self.step_s),
            "tool_deviation_m": {k: float(v) for k, v in self.tool_deviation_m.items()},
            "shape_deviation_m": float(self.shape_deviation_m),
            "joints": {n: {k: float(v) for k, v in d.items()} for n, d in self.joints.items()},
            "duration_s": {k: float(v) for k, v in self.duration_s.items()},
            "jerk": {k: float(v) for k, v in self.jerk.items()},
            "grasps": [{k: float(v) for k, v in g.items()} for g in self.grasps],
            "grasp_counts": {k: int(v) for k, v in self.grasp_counts.items()},
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


def _dtw_mean(pa: np.ndarray, pb: np.ndarray) -> float:
    """Total Euclidean cost along the optimal DTW path divided by the number of matched pairs."""
    n, m = len(pa), len(pb)
    cost = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2))
    acc = np.full((n + 1, m + 1), np.inf)
    length = np.zeros((n + 1, m + 1), int)
    acc[0, 0] = 0.0
    for k in range(2, n + m + 1):  # anti-diagonals: every cell depends only on the two before it
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        cand = np.stack([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        lens = np.stack([length[i - 1, j - 1], length[i - 1, j], length[i, j - 1]])
        pick = cand.argmin(axis=0)
        cols = np.arange(len(i))
        acc[i, j] = cost[i - 1, j - 1] + cand[pick, cols]
        length[i, j] = lens[pick, cols] + 1
    return float(acc[n, m] / length[n, m])


def _jerk(tips: np.ndarray, step_s: float) -> float:
    if len(tips) < 4:
        return 0.0
    d3 = np.diff(tips, n=3, axis=0) / step_s ** 3
    return float(np.sqrt((d3 ** 2).sum(axis=1).mean()))


def _closings(m: Motion, joint: int, threshold: float) -> list:
    """Times at which the joint falls through ``threshold`` (decreasing value), linear between samples."""
    v, t = m.q[:, joint], m.t
    out = []
    for i in range(len(v) - 1):
        if v[i] > threshold >= v[i + 1]:
            out.append(float(t[i] + (t[i + 1] - t[i]) * (v[i] - threshold) / (v[i] - v[i + 1])))
    return out


def compare(scene: Scene, a: Motion, b: Motion, *, step_s: float = 1 / 30,
            grip_threshold: float | None = None) -> Comparison:
    """Numbers for the gaps between motions ``a`` and ``b`` of the same machine placed in ``scene``."""
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
    grasps, counts = [], {"a": 0, "b": 0}
    if "gripper" in names:
        gj = names.index("gripper")
        if grip_threshold is None:
            lo, hi = scene.machines[a.machine].machine.limits[gj]
            grip_threshold = float((lo + hi) / 2)
        ca, cb = _closings(a, gj, grip_threshold), _closings(b, gj, grip_threshold)
        counts = {"a": len(ca), "b": len(cb)}
        for ta, tb in zip(ca, cb):
            gap_g = np.linalg.norm(_tips(scene, a, [ta])[0] - _tips(scene, b, [tb])[0])
            grasps.append({"a_s": ta, "b_s": tb, "tool_gap_m": float(gap_g)})
    else:
        unknown.append("grasps")
        reasons["grasps"] = "グリッパーの関節（gripper）がありません"  # the machine has no joint named gripper

    return Comparison(
        labels={"a": a.label, "b": b.label}, step_s=float(step_s), tool_deviation_m=tool_dev,
        shape_deviation_m=shape, joints=joints,
        duration_s={"a": dur_a, "b": dur_b, "diff": dur_b - dur_a},
        jerk=jerk, grasps=grasps, grasp_counts=counts,
        series={"t": [float(x) for x in t], "tool_gap_m": [float(x) for x in gap]},
        unknown=unknown, unknown_reasons=reasons)
