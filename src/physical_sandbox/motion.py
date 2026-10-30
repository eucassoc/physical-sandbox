"""A motion: one machine's joint angles over time, from numpy or a LeRobot dataset, saved as a recording.

A recording is written in the 0.4 vocabulary (recording schemaVersion 3): each motion is a device that
names its product (``product`` = the motion's machine) and a ``traces`` entry for that device whose
samples carry ``t`` (seconds from the first sample) and ``joints`` (joint name -> radians).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PRODUCT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")  # device.schema.json's pattern for ``product``


@dataclass(frozen=True)
class Motion:
    """``t`` is seconds, rising, starting at 0 (the motion's first sample); ``q`` is (N, J) radians
    in ``joint_names`` order."""

    machine: str
    joint_names: list
    t: np.ndarray
    q: np.ndarray
    source: str = ""
    label: str = ""

    def at(self, t: float) -> np.ndarray:
        """Joint angles (J,) at time ``t``: linear between samples, the first/last held outside."""
        t = float(t)
        return np.array([np.interp(t, self.t, self.q[:, j]) for j in range(self.q.shape[1])])


def motion_from_numpy(q, *, fps: float | None = None, t=None, joint_names, machine: str,
                      label: str = "") -> Motion:
    """A motion from an (N, J) array of radians; give exactly one of ``fps`` and ``t`` (seconds)."""
    if (fps is None) == (t is None):
        raise ValueError("give exactly one of fps and t")
    q = np.array(q, float)
    joint_names = list(joint_names)
    if q.ndim != 2 or q.shape[0] < 1 or q.shape[1] != len(joint_names):
        raise ValueError(f"q must be (N, {len(joint_names)}) for the joints {joint_names}, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must be finite")
    if fps is not None:
        if not fps > 0:
            raise ValueError(f"fps must be positive, got {fps}")
        t = np.arange(q.shape[0]) / float(fps)
    else:
        t = np.array(t, float)
        if t.shape != (q.shape[0],):
            raise ValueError(f"t must have one time per row of q ({q.shape[0]}), got {t.shape[0] if t.ndim else 'a scalar'}")
        if not np.all(np.isfinite(t)):
            raise ValueError("t must be finite")
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be rising")
        t = t - t[0]
    return Motion(machine=machine, joint_names=joint_names, t=t, q=q, source="numpy", label=label)


def motion_from_lerobot(dataset_dir, episode: int, *, machine, key: str = "observation.state",
                        label: str | None = None, machine_name: str = "so101") -> Motion:
    """One episode of a LeRobot v3 dataset as a motion in radians (see ``lerobot.JOINT_MAP``)."""
    from .lerobot import read_episode

    return read_episode(dataset_dir, episode, machine=machine, key=key, label=label,
                        machine_name=machine_name)


def _device_ids(motions) -> list:
    seen: dict = {}
    ids = []
    for m in motions:
        seen[m.machine] = seen.get(m.machine, 0) + 1
        ids.append(f"{m.machine}-{seen[m.machine]}")
    return ids


def save_record(motions, site_id: str, path) -> None:
    """Write the motions as a 0.4 recording (JSON) on the site frame ``site_id``.

    The site frame's geometry lives in ``sites/<site_id>.json``, not here; the two markers are written
    as the frame's own origin and unit point along +x.
    """
    motions = list(motions)
    if not motions:
        raise ValueError("no motions to save")
    if not site_id:
        raise ValueError("site_id is empty")
    for m in motions:
        if not _PRODUCT.match(m.machine):
            raise ValueError(f"machine {m.machine!r} must be lowercase letters, digits, - or _ (it names the product)")
    ids = _device_ids(motions)
    devices, traces = [], []
    for did, m in zip(ids, motions):
        dev = {"id": did, "product": m.machine, "owner": "external", "parent": "site"}
        if m.label:
            dev["displayName"] = m.label
        if m.source:
            dev["_source"] = m.source
        devices.append(dev)
        traces.append({"deviceId": did, "samples": [
            {"t": float(ti), "joints": {n: float(v) for n, v in zip(m.joint_names, row)}}
            for ti, row in zip(m.t, m.q)]})
    record = {
        "schemaVersion": 3,
        "siteFrameId": site_id,
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": "physical-sandbox",
        "frame": {"frameId": "site", "up": "+Y", "unit": "m"},
        "markers": [{"id": "1", "x": 0.0, "y": 0.0, "z": 0.0}, {"id": "2", "x": 1.0, "y": 0.0, "z": 0.0}],
        "equipment": {"site": {"id": site_id}, "devices": devices},
        "names": {d: d for d in ids},
        "traces": traces,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1) + "\n")


def load_record(path) -> list:
    """The motions in a recording: every device that names a product and has a trace of joints."""
    rec = json.loads(Path(path).read_text())
    devices = {d["id"]: d for d in ((rec.get("equipment") or {}).get("devices") or [])}
    motions = []
    for tr in rec.get("traces") or []:
        did = tr.get("deviceId")
        samples = tr.get("samples") or []
        if did not in devices or not samples or not all("joints" in s for s in samples):
            raise ValueError(f"trace of device {did!r} has no joint angles (or no such device in the equipment)")
        names = list(samples[0]["joints"])
        if any(list(s["joints"]) != names for s in samples):
            raise ValueError(f"trace of device {did!r} changes its joints from sample to sample")
        dev = devices[did]
        motions.append(Motion(
            machine=dev.get("product", ""), joint_names=names,
            t=np.array([s["t"] for s in samples], float),
            q=np.array([[s["joints"][n] for n in names] for s in samples], float),
            source=dev.get("_source", ""), label=dev.get("displayName", "") if isinstance(dev.get("displayName", ""), str) else ""))
    return motions
