"""The scene: where each machine stands on the site, so a motion's tool tip can be traced in site coordinates.

The site frame is x forward, y left, z up (the floor is z = 0). A placed machine's base sits at
``(x, y, 0)`` turned by ``yaw`` radians counter-clockwise about +z. A motion names the machine it
drives by id (``Motion.machine``), which is the key in ``Scene.machines``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .machine import Machine, load_machine
from .motion import Motion, load_record, save_record
from .site import Site

FORMAT = "physical-sandbox.scene/1"


@dataclass
class PlacedMachine:
    machine: Machine
    urdf: str  # the file the machine was loaded from (kept so the scene can be saved)
    x: float
    y: float
    yaw: float  # radians, counter-clockwise about +z
    confirmed_by_person: bool = False

    def base_transform(self) -> np.ndarray:
        """(4, 4) transform from the machine's base frame to the site frame."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        T = np.eye(4)
        T[:2, :2] = [[c, -s], [s, c]]
        T[:3, 3] = [self.x, self.y, 0.0]
        return T


@dataclass
class Scene:
    site: Site
    machines: dict[str, PlacedMachine]
    motions: list[Motion] = field(default_factory=list)

    def _placed(self, motion: Motion) -> PlacedMachine:
        placed = self.machines.get(motion.machine)
        if placed is None:
            raise ValueError(f"motion drives machine {motion.machine!r}, which is not placed in the scene "
                             f"(placed: {sorted(self.machines)})")
        if list(motion.joint_names) != list(placed.machine.joint_names):
            raise ValueError(f"motion joints {list(motion.joint_names)} differ from the machine "
                             f"{motion.machine!r} joints {list(placed.machine.joint_names)}")
        return placed

    def tool_path(self, motion: Motion) -> np.ndarray:
        """Tool tip (N, 3) in site coordinates for every sample of ``motion``."""
        placed = self._placed(motion)
        T = placed.base_transform()
        tips = np.array([placed.machine.fk(q) for q in motion.q]).reshape(-1, 3)
        return tips @ T[:3, :3].T + T[:3, 3]

    def link_poses(self, motion: Motion, i: int) -> dict[str, np.ndarray]:
        """World transform (4, 4) of every link at sample ``i`` of ``motion``, in site coordinates."""
        placed = self._placed(motion)
        T = placed.base_transform()
        return {name: T @ P for name, P in placed.machine.link_poses(motion.q[i]).items()}


def save_scene(scene: Scene, path) -> None:
    """Write the scene as JSON; its motions go to a recording ``<name>.record.json`` beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if scene.motions:
        rec = path.with_name(f"{path.stem}.record.json")
        save_record(scene.motions, scene.site.id, rec)
        records.append(rec.name)
    data = {
        "format": FORMAT,
        "site": scene.site.id,
        "machines": {mid: {"urdf": p.urdf, "x": p.x, "y": p.y, "yaw": p.yaw,
                           "confirmedByPerson": p.confirmed_by_person}
                     for mid, p in scene.machines.items()},
        "records": records,
    }
    path.write_text(json.dumps(data, indent=1) + "\n")


def load_scene(path, site: Site) -> Scene:
    """Read a scene file for ``site``; refuses a scene made for another site."""
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("format") != FORMAT:
        raise ValueError(f"{path.name}: format is {data.get('format')!r}, expected {FORMAT!r}")
    if data.get("site") != site.id:
        raise ValueError(f"{path.name} is a scene of site {data.get('site')!r}, not {site.id!r}")
    machines = {}
    for mid, m in (data.get("machines") or {}).items():
        urdf = Path(m["urdf"])
        if not urdf.is_absolute() and not urdf.exists() and (path.parent / urdf).exists():
            urdf = path.parent / urdf
        machines[mid] = PlacedMachine(
            machine=load_machine(urdf), urdf=m["urdf"], x=float(m["x"]), y=float(m["y"]),
            yaw=float(m["yaw"]), confirmed_by_person=bool(m.get("confirmedByPerson", False)))
    motions = []
    for rel in data.get("records") or []:
        motions.extend(load_record(path.parent / rel))
    return Scene(site=site, machines=machines, motions=motions)
