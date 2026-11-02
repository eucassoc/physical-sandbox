"""The scene: where each machine stands on the site, so a motion's tool tip can be traced in site coordinates.

The site frame is x forward, y left, z up (the floor is z = 0). A placed machine's base sits at
``(x, y, 0)`` turned by ``yaw`` radians counter-clockwise about +z. A motion names the machine it
drives by id (``Motion.machine``), which is the key in ``Scene.machines``.
"""
from __future__ import annotations

import json
import math
import os
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
    urdf: str  # the file the machine was loaded from (absolute after load_scene; resolved relative to the cwd on save)
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


def _urdf_for_file(urdf: str, folder: Path) -> str:
    """The urdf path as stored: relative to the scene folder, absolute only if that is impossible."""
    try:
        return os.path.relpath(Path(urdf).expanduser().resolve(), folder.resolve())
    except ValueError:  # e.g. another drive
        return str(Path(urdf).expanduser().resolve())


def _lists_record(scene_path: Path, name: str) -> bool:
    try:
        return name in (json.loads(scene_path.read_text()).get("records") or [])
    except (OSError, ValueError, AttributeError):
        return False


def save_scene(scene: Scene, path) -> None:
    """Write the scene as JSON; its motions go to a recording ``<name>.record.json`` beside it.

    The machines' urdf paths are stored relative to the scene folder, so the folder can be moved or shared.
    An existing ``<name>.record.json`` is overwritten only when the scene file already at ``path`` lists it;
    otherwise ``FileExistsError``. Saving a scene without motions removes the record the previous save of
    this scene listed. Both files are written to temporary names first, then renamed (the
    record, then the scene), so a failure does not leave a half-written pair.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = path.with_name(f"{path.stem}.record.json")
    owned = _lists_record(path, rec.name)
    if scene.motions and rec.exists() and not owned:
        raise FileExistsError(f"{rec} already exists and is not this scene's record; "
                              f"choose another scene name or remove it")
    data = {
        "format": FORMAT,
        "site": scene.site.id,
        "machines": {mid: {"urdf": _urdf_for_file(p.urdf, path.parent), "x": p.x, "y": p.y, "yaw": p.yaw,
                           "confirmedByPerson": p.confirmed_by_person}
                     for mid, p in scene.machines.items()},
        "records": [rec.name] if scene.motions else [],
    }
    tmp_scene = path.with_name(f".{path.name}.tmp")
    tmp_rec = path.with_name(f".{rec.name}.tmp")
    try:
        if scene.motions:
            save_record(scene.motions, scene.site.id, tmp_rec)
        tmp_scene.write_text(json.dumps(data, indent=1) + "\n")
        if scene.motions:
            os.replace(tmp_rec, rec)
        os.replace(tmp_scene, path)
        if not scene.motions and owned:
            rec.unlink(missing_ok=True)  # the scene no longer lists it, and it was this scene's own
    finally:
        tmp_rec.unlink(missing_ok=True)
        tmp_scene.unlink(missing_ok=True)


def save_placement(path, site_id: str, machines: dict[str, PlacedMachine], *, picture: str | None) -> None:
    """Write the scene file for the machines' placement only (the stage's 決定): the same ``format``, ``site`` and
    ``machines`` as ``save_scene``, plus the ``picture`` it was placed on. The ``records`` list of a scene file
    already at ``path`` is kept as it is (``[]`` when there is none), and no record file is touched, so placing
    a machine again never loses the motions that were saved for the scene."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        kept = json.loads(path.read_text()).get("records")
    except (OSError, ValueError, AttributeError):
        kept = None
    data = {
        "format": FORMAT,
        "site": site_id,
        "machines": {mid: {"urdf": _urdf_for_file(p.urdf, path.parent), "x": p.x, "y": p.y, "yaw": p.yaw,
                           "confirmedByPerson": p.confirmed_by_person}
                     for mid, p in machines.items()},
        "records": kept if isinstance(kept, list) else [],
    }
    if picture:
        data["picture"] = picture
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=1) + "\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _need(d, key: str, where: str):
    if not isinstance(d, dict) or key not in d:
        raise ValueError(f"{where}: missing {key!r}")
    return d[key]


def _resolve_urdf(mid: str, urdf: str, folder: Path) -> Path:
    given = Path(urdf).expanduser()
    candidates = [folder / given, given] if not given.is_absolute() else [given]
    for c in candidates:
        if c.is_file():
            return c
    raise ValueError(f"machine {mid!r}: urdf {urdf!r} not found (tried {[str(c) for c in candidates]})")


def load_scene(path, site: Site) -> Scene:
    """Read a scene file for ``site``; refuses a scene made for another site.

    A scene file is treated as untrusted: its records must be plain relative paths that stay inside the
    scene folder. A machine's urdf is looked up in the scene folder first, then as written, and may point
    outside the folder on purpose (robot folders are shared, e.g. ``../robots/arm.urdf``). The urdf and the
    meshes it names are read and copied into the cache, so a scene from an untrusted source can make this
    tool read the files its urdf points at (nothing is sent anywhere).
    """
    path = Path(path)
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise ValueError(f"{path.name}: format is {data.get('format') if isinstance(data, dict) else None!r}, "
                         f"expected {FORMAT!r}")
    if data.get("site") != site.id:
        raise ValueError(f"{path.name} is a scene of site {data.get('site')!r}, not {site.id!r}")
    folder = path.parent
    machines = {}
    for mid, m in (data.get("machines") or {}).items():
        where = f"{path.name}: machine {mid!r}"
        given, x, y, yaw = (_need(m, k, where) for k in ("urdf", "x", "y", "yaw"))
        urdf = _resolve_urdf(mid, str(given), folder).resolve()
        machines[mid] = PlacedMachine(
            machine=load_machine(urdf), urdf=str(urdf), x=float(x), y=float(y), yaw=float(yaw),
            confirmed_by_person=bool(m.get("confirmedByPerson", False)))
    motions = []
    root = folder.resolve()
    for rel in data.get("records") or []:
        if not isinstance(rel, str) or Path(rel).is_absolute():
            raise ValueError(f"{path.name}: record {rel!r} must be a relative path")
        target = (folder / rel).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"{path.name}: record {rel!r} is outside the scene folder")
        if not target.is_file():
            raise ValueError(f"{path.name}: record {rel!r} not found at {target}")
        motions.extend(load_record(target))
    return Scene(site=site, machines=machines, motions=motions)
