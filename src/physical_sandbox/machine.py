"""A machine (robot) loaded from URDF into MuJoCo: joints, limits, forward kinematics, link poses.

The machine's base frame is the MuJoCo world. Mesh files the URDF points at are first copied
into the cache (``PS_CACHE_DIR``, default ``~/.cache/physical-sandbox``) with OBJ material lines
removed, because MuJoCo draws only some faces of OBJ files that keep ``mtllib``/``usemtl``.
"""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

_MOVABLE = ("revolute", "prismatic", "continuous")


def _cache_root() -> Path:
    return Path(os.environ.get("PS_CACHE_DIR", "~/.cache/physical-sandbox")).expanduser()


def _find_mesh(urdf_dir: Path, name: str) -> Path:
    """Resolve a URDF mesh reference: as written, with .obj for .stl, then under ``meshes/``."""
    ref = name
    if "://" in name:  # package://pkg/path -> path (the package folder is not known here)
        ref = name.split("://", 1)[1]
        ref = ref.split("/", 1)[1] if "/" in ref else ref
    rel = Path(ref)
    variants = [rel]
    if rel.suffix.lower() == ".stl":
        variants.append(rel.with_suffix(".obj"))
        variants.append(rel.with_suffix(".OBJ"))
    candidates = []
    for v in variants:
        candidates.append(v if v.is_absolute() else urdf_dir / v)
    for v in variants:
        if not v.is_absolute():
            candidates.append(urdf_dir / "meshes" / v)
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"mesh {name!r} not found next to {urdf_dir} (tried {[str(c) for c in candidates]})")


def _strip_mtl(data: bytes) -> bytes:
    keep = [ln for ln in data.splitlines(keepends=True)
            if not ln.lstrip().startswith((b"mtllib", b"usemtl"))]
    return b"".join(keep)


def _ascii_stl_to_binary(data: bytes) -> bytes:
    """MuJoCo reads binary STL only; rewrite an ASCII STL, leave a binary one untouched."""
    if len(data) >= 84 and len(data) == 84 + 50 * int.from_bytes(data[80:84], "little"):
        return data
    if not data.lstrip().startswith(b"solid"):
        return data
    verts = []
    for ln in data.decode("ascii", "replace").splitlines():
        p = ln.split()
        if len(p) == 4 and p[0] == "vertex":
            verts.append([float(x) for x in p[1:]])
    tris = np.array(verts, "<f4").reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    rec = np.zeros(len(tris), [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tris
    return bytes(80) + len(tris).to_bytes(4, "little") + rec.tobytes()


def _prepare(urdf_path: Path) -> Path:
    """Return the rewritten URDF (mesh copies, MuJoCo compiler settings) in the cache.

    The folder is named by a hash of the URDF and of every mesh file it uses (path, mtime, size),
    so an edit to any of them gives a new folder. A complete folder is reused as it is; a new one
    is built beside it under a temporary name and renamed into place in one step, so another
    process (the MCP server, the CLI) never sees a half-written folder.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    st = urdf_path.stat()
    key = hashlib.sha1()
    key.update(f"{urdf_path}|{st.st_mtime_ns}|{st.st_size}".encode())
    srcs: dict[str, Path] = {}  # mesh reference as written -> resolved file
    for mesh in root.iter("mesh"):
        name = mesh.get("filename")
        if name and name not in srcs:
            srcs[name] = _find_mesh(urdf_path.parent, name).resolve()
    for src in dict.fromkeys(srcs.values()):
        s = src.stat()
        key.update(f"|{src}|{s.st_mtime_ns}|{s.st_size}".encode())

    base = _cache_root() / "meshes"
    final = base / key.hexdigest()
    if (final / urdf_path.name).is_file():
        return final / urdf_path.name

    base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=base))
    try:
        copies: dict[Path, str] = {}
        used: set[str] = set()  # MuJoCo names a mesh by its file stem, compared without case
        for mesh in root.iter("mesh"):
            name = mesh.get("filename")
            if not name:
                continue
            src = srcs[name]
            if src not in copies:
                stem, suffix = src.stem, src.suffix.lower()
                fname, n = f"{stem}{suffix}", 1
                while Path(fname).stem.lower() in used:
                    n += 1
                    fname = f"{stem}_{n}{suffix}"
                used.add(Path(fname).stem.lower())
                data = src.read_bytes()
                if suffix == ".obj":
                    data = _strip_mtl(data)
                elif suffix == ".stl":
                    data = _ascii_stl_to_binary(data)
                (tmp / fname).write_bytes(data)
                copies[src] = fname
            mesh.set("filename", copies[src])

        for old in root.findall("mujoco"):
            root.remove(old)
        mj = ET.Element("mujoco")
        ET.SubElement(mj, "compiler", meshdir=".", balanceinertia="true",
                      discardvisual="false", fusestatic="false")
        root.insert(0, mj)
        tree.write(tmp / urdf_path.name, encoding="utf-8", xml_declaration=True)
        try:
            os.replace(tmp, final)
        except OSError:
            # another process renamed its folder into place first: use theirs, drop ours
            if not (final / urdf_path.name).is_file():
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final / urdf_path.name


def _urdf_limits(urdf_path: Path) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        kind = j.get("type")
        if kind not in _MOVABLE:
            continue
        lim = j.find("limit")
        if kind == "continuous" or lim is None or lim.get("lower") is None or lim.get("upper") is None:
            span = math.pi if kind != "prismatic" else math.inf
            limits[j.get("name")] = (-span, span)
        else:
            limits[j.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
    return limits


@dataclass
class Machine:
    joint_names: list[str]
    limits: np.ndarray  # (J, 2), URDF joint limits in joint_names order
    model: mujoco.MjModel
    tool_link: str
    _data: mujoco.MjData = field(repr=False)
    _qadr: np.ndarray = field(repr=False)

    def _pose(self, q) -> None:
        q = np.asarray(q, float).reshape(-1)
        if q.shape[0] != len(self.joint_names):
            raise ValueError(f"q must have {len(self.joint_names)} values, got {q.shape[0]}")
        self._data.qpos[self._qadr] = q
        mujoco.mj_kinematics(self.model, self._data)

    def fk(self, q: np.ndarray) -> np.ndarray:
        """Tool link position (3,) in the machine's base frame for joint values ``q`` (J,)."""
        self._pose(q)
        return self._data.xpos[self.model.body(self.tool_link).id].copy()

    def link_poses(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """World transform (4, 4) of every body except the world, keyed by body name."""
        self._pose(q)
        poses = {}
        for i in range(1, self.model.nbody):
            T = np.eye(4)
            T[:3, :3] = self._data.xmat[i].reshape(3, 3)
            T[:3, 3] = self._data.xpos[i]
            poses[self.model.body(i).name] = T
        return poses


def _default_tool(model: mujoco.MjModel, urdf_path: Path) -> str:
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        if j.get("name") == "gripper_frame_joint" and j.get("type") == "fixed":
            child = j.find("child")
            if child is not None and child.get("link"):
                return child.get("link")
    depth = np.zeros(model.nbody, int)
    for i in range(1, model.nbody):
        depth[i] = depth[model.body_parentid[i]] + 1
    best = max(range(1, model.nbody), key=lambda i: (depth[i], i))
    return model.body(best).name


def load_machine(urdf_path: str | Path, *, tool_link: str | None = None) -> Machine:
    urdf_path = Path(urdf_path).expanduser().resolve()
    model = mujoco.MjModel.from_xml_path(str(_prepare(urdf_path)))
    wanted = _urdf_limits(urdf_path)
    names, qadr = [], []
    for j in range(model.njnt):
        name = model.joint(j).name
        if int(model.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)) and name in wanted:
            names.append(name)
            qadr.append(model.jnt_qposadr[j])
    tool = tool_link or _default_tool(model, urdf_path)
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool) < 0:
        raise ValueError(f"tool link {tool!r} is not a link of {urdf_path.name}")
    return Machine(
        joint_names=names,
        limits=np.array([wanted[n] for n in names], float).reshape(-1, 2),
        model=model, tool_link=tool,
        _data=mujoco.MjData(model), _qadr=np.array(qadr, int))
