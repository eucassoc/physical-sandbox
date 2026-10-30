"""A machine's shapes, thinned and packed for the in-chat screen.

The screen holds no kinematics: it draws each link's shape at the 4x4 pose the server sends. So the shapes go
once, link by link, in the link's own frame: MuJoCo's mesh vertices and faces (the visuals, group 1; every group
when there are none, as ``render.py`` does) welded by position, thinned to about a ninth of the triangles with
``fast-simplification``, and the primitives (boxes, spheres, cylinders, capsules, ellipsoids) tessellated.

The pack is one binary: per link, ``vCount`` float32 positions (x, y, z) and then ``iCount`` uint32 vertex
indices, little endian, at the byte ``offset`` the layout names. It is built once and kept on disk under the
cache directory (``PS_CACHE_DIR``) as ``models/<key>.json`` (the layout) and ``models/<key>.b64`` (the binary as
base64, so a part is a slice of text). The key is a hash of everything that is drawn, so a changed mesh file
gives a new pack. The server keeps nothing in memory: a different start of it finds the files.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
from pathlib import Path

import mujoco
import numpy as np

from ..machine import Machine

KEEP = 1 / 9  # of the triangles of a mesh
MIN_FACES = 300  # a mesh with fewer triangles is not thinned below this
_ROUNDS = 10  # the simplifier stalls on some meshes in one go; each further round takes more off
_VISUAL_GROUP = 1  # see render.py: with discardvisual="false" visuals are in group 1, collisions in group 0
_VERSION = "1"
_KEY = re.compile(r"^[0-9a-f]{40}$")

_MESH = int(mujoco.mjtGeom.mjGEOM_MESH)
_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
_CAPSULE = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
_CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
_ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)
_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)


def _models_dir() -> Path:
    d = Path(os.environ.get("PS_CACHE_DIR", "~/.cache/physical-sandbox")).expanduser() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # two starts of the server may race
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---- the shapes ----

def _box(size) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = (float(s) for s in size)
    v = np.array([[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    # vertex index = 4 * (sx > 0) + 2 * (sy > 0) + (sz > 0); every face counter-clockwise seen from outside
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],   # -x, +x
                  [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],   # -y, +y
                  [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],  # -z, +z
                 np.int64)
    return v, f


def _sphere(radii, half: float = 0.0, rings: int = 9, segs: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """A UV sphere (an ellipsoid with three radii); ``half`` pulls the two hemispheres apart along z: a capsule."""
    rx, ry, rz = (float(r) for r in radii)
    verts = [[0.0, 0.0, rz + half]]
    for i in range(1, rings):
        phi = math.pi * i / rings
        z = rz * math.cos(phi) + (half if phi < math.pi / 2 - 1e-9 else -half if phi > math.pi / 2 + 1e-9 else 0.0)
        for j in range(segs):
            th = 2 * math.pi * j / segs
            verts.append([rx * math.sin(phi) * math.cos(th), ry * math.sin(phi) * math.sin(th), z])
    verts.append([0.0, 0.0, -rz - half])
    faces = []
    south = len(verts) - 1
    for j in range(segs):
        faces.append([0, 1 + j, 1 + (j + 1) % segs])
    for i in range(rings - 2):
        a, b = 1 + i * segs, 1 + (i + 1) * segs
        for j in range(segs):
            j2 = (j + 1) % segs
            faces += [[a + j, b + j, b + j2], [a + j, b + j2, a + j2]]
    last = 1 + (rings - 2) * segs
    for j in range(segs):
        faces.append([south, last + (j + 1) % segs, last + j])
    return np.array(verts, float), np.array(faces, np.int64)


def _cylinder(size, segs: int = 16) -> tuple[np.ndarray, np.ndarray]:
    r, h = float(size[0]), float(size[1])
    ring = [[r * math.cos(2 * math.pi * j / segs), r * math.sin(2 * math.pi * j / segs)] for j in range(segs)]
    verts = [[x, y, -h] for x, y in ring] + [[x, y, h] for x, y in ring] + [[0.0, 0.0, -h], [0.0, 0.0, h]]
    faces = []
    for j in range(segs):
        j2 = (j + 1) % segs
        faces += [[j, j2, segs + j2], [j, segs + j2, segs + j]]  # the side
        faces.append([2 * segs, j2, j])  # the bottom, seen from below
        faces.append([2 * segs + 1, segs + j, segs + j2])  # the top
    return np.array(verts, float), np.array(faces, np.int64)


def _primitive(t: int, size) -> tuple[np.ndarray, np.ndarray] | None:
    if t == _BOX:
        return _box(size)
    if t == _SPHERE:
        return _sphere((size[0],) * 3)
    if t == _ELLIPSOID:
        return _sphere(size[:3])
    if t == _CAPSULE:
        return _sphere((size[0],) * 3, half=float(size[1]))
    if t == _CYLINDER:
        return _cylinder(size)
    return None  # planes, heightfields, SDFs: not part of a robot's look


def _thinned(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weld by position (a mesh split at every texture seam stops the simplifier), drop the faces that
    collapsed, then thin to ``KEEP`` of the triangles (not below ``MIN_FACES``)."""
    import fast_simplification

    u, inv = np.unique(verts.astype(np.float32), axis=0, return_inverse=True)
    tri = inv.reshape(-1)[faces]
    tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
    pts = u.astype(np.float64)
    target = max(MIN_FACES, int(len(tri) * KEEP))
    for _ in range(_ROUNDS):
        if len(tri) <= target:
            break
        p2, t2 = fast_simplification.simplify(pts, tri, target_reduction=1 - target / len(tri), agg=7)
        if len(t2) < 4 or len(t2) >= len(tri) * 0.98:
            break  # stalled (or would leave nothing)
        pts, tri = p2, t2
    # only the vertices still used, renumbered
    used, remap = np.unique(tri.reshape(-1), return_inverse=True)
    return pts[used].astype(np.float32), remap.reshape(-1, 3).astype(np.uint32)


def _drawn_geoms(model: mujoco.MjModel) -> list[int]:
    groups = (_VISUAL_GROUP,) if (model.geom_group == _VISUAL_GROUP).any() else tuple(range(6))
    return [g for g in range(model.ngeom) if int(model.geom_group[g]) in groups and int(model.geom_bodyid[g]) > 0]


def draw_key(machine: Machine) -> str:
    """A hash of everything the pack is made from."""
    m = machine.model
    h = hashlib.sha1(f"physical-sandbox pack {_VERSION} {KEEP} {MIN_FACES}".encode())
    meshes = set()
    for g in _drawn_geoms(m):
        t = int(m.geom_type[g])
        h.update(f"|{m.body(int(m.geom_bodyid[g])).name}|{t}".encode())
        for a in (m.geom_pos[g], m.geom_quat[g], m.geom_size[g]):
            h.update(np.asarray(a, np.float64).tobytes())
        if t == _MESH:
            meshes.add(int(m.geom_dataid[g]))
    for mid in sorted(meshes):
        v0, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        f0, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        h.update(np.ascontiguousarray(m.mesh_vert[v0:v0 + vn]).tobytes())
        h.update(np.ascontiguousarray(m.mesh_face[f0:f0 + fn]).tobytes())
    return h.hexdigest()


def _build(machine: Machine) -> tuple[list[dict], bytes]:
    m = machine.model
    per_body: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for g in _drawn_geoms(m):
        t = int(m.geom_type[g])
        if t == _MESH:
            mid = int(m.geom_dataid[g])
            if mid not in mesh_cache:
                v0, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                f0, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                mesh_cache[mid] = _thinned(np.asarray(m.mesh_vert[v0:v0 + vn], float),
                                           np.asarray(m.mesh_face[f0:f0 + fn], np.int64))
            v, f = mesh_cache[mid]
            v = v.astype(np.float64)
        else:
            prim = _primitive(t, m.geom_size[g])
            if prim is None:
                continue
            v, f = prim
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(m.geom_quat[g], np.float64))
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]  # into the link's own frame
        per_body.setdefault(int(m.geom_bodyid[g]), []).append((v, f))
    links, chunks, offset = [], [], 0
    for b in sorted(per_body):
        vs, fs, base = [], [], 0
        for v, f in per_body[b]:
            vs.append(v)
            fs.append(np.asarray(f, np.int64) + base)
            base += len(v)
        pos = np.concatenate(vs).astype("<f4")
        idx = np.concatenate(fs).astype("<u4")
        links.append({"name": m.body(b).name, "vCount": int(len(pos)), "iCount": int(idx.size), "offset": offset})
        for a in (pos.reshape(-1), idx.reshape(-1)):
            chunks.append(a.tobytes())
            offset += a.nbytes
    return links, b"".join(chunks)


def pack_model(machine: Machine, chunk: int) -> dict:
    """The machine's pack on disk (made if it is not there yet): ``{key, links, binParts}``. ``links`` is the
    layout, one entry per drawn link in the order the poses use: ``{name, vCount, iCount, offset}``;
    ``binParts`` is how many ``chunk``-character parts the base64 text makes."""
    key = draw_key(machine)
    d = _models_dir()
    meta, b64 = d / f"{key}.json", d / f"{key}.b64"
    if not (meta.exists() and b64.exists()):
        links, blob = _build(machine)
        _write_atomic(b64, base64.b64encode(blob))
        _write_atomic(meta, json.dumps({"links": links}).encode())  # the meta last: it marks "complete"
    links = json.loads(meta.read_text())["links"]
    return {"key": key, "links": links, "binParts": -(-b64.stat().st_size // chunk)}


def model_part(key: str, part: int, chunk: int) -> str:
    """One ``chunk``-character part of the pack's base64 text ('' past the end)."""
    if not isinstance(key, str) or not _KEY.match(key):
        raise ValueError("bad model key")
    if part < 0:
        raise ValueError("part must be >= 0")
    path = _models_dir() / f"{key}.b64"
    meta = _models_dir() / f"{key}.json"
    if not (path.is_file() and meta.is_file()):
        raise ValueError(f"no model {key!r}: run ps_compare first")
    with path.open("rb") as f:
        f.seek(part * chunk)
        return f.read(chunk).decode("ascii")
