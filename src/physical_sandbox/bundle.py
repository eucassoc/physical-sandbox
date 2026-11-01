"""The display bundle: what the browser draws a machine from, made from its URDF.

The browser never reads URDF. It assembles ``bundle.json`` (links with their visual meshes, joints, chains)
and fetches each mesh as an RBM1 binary from ``assets/``. Ported from the old open build's
``tools/web_mesh.py`` and ``tools/export-web-robots.py`` (the bundle's keys are those of ``app/data/<id>.json``)
and ``robots/mesh_convert.py`` / ``robots/mesh_bake.py`` (reading STL, DAE and OBJ; baking a mesh scale).

RBM1, little-endian throughout::

    offset  size  contents
    0       4     magic "RBM1"
    4       4     flags uint32: bit0 = has UV, bit1 = has normals
    8       4     vertex count uint32
    12      4     index count uint32
    16      12V   positions float32 x 3
    ...     12V   normals float32 x 3   (when bit1)
    ...     8V    UVs float32 x 2       (when bit0)
    ...     4I    indices uint32

Every array starts on a multiple of 4, so the browser maps typed arrays straight onto the buffer.

A mesh with more than ``MAX_FACES`` triangles is thinned by vertex clustering: vertices are snapped to a grid of
cubes, those in one cube merge into their mean, and triangles that collapse are dropped. The finest grid that
gets the mesh under ``MAX_FACES`` is used: it starts at 1024 cubes along the mesh's longest side and halves
while the result is still over, but never goes coarser than 64 (a mesh still over then stays over). A mesh at or
under ``MAX_FACES`` is left exactly as it is.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

from ._atomic import write_text_atomic
from .errors import InputError
from .machine import _find_mesh

MAGIC = b"RBM1"
HAS_UV = 1
HAS_NORMAL = 2
MAX_FACES = 20_000
GRID_FINEST = 1024  # cubes along the longest side the thinning starts from
GRID_COARSEST = 64  # and never goes below

COLLADA = "http://www.collada.org/2005/11/COLLADASchema"
_NS = {"c": COLLADA}


@dataclass
class BundleResult:
    bundle_path: Path
    mesh_paths: list[Path]  # every file written beside bundle.json (the .bin meshes and any texture images)
    links: int
    joints: list[str]
    warnings: list[str] = field(default_factory=list)


@dataclass
class Mesh:
    positions: np.ndarray  # (V, 3) float64
    faces: np.ndarray  # (F, 3) int64, indices into positions
    uvs: np.ndarray | None = None  # (V, 2)
    normals: np.ndarray | None = None  # (V, 3)
    texture: Path | None = None  # the image an OBJ's material applies


class _Unreadable(Exception):
    """A mesh format this machine cannot read (GLB without trimesh): the visual is skipped with a warning."""


# ---- reading meshes ----

def _first_seen_unique(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows in the order they first appear, and each row's index among them (as a dict would number them)."""
    if len(rows) == 0:
        return rows, np.zeros(0, np.int64)
    uniq, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return uniq[order], rank[inverse.reshape(-1)]


def read_stl(path) -> tuple[np.ndarray, np.ndarray]:
    """An STL (binary or ASCII) as (vertices (V, 3), faces (F, 3)), with repeated corners shared.

    Starting with ``solid`` does not mean ASCII: a file is binary when 84 + 50 x its triangle count is its size.
    """
    data = Path(path).read_bytes()
    binary = len(data) >= 84 and 84 + 50 * struct.unpack("<I", data[80:84])[0] == len(data)
    if binary:
        count = struct.unpack("<I", data[80:84])[0]
        rec = np.frombuffer(data, [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count, 84)
        corners = rec["v"].reshape(-1, 3)
    else:
        values = []
        for line in data.decode("utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[0] == "vertex":
                values.append([float(v) for v in parts[1:]])
        corners = np.array(values[: len(values) // 3 * 3], np.float32).reshape(-1, 3)
    if len(corners) == 0:
        raise ValueError(f"{path}: not one triangle could be read")
    vertices, index = _first_seen_unique(corners)
    return vertices.astype(np.float64), index.reshape(-1, 3)


def _obj_corner(token: str) -> tuple[int, int, int]:
    fields = (token.split("/") + ["", ""])[:3]
    return tuple(int(f) - 1 if f else -1 for f in fields)


def _obj_texture(obj_path: Path, text: str) -> Path | None:
    """The image the OBJ's material applies (mtllib -> map_Kd), or None. Names are never guessed."""
    library = next((ln.split(None, 1)[1].strip() for ln in text.splitlines()
                    if ln.startswith("mtllib ") and len(ln.split(None, 1)) == 2), None)
    if library is None or not (obj_path.parent / library).is_file():
        return None
    mtl = obj_path.parent / library
    for line in mtl.read_text(errors="replace").splitlines():
        if line.strip().startswith("map_Kd "):
            image = mtl.parent / line.strip().split(None, 1)[1].strip()
            if not image.is_file():
                raise FileNotFoundError(f"{mtl}: no image for map_Kd, {image}")
            return image
    return None


def read_obj(path) -> Mesh:
    """An OBJ with its UVs and normals, its three index streams collapsed into the one a GPU wants.

    Faces with more than three corners are fan-triangulated. A corner without a UV or normal gets (0, 0) or
    (0, 0, 1) when the file has any.
    """
    path = Path(path)
    text = path.read_text(errors="replace")
    positions, uvs, normals, corners = [], [], [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v":
            positions.append([float(v) for v in parts[1:4]])
        elif tag == "vt":
            uvs.append([float(v) for v in parts[1:3]])
        elif tag == "vn":
            normals.append([float(v) for v in parts[1:4]])
        elif tag == "f":
            if len(parts) < 4:
                raise ValueError(f"{path}: a face has only {len(parts) - 1} vertices")
            cs = [_obj_corner(p) for p in parts[1:]]
            for i in range(1, len(cs) - 1):
                corners.extend((cs[0], cs[i], cs[i + 1]))
    if not corners:
        raise ValueError(f"{path}: not one face could be read")
    triples = np.array(corners, np.int64)
    n_pos = len(positions)
    triples[:, 0] = np.where(triples[:, 0] < -1, triples[:, 0] + 1 + n_pos, triples[:, 0])  # negative = relative
    uniq, index = _first_seen_unique(triples)
    pos = np.array(positions, np.float64)[uniq[:, 0]]
    out_uv = out_n = None
    if uvs:
        table = np.vstack([np.array(uvs, np.float64), [[0.0, 0.0]]])
        out_uv = table[np.where(uniq[:, 1] >= 0, uniq[:, 1], len(uvs))]
    if normals:
        table = np.vstack([np.array(normals, np.float64), [[0.0, 0.0, 1.0]]])
        out_n = table[np.where(uniq[:, 2] >= 0, uniq[:, 2], len(normals))]
    return Mesh(pos, index.reshape(-1, 3), out_uv, out_n, _obj_texture(path, text))


def _node_transform(node) -> np.ndarray:
    """The transform elements directly under a Collada <node>, composed in order."""
    m = np.eye(4)
    for child in node:
        tag = child.tag.split("}")[-1]
        values = [float(v) for v in (child.text or "").split()]
        if tag == "matrix":
            local = np.array(values[:16]).reshape(4, 4)
        elif tag == "translate":
            local = np.eye(4)
            local[:3, 3] = values[:3]
        elif tag == "rotate":
            local = np.eye(4)
            local[:3, :3] = _axis_angle(values[:3], math.radians(values[3]))
        elif tag == "scale":
            local = np.diag([*values[:3], 1.0])
        else:
            continue
        m = m @ local
    return m


def _geometry_transforms(root) -> dict[str, list[np.ndarray]]:
    """A geometry id -> the transforms that place it in the scene. Ignoring them tilts or displaces meshes."""
    table: dict[str, list[np.ndarray]] = {}

    def walk(node, parent):
        current = parent @ _node_transform(node)
        for inst in node.findall("c:instance_geometry", _NS):
            url = (inst.get("url") or "").lstrip("#")
            if url:
                table.setdefault(url, []).append(current)
        for child in node.findall("c:node", _NS):
            walk(child, current)

    scenes = root.find("c:library_visual_scenes", _NS)
    if scenes is not None:
        for scene in scenes.findall("c:visual_scene", _NS):
            for node in scene.findall("c:node", _NS):
                walk(node, np.eye(4))
    return table


def read_dae(path) -> tuple[np.ndarray, np.ndarray]:
    """Triangle meshes out of a Collada file, the scene graph's transforms applied. Non-triangles are refused."""
    root = ET.parse(path).getroot()
    placements = _geometry_transforms(root)
    corners = []
    for geometry in root.iter(f"{{{COLLADA}}}geometry"):
        mesh = geometry.find("c:mesh", _NS)
        if mesh is None:
            continue
        positions = None
        for source in mesh.findall("c:source", _NS):
            array = source.find("c:float_array", _NS)
            if array is not None:
                positions = np.array([float(v) for v in array.text.split()]).reshape(-1, 3)
                break
        if positions is None:
            raise ValueError(f"{path}: no POSITION source")
        for matrix in placements.get(geometry.get("id"), [np.eye(4)]):
            placed = positions @ matrix[:3, :3].T + matrix[:3, 3]
            for tris in mesh.findall("c:triangles", _NS) + mesh.findall("c:polylist", _NS):
                vcount = tris.find("c:vcount", _NS)
                if vcount is not None and any(int(v) != 3 for v in vcount.text.split()):
                    raise ValueError(f"{path}: a face is not a triangle (unsupported)")
                inputs = tris.findall("c:input", _NS)
                stride = max(int(i.get("offset", 0)) for i in inputs) + 1
                offset = next((int(i.get("offset", 0)) for i in inputs if i.get("semantic") == "VERTEX"), 0)
                idx = np.array([int(v) for v in tris.find("c:p", _NS).text.split()], np.int64)[offset::stride]
                idx = idx[: len(idx) // 3 * 3]
                corners.append(placed[idx])
    if not corners or sum(len(c) for c in corners) == 0:
        raise ValueError(f"{path}: not one triangle could be read")
    vertices, index = _first_seen_unique(np.vstack(corners))
    return vertices, index.reshape(-1, 3)


def _trimesh():
    try:
        import trimesh
    except ImportError:
        return None
    return trimesh


def read_mesh(path) -> Mesh:
    """Any mesh a URDF may point at: STL, OBJ, DAE; GLB, glTF and 3MF when trimesh is installed."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return Mesh(*read_stl(path))
    if suffix == ".obj":
        return read_obj(path)
    if suffix == ".dae":
        return Mesh(*read_dae(path))
    if suffix in (".glb", ".gltf", ".3mf"):
        trimesh = _trimesh()
        if trimesh is None:
            raise _Unreadable(f"{path.name}: {suffix} needs trimesh (pip install trimesh); this visual is skipped")
        loaded = trimesh.load(str(path), force="mesh")
        return Mesh(np.asarray(loaded.vertices, np.float64), np.asarray(loaded.faces, np.int64))
    raise _Unreadable(f"{path.name}: the mesh format {suffix!r} is not read; this visual is skipped")


# ---- scale, thinning, RBM1 ----

def parse_scale(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (1.0, 1.0, 1.0)
    parts = [float(v) for v in text.split()]
    if len(parts) != 3:
        raise ValueError(f"a mesh scale has three numbers, not {text!r}")
    return tuple(parts)


def _scale_suffix(scale) -> str:
    """Added to a mesh's name, so one mesh used at two scales becomes two files."""
    return "" if tuple(scale) == (1.0, 1.0, 1.0) else "__s" + "_".join(f"{v:g}" for v in scale)


def bake_scale(mesh: Mesh, scale) -> Mesh:
    """The URDF's <mesh scale> baked into the vertices. A mirror (negative determinant) reverses the winding,
    or the face would be lit from inside."""
    s = np.asarray(scale, float)
    if np.all(s == 1.0):
        return mesh
    normals = None
    if mesh.normals is not None:
        normals = mesh.normals / np.where(s == 0, 1.0, s)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)
    faces = mesh.faces[:, [0, 2, 1]] if np.prod(s) < 0 else mesh.faces
    return Mesh(mesh.positions * s, faces, mesh.uvs, normals, mesh.texture)


def decimate(positions, faces, *, max_faces: int = MAX_FACES, uvs=None, normals=None):
    """Thin a mesh to at most ``max_faces`` triangles by vertex clustering.

    Returns (positions, faces, (uvs, normals)). The grid starts at 1024 cubes along the longest side and halves
    while the result is over ``max_faces``, stopping at 64 even if it is still over. With UVs, vertices merge only
    when their UVs also fall in the same square of the same grid, so a texture seam stays a seam.
    """
    positions = np.asarray(positions, float)
    faces = np.asarray(faces, np.int64)
    if len(faces) <= max_faces:
        return positions, faces, (uvs, normals)
    lo = positions.min(0)
    extent = float((positions.max(0) - lo).max()) or 1.0
    divisions = GRID_FINEST
    while True:
        key = np.floor((positions - lo) / extent * divisions).astype(np.int64)
        if uvs is not None:
            key = np.hstack([key, np.floor(np.asarray(uvs) * divisions).astype(np.int64)])
        _, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        f = inverse[faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        if len(f):
            _, keep = np.unique(np.sort(f, axis=1), axis=0, return_index=True)
            f = f[np.sort(keep)]
        if len(f) <= max_faces or divisions <= GRID_COARSEST:
            break
        divisions //= 2
    n = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n).astype(float)[:, None]

    def mean(values):
        out = np.zeros((n, values.shape[1]))
        np.add.at(out, inverse, values)
        return out / counts

    p = mean(positions)
    uv = mean(np.asarray(uvs, float)) if uvs is not None else None
    nr = None
    if normals is not None:
        nr = mean(np.asarray(normals, float))
        nr /= np.maximum(np.linalg.norm(nr, axis=1, keepdims=True), 1e-30)
    used = np.unique(f)  # drop the vertices no triangle uses any more
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return p[used], remap[f], (uv[used] if uv is not None else None, nr[used] if nr is not None else None)


def to_binary(mesh: Mesh) -> bytes:
    flags = (HAS_UV if mesh.uvs is not None else 0) | (HAS_NORMAL if mesh.normals is not None else 0)
    idx = np.asarray(mesh.faces, "<u4").reshape(-1)
    chunks = [MAGIC, struct.pack("<III", flags, len(mesh.positions), len(idx)),
              np.asarray(mesh.positions, "<f4").tobytes()]
    if mesh.normals is not None:
        chunks.append(np.asarray(mesh.normals, "<f4").tobytes())
    if mesh.uvs is not None:
        chunks.append(np.asarray(mesh.uvs, "<f4").tobytes())
    chunks.append(idx.tobytes())
    return b"".join(chunks)


# ---- reading the URDF ----

def parse_origin(element) -> dict:
    """<origin xyz rpy>; the identity when there is none."""
    if element is None:
        return {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]}
    return {"xyz": [float(v) for v in (element.get("xyz") or "0 0 0").split()],
            "rpy": [float(v) for v in (element.get("rpy") or "0 0 0").split()]}


def parse_mimic(joint) -> dict | None:
    """<mimic joint multiplier offset>; None for a joint that follows nothing."""
    mimic = joint.find("mimic")
    if mimic is None or not mimic.get("joint"):
        return None
    return {"joint": mimic.get("joint"), "multiplier": float(mimic.get("multiplier") or 1.0),
            "offset": float(mimic.get("offset") or 0.0)}


def find_root_link(links, joints) -> str:
    """The link that is no joint's child but some joint's parent; with no joints, the one link there is."""
    if not joints:
        if len(links) != 1:
            raise InputError(f"no joints, so exactly one link is expected, not {len(links)}")
        return next(iter(links))
    children = {j["child"] for j in joints}
    parents = {j["parent"] for j in joints}
    roots = [n for n in links if n not in children and n in parents]
    if len(roots) != 1:
        raise InputError(f"{len(roots)} root links: {roots}")
    return roots[0]


def _mesh_key(reference: str) -> str:
    """A URDF mesh reference -> its name in the bundle, without suffix: package:// and a leading meshes/ or
    assets/ dropped (``assets/base.stl`` -> ``base``); a path that leaves the folder becomes its file stem."""
    ref = reference.replace("\\", "/")
    if "://" in ref:
        ref = ref.split("://", 1)[1]
        ref = ref.split("/", 1)[1] if "/" in ref and not ref.startswith("/") else ref
    p = PurePosixPath(ref)
    parts = [x for x in p.with_suffix("").parts if x not in ("", ".")]
    if p.is_absolute() or ".." in parts or not parts:
        parts = [p.stem or "mesh"]
    while len(parts) > 1 and parts[0] in ("meshes", "assets"):
        parts.pop(0)
    return "/".join(parts)


def _joint(joint) -> dict:
    kind = joint.get("type")
    limit = joint.find("limit")
    axis = joint.find("axis")
    parent, child = joint.find("parent"), joint.find("child")
    if parent is None or child is None:
        raise InputError(f"joint {joint.get('name')!r} has no parent or child link")
    mimic = parse_mimic(joint)
    return {
        "name": joint.get("name"),
        "type": kind,
        "parent": parent.get("link"),
        "child": child.get("link"),
        "origin": parse_origin(joint.find("origin")),
        "axis": [float(v) for v in ((axis.get("xyz") if axis is not None else None) or "0 0 1").split()],
        # continuous has no limits: +/-pi stands in for the screen's sake
        "lower": float(limit.get("lower")) if limit is not None and limit.get("lower")
        else (-math.pi if kind != "fixed" else 0.0),
        "upper": float(limit.get("upper")) if limit is not None and limit.get("upper")
        else (math.pi if kind != "fixed" else 0.0),
        **({"mimic": mimic} if mimic is not None else {}),
    }


def _public(block):
    """A profile block without its '_' notes (the web gets figures, not their provenance)."""
    if not isinstance(block, dict):
        return None
    return {k: v for k, v in block.items() if not k.startswith("_")}


def _public_wheels(profile):
    wheels = profile.get("wheels")
    if not isinstance(wheels, dict):
        return None
    out = {k: v for k, v in wheels.items() if not k.startswith("_") and k != "joints"}
    if isinstance(wheels.get("joints"), list):
        out["joints"] = [_public(j) for j in wheels["joints"] if isinstance(j, dict)]
    return out


# ---- measuring ----

def _axis_angle(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) or 1.0)
    x, y, z = a
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
                     [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
                     [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])


def _pose(origin) -> np.ndarray:
    """A URDF origin as a 4x4: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = origin["rpy"]
    m = np.eye(4)
    m[:3, :3] = _axis_angle([0, 0, 1], y) @ _axis_angle([0, 1, 0], p) @ _axis_angle([1, 0, 0], r)
    m[:3, 3] = origin["xyz"]
    return m


def _dimensions(root: str, links: dict, joints: list, points: dict[str, np.ndarray], motion) -> dict | None:
    """The size in metres at rest (every joint at 0; x length, y width, z height, ROS), rounded outward to the
    millimetre; a footprint the profile declares wins for length and width. None when no mesh gave a vertex."""
    children: dict[str, list] = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    world = {root: np.eye(4)}
    stack = [root]
    while stack:
        parent = stack.pop()
        for j in children.get(parent, []):
            if j["child"] not in world:
                world[j["child"]] = world[parent] @ _pose(j["origin"])
                stack.append(j["child"])
    gathered = []
    for name, link in links.items():
        if name not in world:
            continue  # a branch the root cannot reach is never drawn
        for v in link["visuals"]:
            pts = points.get(v["mesh"])
            if pts is not None and len(pts):
                m = world[name] @ _pose(v["origin"])
                gathered.append(pts @ m[:3, :3].T + m[:3, 3])
    if not gathered:
        return None
    allp = np.vstack(gathered)
    size = [math.ceil(round(float(d) * 1000, 6)) / 1000 for d in allp.max(0) - allp.min(0)]
    footprint = (motion or {}).get("footprint") or {}
    return {"lengthM": footprint.get("lengthM", round(size[0], 3)),
            "widthM": footprint.get("widthM", round(size[1], 3)), "heightM": round(size[2], 3)}


# ---- the bundle ----

def _display_name(product: dict, profile: dict, name: str) -> str:
    shown = product.get("displayName")
    if isinstance(shown, dict) and shown.get("en"):
        return shown["en"]
    if isinstance(shown, str) and shown:
        return shown
    return profile.get("displayName") or name


def make_bundle(urdf_path: Path, *, product: dict, profile: dict | None, name: str, out_dir: Path) -> BundleResult:
    """Write ``out_dir/bundle.json`` and its meshes under ``out_dir/assets/`` from a URDF.

    ``product`` is the machine's product.json (display name, licence), ``profile`` its profile.json (vendor, mass,
    locomotion, motion, chains, wheels) or None. Every mesh is found and read before anything is written, and the
    JSON goes last, so a missing mesh (FileNotFoundError naming it) leaves no bundle pointing at nothing.
    """
    urdf_path = Path(urdf_path).expanduser().resolve()
    out_dir = Path(out_dir)
    product = product or {}
    profile = profile or {}
    root = ET.parse(urdf_path).getroot()
    if root.tag != "robot":
        raise InputError(f"{urdf_path.name}: the display bundle is made from a URDF (<robot>), not <{root.tag}>")
    warnings: list[str] = []

    # read every visual mesh first: (resolved file, scale) -> its name in the bundle and the mesh itself
    names: dict[tuple[Path, tuple], str] = {}
    meshes: dict[str, Mesh] = {}
    taken: set[str] = set()
    links: dict[str, dict] = {}
    for link in root.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            geom = visual.find("geometry")
            mesh_el = geom.find("mesh") if geom is not None else None
            if mesh_el is None or not mesh_el.get("filename"):
                shape = next((c.tag for c in (geom if geom is not None else [])), None)
                if shape:
                    warnings.append(f"link {link.get('name')!r}: a <{shape}> visual is not drawn (meshes only)")
                continue
            ref = mesh_el.get("filename")
            source = _find_mesh(urdf_path.parent, ref).resolve()  # FileNotFoundError names the reference
            scale = parse_scale(mesh_el.get("scale"))
            key = (source, scale)
            if key not in names:
                try:
                    mesh = read_mesh(source)
                except _Unreadable as e:
                    warnings.append(str(e))
                    names[key] = ""
                    continue
                mesh = bake_scale(mesh, scale)
                if len(mesh.faces) > MAX_FACES:
                    before = len(mesh.faces)
                    p, f, (uv, nr) = decimate(mesh.positions, mesh.faces, uvs=mesh.uvs, normals=mesh.normals)
                    mesh = Mesh(p, f, uv, nr, mesh.texture)
                    note = "" if len(f) <= MAX_FACES else f" (still over {MAX_FACES} at the coarsest grid)"
                    warnings.append(f"{source.name}: thinned from {before} to {len(f)} triangles{note}")
                base = "assets/" + _mesh_key(ref) + _scale_suffix(scale)
                label, n = base, 1
                while label in taken:
                    n += 1
                    label = f"{base}_{n}"
                taken.add(label)
                names[key] = label + ".bin"
                meshes[names[key]] = mesh
            if not names[key]:
                continue
            entry = {"mesh": names[key], "origin": parse_origin(visual.find("origin"))}
            texture = meshes[names[key]].texture
            if texture is not None:
                entry["texture"] = str(PurePosixPath(names[key]).parent / texture.name)
            visuals.append(entry)
        links[link.get("name")] = {"visuals": visuals}

    joints = [_joint(j) for j in root.findall("joint")]
    root_link = find_root_link(links, joints)
    chains = [{"name": c["name"], "baseLink": c["baseLink"], "tipLink": c["tipLink"], "joints": c["jointNames"]}
              for c in profile.get("chains", [])]
    motion = _public(profile.get("motion"))
    wheels = _public_wheels(profile)
    env = profile.get("environment")

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for label, mesh in meshes.items():
        dest = out_dir / label
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(to_binary(mesh))
        written.append(dest)
        if mesh.texture is not None:
            image = dest.parent / mesh.texture.name
            if image not in written:
                shutil.copyfile(mesh.texture, image)
                written.append(image)

    data = {
        "id": name,
        # put on mesh URLs to defeat the browser's cache after a mesh changes
        "version": str(int(time.time())),
        "displayName": _display_name(product, profile, name),
        "vendor": profile.get("vendor") or product.get("maker", ""),
        "licenseId": product.get("licenseId") or profile.get("licenseId", ""),
        "urdfSourceUrl": profile.get("urdfSourceUrl", ""),
        "totalMassKg": profile.get("totalMassKg", 0.0),
        "locomotion": profile.get("locomotion", "Static"),
        "hoverHeightM": profile.get("hoverHeightM", 0.0),
        # water: a depth and whether the maker rates it, both or neither
        **({"fordingDepthM": env["fordingDepthM"], "fordingRated": bool(env.get("fordingRated", False))}
           if isinstance(env, dict) and env.get("fordingDepthM") is not None else {}),
        **({"motion": motion} if motion is not None else {}),
        **({"wheels": wheels} if wheels is not None else {}),
        "rootLink": root_link,
        "links": links,
        "joints": joints,
        "chains": chains,
    }
    dims = _dimensions(root_link, links, joints, {k: m.positions for k, m in meshes.items()}, motion)
    if dims is not None:
        data["dimensions"] = dims
    bundle_path = out_dir / "bundle.json"
    write_text_atomic(bundle_path, json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    return BundleResult(bundle_path, written, len(links), [j["name"] for j in joints], warnings)
