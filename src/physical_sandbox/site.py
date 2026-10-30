"""The site file: where things are compared.

A site holds its cameras, its floor, and the fixed things chosen as whole objects. This module is only
the site's data and its file I/O; solving a floor and cutting out objects live elsewhere.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .camera import Camera
from .floor import MAX_FOV_DEG, MIN_FOV_DEG

FORMAT = "physical-sandbox.site/1"

# three's world (y up) <- ROS world (z up); only the old open build's site files need it.
_M = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


@dataclass
class ObjectView:
    """One object as one camera sees it."""
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in pixels
    mask_rle: dict  # COCO uncompressed RLE: {"size": [h, w], "counts": [...]}
    ground_px: list[tuple[float, float]]  # where the object touches the ground, in pixels


@dataclass
class SiteObject:
    name: str
    kind: Literal["fixed", "target"]
    views: dict[str, ObjectView]  # camera name -> view
    ground_m: np.ndarray | None = None  # (3,) site coordinates, if known


@dataclass
class Floor:
    camera: str  # the camera the corners were picked in
    corners_px: list[tuple[float, float]]
    edges_m: tuple[float, float]
    corners_m: np.ndarray  # (4, 3), z = 0


@dataclass
class Site:
    id: str
    cameras: dict[str, Camera]
    floor: Floor | None = None
    objects: list[SiteObject] = field(default_factory=list)
    residual_px: dict[str, float] = field(default_factory=dict)  # camera name -> reprojection error
    confirmed_by_person: bool = False
    dimensions: Literal["measured", "estimated"] = "estimated"
    history: list[dict] = field(default_factory=list)  # each has at least "at" (ISO 8601 UTC) and "what"


# ---- masks: COCO uncompressed RLE, column-major, starting with a run of False ----

def mask_to_rle(mask: np.ndarray) -> dict:
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {m.shape}")
    flat = m.ravel(order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1  # where a run ends
    bounds = np.concatenate(([0], edges, [flat.size]))
    counts = np.diff(bounds).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)  # the first run is always of False
    if not flat.size:
        counts = [0]
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def rle_to_mask(rle: dict) -> np.ndarray:
    h, w = rle["size"]
    counts = rle["counts"]
    if sum(counts) != h * w:
        raise ValueError(f"RLE runs add up to {sum(counts)}, expected {h * w}")
    flat = np.zeros(h * w, dtype=bool)
    pos = 0
    value = False
    for c in counts:
        if value:
            flat[pos:pos + c] = True
        pos += c
        value = not value
    return flat.reshape((h, w), order="F")


# ---- the file ----

def _view_to_dict(v: ObjectView) -> dict:
    return {"box": list(v.box), "mask_rle": {"size": list(v.mask_rle["size"]), "counts": list(v.mask_rle["counts"])},
            "ground_px": [list(p) for p in v.ground_px]}


def _view_from_dict(d: dict) -> ObjectView:
    return ObjectView(tuple(int(x) for x in d["box"]), d["mask_rle"], [(float(x), float(y)) for x, y in d["ground_px"]])


def _object_to_dict(o: SiteObject) -> dict:
    return {"name": o.name, "kind": o.kind, "views": {k: _view_to_dict(v) for k, v in o.views.items()},
            "ground_m": None if o.ground_m is None else np.asarray(o.ground_m, float).tolist()}


def _object_from_dict(d: dict) -> SiteObject:
    g = d.get("ground_m")
    return SiteObject(d["name"], d["kind"], {k: _view_from_dict(v) for k, v in d["views"].items()},
                      None if g is None else np.array(g, float))


def _floor_to_dict(f: Floor) -> dict:
    return {"camera": f.camera, "corners_px": [list(p) for p in f.corners_px], "edges_m": list(f.edges_m),
            "corners_m": np.asarray(f.corners_m, float).tolist()}


def _floor_from_dict(d: dict) -> Floor:
    return Floor(d["camera"], [(float(x), float(y)) for x, y in d["corners_px"]],
                 (float(d["edges_m"][0]), float(d["edges_m"][1])), np.array(d["corners_m"], float))


def save_site(site: Site, path) -> None:
    d = {
        "format": FORMAT,
        "id": site.id,
        "cameras": {k: c.to_dict() for k, c in site.cameras.items()},
        "floor": None if site.floor is None else _floor_to_dict(site.floor),
        "objects": [_object_to_dict(o) for o in site.objects],
        "residual_px": {k: float(v) for k, v in site.residual_px.items()},
        "confirmed_by_person": bool(site.confirmed_by_person),
        "dimensions": site.dimensions,
        "history": site.history,
    }
    Path(path).write_text(json.dumps(d, indent=2))


def load_site(path) -> Site:
    d = json.loads(Path(path).read_text())
    fmt = d.get("format") if isinstance(d, dict) else None
    if fmt != FORMAT:
        raise ValueError(f"not a site file this version reads: format is {fmt!r}, expected {FORMAT!r}")
    return Site(
        id=d["id"],
        cameras={k: Camera.from_dict(c) for k, c in d["cameras"].items()},
        floor=None if d.get("floor") is None else _floor_from_dict(d["floor"]),
        objects=[_object_from_dict(o) for o in d.get("objects", [])],
        residual_px={k: float(v) for k, v in d.get("residual_px", {}).items()},
        confirmed_by_person=bool(d.get("confirmed_by_person", False)),
        dimensions=d.get("dimensions", "estimated"),
        history=list(d.get("history", [])),
    )


def load_old_site(path) -> Site:
    """A site file of the old open build (``sites/*.json``): only its camera is copied, as ``'camera'``."""
    path = Path(path)
    d = json.loads(path.read_text())
    intr, extr = d["camera"]["intr"], d["camera"]["extr"]
    K = [[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]]
    R = np.array(extr["R"], float).reshape(3, 3) @ _M  # R_ros = R_three @ M
    camera = Camera(intr["width"], intr["height"], K, intr["dist"], R, extr["t"])
    return Site(id=d.get("id") or path.stem, cameras={"camera": camera}, confirmed_by_person=True,
                dimensions="estimated")


# ---- tying more cameras to one site ----

def _middle_point(view: ObjectView) -> tuple[float, float]:
    """The middle of an object view's ground points (left end, middle, right end)."""
    return view.ground_px[len(view.ground_px) // 2]


def locate_objects(site: Site, camera_name: str) -> Site:
    """A new site where each object seen by ``camera_name`` has its ``ground_m`` set to where the middle of its
    ground points meets the floor. Objects with no view in that camera (or whose ray misses the floor) keep
    what they had."""
    camera = site.cameras[camera_name]
    objects = []
    for obj in site.objects:
        view = obj.views.get(camera_name)
        if view is not None and view.ground_px:
            p = camera.floor_point(*_middle_point(view))
            if p is not None:
                obj = dataclasses.replace(obj, ground_m=p)
        objects.append(obj)
    return dataclasses.replace(site, objects=objects)


def _known_positions(site: Site) -> dict[str, np.ndarray]:
    known = {o.name: np.asarray(o.ground_m, float) for o in site.objects if o.ground_m is not None}
    if site.floor is not None:
        for i, c in enumerate(site.floor.corners_m):
            known[f"corner-{i + 1}"] = np.asarray(c, float)
    return known


def _pixel(value) -> tuple[float, float]:
    """One pixel point, or a list of ground points (then its middle one)."""
    a = np.asarray(value, float)
    if a.ndim == 2:
        a = a[len(a) // 2]
    if a.shape != (2,):
        raise ValueError(f"a pixel point must be (u, v) or a list of them, got shape {a.shape}")
    return float(a[0]), float(a[1])


def _focal_from_homography(world_xy: np.ndarray, px_c: np.ndarray) -> float | None:
    """The focal length that makes the first two columns of the floor's homography orthogonal and equally
    long, or None when the points do not allow it (the same two conditions as ``floor.solve_floor``)."""
    H, _ = cv2.findHomography(world_xy, px_c)
    if H is None:
        return None
    h1, h2 = H[:, 0], H[:, 1]
    A = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    B = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    aa = float(A @ A)
    w = -float(A @ B) / aa if aa > 0 else -1.0
    return float(1 / np.sqrt(w)) if w > 0 else None


def _fit_from_floor_points(width: int, height: int, world: np.ndarray, px: np.ndarray, f0: float):
    """calibrateCamera on one planar view from focal guess ``f0``: (camera, residual) or None if it is no solution."""
    cx, cy = width / 2, height / 2
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    K0 = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1]])
    try:
        _, K, _, rvecs, tvecs = cv2.calibrateCamera(
            [world.astype(np.float32)], [px.astype(np.float32)], (int(width), int(height)), K0, np.zeros(5),
            flags=flags)
    except cv2.error:
        return None
    fov = np.degrees(2 * np.arctan(width / (2 * K[0, 0])))
    if not (MIN_FOV_DEG <= fov <= MAX_FOV_DEG):
        return None
    R, _ = cv2.Rodrigues(rvecs[0])
    t = tvecs[0].reshape(3)
    if not ((world @ R.T + t)[:, 2] > 0).all():
        return None  # a used point is behind the camera
    camera = Camera(width, height, K, np.zeros(5), R, t)
    if camera.position[2] <= 0:
        return None  # a planar view has a mirror solution under the floor; the camera must be above it
    residual = float(np.linalg.norm(camera.project(world) - px, axis=1).max())
    if not np.isfinite(residual):
        return None
    return camera, residual


def add_camera(site: Site, name: str, width: int, height: int,
               seen: dict[str, object]) -> tuple[Site, float]:
    """Tie a further camera to the site from what it shares with what is already known.

    ``seen`` maps a name to a pixel point (or a list of ground points, of which the middle one is used) in the
    new picture: an object's name (its ground point) or ``corner-1`` ... ``corner-4`` (the floor's corners).
    Only names whose site position is known are used; there must be 4 or more. Everything used is on the floor,
    so the fit is planar: principal point at the centre, square pixels, no distortion, the focal length fitted
    together with the pose. Returns the new site (``cameras[name]`` and ``residual_px[name]`` set; the input is
    untouched) and the largest reprojection distance in pixels over the points used."""
    known = _known_positions(site)
    used = [n for n in seen if n in known]
    unused = [n for n in seen if n not in known]
    if len(used) < 4:
        raise ValueError(
            f"a camera needs 4 or more shared points whose place in the site is known; got {len(used)} "
            f"({', '.join(used) or 'none'}). Unknown or missing: {', '.join(unused) or 'none'}"
        )
    world = np.array([known[n] for n in used], float)
    px = np.array([_pixel(seen[n]) for n in used], float)
    if np.abs(world[:, 2]).max() > 1e-9:
        raise ValueError("every shared point must be on the floor (z = 0)")

    guesses = []
    f_h = _focal_from_homography(world[:, :2], px - [width / 2, height / 2])
    if f_h is not None:
        guesses.append(f_h)
    guesses.append(width / (2 * np.tan(np.radians(60.0) / 2)))  # a 60 degree field of view
    best = None
    for f0 in guesses:
        fit = _fit_from_floor_points(width, height, world, px, f0)
        if fit is not None and (best is None or fit[1] < best[1]):
            best = fit
    if best is None:
        raise ValueError(
            "no camera fits these points: they may lie in one line, be seen square-on, or put the camera below "
            "the floor or a point behind it"
        )
    camera, residual = best
    new_site = dataclasses.replace(
        site, cameras={**site.cameras, name: camera}, residual_px={**site.residual_px, name: residual})
    return new_site, residual
