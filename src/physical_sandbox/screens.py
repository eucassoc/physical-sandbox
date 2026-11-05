"""The data the screens draw from: the compare screen's data for one or two motions, and the store it waits in.

The in-chat screens (the MCP server, ``mcp/server.py``) and a local server answer from the same builders, so
nothing here needs the ``[mcp]`` extra. The screen holds no kinematics: the server sends each link's shape once
(``mcp.pack``) and, per frame and record, each link's 4x4 pose in the machine's own frame, with the machine's
base transform (where it stands on the site) sent apart. A screen composes ``base @ pose`` itself, so it can
move the machine without asking the server again.

What the screen draws from goes to disk (``PS_CACHE_DIR``, default ``~/.cache/physical-sandbox``) and comes
back in parts: the host may answer the screen's calls from a different start of the server, and the text the
model reads must stay short.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .comparison import _grid, compare, grasps_of
from .errors import InputError
from .fit_screen import CHUNK, cache_dir, photo, write_atomic
from .mcp.pack import is_key, pack_model
from .motion import Motion
from .report import _header
from .scene import Scene
from .site import Site
from .sources import Source, camera_for

POSES_BUDGET = 1_600_000  # characters of JSON for all the poses; a longer motion is sampled every n-th frame
MAX_TRAIL_POINTS = 600
STEP_S = 1 / 30  # the common clock of a comparison (``compare``'s default)


def _num(x: float):
    """A number for the poses' JSON: an integral value is written without its '.0' (a quarter of the bytes are
    zeros and ones)."""
    return int(x) if x == int(x) else x


def compact(a: np.ndarray, digits: int) -> list:
    """The numbers of ``a`` row by row, rounded to ``digits`` decimals, integral ones written as integers."""
    return [_num(float(x)) for x in np.round(np.asarray(a, float), digits).reshape(-1)]


def pixels(cam: Camera, pts: np.ndarray) -> list:
    """Site points (N, 3) as pixel pairs with two decimals; null for a point that is not in front of the camera."""
    out = []
    for p in cam.project(pts):
        out.append(None if not np.isfinite(p).all() else [round(float(p[0]), 2), round(float(p[1]), 2)])
    return out


def scene_picture(scene_path: str) -> str | None:
    """The picture a scene file names (``"picture"``, relative to the scene's folder), or None."""
    try:
        d = json.loads(Path(scene_path).expanduser().read_text())
    except (OSError, ValueError):
        d = None
    name = d.get("picture") if isinstance(d, dict) else None
    if not isinstance(name, str) or not name:
        return None
    given = Path(name).expanduser()
    return str(given if given.is_absolute() else Path(scene_path).expanduser().parent / given)


def local_video(spec) -> str:
    """The picture the model asked for, limited to what is on this machine: a local file, an ``rtsp://`` camera
    or a camera number. An ``http(s)://`` address (or any other ``scheme://``) is refused here: a tool call must
    not make this machine fetch from the web (``psbx compare`` on the command line still opens them)."""
    text = str(spec).strip()
    if text.isdigit() or text.lower().startswith("rtsp://"):
        return text
    if "://" in text:
        raise InputError(f"video {text!r}: only a local file, an rtsp:// camera or a camera number can be "
                         f"opened from the chat (http(s) addresses are not)")
    return str(spec)


def write_still(frame: np.ndarray, spec, start_s: float, cam: Camera) -> Path:
    """The picture's frame as a PNG in the cache directory, so the screen's picture calls (and a different start
    of the server) find it. ``cam`` is the site camera the frame was read for, before any undistortion: the key
    holds its K and lens numbers, so a site re-fitted on the same picture never reuses a still undistorted for
    the old lens."""
    p = Path(str(spec)).expanduser()
    stamp = f"{p.resolve()}:{p.stat().st_mtime_ns}:{p.stat().st_size}" if p.is_file() else f"{spec}:{time.time_ns()}"
    lens = f"{np.asarray(cam.K, float).tobytes().hex()}:{np.asarray(cam.dist, float).tobytes().hex()}"
    key = hashlib.sha1(f"{stamp}:{start_s}:{lens}:{frame.shape}".encode()).hexdigest()
    d = cache_dir().parent / "frames"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.png"
    if not path.exists():
        ok, buf = cv2.imencode(".png", frame)
        if not ok:
            raise ValueError(f"cannot write the picture {str(spec)!r} as a still")
        write_atomic(path, buf.tobytes())
    return path


def compare_dir() -> Path:
    """Where the screens' data waits for the part tools (the comparison's poses, the proposal's outlines)."""
    d = cache_dir().parent / "compare"
    d.mkdir(parents=True, exist_ok=True)
    return d


def store_data(data: dict) -> tuple[str, int]:
    """The screen's data (poses, times, pixel paths, the series) as JSON text under the cache directory, named by
    the hash of its text. Returns (key, number of CHUNK-character parts). Written atomically; the same data is
    found, not written again."""
    text = json.dumps(data, separators=(",", ":"))  # ASCII only: a part is a slice of bytes
    key = hashlib.sha1(text.encode()).hexdigest()
    path = compare_dir() / f"{key}.json"
    if not path.exists():
        write_atomic(path, text.encode())
    return key, -(-len(text) // CHUNK)


def data_part(key: str, part: int, maker: str) -> str:
    """One CHUNK of the stored data ``key`` (40 hex characters only: no path gets through), as JSON text
    ``{"part", "data"}``. ``maker`` names the tool to run first, for the message when there is no such data."""
    if not is_key(key):
        raise ValueError("bad data key")
    if part < 0:
        raise ValueError("part must be >= 0")
    path = compare_dir() / f"{key}.json"
    if not path.is_file():
        raise ValueError(f"no data {key!r}: run {maker} first")
    with path.open("rb") as f:
        f.seek(part * CHUNK)
        return json.dumps({"part": part, "data": f.read(CHUNK).decode("ascii")})


def screen_data(site: Site, scene: Scene, camera_name: str, motions: list[Motion], source: Source, *,
                start_s: float = 0.0, grip_threshold: float | None = None, labels: list | None = None,
                poses_budget: int | None = None) -> dict:
    """Everything the compare screen needs for one or two motions of a machine on a site, as a plain dict.

    ``source`` is the open picture (a still or a recording; its frame at ``start_s`` is the screen's picture),
    ``labels`` name the motions on the screen (None or an empty entry keeps the motion's own). What the screen
    draws from (the poses, the times, the pixel paths, the numbers' series) is stored on disk and named by
    ``dataKey``; what a model reads stays short.

    The poses are each link's 4x4 in the machine's own frame, in ``compare``'s time grid (every n-th frame
    when they would pass ``poses_budget`` characters, default ``POSES_BUDGET``; the last frame is always
    there); ``base`` is the machine's base transform (16 numbers, row by row) and ``placed`` where it stands.
    The pixel paths (the tool tips, their trails, the floor) are in the picture, from the site camera. With
    two motions ``figures`` is ``compare``'s numbers; with one it is only the duration, the grasp count (by
    ``compare.grasps_of``) and the grip threshold.
    """
    if not motions or len(motions) > 2:
        raise ValueError(f"the compare screen shows one or two motions, got {len(motions)}")
    if camera_name not in site.cameras:
        raise InputError(f"the site has no camera {camera_name!r} (it has {', '.join(sorted(site.cameras))})")
    if grip_threshold is not None and not math.isfinite(float(grip_threshold)):
        raise InputError(f"gripThreshold must be a finite number, got {grip_threshold}")
    motions = [replace(m, label=label) if label else m for m, label in zip(motions, list(labels or []) + [None, None])]
    poses_budget = POSES_BUDGET if poses_budget is None else poses_budget

    try:
        cam = camera_for(source, site.cameras[camera_name])
    except ValueError as e:
        raise InputError(f"video {str(source.spec)!r}: {e}") from e
    frame = source.at(start_s)
    read_for = cam  # the camera the frame was read for, lens and all: the still's key
    if np.any(cam.dist):
        # three draws a pinhole: undo the lens in the picture, and use the pinhole camera for everything else
        frame = cv2.undistort(frame, cam.K, cam.dist)
        cam = Camera(cam.width, cam.height, cam.K, np.zeros(5), cam.R, cam.t)
    still = write_still(frame, source.spec, start_s, read_for)
    img = photo(str(still), frame=frame)

    first = motions[0]
    placed = scene.machines[first.machine]
    if len(motions) == 2:
        result = compare(scene, motions[0], motions[1], grip_threshold=grip_threshold)
        figures = result.to_dict()
        series = figures.pop("series")
        times, step_s = series["t"], float(result.step_s)
    else:
        grasps = grasps_of(first, scene, grip_threshold=grip_threshold)
        duration = float(first.t[-1] - first.t[0])
        times, step_s = [float(x) for x in _grid(duration, STEP_S, cover=True)], STEP_S
        figures = {"duration_s": {"a": duration}, "grasp_counts": {"a": grasps["count"]},
                   "grip_threshold": grasps["threshold"]}
        series = {"t": times}

    model = pack_model(placed.machine, CHUNK)
    names = [l["name"] for l in model["links"]]
    base = placed.base_transform()
    poses, tips = [], []  # per record, per frame
    for m in motions:
        rows, tip = [], []
        for t in times:
            local = placed.machine.link_poses(m.at(t))
            rows.append([compact(local[n], 5) for n in names])
            tip.append((base @ local[placed.machine.tool_link])[:3, 3])
        poses.append(rows)
        tips.append(np.array(tip))
    stride = max(1, -(-len(json.dumps(poses, separators=(",", ":"))) // poses_budget))
    keep = list(range(0, len(times), stride))
    if keep[-1] != len(times) - 1:
        keep.append(len(times) - 1)  # the end of the motion is always shown (the last step may be shorter)
    poses = [[rows[k] for k in keep] for rows in poses]

    records = []
    for i, (m, tip) in enumerate(zip(motions, tips)):
        label = m.label or "AB"[i]
        path = scene.tool_path(m)
        path = path[::max(1, -(-len(path) // MAX_TRAIL_POINTS))]
        records.append({"label": label, "devices": [{
            "id": m.machine, "centers": pixels(cam, tip[keep]), "tool_trail": pixels(cam, path)}]})
    floor = None
    if site.floor is not None:
        box = pixels(cam, site.floor.corners_m)
        floor = box if all(box) else None
    # What the screen draws from goes to disk and comes back through the part tool; what the model reads stays short
    data_key, data_parts = store_data({
        "poses": poses, "t": [times[k] for k in keep], "series": series,
        "records": [{"devices": r["devices"]} for r in records]})
    return {
        "imagePath": str(still), "width": img["width"], "height": img["height"],
        "parts": -(-len(img["b64"]) // CHUNK),
        "stepS": step_s * stride, "stride": stride, "frames": len(keep),
        "records": [{"label": r["label"], "devices": [{"id": d["id"]} for d in r["devices"]]} for r in records],
        "figures": figures, "site": _header(site),
        "camera": {"intr": {"width": cam.width, "height": cam.height, "fx": float(cam.K[0, 0]),
                            "fy": float(cam.K[1, 1]), "cx": float(cam.K[0, 2]), "cy": float(cam.K[1, 2])},
                   "extr": {"R": [float(x) for x in cam.R.reshape(-1)], "t": [float(x) for x in cam.t]}},
        "floor": floor, "model": model, "dataKey": data_key, "dataParts": data_parts,
        "base": compact(base, 6),
        "placed": {"id": first.machine, "x": float(placed.x), "y": float(placed.y), "yaw": float(placed.yaw)},
    }
