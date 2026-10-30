"""MCP server (stdio) with the in-chat screen to fix a site.

An AI agent looks at a photo, proposes the floor's four corners and the fixed objects; the screen shows them
over the photo; the person drags a corner if needed and presses 決定 (Decide); the site file is written.

Nothing is kept in memory. The host may answer the screen's calls from a different start of this server, so
what the screen needs later goes to disk: the photo's display copy in the cache directory
(``PS_CACHE_DIR``, default ``~/.cache/physical-sandbox``), and the proposal (the cut-out objects) and the
agent's first corners and lengths under ``<siteDir>/.work/``.

Tool inputs are plain lists, dicts, strings and numbers: no ``$ref`` and no tuples in their schemas (Claude
Desktop hid tools with those).
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from mcp.server.apps import Apps
    from mcp.server.mcpserver import MCPServer
except ImportError as e:  # pragma: no cover
    raise ImportError("The MCP server needs the [mcp] extra: pip install 'physical-sandbox[mcp]'") from e

from ..floor import solve_floor
from ..objects import ground_px, sam_segmenter
from ..site import Floor, ObjectView, Site, SiteObject, load_site, locate_objects, mask_to_rle, save_site
from ..sources import open_source

URI = "ui://physical-sandbox/fit-floor.html"
CHUNK = 20_000  # characters of base64 per photo part: each result stays under the host's size cap
MAX_SIDE = 1280  # the display copy's longer side
MAX_OUTLINE_POINTS = 200
CAMERA = "camera"
KINDS = ("fixed", "target")

_HERE = Path(__file__).parent
BUNDLE_MARKER = "/*__EXT_APPS_BUNDLE__*/"


def make_segmenter():
    """The segmenter used to cut objects out (SAM 2.1). A module-level function so tests can swap it."""
    return sam_segmenter()


# ---- the photo's display copy, on disk ----

def _cache_dir() -> Path:
    d = Path(os.environ.get("PS_CACHE_DIR", "~/.cache/physical-sandbox")).expanduser() / "photos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # two starts of the server may race
    tmp.write_bytes(data)
    os.replace(tmp, path)


def photo(image_path: str) -> dict:
    """The photo's size and a JPEG display copy as base64: ``{width, height, b64}``. The copy is made once and
    kept in the cache directory, so a different start of the server finds it. The size is that of the frame the
    calculations use (EXIF rotation applied, as ``open_source`` reads it)."""
    p = Path(image_path).expanduser()
    if not p.is_file():
        raise ValueError(f"cannot find the picture {image_path!r}")
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()
    cache = _cache_dir()
    jpg, meta = cache / f"{key}.jpg", cache / f"{key}.json"
    if not (jpg.exists() and meta.exists()):
        with open_source(p) as src:
            img = src.frame(0)
        h, w = img.shape[:2]
        k = min(1.0, MAX_SIDE / max(w, h))
        small = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise ValueError(f"cannot make a display copy of {image_path!r}")
        _write_atomic(jpg, buf.tobytes())
        _write_atomic(meta, json.dumps({"width": w, "height": h}).encode())  # the meta last: it marks "complete"
    return {**json.loads(meta.read_text()), "b64": base64.b64encode(jpg.read_bytes()).decode()}


# ---- working files under the site directory ----

def _work(site_dir: str) -> Path:
    return Path(site_dir).expanduser() / ".work"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps(data).encode())


def _proposal(site_dir: str, image_path: str, width: int, height: int) -> dict:
    """The stored proposal when it is for this picture, else an empty one."""
    d = _read_json(_work(site_dir) / "proposal.json")
    if isinstance(d, dict) and d.get("imagePath") == image_path and (d.get("width"), d.get("height")) == (width, height):
        return d
    return {"imagePath": image_path, "width": width, "height": height, "objects": []}


def _screen_object(o: dict) -> dict:
    """What the screen needs of a stored object (not the mask)."""
    return {k: o[k] for k in ("name", "kind", "outline", "ground_px", "box")}


# ---- checking what comes in ----

def _corners(value) -> list[tuple[float, float]]:
    try:
        pts = [(float(c["u"]), float(c["v"])) for c in value]
    except (TypeError, KeyError, ValueError) as e:
        raise ValueError("corners must be four {u, v} pixels") from e
    if len(pts) != 4 or not all(math.isfinite(x) for p in pts for x in p):
        raise ValueError("corners must be four {u, v} pixels")
    return pts


def _edges(value) -> tuple[float, float]:
    try:
        a, b = (float(x) for x in value)
    except (TypeError, ValueError) as e:
        raise ValueError("edgesM must be two lengths in metres") from e
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError("edgesM must be two positive lengths in metres")
    return a, b


def _default_corners(w: int, h: int) -> list[dict]:
    return [{"u": w * 0.1, "v": h * 0.9}, {"u": w * 0.9, "v": h * 0.9}, {"u": w * 0.7, "v": h * 0.6},
            {"u": w * 0.3, "v": h * 0.6}]


def _json(d: dict) -> str:
    return json.dumps(d)


def _state(image_path: str, edges_m, corners, site_dir: str, objects: list[dict]) -> dict:
    img = photo(image_path)
    w, h = img["width"], img["height"]
    return {"imagePath": image_path, "width": w, "height": h, "parts": -(-len(img["b64"]) // CHUNK),
            "edgesM": list(edges_m), "corners": corners, "siteDir": site_dir, "objects": objects}


def _outline(mask: np.ndarray) -> list[list[float]]:
    """The biggest outer contour of a mask as plain float pairs, simplified to at most 200 points."""
    contours, _ = cv2.findContours(np.asarray(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    eps = 0.5
    pts = contour
    while len(pts) > MAX_OUTLINE_POINTS:
        pts = cv2.approxPolyDP(contour, eps, True)
        eps *= 1.5
    return [[float(x), float(y)] for x, y in pts.reshape(-1, 2)]


# ---- the tools ----

apps = Apps()


@apps.tool(resource_uri=URI, description=(
    "Open a screen inside the chat to fix the floor on a photo: the photo with four orange corners the person "
    "can drag, and the floor solved from them drawn as a 1 m grid. imagePath is a photo file; edgesM the floor's "
    "two edge lengths in metres (1->2, 2->3); corners optional pixels {u, v}, four, going round; siteDir the "
    "folder where the site file will be written. Objects already proposed for this photo are drawn too."))
def ps_fit_floor(imagePath: str, edgesM: list[float], siteDir: str, corners: list[dict] | None = None) -> str:
    edges = _edges(edgesM)
    img = photo(imagePath)
    w, h = img["width"], img["height"]
    corners = [{"u": u, "v": v} for u, v in _corners(corners)] if corners else _default_corners(w, h)
    _write_json(_work(siteDir) / "fit.json", {"imagePath": imagePath, "edgesM": list(edges), "corners": corners})
    objects = [_screen_object(o) for o in _proposal(siteDir, imagePath, w, h)["objects"]]
    return _json(_state(imagePath, edges, corners, siteDir, objects))


@apps.tool(resource_uri=URI, visibility=["app"], description="One part of the photo (screen only).")
def ps_spike_image(imagePath: str, part: int) -> str:
    b64 = photo(imagePath)["b64"]
    return _json({"part": part, "data": b64[part * CHUNK:(part + 1) * CHUNK]})


@apps.tool(resource_uri=URI, visibility=["app"], description=(
    "Solve the camera from four floor corners and give a 1 m grid in the photo's pixels (screen only)."))
def ps_floor_solve(width: float, height: float, corners: list[dict], edgesM: list[float]) -> str:
    try:
        px, edges = _corners(corners), _edges(edgesM)
        w, h = int(round(width)), int(round(height))
        if w <= 0 or h <= 0:
            raise ValueError("width and height must be positive")
    except ValueError as e:
        return _json({"ok": False, "reason": f"bad-input: {e}"})
    sol = solve_floor(w, h, px, edges)
    if sol.camera is None:
        return _json({"ok": False, "reason": sol.reason})
    a = float(sol.corners_m[1][0])
    b = float(sol.corners_m[2][1])  # negative when the floor was mirrored
    ts = (0, 0.25, 0.5, 0.75, 1)
    lines = [[[m, t * b, 0.0] for t in ts] for m in np.arange(0, abs(a) + 1e-9, 1.0)]
    lines += [[[t * a, m * math.copysign(1.0, b), 0.0] for t in ts] for m in np.arange(0, abs(b) + 1e-9, 1.0)]
    out = []
    for line in lines:
        pix = sol.camera.project(np.array(line, float))
        out.append([None if not np.isfinite(p).all() else [round(float(p[0]), 1), round(float(p[1]), 1)] for p in pix])
    fov = 2 * math.degrees(math.atan(w / 2 / float(sol.camera.K[0, 0])))
    return _json({"ok": True, "lines": out, "residualPx": round(sol.residual_px, 2), "fovDeg": round(fov),
                  "mirrored": bool(sol.mirrored)})


@apps.tool(resource_uri=URI, description=(
    "Cut out fixed objects the agent chose, from rough boxes, and show their outlines and names in the same "
    "screen. boxes: [{name, kind ('fixed' or 'target'), x0, y0, x1, y1}] in the photo's pixels. The cut-outs "
    "are kept under siteDir/.work; a second call adds to them (the same name is replaced). The floor corners "
    "and lengths come from the earlier ps_fit_floor on this siteDir, or from edgesM and corners given here. "
    "The result lists the names SAM found no mask for in 'missing'."))
def ps_objects_propose(imagePath: str, boxes: list[dict], siteDir: str, edgesM: list[float] | None = None,
                       corners: list[dict] | None = None) -> str:
    fit = _read_json(_work(siteDir) / "fit.json")
    fit = fit if isinstance(fit, dict) and fit.get("imagePath") == imagePath else {}
    edges_value = edgesM if edgesM is not None else fit.get("edgesM")
    if edges_value is None:
        raise ValueError("the floor's edge lengths are not known: call ps_fit_floor first, or pass edgesM")
    edges = _edges(edges_value)

    asked: dict[str, tuple[str, tuple[int, int, int, int]]] = {}
    for b in boxes:
        try:
            name = str(b["name"]).strip()
            kind = str(b.get("kind", "fixed"))
            box = tuple(int(round(float(b[k]))) for k in ("x0", "y0", "x1", "y1"))
        except (TypeError, KeyError, ValueError) as e:
            raise ValueError("each box needs name, x0, y0, x1, y1 (and kind 'fixed' or 'target')") from e
        if not name or kind not in KINDS or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"bad box {b!r}: a name, kind 'fixed' or 'target', and x1 > x0, y1 > y0")
        if name in asked:
            raise ValueError(f"the name {name!r} is used twice")
        asked[name] = (kind, box)
    if not asked:
        raise ValueError("boxes is empty")

    with open_source(Path(imagePath).expanduser()) as src:
        image = src.frame(0)
    h, w = image.shape[:2]
    masks = make_segmenter()(image, {n: box for n, (_, box) in asked.items()})

    proposal = _proposal(siteDir, imagePath, w, h)
    kept = {o["name"]: o for o in proposal["objects"]}
    missing = []
    for name, (kind, box) in asked.items():
        mask = masks.get(name)
        if mask is None or not np.asarray(mask, bool).any():
            missing.append(name)
            continue
        mask = np.asarray(mask, bool)
        kept[name] = {"name": name, "kind": kind, "box": list(box), "outline": _outline(mask),
                      "ground_px": [[float(u), float(v)] for u, v in ground_px(mask)],
                      "mask_rle": mask_to_rle(mask)}
    proposal["objects"] = list(kept.values())
    _write_json(_work(siteDir) / "proposal.json", proposal)

    pts = _corners(corners if corners is not None else fit.get("corners") or _default_corners(w, h))
    state = _state(imagePath, edges, [{"u": u, "v": v} for u, v in pts], siteDir,
                   [_screen_object(o) for o in proposal["objects"]])
    state["missing"] = missing
    return _json(state)


@apps.tool(resource_uri=URI, visibility=["app"], description=(
    "Keep the decided site: write siteDir/site.json from the floor and the objects the person kept (screen only)."))
def ps_site_decide(siteDir: str, imagePath: str, corners: list[dict], edgesM: list[float], objects: list[dict],
                   measured: bool = False) -> str:
    try:
        px, edges = _corners(corners), _edges(edgesM)
        with open_source(Path(imagePath).expanduser()) as src:
            w, h = src.size
        sol = solve_floor(w, h, px, edges)
        if sol.camera is None:
            return _json({"ok": False, "reason": sol.reason})
        proposal = _proposal(siteDir, imagePath, w, h)
        stored = {o["name"]: o for o in proposal["objects"]}
        site_objects = []
        for o in objects:
            name = str(o.get("name", ""))
            if name not in stored:
                return _json({"ok": False, "reason": f"no cut-out for {name!r}: run ps_objects_propose for it"})
            s = stored[name]
            kind = str(o.get("kind", s["kind"]))
            if kind not in KINDS:
                return _json({"ok": False, "reason": f"bad kind {kind!r} for {name!r}"})
            view = ObjectView(tuple(int(x) for x in s["box"]), s["mask_rle"],
                              [(float(u), float(v)) for u, v in s["ground_px"]])
            site_objects.append(SiteObject(name, kind, {CAMERA: view}))
        floor = Floor(CAMERA, px, edges, sol.corners_m)
        site = Site(
            id=Path(siteDir).expanduser().resolve().name or "site",
            cameras={CAMERA: sol.camera}, floor=floor, objects=site_objects,
            residual_px={CAMERA: float(sol.residual_px)}, confirmed_by_person=True,
            dimensions="measured" if measured else "estimated",
            history=[{"at": datetime.now(timezone.utc).isoformat(), "what": "floor fixed and confirmed by a person",
                      "image": imagePath, "mirrored": bool(sol.mirrored),
                      "objects": [o.name for o in site_objects]}])
        site = locate_objects(site, CAMERA)
        path = Path(siteDir).expanduser() / "site.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_site(site, path)
    except (ValueError, OSError) as e:
        return _json({"ok": False, "reason": str(e)})
    return _json({"ok": True, "sitePath": str(path), "residualPx": round(float(sol.residual_px), 2),
                  "mirrored": bool(sol.mirrored)})


def ps_site_status(siteDir: str) -> str:
    """Whether the site in siteDir is decided, with its residual (pixels), object names and how its lengths
    were obtained."""
    path = Path(siteDir).expanduser() / "site.json"
    empty = {"decided": False, "residualPx": None, "objects": [], "dimensions": None, "confirmedByPerson": False}
    if not path.is_file():
        return _json(empty)
    try:
        site = load_site(path)
    except (ValueError, KeyError, OSError) as e:
        return _json({**empty, "error": str(e)})
    return _json({"decided": site.confirmed_by_person, "residualPx": site.residual_px.get(CAMERA),
                  "objects": [o.name for o in site.objects], "dimensions": site.dimensions,
                  "confirmedByPerson": site.confirmed_by_person})


def _widget() -> str:
    bundle = (_HERE / "widgets/vendor/ext-apps-app-with-deps.js").read_text()
    return (_HERE / "widgets/fit-floor.html").read_text().replace(BUNDLE_MARKER, bundle)


apps.add_html_resource(URI, _widget())
server = MCPServer("physical-sandbox", extensions=[apps])
server.tool(description="Whether the site in siteDir is decided (confirmed by the person), with its residual in "
                        "pixels, its object names and whether its lengths are measured or estimated.")(ps_site_status)


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
