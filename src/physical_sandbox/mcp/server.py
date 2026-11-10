"""MCP server (stdio) with the in-chat screens: fix a site, and compare two motions on it.

Fix a site: an AI agent looks at a photo, proposes the floor's four corners and the fixed objects; the screen
shows them over the photo; the person drags a corner if needed and presses 決定 (Decide); the site file is written.

Compare: the agent calls ``ps_compare`` on a site, a scene and two motions; the screen shows the site's picture
with both motions' machines as see-through shells in the records' colours, the tool-tip trails, a slider and the
numbers. The screen holds no kinematics: the server sends each link's shape once (``ps_model_part``) and, per
frame and record, each link's 4x4 pose in the machine's own frame, with the machine's base transform apart
(the data is built by ``screens.screen_data``).

Nothing is kept in memory. The host may answer the screen's calls from a different start of this server, so
what the screen needs later goes to disk: the photo's display copy in the cache directory
(``PS_CACHE_DIR``, default ``~/.cache/physical-sandbox``), and the proposal (the cut-out objects) and the
agent's first corners and lengths under ``<siteDir>/.work/``.

Tool inputs are plain lists, dicts, strings and numbers: no ``$ref`` and no tuples in their schemas (Claude
Desktop hid tools with those).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from mcp.server.apps import Apps
    from mcp.server.mcpserver import MCPServer
except ImportError as e:  # pragma: no cover
    raise ImportError("The MCP server needs the [mcp] extra: pip install 'physical-sandbox[mcp]'") from e

from .. import screens
from ..fit_screen import (BUNDLE_MARKER, CHUNK, MAX_SIDE, THREE_MARKER, cache_dir, check_corners, check_edges,  # noqa: F401
                          default_corners, ext_apps_bundle, floor_grid, photo, photo_part, widget_html,
                          write_atomic)
from ..floor import solve_floor
from ..objects import ground_px, sam_segmenter
from ..site import Floor, ObjectView, Site, SiteObject, load_site, locate_objects, mask_to_rle, save_site
from ..sources import open_source
from .pack import model_part

URI = "ui://physical-sandbox/fit-floor.html"
COMPARE_URI = "ui://physical-sandbox/compare.html"
MAX_OUTLINE_POINTS = 200
CAMERA = "camera"
KINDS = ("fixed", "target")

_HERE = Path(__file__).parent
POSES_BUDGET = screens.POSES_BUDGET  # tests lower it: ps_compare passes the current value on


def make_segmenter():
    """The segmenter used to cut objects out (SAM 2.1). A module-level function so tests can swap it."""
    return sam_segmenter()


_write_atomic = write_atomic  # the proposal and model caches are written the same way


def _resolved(image_path: str) -> str:
    """The picture's resolved path: proposals and the floor state are matched on it, not on the raw string."""
    return str(Path(image_path).expanduser().resolve())


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


_OBJECT_KEYS = ("name", "kind", "box", "outline", "ground_px", "mask_rle")


def _proposal(site_dir: str, image_path: str, width: int, height: int) -> dict:
    """The stored proposal when it is for this picture (by resolved path and size), else an empty one. Entries
    that lack a field are dropped, so a damaged file never breaks a caller."""
    key = _resolved(image_path)
    d = _read_json(_work(site_dir) / "proposal.json")
    ok = (isinstance(d, dict) and d.get("imageKey") == key and (d.get("width"), d.get("height")) == (width, height)
          and isinstance(d.get("objects"), list))
    objects = [o for o in d["objects"] if isinstance(o, dict) and all(k in o for k in _OBJECT_KEYS)] if ok else []
    return {"imagePath": image_path, "imageKey": key, "width": width, "height": height, "objects": objects}


def _fit(site_dir: str, image_path: str) -> dict:
    """The floor state ``ps_fit_floor`` stored for this picture (by resolved path), or {}."""
    d = _read_json(_work(site_dir) / "fit.json")
    return d if isinstance(d, dict) and d.get("imageKey") == _resolved(image_path) else {}


def _write_fit(site_dir: str, image_path: str, edges, corners: list[dict], camera: str) -> None:
    _write_json(_work(site_dir) / "fit.json", {"imagePath": image_path, "imageKey": _resolved(image_path),
                                               "edgesM": list(edges), "corners": corners, "camera": camera})


def _camera_name(value, fit: dict | None = None) -> str:
    """The site camera's name: ``value`` when given, else the one ``ps_fit_floor`` stored, else ``camera``."""
    if value is None:
        stored = (fit or {}).get("camera")
        value = stored if isinstance(stored, str) and stored else CAMERA
    if not isinstance(value, str) or not value.strip() or len(value) > 64 or not value.isprintable():
        raise ValueError(f"camera must be a short name, got {value!r}")
    return value.strip()


def _model_object(o: dict) -> dict:
    """What the model reads of a stored object: its name, kind and box (the outline goes to the screen only)."""
    return {k: o[k] for k in ("name", "kind", "box")}


def _with_objects(state: dict, objects: list[dict]) -> dict:
    """The state with the objects as the model reads them, and the key to the screen's outlines (or None)."""
    state["objects"] = [_model_object(o) for o in objects]
    if objects:  # outlines are up to 200 points each: they reach the screen through ps_objects_part
        key, parts = screens.store_data({"objects": [{k: o[k] for k in ("name", "outline", "ground_px")} for o in objects]})
    else:
        key, parts = None, 0
    state["objectsKey"], state["objectsParts"] = key, parts
    return state


_corners, _edges, _default_corners = check_corners, check_edges, default_corners


def _json(d: dict) -> str:
    return json.dumps(d)


def _state(image_path: str, img: dict, edges_m, corners, site_dir: str, objects: list[dict]) -> dict:
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


def ps_catalog_search(q: str = "", category: str | None = None, kind: str | None = None) -> str:
    from .. import catalog
    from ..push import kept_token

    return _json({"items": catalog.search(q, category=category, kind=kind, token=kept_token(catalog._base()))})


def ps_catalog_pull(ref: str) -> str:
    from .. import catalog
    from ..errors import CatalogError
    from ..push import kept_token

    d = catalog.pull(ref, token=kept_token(catalog._base()))
    try:
        urdf = str(catalog.urdf_path(d))
    except CatalogError:  # an item without a machine (a recording, say)
        urdf = None
    return _json({"dir": str(d), "urdf": urdf})


def ps_catalog_push(ref: str, path: str, private: bool = True) -> str:
    from ..push import PushError, push

    said: list[str] = []
    try:
        # never a login from here: the code to enter would not reach the person
        result = push(ref, Path(path).expanduser(), private=private, log=said.append, interactive=False)
    except PushError as e:
        if e.code != "signin":
            raise
        return _json({"error": "signin", "message": "not signed in to PS.com: run `psbx login` in a terminal once, "
                                                    "then ask again"})
    return _json({"ref": result.ref, "bytes": result.bytes, "companions": result.companions,
                  "private": result.visibility == "private", "visibility": result.visibility,
                  "log": said})


def ps_machine_check(path: str) -> str:
    from ..machine_tools import check

    result = check(Path(path).expanduser())
    return _json({"ok": result.ok, "items": [{"ok": ok, "text": text} for ok, text in result.items]})


@apps.tool(resource_uri=URI, description=(
    "Open a screen inside the chat to fix the floor on a photo: the photo with four orange corners the person "
    "can drag, and the floor solved from them drawn as a 1 m grid. imagePath is a photo file; edgesM the floor's "
    "two edge lengths in metres (1->2, 2->3); corners optional pixels {u, v}, four, going round; siteDir the "
    "folder where the site file will be written; camera the camera's name in the site (default 'camera'; "
    "ps_compare's camera). Objects already proposed for this photo are drawn too."))
def ps_fit_floor(imagePath: str, edgesM: list[float], siteDir: str, corners: list[dict] | None = None,
                 camera: str | None = None) -> str:
    edges = _edges(edgesM)
    camera = _camera_name(camera)
    img = photo(imagePath)
    w, h = img["width"], img["height"]
    corners = [{"u": u, "v": v} for u, v in _corners(corners)] if corners else _default_corners(w, h)
    _write_fit(siteDir, imagePath, edges, corners, camera)
    objects = _proposal(siteDir, imagePath, w, h)["objects"]
    return _json(_with_objects(_state(imagePath, img, edges, corners, siteDir, []), objects))


@apps.tool(resource_uri=URI, visibility=["app"], description="One part of the photo (screen only).")
def ps_spike_image(imagePath: str, part: int) -> str:
    return _json(photo_part(imagePath, part))


@apps.tool(resource_uri=URI, visibility=["app"], description=(
    "Solve the camera from four floor corners and give a 1 m grid in the photo's pixels (screen only)."))
def ps_floor_solve(width: float, height: float, corners: list[dict], edgesM: list[float]) -> str:
    return _json(floor_grid(width, height, corners, edgesM))


@apps.tool(resource_uri=URI, description=(
    "Cut out fixed objects the agent chose, from rough boxes, and show their outlines and names in the same "
    "screen (the result gives each object's name, kind and box; the outlines go to the screen only). boxes: [{name, kind ('fixed' or 'target'), x0, y0, x1, y1}] in the photo's pixels. The cut-outs "
    "are kept under siteDir/.work; a second call adds to them (the same name is replaced). The floor corners "
    "and lengths come from the earlier ps_fit_floor on this siteDir, or from edgesM and corners given here; so "
    "does camera (the camera's name in the site, default 'camera'). "
    "The result lists the names SAM found no mask for in 'missing'."))
def ps_objects_propose(imagePath: str, boxes: list[dict], siteDir: str, edgesM: list[float] | None = None,
                       corners: list[dict] | None = None, camera: str | None = None) -> str:
    fit = _fit(siteDir, imagePath)
    camera = _camera_name(camera, fit)
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
    img = photo(imagePath, frame=image)  # the display copy from this same frame
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
    shown = [{"u": u, "v": v} for u, v in pts]
    _write_fit(siteDir, imagePath, edges, shown, camera)  # so the screen's decide finds the floor, whoever opened it
    state = _with_objects(_state(imagePath, img, edges, shown, siteDir, []), proposal["objects"])
    state["missing"] = missing
    return _json(state)


@apps.tool(resource_uri=URI, visibility=["app"], description=(
    "Keep the decided site: write siteDir/site.json from the floor and the objects the person kept (screen only). "
    "camera is the camera's name in the site (default: the one ps_fit_floor was given, else 'camera')."))
def ps_site_decide(siteDir: str, imagePath: str, corners: list[dict], edgesM: list[float], objects: list[dict],
                   measured: bool = False, camera: str | None = None) -> str:
    try:
        px, edges = _corners(corners), _edges(edgesM)
        fit = _fit(siteDir, imagePath)
        if not fit:
            return _json({"ok": False, "reason": "this picture was not opened for this site: use ps_fit_floor first"})
        cam_name = _camera_name(camera, fit)
        if not isinstance(objects, list) or not all(isinstance(o, dict) for o in objects):
            return _json({"ok": False, "reason": "objects must be a list of {name, kind}"})
        names = [str(o.get("name", "")) for o in objects]
        if len(set(names)) != len(names):
            return _json({"ok": False, "reason": f"an object name is used twice: {sorted(names)}"})
        with open_source(Path(imagePath).expanduser()) as src:
            w, h = src.size
        sol = solve_floor(w, h, px, edges)
        if sol.camera is None:
            return _json({"ok": False, "reason": sol.reason})
        stored = {o["name"]: o for o in _proposal(siteDir, imagePath, w, h)["objects"]}
        site_objects = []
        for o, name in zip(objects, names):
            if name not in stored:
                return _json({"ok": False, "reason": f"no cut-out for {name!r}: run ps_objects_propose for it"})
            s = stored[name]
            kind = str(o.get("kind", s["kind"]))
            if kind not in KINDS:
                return _json({"ok": False, "reason": f"bad kind {kind!r} for {name!r}"})
            view = ObjectView(tuple(int(x) for x in s["box"]), s["mask_rle"],
                              [(float(u), float(v)) for u, v in s["ground_px"]])
            site_objects.append(SiteObject(name, kind, {cam_name: view}))
        floor = Floor(cam_name, px, edges, sol.corners_m)
        site = Site(
            id=Path(siteDir).expanduser().resolve().name or "site",
            cameras={cam_name: sol.camera}, floor=floor, objects=site_objects,
            residual_px={cam_name: float(sol.residual_px)}, confirmed_by_person=True,
            dimensions="measured" if measured else "estimated",
            history=[{"at": datetime.now(timezone.utc).isoformat(), "what": "floor fixed and confirmed by a person",
                      "image": Path(imagePath).name, "mirrored": bool(sol.mirrored), "objects": names}])
        site = locate_objects(site, cam_name)
        path = Path(siteDir).expanduser() / "site.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_site(site, path)
    except (ValueError, OSError, KeyError, TypeError, AttributeError) as e:
        return _json({"ok": False, "reason": f"{type(e).__name__}: {e}"})
    return _json({"ok": True, "sitePath": str(path), "residualPx": round(float(sol.residual_px), 2),
                  "mirrored": bool(sol.mirrored), "camera": cam_name})


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
    cam_name = site.floor.camera if site.floor is not None else next(iter(site.residual_px), CAMERA)
    return _json({"decided": site.confirmed_by_person, "residualPx": site.residual_px.get(cam_name),
                  "objects": [o.name for o in site.objects], "dimensions": site.dimensions,
                  "confirmedByPerson": site.confirmed_by_person})


# ---- place a machine on a site ----

def _finite(name: str, value) -> float:
    """``value`` as a finite number: a bool, text, NaN or infinity is bad input."""
    import math

    from ..errors import InputError

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise InputError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _scene_path_in(site_dir: str, scene_path: str | None) -> Path:
    """Where ps_scene_place writes: siteDir/scene.json, or the .json file scenePath names inside siteDir (a relative
    one is taken in siteDir). It replaces the file there, so a path anywhere else is refused: what the agent
    passes may come from a page or a file it read, and must not overwrite the person's other files."""
    from ..errors import InputError

    folder = Path(site_dir).expanduser().resolve()
    if not scene_path:
        return folder / "scene.json"
    given = Path(scene_path).expanduser()
    path = (given if given.is_absolute() else folder / given).resolve()
    if not path.is_relative_to(folder) or path.suffix.lower() != ".json":
        raise InputError(f"scenePath {scene_path!r}: the scene file has to be a .json file inside siteDir ({folder})")
    return path


def ps_scene_place(siteDir: str, machine: str, x: float, y: float, yaw: float, scenePath: str | None = None,
                   urdf: str | None = None) -> str:
    from .. import catalog
    from ..cli import _reading
    from ..errors import InputError
    from ..machine import load_machine
    from ..push import kept_token
    from ..scene import PlacedMachine, save_placement

    x, y, yaw = _finite("x", x), _finite("y", y), _finite("yaw", yaw)
    site = _reading(f"siteDir {siteDir!r}", load_site, Path(siteDir).expanduser() / "site.json")
    if urdf is not None:  # a machine file on this computer: machine is the id to give it
        mid, path = machine, Path(urdf).expanduser().resolve()
        if not isinstance(mid, str) or not mid.strip() or len(mid) > 64 or not mid.isprintable():
            raise InputError(f"machine must be a short name, got {machine!r}")
        if path.suffix.lower() != ".urdf" or not path.is_file():
            raise InputError(f"urdf {urdf!r}: no such .urdf file")
    else:  # a catalog ref: the folder it was pulled to (no request when it is already here)
        mid = catalog.parse_ref(machine)[1]
        item = catalog.cached(machine) or catalog.pull(machine, token=kept_token(catalog._base()))
        path = catalog.urdf_path(item)  # CatalogError for an item without a machine
    fit = _read_json(_work(siteDir) / "fit.json")
    picture = fit.get("imagePath") if isinstance(fit, dict) else None  # the picture the floor was fixed on
    placed = PlacedMachine(load_machine(path), str(path), x, y, yaw, confirmed_by_person=False)
    scene_path = _scene_path_in(siteDir, scenePath)
    save_placement(scene_path, site.id, {mid: placed}, picture=picture if isinstance(picture, str) else None)
    return _json({"scenePath": str(scene_path), "machine": mid, "x": x, "y": y, "yaw": yaw})


# ---- compare two motions ----

@apps.tool(resource_uri=COMPARE_URI, description=(
    "Open a screen inside the chat to compare two motions of a machine on a site: the site's picture with the "
    "machine of each motion as a see-through shell in its own colour (A blue, B orange), the tool-tip paths, a "
    "slider and a play button, and the table of numbers (the gaps, with the times they happen; click a time to "
    "jump there). Leave b out to look at one motion alone: the same screen without the pair's gaps (it shows "
    "the duration and the grasps). A record made for another stand of the same robot is read onto the "
    "scene's only machine, and 'notes' in the answer says so. siteDir is the folder with site.json; scenePath the "
    "scene file (where the machine stands); "
    "camera the site camera's name; a and b the motions: a recording file[#label] or "
    "lerobot:<dataset_dir>#<episode>. labelA and labelB name them on the screen. video is the picture to draw on "
    "(a still or a recording); without it, A's own recording when A is a LeRobot episode, else the picture the "
    "scene file names, else it asks for one. videoCamera is the camera of A's LeRobot dataset to take that "
    "recording from (default: the same name as camera; a site fixed in the chat screen calls its camera "
    "'camera', so give the dataset's camera name here, such as 'side'). gripThreshold is the gripper joint value "
    "(radians) a grasp closes through (default: the middle of the range the gripper went through in the two "
    "motions). What the numbers cannot tell yet is listed with its reason."))
def ps_compare(siteDir: str, scenePath: str, camera: str, a: str, b: str | None = None,
               labelA: str | None = None, labelB: str | None = None, video: str | None = None,
               videoCamera: str | None = None, gripThreshold: float | None = None) -> str:
    from ..cli import _check_motion, _check_motions, _lerobot_picture, _load_motion, _own_episode_span, _reading
    from ..errors import InputError
    from ..scene import load_scene

    site = _reading(f"siteDir {siteDir!r}", load_site, Path(siteDir).expanduser() / "site.json")
    if camera not in site.cameras:
        raise InputError(f"the site has no camera {camera!r} (it has {', '.join(sorted(site.cameras))})")
    scene = _reading(f"scenePath {scenePath!r}", load_scene, Path(scenePath).expanduser(), site)
    notes: list[str] = []  # what the read-over of a one-device record assumed
    ma, a_lerobot = _load_motion(a, scene, notes)
    if b is None:  # one record alone: no pair to check, only the motion against the placed machine
        mb = None
        _check_motion(scene, ma)
    else:
        mb, _ = _load_motion(b, scene, notes)
        _check_motions(scene, ma, mb)

    start_s = 0.0
    if video:
        spec = video
        if a_lerobot is not None:  # A's own episode file given by name: the still is at A's episode's start
            start_s = (_own_episode_span(a_lerobot, video, videoCamera) or (0.0, None))[0]
    elif a_lerobot is not None:
        spec, start_s, _ = _lerobot_picture(a_lerobot, videoCamera or camera, "videoCamera", bool(videoCamera))
    else:
        spec = screens.scene_picture(scenePath)
        if spec is None:
            raise InputError("give video: the picture to draw on (a still or a recording). The site file keeps no "
                             "picture, A is not a LeRobot episode whose recording could be used, and the scene "
                             "file names none")
    spec = screens.local_video(spec)
    with _reading(f"video {str(spec)!r}", open_source, spec) as source:
        out = screens.screen_data(site, scene, camera, [ma] if mb is None else [ma, mb], source, start_s=start_s,
                                  grip_threshold=gripThreshold,
                                  labels=[labelA] if mb is None else [labelA, labelB],
                                  poses_budget=POSES_BUDGET)
    out["notes"] = notes
    return json.dumps(out, separators=(",", ":"))


@apps.tool(resource_uri=COMPARE_URI, visibility=["app"], description=(
    "One part of the comparison's data (poses, times, pixel paths) as JSON text (screen only). key is ps_compare's "
    "dataKey."))
def ps_compare_part(key: str, part: int) -> str:
    return screens.data_part(key, part, "ps_compare")


@apps.tool(resource_uri=URI, visibility=["app"], description=(
    "One part of the proposed objects' outlines and ground points as JSON text (screen only). key is "
    "ps_fit_floor's or ps_objects_propose's objectsKey."))
def ps_objects_part(key: str, part: int) -> str:
    return screens.data_part(key, part, "ps_objects_propose")


@apps.tool(resource_uri=COMPARE_URI, visibility=["app"], description=(
    "One part of a machine's thinned shapes, as base64 text (screen only). key comes from ps_compare's model."))
def ps_model_part(key: str, part: int) -> str:
    return _json({"part": part, "data": model_part(key, part, CHUNK)})


def _widget(name: str, three: bool = False) -> str:
    return widget_html(name, ext_apps_bundle(), three)


apps.add_html_resource(URI, _widget("fit-floor.html"))
apps.add_html_resource(COMPARE_URI, _widget("compare.html", three=True))
server = MCPServer("physical-sandbox", extensions=[apps])
server.tool(description="Search the PS.com catalog of machines and other items by text, category or kind.")(
    ps_catalog_search)
server.tool(description="Download a catalog item (handle/name[:version]) into the local cache; gives its folder "
                        "and its URDF file.")(ps_catalog_pull)
server.tool(description="Upload a machine folder, an object (one mesh and object.json) or a record (.json) to the "
                        "PS.com catalog as handle/name[:version], private unless private is false. A record's "
                        "machines that the catalog does not have go up with it, privately.")(ps_catalog_push)
server.tool(description="Check a machine folder (URDF, meshes, product.json, profile.json) or an object (one mesh "
                        "and object.json) before pushing it to the catalog: ok, and one line per check.")(
    ps_machine_check)
server.tool(description="Whether the site in siteDir is decided (confirmed by the person), with its residual in "
                        "pixels, its object names and whether its lengths are measured or estimated.")(ps_site_status)
server.tool(description="Place a machine on a site, in the site's floor frame in metres, and write the scene file "
                        "(default siteDir/scene.json; ps_compare's scenePath; a scenePath has to be a .json file inside siteDir). "
                        "machine is a catalog ref "
                        "(handle/name[:version]; pulled first when it is not on this computer) or, with urdf (a "
                        "machine file on this computer), the id to give that machine. x and y are the base's "
                        "position on the floor in metres, yaw its turn in radians. It replaces any machine the "
                        "scene file already places (one machine per scene). It answers the scene file, "
                        "the machine's id and where it stands. Nobody has looked at the placement yet: say so "
                        "and ask the person to check it in the compare screen or in `psbx stage`, where they "
                        "confirm it.")(ps_scene_place)


def main() -> None:
    # While serving, the SDK (mcp >= 2.3) keeps the protocol on a private duplicate of fd 1 and points fd 1 at
    # stderr, so what a library prints or logs to stdout (ultralytics does, on its first run) cannot corrupt it.
    # tests/test_mcp_site.py checks this over raw pipes, so an SDK that stops doing it is noticed.
    server.run("stdio")


if __name__ == "__main__":
    main()
