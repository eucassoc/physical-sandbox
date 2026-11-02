"""The screen that fixes a floor on a photo, shared by the chat and the browser.

In the chat (the MCP server, ``psbx mcp``) an AI agent opens it with its proposal; in the browser (``psbx site
fit`` without ``--corners``) the person opens it on their own photo. Both show the same widget,
``mcp/widgets/fit-floor.html``: the photo, four orange corners to drag, the two edge lengths, and the solved
floor drawn as a grid. This module holds what both need: the photo's display copy, the input checks, the grid,
the widget's HTML, and a small HTTP server on 127.0.0.1 that stands in for the chat, with a bridge of its own in
place of the MCP Apps one. None of it needs the ``[mcp]`` extra.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import secrets
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .floor import solve_floor
from .site import Floor, Site, load_site, save_site
from .sources import open_source

CHUNK = 20_000  # characters of base64 per photo part: each result stays under the chat host's size cap
MAX_SIDE = 1280  # the display copy's longer side
A4_M = (0.21, 0.297)  # the lengths the browser screen starts with: a sheet of A4 paper on the floor
WIDGETS = Path(__file__).parent / "mcp" / "widgets"
BUNDLE_MARKER = "/*__EXT_APPS_BUNDLE__*/"
THREE_MARKER = "/*__THREE__*/"
LIB_MARKER = re.compile(r"/\*__LIB:([a-z]+)__\*/")


# ---- the photo's display copy, on disk ----

def cache_dir() -> Path:
    d = Path(os.environ.get("PS_CACHE_DIR", "~/.cache/physical-sandbox")).expanduser() / "photos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")  # two starts of the server may race
    tmp.write_bytes(data)
    os.replace(tmp, path)


def photo(image_path: str, frame: np.ndarray | None = None, make: bool = True) -> dict:
    """The photo's size and a JPEG display copy as base64: ``{width, height, b64}``. The copy is made once and
    kept in the cache directory, so a different start of the server finds it. The size is that of the frame the
    calculations use (EXIF rotation applied, as ``open_source`` reads it). ``frame`` is that frame when the
    caller has already decoded it (so it is not decoded twice); with ``make=False`` an image that has no copy
    yet is refused instead of decoded (the screen's own calls only fetch what a tool already opened)."""
    p = Path(image_path).expanduser()
    if not p.is_file():
        raise ValueError(f"cannot find the picture {image_path!r}")
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}:{st.st_mtime_ns}:{st.st_size}".encode()).hexdigest()
    cache = cache_dir()
    jpg, meta = cache / f"{key}.jpg", cache / f"{key}.json"
    if not (jpg.exists() and meta.exists()):
        if not make:
            raise ValueError(f"{image_path!r} has not been opened: use ps_fit_floor on it first")
        if frame is None:
            with open_source(p) as src:
                frame = src.frame(0)
        img = frame
        h, w = img.shape[:2]
        k = min(1.0, MAX_SIDE / max(w, h))
        small = cv2.resize(img, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise ValueError(f"cannot make a display copy of {image_path!r}")
        write_atomic(jpg, buf.tobytes())
        write_atomic(meta, json.dumps({"width": w, "height": h}).encode())  # the meta last: it marks "complete"
    return {**json.loads(meta.read_text()), "b64": base64.b64encode(jpg.read_bytes()).decode()}


def photo_part(image_path: str, part: int) -> dict:
    """One part of an already opened photo's display copy."""
    if part < 0:
        raise ValueError("part must be >= 0")
    b64 = photo(image_path, make=False)["b64"]
    return {"part": part, "data": b64[part * CHUNK:(part + 1) * CHUNK]}


# ---- checking what comes in ----

def check_corners(value) -> list[tuple[float, float]]:
    try:
        pts = [(float(c["u"]), float(c["v"])) for c in value]
    except (TypeError, KeyError, ValueError) as e:
        raise ValueError("corners must be four {u, v} pixels") from e
    if len(pts) != 4 or not all(math.isfinite(x) for p in pts for x in p):
        raise ValueError("corners must be four {u, v} pixels")
    return pts


def check_edges(value) -> tuple[float, float]:
    try:
        a, b = (float(x) for x in value)
    except (TypeError, ValueError) as e:
        raise ValueError("edgesM must be two lengths in metres") from e
    if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
        raise ValueError("edgesM must be two positive lengths in metres")
    return a, b


def default_corners(w: int, h: int) -> list[dict]:
    return [{"u": w * 0.1, "v": h * 0.9}, {"u": w * 0.9, "v": h * 0.9}, {"u": w * 0.7, "v": h * 0.6},
            {"u": w * 0.3, "v": h * 0.6}]


# ---- the floor as the screen draws it ----

def floor_grid(width: float, height: float, corners: list[dict], edges_m) -> dict:
    """The camera solved from four corners, as the screen draws it: grid lines in the photo's pixels (``None``
    for a point that does not project), the grid's step (1 m, or 10 cm on a floor under 2 m, a sheet of paper
    on a table), the residual, the field of view; or ``{ok: False, reason}``."""
    try:
        px, edges = check_corners(corners), check_edges(edges_m)
        w, h = int(round(width)), int(round(height))
        if w <= 0 or h <= 0:
            raise ValueError("width and height must be positive")
    except ValueError as e:
        return {"ok": False, "reason": f"bad-input: {e}"}
    sol = solve_floor(w, h, px, edges)
    if sol.camera is None:
        return {"ok": False, "reason": sol.reason}
    a = float(sol.corners_m[1][0])
    b = float(sol.corners_m[2][1])  # negative when the floor was mirrored
    step = 1.0 if max(abs(a), abs(b)) >= 2.0 else 0.1
    ts = (0, 0.25, 0.5, 0.75, 1)
    lines = [[[m, t * b, 0.0] for t in ts] for m in np.arange(0, abs(a) + 1e-9, step)]
    lines += [[[t * a, m * math.copysign(1.0, b), 0.0] for t in ts] for m in np.arange(0, abs(b) + 1e-9, step)]
    out = []
    for line in lines:
        pix = sol.camera.project(np.array(line, float))
        out.append([None if not np.isfinite(p).all() else [round(float(p[0]), 1), round(float(p[1]), 1)] for p in pix])
    fov = 2 * math.degrees(math.atan(w / 2 / float(sol.camera.K[0, 0])))
    return {"ok": True, "lines": out, "stepM": step, "residualPx": round(sol.residual_px, 2), "fovDeg": round(fov),
            "mirrored": bool(sol.mirrored)}


# ---- the widget ----

def widget_html(name: str, bridge: str, three: bool = False) -> str:
    """A widget's HTML with ``bridge`` (JavaScript defining ``globalThis.ExtApps``) in place of the bundle marker,
    and every ``/*__LIB:<name>__*/`` marker replaced by ``widgets/lib/<name>.js`` (a missing lib raises
    ``FileNotFoundError``)."""
    html = (WIDGETS / name).read_text().replace(BUNDLE_MARKER, bridge)
    if three:
        html = html.replace(THREE_MARKER, (WIDGETS / "vendor/three.iife.js").read_text())
    return LIB_MARKER.sub(lambda m: (WIDGETS / "lib" / f"{m.group(1)}.js").read_text(), html)


def ext_apps_bundle() -> str:
    """The MCP Apps bridge, as the chat host expects it."""
    return (WIDGETS / "vendor/ext-apps-app-with-deps.js").read_text()


# The bridge the browser screen gets instead: the same four calls the widget makes, answered by FitServer.
# ``kind`` tells the widget it is not in a chat (there is no message to send back after 決定).
LOCAL_BRIDGE = """
globalThis.ExtApps = { App: class {
  kind = 'local';
  async connect() {
    const text = await (await fetch('first')).text();
    this.ontoolresult({ content: [{ type: 'text', text }] });
  }
  getHostContext() { return { theme: matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light' }; }
  async sendMessage() { return {}; }
  async callServerTool({ name, arguments: args }) {
    const res = await fetch('call', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                     body: JSON.stringify({ name, args }) });
    const out = await res.json();
    return { isError: 'error' in out, content: [{ type: 'text', text: out.error ?? out.text }] };
  }
} };
"""


# ---- the browser screen's server ----

def decide_floor(image: str, width: int, height: int, out: Path, camera: str, args: dict, *, what: str) -> dict:
    """The browser screens' 決定: solve the floor from ``args`` (``corners``, ``edgesM``, ``measured``), write the
    site file ``out`` (confirmed by a person, no objects; ``what`` is its history text) and answer
    ``{ok, sitePath, residualPx, mirrored, camera}``, or ``{ok: False, reason}`` without writing anything."""
    try:
        px, edges = check_corners(args["corners"]), check_edges(args["edgesM"])
    except (KeyError, ValueError) as e:
        return {"ok": False, "reason": f"bad-input: {e}"}
    if args.get("objects"):
        return {"ok": False, "reason": "the browser screen has no objects: cut them out in the chat"}
    sol = solve_floor(width, height, px, edges)
    if sol.camera is None:
        return {"ok": False, "reason": sol.reason}
    measured = bool(args.get("measured", False))
    site = Site(
        id=out.stem, cameras={camera: sol.camera},
        floor=Floor(camera, px, edges, sol.corners_m),
        residual_px={camera: float(sol.residual_px)},
        confirmed_by_person=True, dimensions="measured" if measured else "estimated",
        history=[{"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "what": what,
                  "image": Path(image).name, "mirrored": bool(sol.mirrored)}])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_site(site, out)
    return {"ok": True, "sitePath": str(out), "residualPx": round(float(sol.residual_px), 2),
            "mirrored": bool(sol.mirrored), "camera": camera}


class FitServer:
    """Serves the floor-fitting screen on 127.0.0.1 for one photo. 決定 in the screen writes the site file
    ``out`` (confirmed by a person, no objects); ``on_decide(site, path)`` is called each time.

    Only this process's browser is meant to reach it: the address carries a random token, and the one call
    that writes (``call``) takes JSON only, so a page from another origin cannot make it without the browser's
    permission check (CORS), which is never granted.
    """

    def __init__(self, image, out, camera: str = "camera", edges=None, measured: bool = False, port: int = 0,
                 on_decide=None):
        self.image = str(image)
        self.out = Path(out)
        self.camera = camera
        self.edges = check_edges(edges) if edges is not None else A4_M
        self.measured = measured
        self.on_decide = on_decide
        self.decided = 0
        self.token = secrets.token_urlsafe(12)
        with open_source(Path(self.image).expanduser()) as source:  # a bad picture fails before the browser opens
            if source.kind != "still":
                raise ValueError(f"{self.image!r} is not a still image (.jpg .jpeg .png)")
            self.width, self.height = source.size
            img = photo(self.image, frame=source.frame(0))
        self._first = json.dumps({
            "imagePath": self.image, "width": self.width, "height": self.height,
            "parts": -(-len(img["b64"]) // CHUNK), "edgesM": list(self.edges), "measured": measured,
            "corners": default_corners(self.width, self.height), "siteDir": "", "objects": [],
            "objectsKey": None, "objectsParts": 0})
        self._html = widget_html("fit-floor.html", LOCAL_BRIDGE)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler_for(self))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/{self.token}/"

    def serve_forever(self) -> None:
        """Answer the browser until ``shutdown`` (from another thread) or KeyboardInterrupt."""
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def close(self) -> None:
        self._httpd.server_close()

    def first(self) -> str:
        return self._first

    def html(self) -> str:
        return self._html

    def call(self, name: str, args: dict) -> str:
        """One of the widget's calls, by the tool name it uses in the chat; the JSON text it expects back."""
        if name == "ps_spike_image":
            return json.dumps(photo_part(str(args["imagePath"]), int(args["part"])))
        if name == "ps_floor_solve":
            return json.dumps(floor_grid(args["width"], args["height"], args["corners"], args["edgesM"]))
        if name == "ps_site_decide":
            return json.dumps(self._decide(args))
        raise ValueError(f"the browser screen has no tool {name!r}")

    def _decide(self, args: dict) -> dict:
        out = decide_floor(self.image, self.width, self.height, self.out, self.camera, args,
                           what="floor fixed in the browser (psbx site fit) and confirmed by a person")
        if out["ok"]:
            self.decided += 1
            if self.on_decide is not None:
                self.on_decide(load_site(self.out), self.out)
        return out


def _handler_for(screen):
    """The HTTP handler for any screen object with ``token``, ``html()``, ``first()`` and ``call(name, args)``."""
    prefix = f"/{screen.token}/"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # the terminal shows the address and the decisions, not every request
            pass

        def _send(self, body: str, ctype: str, code: int = 200) -> None:
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == prefix:
                self._send(screen.html(), "text/html; charset=utf-8")
            elif self.path == prefix + "first":
                self._send(screen.first(), "application/json")
            else:
                self._send("not found", "text/plain", 404)

        def do_POST(self) -> None:
            ctype = self.headers.get("Content-Type", "")
            if self.path != prefix + "call" or not ctype.startswith("application/json"):
                self._send("not found", "text/plain", 404)
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                out = {"text": screen.call(str(req["name"]), dict(req["args"]))}
            except Exception as e:  # noqa: BLE001 - whatever went wrong is shown in the screen's bottom line
                out = {"error": f"{type(e).__name__}: {e}"}
            self._send(json.dumps(out), "application/json")

    return Handler
