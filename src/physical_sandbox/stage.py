"""The stage: one browser screen over a work folder, where a person fixes the floor on a photo, places a machine,
plays a motion over the photo and compares two motions (``psbx stage``).

``Stage`` holds the work folder's rules (which picture, which files) and answers the page's calls; ``StageServer``
serves the page on 127.0.0.1 with the same small server the floor-only screen uses (``fit_screen``). In this
first step the page fixes the floor; its other sections are filled in by later steps.
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from . import catalog, fit_screen, push
from .errors import InputError
from .fit_screen import A4_M, CHUNK, LOCAL_BRIDGE, decide_floor, default_corners, floor_grid, photo, photo_part
from .site import Site, load_site
from .sources import open_source

STAGE_FORMAT = "physical-sandbox.stage/1"


def work_dir_for(picture: Path) -> Path:
    """The work folder a picture gets when none is given: next to it, named after it (``desk.jpg`` -> ``desk``)."""
    return picture.parent / picture.stem


def _picture_named_in(work: Path) -> Path | None:
    """The picture ``work/stage.json`` (else ``work/scene.json``) names (``"picture"``; a relative name is
    relative to the folder), or None."""
    for name in ("stage.json", "scene.json"):
        try:
            d = json.loads((work / name).read_text())
        except (OSError, ValueError):
            continue
        given = d.get("picture") if isinstance(d, dict) else None
        if isinstance(given, str) and given:
            p = Path(given).expanduser()
            return (p if p.is_absolute() else work / p).resolve()
    return None


class Stage:
    """A picture and the work folder that belongs to it. Exactly one of ``picture`` and ``work`` may be None:
    with only the folder, its ``stage.json`` or ``scene.json`` says which picture it is for."""

    def __init__(self, picture=None, work=None, camera: str = "camera", machine_dirs=(), lerobot_dirs=()):
        if picture is None and work is None:
            raise InputError("the stage needs a picture or a work folder")
        self.camera = camera
        self.machine_dirs = tuple(Path(d) for d in machine_dirs)
        self.lerobot_dirs = tuple(Path(d) for d in lerobot_dirs)
        if picture is None:
            self.work = Path(work)
            found = _picture_named_in(self.work)
            if found is None:
                raise InputError(f"no picture in {self.work}: give the picture")
            self.picture = found
        else:
            self.picture = Path(picture).expanduser().resolve()
            self.work = Path(work) if work is not None else work_dir_for(self.picture)
        if not self.picture.is_file():
            raise InputError(f"picture {str(self.picture)!r}: no such file")
        self._check_still()
        self._check_site_is_for_this_picture()

    @property
    def site_path(self) -> Path:
        return self.work / "site.json"

    @property
    def scene_path(self) -> Path:
        return self.work / "scene.json"

    @property
    def stage_path(self) -> Path:
        return self.work / "stage.json"

    @property
    def site(self) -> Site | None:
        """The work folder's site file, read when it is there."""
        if not self.site_path.is_file():
            return None
        try:
            return load_site(self.site_path)
        except (OSError, ValueError, KeyError) as e:
            raise InputError(f"{self.site_path}: cannot read it ({e})") from e

    def _check_still(self) -> None:
        try:
            with open_source(self.picture) as source:
                if source.kind != "still":
                    raise InputError(f"{str(self.picture)!r} is not a still image (.jpg .jpeg .png)")
                self.width, self.height = source.size
                self._photo = photo(str(self.picture), frame=source.frame(0))  # the display copy, made once
        except InputError:
            raise
        except (ValueError, OSError) as e:
            raise InputError(f"picture {str(self.picture)!r}: {e}") from e

    def _check_site_is_for_this_picture(self) -> None:
        site = self.site
        if site is None or not site.history:
            return
        fixed_on = site.history[-1].get("image")
        if isinstance(fixed_on, str) and fixed_on != self.picture.name:
            raise InputError(f"{self.site_path} was fixed on {fixed_on}, not on {self.picture.name}: "
                             f"use another work folder for this picture (--dir)")

    def first(self) -> dict:
        """What the page starts from."""
        site = self.site
        floor = site.floor if site is not None else None
        if floor is not None:
            corners = [{"u": float(u), "v": float(v)} for u, v in floor.corners_px]
            edges, measured = [float(e) for e in floor.edges_m], site.dimensions == "measured"
        else:
            corners, edges, measured = default_corners(self.width, self.height), list(A4_M), False
        return {
            "imagePath": str(self.picture), "width": self.width, "height": self.height,
            "parts": -(-len(self._photo["b64"]) // CHUNK), "camera": self.camera,
            "floor": {"corners": corners, "edgesM": edges, "measured": measured,
                      "decided": floor is not None and site.confirmed_by_person},
            "machine": None, "motionA": None, "motionB": None, "gripThreshold": None,
            "login": push.login_status(catalog._base()),  # the token file only: no request
            "workDir": str(self.work),
        }

    def call(self, name: str, args: dict) -> str:
        """One of the page's calls, by name; the JSON text it expects back."""
        if name == "ps_spike_image":
            return json.dumps(photo_part(str(args["imagePath"]), int(args["part"])))
        if name == "ps_floor_solve":
            return json.dumps(floor_grid(args["width"], args["height"], args["corners"], args["edgesM"]))
        if name == "ps_site_decide":
            return json.dumps(self._decide(args))
        raise ValueError(f"the stage has no tool {name!r}")

    def _decide(self, args: dict) -> dict:
        out = decide_floor(str(self.picture), self.width, self.height, self.site_path, self.camera, args,
                           what="floor fixed in the browser (psbx stage) and confirmed by a person")
        if out["ok"]:  # stage.json lets the folder alone find its picture next time
            fit_screen.write_atomic(self.stage_path, json.dumps(
                {"format": STAGE_FORMAT, "picture": str(self.picture)}, indent=2).encode())
        return out


class StageServer:
    """Serves the stage page on 127.0.0.1 (the address carries a random token; the one call that writes takes
    JSON only: see ``fit_screen._handler_for``). ``on_decide(site, path)`` is called after each 決定."""

    def __init__(self, stage: Stage, port: int = 0, on_decide=None):
        self.stage = stage
        self.on_decide = on_decide
        self.token = secrets.token_urlsafe(12)
        self._html = fit_screen.widget_html("stage.html", LOCAL_BRIDGE, three=True)
        self._httpd = fit_screen.ThreadingHTTPServer(("127.0.0.1", port), fit_screen._handler_for(self))

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/{self.token}/"

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def close(self) -> None:
        self._httpd.server_close()

    def html(self) -> str:
        return self._html

    def first(self) -> str:
        return json.dumps(self.stage.first())

    def call(self, name: str, args: dict) -> str:
        text = self.stage.call(name, args)
        if name == "ps_site_decide" and self.on_decide is not None and json.loads(text).get("ok"):
            self.on_decide(self.stage.site, self.stage.site_path)
        return text
