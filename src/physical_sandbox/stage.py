"""The stage: one browser screen over a work folder, where a person fixes the floor on a photo, places a machine,
plays a motion over the photo and compares two motions (``psbx stage``).

``Stage`` holds the work folder's rules (which picture, which files) and answers the page's calls; ``StageServer``
serves the page on 127.0.0.1 with the same small server the floor-only screen uses (``fit_screen``). In this
first step the page fixes the floor; its other sections are filled in by later steps.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import secrets
import threading
from pathlib import Path

import numpy as np

from . import catalog, fit_screen, push, screens
from .errors import CatalogError, InputError
from .fit_screen import A4_M, CHUNK, LOCAL_BRIDGE, decide_floor, default_corners, floor_grid, photo, photo_part
from .machine import load_machine
from .mcp.pack import model_part, pack_model
from .motion import load_record
from .render import render_overlay
from .scene import PlacedMachine, load_scene, save_placement
from .screens import compact, pixels
from .site import Site, load_site
from .sources import open_source

STAGE_FORMAT = "physical-sandbox.stage/1"
_UNREADABLE = (OSError, ValueError, KeyError, TypeError, AttributeError)  # a file that is not what it should be


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


def _urdf_in(folder: Path) -> Path | None:
    """The machine file of a folder: the first ``.urdf`` its catalog manifest lists, else the first ``.urdf`` in it."""
    if (folder / "manifest.json").is_file():
        try:
            found = catalog.urdf_path(folder)
            if found.is_file():
                return found
        except (CatalogError, OSError, ValueError, KeyError, TypeError):
            pass
    found = sorted(p for p in folder.glob("*.urdf") if p.is_file())
    return found[0] if found else None


def _subfolders(parent: Path, depth: int) -> list[Path]:
    """The folders exactly ``depth`` levels under ``parent`` (none when it is not there), by name."""
    level = [parent]
    for _ in range(depth):
        level = [d for p in level if p.is_dir() for d in sorted(p.iterdir()) if d.is_dir()]
    return level


def _safe_name(label: str) -> str:
    """A label as a file name part: every character outside ``A-Za-z0-9._-`` (non-ASCII too) becomes ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", label) or "motion"


def _hub_root() -> Path:
    """Where LeRobot keeps the datasets it downloaded: ``$HF_HOME/lerobot/hub``, else under ``~/.cache/huggingface``."""
    home = os.environ.get("HF_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface") / "lerobot" / "hub"


def _dataset_name(folder: Path) -> str:
    """A dataset's name: ``owner/name`` for a hub snapshot (``datasets--owner--name/snapshots/<hash>``), else the folder's."""
    hub = folder.parent.parent.name
    if folder.parent.name == "snapshots" and hub.startswith("datasets--"):
        return hub[len("datasets--"):].replace("--", "/")
    return folder.name


def _number(args: dict, key: str) -> float:
    """``args[key]`` as a finite number (a bool or text is not one)."""
    value = args.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return float(value)


class Stage:
    """A picture and the work folder that belongs to it. Exactly one of ``picture`` and ``work`` may be None:
    with only the folder, its ``stage.json`` or ``scene.json`` says which picture it is for."""

    def __init__(self, picture=None, work=None, camera: str = "camera", machine_dirs=(), lerobot_dirs=()):
        if picture is None and work is None:
            raise InputError("the stage needs a picture or a work folder")
        self.camera = camera
        self.machine_dirs = tuple(Path(d) for d in machine_dirs)
        self.lerobot_dirs = tuple(Path(d) for d in lerobot_dirs)
        self._opened: set[Path] = set()  # machine files the page opened (a typed path may be placed after it)
        self._jobs: dict[str, dict] = {}  # exports by job id; one runs at a time
        self._jobs_lock = threading.Lock()
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
            "machine": self._placed_machine(), "motionA": self._kept("a"), "motionB": self._kept("b"), "gripThreshold": None,
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
        if name == "ps_machines_list":
            return json.dumps({"machines": self._machines()})
        if name == "ps_catalog_search":
            return json.dumps(self._catalog_search(str(args.get("q", ""))))
        if name == "ps_catalog_pull":
            return json.dumps(self._catalog_pull(str(args["ref"])))
        if name == "ps_machine_open":
            return json.dumps(self._machine_open(args))
        if name == "ps_scene_place":
            return json.dumps(self._scene_place(args))
        if name == "ps_motions_list":
            return json.dumps(self._motions_list())
        if name == "ps_motion_open":
            return json.dumps(self._motion_open(args))
        if name == "ps_compare_part":
            return screens.data_part(str(args["key"]), int(args["part"]), "ps_motion_open")
        if name == "ps_export":
            return json.dumps(self._export(args))
        if name == "ps_export_status":
            return json.dumps(self._export_status(str(args.get("job", ""))))
        if name == "ps_model_part":
            return json.dumps({"part": int(args["part"]), "data": model_part(str(args["key"]), int(args["part"]), CHUNK)})
        raise ValueError(f"the stage has no tool {name!r}")

    def _decide(self, args: dict) -> dict:
        out = decide_floor(str(self.picture), self.width, self.height, self.site_path, self.camera, args,
                           what="floor fixed in the browser (psbx stage) and confirmed by a person")
        if out["ok"]:  # stage.json lets the folder alone find its picture next time
            fit_screen.write_atomic(self.stage_path, json.dumps(
                {"format": STAGE_FORMAT, "picture": str(self.picture)}, indent=2).encode())
        return out

    # ---- the machine step ----

    def _update_stage_json(self, **keys) -> None:
        """Add ``keys`` to ``stage.json``, keeping every other key it has (it is made when it is not there)."""
        try:
            d = json.loads(self.stage_path.read_text())
        except (OSError, ValueError):
            d = None
        if not isinstance(d, dict):
            d = {"format": STAGE_FORMAT, "picture": str(self.picture)}
        d.update(keys)
        self.work.mkdir(parents=True, exist_ok=True)
        fit_screen.write_atomic(self.stage_path, json.dumps(d, indent=2).encode())

    def _kept(self, key: str) -> str | None:
        """A text ``stage.json`` keeps under ``key`` (the motion spec of slot ``a`` or ``b``), or None."""
        try:
            d = json.loads(self.stage_path.read_text())
        except (OSError, ValueError):
            return None
        value = d.get(key) if isinstance(d, dict) else None
        return value if isinstance(value, str) and value else None

    def _scene_machines(self) -> dict:
        """The ``machines`` of ``scene.json`` as the file has them (none when it is not there or not readable)."""
        try:
            d = json.loads(self.scene_path.read_text())
        except (OSError, ValueError):
            return {}
        machines = d.get("machines") if isinstance(d, dict) else None
        return machines if isinstance(machines, dict) else {}

    def _placed_machine(self) -> dict | None:
        """``{id, urdf, placed: {x, y, yaw}}`` when ``scene.json`` places exactly one machine, else None."""
        machines = self._scene_machines()
        if len(machines) != 1:
            return None
        (mid, m), = machines.items()
        try:
            urdf = (self.work / Path(str(m["urdf"])).expanduser()).resolve()
            return {"id": mid, "urdf": str(urdf),
                    "placed": {"x": float(m["x"]), "y": float(m["y"]), "yaw": float(m["yaw"])}}
        except (KeyError, TypeError, ValueError):
            return None

    def _machines(self) -> list[dict]:
        """The machines the person can pick: the cache, the catalog's cache, then the ``--machine`` folders."""
        root = catalog.cache_root()
        found = []
        for folder in _subfolders(root / "machines", 1):
            if (urdf := _urdf_in(folder)) is not None:
                found.append({"id": folder.name, "urdf": str(urdf), "source": "cache", "label": folder.name})
        for folder in _subfolders(root / "catalog", 3):
            if (urdf := _urdf_in(folder)) is not None:
                ref = f"{folder.parent.parent.name}/{folder.parent.name}:{folder.name}"
                found.append({"id": folder.parent.name, "urdf": str(urdf), "source": "catalog", "label": ref})
        for d in self.machine_dirs:
            for folder in [d, *_subfolders(d, 1)]:
                if (urdf := _urdf_in(folder)) is not None:
                    found.append({"id": folder.name, "urdf": str(urdf), "source": "dir", "label": folder.name})
        return found

    def _catalog_search(self, q: str) -> dict:
        try:
            items = catalog.search(q, token=push.kept_token(catalog._base()))
        except (CatalogError, InputError) as e:  # offline, a refusal: the screen shows it in its bottom line
            return {"error": str(e)}
        out = []
        for item in items:
            ref = f"{item.get('handle')}/{item.get('name')}"
            display = item.get("display") if isinstance(item.get("display"), dict) else {}
            out.append({"ref": ref, "version": item.get("latest_version"), "private": bool(item.get("private")),
                        "label": display.get("ja") or display.get("en") or ref})
        return {"items": out}

    def _catalog_pull(self, ref: str) -> dict:
        try:
            _, name, _ = catalog.parse_ref(ref)
            folder = catalog.pull(ref, token=push.kept_token(catalog._base()))
            return {"id": name, "urdf": str(catalog.urdf_path(folder))}
        except (CatalogError, InputError) as e:
            return {"error": str(e)}

    def _may_open(self, urdf: Path) -> bool:
        """Whether a machine file the page names is one it was offered: under the cache, a ``--machine`` folder or
        the work folder, or one this stage already opened or the work folder's scene places."""
        if urdf in self._opened:
            return True
        placed = self._placed_machine()
        if placed is not None and Path(placed["urdf"]) == urdf:
            return True
        for root in (catalog.cache_root(), self.work, *self.machine_dirs):
            try:
                urdf.relative_to(Path(root).expanduser().resolve())
                return True
            except ValueError:
                continue
        return False

    def _checked_urdf(self, args: dict) -> tuple[str, Path]:
        """``(id, urdf)`` of a call that names a machine: an existing ``.urdf`` file the page was offered
        (any, with ``typed``)."""
        mid, given = args.get("id"), args.get("urdf")
        if not isinstance(mid, str) or not mid or len(mid) > 64:
            raise ValueError("id must be a short name")
        if not isinstance(given, str) or not given:
            raise ValueError("urdf must be a path")
        urdf = Path(given).expanduser().resolve()
        if urdf.suffix.lower() != ".urdf" or not urdf.is_file():
            raise ValueError(f"{given!r}: no such .urdf file")
        if not (args.get("typed") is True or self._may_open(urdf)):
            raise ValueError("urdf must be a listed machine")
        return mid, urdf

    def _decided_site(self) -> Site:
        site = self.site
        if site is None or site.floor is None or not site.confirmed_by_person or self.camera not in site.cameras:
            raise ValueError("fix the floor first: decide it in the floor step")
        return site

    def _machine_open(self, args: dict) -> dict:
        """Everything the page draws a machine from: its shapes, its zero pose, the camera, the floor in the picture."""
        site = self._decided_site()
        mid, urdf = self._checked_urdf(args)
        machine = load_machine(urdf)
        model = pack_model(machine, CHUNK)
        q0 = np.clip(np.zeros(len(machine.joint_names)), machine.limits[:, 0], machine.limits[:, 1])
        local = machine.link_poses(q0)
        self._opened.add(urdf)
        cam = site.cameras[self.camera]
        corners = np.asarray(site.floor.corners_m, float)
        placed = self._scene_machines().get(mid)
        try:
            placed = {k: float(placed[k]) for k in ("x", "y", "yaw")} if isinstance(placed, dict) else None
        except (KeyError, TypeError, ValueError):
            placed = None
        return {
            "id": mid, "urdf": str(urdf), "model": model,
            "zeroPoses": [compact(local[l["name"]], 5) for l in model["links"]],
            "camera": {"intr": {"width": cam.width, "height": cam.height, "fx": float(cam.K[0, 0]),
                                "fy": float(cam.K[1, 1]), "cx": float(cam.K[0, 2]), "cy": float(cam.K[1, 2])},
                       "extr": {"R": [float(x) for x in cam.R.reshape(-1)], "t": [float(x) for x in cam.t]}},
            "floor": pixels(cam, corners),
            "center": {"x": float(corners[:, 0].mean()), "y": float(corners[:, 1].mean())},
            "placed": placed,
        }

    def _scene_place(self, args: dict) -> dict:
        """決定 of the machine step: ``scene.json`` places the machine (the scene's record list stays)."""
        site = self._decided_site()
        mid, urdf = self._checked_urdf(args)
        x, y, yaw = _number(args, "x"), _number(args, "y"), _number(args, "yaw")
        placed = PlacedMachine(load_machine(urdf), str(urdf), x, y, yaw, confirmed_by_person=True)
        save_placement(self.scene_path, site.id, {mid: placed}, picture=str(self.picture))
        self._opened.add(urdf)
        self._update_stage_json(machine={"id": mid, "urdf": str(urdf)})
        return {"scenePath": str(self.scene_path), "base": compact(placed.base_transform(), 6)}

    # ---- the motion step ----

    def _motions_list(self) -> dict:
        """The motions the page can open: every ``*.record.json`` of the work folder (one entry per motion) and
        every LeRobot dataset (the hub's downloads, then the ``--lerobot`` folders)."""
        records = []
        for path in sorted(self.work.glob("*.record.json")):
            try:
                motions = load_record(path)
            except _UNREADABLE:
                continue
            for m in motions:
                label = m.label or m.machine
                records.append({"spec": str(path) if len(motions) == 1 else f"{path}#{label}", "label": label})
        found: list[Path] = []
        hub = _hub_root()
        if hub.is_dir():
            found += sorted(hub.glob("datasets--*/snapshots/*"))
        found += self.lerobot_dirs
        lerobot, seen = [], set()
        for folder in found:
            info_path = folder / "meta" / "info.json"
            if str(folder) in seen or not info_path.is_file():
                continue
            seen.add(str(folder))
            try:
                total = json.loads(info_path.read_text()).get("total_episodes")
            except (OSError, ValueError, AttributeError):
                continue
            lerobot.append({"dir": str(folder), "name": _dataset_name(folder),
                            "episodes": total if isinstance(total, int) and not isinstance(total, bool) else None})
        return {"records": records, "lerobot": lerobot}

    def _load_motion(self, spec: str):
        """``(scene, motion, notes)`` for a motion spec (a record file with ``#label``, or ``lerobot:<dir>#<n>``) on
        the one machine the scene places. A bad spec raises ``InputError`` (the page shows it)."""
        from .cli import _load_motion, _split_label

        site = self._decided_site()
        if self._placed_machine() is None:
            raise ValueError("ロボットを置いてから動きを読み込んでください")
        scene = load_scene(self.scene_path, site)
        if not spec.startswith("lerobot:"):
            path, label = _split_label(spec)
            if not Path(path).expanduser().is_absolute() and (self.work / path).is_file():
                spec = str(self.work / path) + (f"#{label}" if label is not None else "")
        notes: list[str] = []
        motion, lerobot = _load_motion(spec, scene, notes)
        if lerobot is not None:
            motion = dataclasses.replace(
                motion, label=f"{_dataset_name(Path(lerobot[0]).expanduser()).rsplit('/', 1)[-1]} ep{lerobot[1]}")
        elif not motion.label:
            motion = dataclasses.replace(motion, label=motion.machine)
        return scene, motion, notes

    def _motion_open(self, args: dict) -> dict:
        """One motion on the placed machine: what the compare screen draws (``screens.screen_data`` with one record)."""
        slot, spec = args.get("slot"), args.get("spec")
        if slot not in ("a", "b"):
            raise ValueError("slot must be 'a' or 'b'")
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("spec must be a record file or lerobot:<dir>#<episode>")
        spec = spec.strip()
        scene, motion, notes = self._load_motion(spec)
        with open_source(self.picture) as source:
            out = screens.screen_data(scene.site, scene, self.camera, [motion], source)
        out["note"] = notes[0] if notes else None
        out["spec"] = spec
        self._update_stage_json(**{slot: spec})
        return out

    def _export(self, args: dict) -> dict:
        """Start writing the motion over the photo as an mp4 in the work folder (one at a time); answers the job id."""
        if args.get("what") != "replay":
            raise ValueError("what must be 'replay'")
        spec = args.get("spec")
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("spec must be a record file or lerobot:<dir>#<episode>")
        scene, motion, _ = self._load_motion(spec.strip())
        out = self.work / f"replay-{_safe_name(motion.label)}.mp4"
        job = secrets.token_hex(4)
        with self._jobs_lock:
            if any(not j["done"] for j in self._jobs.values()):
                return {"error": "書き出し中です"}
            self._jobs[job] = state = {"done": False, "progress": 0, "paths": [], "error": None}

        def progress(k: int, n: int) -> None:
            with self._jobs_lock:
                state["progress"] = k / n

        def run() -> None:
            error, paths = None, []
            try:
                with open_source(self.picture) as source:
                    render_overlay(scene, self.camera, source, [motion], out, progress=progress)
                paths = [str(out)]
            except Exception as e:  # the page shows one line; the thread must not die silently
                text = str(e).strip()
                error = text.splitlines()[0] if text else type(e).__name__
            with self._jobs_lock:
                state.update(done=True, paths=paths, error=error)
                if error is None:
                    state["progress"] = 1.0
        threading.Thread(target=run, daemon=True).start()
        return {"job": job}

    def _export_status(self, job: str) -> dict:
        with self._jobs_lock:
            state = self._jobs.get(job)
            if state is None:
                raise ValueError(f"no export {job!r}")
            return {"done": state["done"], "progress": state["progress"], "paths": list(state["paths"]),
                    "error": state["error"]}


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
