"""The stage: one browser screen over a work folder, where a person fixes the floor on a photo, places a machine,
plays a motion over the photo and compares two motions (``psbx stage``).

``Stage`` holds the work folder's rules (which picture, which files) and answers the page's calls; ``StageServer``
serves the page on 127.0.0.1 with the same small server the floor-only screen uses (``fit_screen``). The page has
four sections: the floor (fixed from four corners), the machine (placed on the floor), the motion (one record or
LeRobot episode played over the photo) and the comparison of two motions, with exports of the replay and the report.
The motion in A is saved to PS.com (over the person's own record, or as a new one) or exported as a record file; a
save PS.com does not get is kept in the work folder (``unsent``) and sent again later.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import secrets
import tempfile
import threading
import urllib.error
from pathlib import Path

import numpy as np

from . import catalog, fit_screen, push, record_edit, report, screens, unsent
from ._atomic import write_text_atomic
from .comparison import compare
from .errors import CatalogError, InputError
from .push import PushError
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
PS_PREFIX = "ps:"  # a motion spec naming a PS.com record: ps:<handle>/<name>[:<version>][#<label>]
_QUICK_S = 5  # seconds a PS.com request of the motion step may wait: the page waits on it before anything else
_REF_WORDS = "PS.com の記録は handle/name か handle/name:バージョン で指定してください"
_UNREADABLE = (OSError, ValueError, KeyError, TypeError, AttributeError)  # a file that is not what it should be
_NO_A = "動きを読み込んでから保存してください"
_NOT_OWN = "上書き保存できるのは、PS.com から読み込んだ自分の記録だけです"
_NAME_WORDS = "名前は英小文字・数字・- の 2〜39 文字で、先頭と末尾は英小文字か数字にしてください"


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
    """A label as a file name part: letters and digits of any script (a Japanese label stays readable), ``.``, ``_``
    and ``-`` stay; everything else becomes ``_``. Leading dots go (no hidden files); an empty result is ``motion``."""
    return re.sub(r"[^\w.-]", "_", label).lstrip(".") or "motion"


def _catalog_name(label: str) -> str:
    """A label as a catalog name (lowercase letters, digits and ``-``, 2 to 39 characters): ``dance:2`` ->
    ``dance-2``; a label with too little of that (a Japanese one) is ``record``."""
    name = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:39].rstrip("-")
    return name if len(name) >= 2 else "record"


def _record_stem(path: Path) -> str:
    """A record file's name without ``.record.json`` (else without its last suffix): ``demo.record.json`` -> ``demo``."""
    name = Path(path).name
    for suffix in (".record.json", ".json"):
        if name.endswith(suffix) and len(name) > len(suffix):
            return name[: -len(suffix)]
    return Path(path).stem


def _hub_roots() -> list[Path]:
    """Where downloaded datasets wait: LeRobot's own ``<HF_HOME>/lerobot/hub`` and the Hugging Face hub's
    ``<HF_HOME>/hub`` (where ``huggingface-cli download`` writes); ``HF_HOME`` defaults to ``~/.cache/huggingface``."""
    home = os.environ.get("HF_HOME")
    base = Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface"
    return [base / "lerobot" / "hub", base / "hub"]


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


def _shown(p: dict) -> dict:
    """A kept copy as the page lists it."""
    return {"name": p["name"], "ref": p["ref"], "mode": p["mode"]}


def _joint_device(raw: dict, spec: str) -> str | None:
    """The device of a record whose motion a spec reads: the one labelled ``#<label>`` in the spec, else the first
    device that names a product and has a trace of joint angles (as ``load_record`` reads them)."""
    _, sep, label = spec.partition("#")
    try:
        devices = {d.get("id"): d for d in raw["equipment"]["devices"] if isinstance(d, dict)}
        for tr in raw.get("traces") or []:
            dev = devices.get(tr.get("deviceId"))
            samples = tr.get("samples") or []
            if (dev is None or not isinstance(dev.get("product"), str) or not dev["product"] or not samples
                    or not all(isinstance(s, dict) and isinstance(s.get("joints"), dict) and s["joints"]
                               for s in samples)):
                continue
            if not sep or dev.get("displayName") == label:
                return dev["id"] if isinstance(dev.get("id"), str) else None
    except (KeyError, TypeError, AttributeError):
        return None
    return None


class Stage:
    """A picture and the work folder that belongs to it. Exactly one of ``picture`` and ``work`` may be None:
    with only the folder, its ``stage.json`` or ``scene.json`` says which picture it is for."""

    def __init__(self, picture=None, work=None, camera: str | None = None, machine_dirs=(), lerobot_dirs=()):
        if picture is None and work is None:
            raise InputError("the stage needs a picture or a work folder")
        self.machine_dirs = tuple(Path(d).expanduser().resolve() for d in machine_dirs)
        self.lerobot_dirs = tuple(Path(d).expanduser().resolve() for d in lerobot_dirs)
        self._opened: set[Path] = set()  # machine files the page opened (a typed path may be placed after it)
        self._jobs: dict[str, dict] = {}  # exports by job id; one runs at a time
        self._jobs_lock = threading.Lock()
        self._stage_lock = threading.Lock()  # stage.json is read, changed and written back: one call at a time
        self._save_lock = threading.Lock()  # one save or send at a time: a kept copy is never sent twice
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
        self.camera = camera if camera is not None else self._site_camera()

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

    def _site_camera(self) -> str:
        """The camera name to use when none was given: the one the work folder's floor was decided with, else ``camera``."""
        site = self.site
        floor = site.floor if site is not None else None
        return floor.camera if floor is not None and floor.camera else "camera"

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
            "machine": self._placed_machine(), "motionA": self._kept("a"), "motionB": self._kept("b"), "gripThreshold": self._kept_grip(),
            "aOpened": self._kept_opened(),
            "unsent": [_shown(p) for p in unsent.pending(self.work)],
            "login": push.login_status(catalog._base()),  # the token file only: no request
            "workDir": str(self.work),
        }

    def call(self, name: str, args: dict) -> str:
        """One of the page's calls, by name; the JSON text it expects back."""
        if name == "ps_spike_image":
            return json.dumps(photo_part(str(self.picture), int(args["part"])))
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
        if name == "ps_compare":
            return json.dumps(self._compare(args))
        if name == "ps_compare_part":
            return screens.data_part(str(args["key"]), int(args["part"]), "ps_motion_open")
        if name == "ps_record_save":
            return json.dumps(self._record_save(args))
        if name == "ps_record_export":
            return json.dumps(self._record_export())
        if name == "ps_unsent_send":
            return json.dumps(self._unsent_send())
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
            self._update_stage_json(picture=str(self.picture))
        return out

    # ---- the machine step ----

    def _update_stage_json(self, _drop=(), **keys) -> None:
        """Add ``keys`` to ``stage.json`` and remove the keys named in ``_drop``, keeping every other key it has
        (it is made when it is not there)."""
        with self._stage_lock:
            try:
                d = json.loads(self.stage_path.read_text())
            except (OSError, ValueError):
                d = None
            if not isinstance(d, dict):
                d = {"format": STAGE_FORMAT, "picture": str(self.picture)}
            d.update(keys)
            for key in _drop:
                d.pop(key, None)
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

    def _kept_opened(self) -> dict | None:
        """``aOpened`` of ``stage.json`` (what slot A opened from PS.com) while slot A is a ``ps:`` spec, else None."""
        a = self._kept("a")
        if a is None or not a.startswith(PS_PREFIX):
            return None
        try:
            opened = json.loads(self.stage_path.read_text()).get("aOpened")
        except (OSError, ValueError, AttributeError):
            return None
        return opened if isinstance(opened, dict) else None

    def _a_keys(self, spec: str, opened: dict | None) -> dict:
        """The ``stage.json`` change for slot A holding ``spec``: ``aOpened`` is written with a ``ps:`` spec and
        removed otherwise."""
        if opened is None:
            return {"a": spec, "_drop": ("aOpened",)}
        return {"a": spec, "aOpened": opened}

    def _kept_grip(self) -> float | None:
        """The grip threshold ``stage.json`` keeps (a finite number), or None for the automatic one."""
        try:
            value = json.loads(self.stage_path.read_text()).get("gripThreshold")
        except (OSError, ValueError, AttributeError):
            return None
        ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        return float(value) if ok else None

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
            items = catalog.search(q, kind="machine", token=push.kept_token(catalog._base()))
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
        for hub in _hub_roots():
            if hub.is_dir():
                found += sorted(hub.glob("datasets--*/snapshots/*"))
        found += self.lerobot_dirs
        lerobot, seen = [], set()
        for folder in found:
            info_path = folder / "meta" / "info.json"
            if not info_path.is_file() or folder.resolve() in seen:
                continue
            seen.add(folder.resolve())
            try:
                total = json.loads(info_path.read_text()).get("total_episodes")
            except (OSError, ValueError, AttributeError):
                continue
            lerobot.append({"dir": str(folder), "name": _dataset_name(folder),
                            "episodes": total if isinstance(total, int) and not isinstance(total, bool) else None})
        return {"records": records, "lerobot": lerobot, "catalog": self._catalog_records()}

    def _catalog_records(self) -> list[dict]:
        """The signed-in person's records on PS.com (``[{spec, label, own}]``, ``spec`` at the latest version).
        Signed out, offline or refused: none, and nothing to show (the page lists the rest as usual)."""
        base = catalog._base()
        token, handle = push.kept_token(base), push.login_status(base)["handle"]
        if not token or not handle:
            return []
        try:
            items = catalog.search(kind="record", owner="me", limit=500, token=token, timeout=_QUICK_S)
        except (CatalogError, InputError):
            return []
        out = []
        for item in items:
            ref = f"{item.get('handle')}/{item.get('name')}:{item.get('latest_version')}"
            if item.get("handle") != handle or item.get("kind") != "record":
                continue
            try:
                catalog.parse_ref(ref)
            except InputError:  # no finished version, or names the server should not have given
                continue
            display = item.get("display") if isinstance(item.get("display"), dict) else {}
            title = display.get("ja") or display.get("en")
            out.append({"spec": PS_PREFIX + ref, "label": f"{title}（{ref}）" if title else ref, "own": True})
        return out

    def _catalog_record(self, spec: str, ask_latest: bool) -> tuple[Path, str | None, dict]:
        """A ``ps:<handle>/<name>[:<version>][#<label>]`` spec as ``(record.json in the catalog cache, label,
        opened)``; ``opened`` is ``{spec, ref, version, latest, own}`` with ``spec`` at the version read (a spec
        without one is the latest). ``ask_latest`` reads the item's latest version from PS.com (offline: the version
        read); otherwise ``latest`` is the version read. Errors are in the page's words."""
        body, sep, label = spec[len(PS_PREFIX):].partition("#")
        try:
            handle, name, version = catalog.parse_ref(body.strip())
        except InputError:
            raise ValueError(_REF_WORDS) from None
        base = catalog._base()
        token, login = push.kept_token(base), push.login_status(base)

        def refused(e: CatalogError, what: str) -> ValueError:
            if isinstance(e.__cause__, urllib.error.HTTPError) and e.__cause__.code == 404:
                hint = "" if login["signedIn"] else \
                    "（非公開の記録は、端末で psbx login を実行してサインインしてから読み込んでください）"
                return ValueError(f"PS.com に記録 {what} が見つかりません{hint}")
            if isinstance(e, catalog._Offline) or isinstance(e.__cause__, catalog._Offline):  # nothing cached either
                return ValueError(f"PS.com に接続できず、記録 {what} を取得できませんでした（{e}）")
            return ValueError(f"PS.com から記録 {what} を取得できませんでした（{e}）")

        def not_a_record() -> ValueError:
            return ValueError(f"{handle}/{name} は記録ではありません")

        latest = None
        if ask_latest and version != "latest":
            try:
                item = catalog._get_json(f"{base}/api/catalog/{handle}/{name}", token, _QUICK_S)["item"]
            except catalog._Offline:
                item = None  # the cache may still hold the version asked for
            except CatalogError as e:
                raise refused(e, body.strip()) from None
            except (KeyError, TypeError):
                raise ValueError(f"PS.com から記録 {body.strip()} を取得できませんでした（答えの形が違います）") from None
            if isinstance(item, dict):
                if item.get("kind") != "record":
                    raise not_a_record()
                latest = item.get("latest_version")
                latest = latest if isinstance(latest, str) and latest else None
        try:  # without a version, pull asks PS.com for the latest (offline: the one it read last)
            folder = catalog.pull(f"{handle}/{name}" + ("" if version == "latest" else f":{version}"), token=token,
                                  timeout=_QUICK_S)
        except CatalogError as e:
            raise refused(e, body.strip()) from None
        version = folder.name
        try:
            kind = json.loads((folder / "manifest.json").read_text()).get("kind")
        except (OSError, ValueError, AttributeError):
            kind = None
        path = (folder / "record.json").resolve()
        try:  # a ps: spec opens only a file of the catalog cache
            path.relative_to((catalog.cache_root() / "catalog").resolve())
        except ValueError:
            raise ValueError(f"{handle}/{name}:{version} は記録ではありません") from None
        if kind != "record" or not path.is_file():
            raise not_a_record()
        ref = f"{handle}/{name}"
        opened = {"spec": f"{PS_PREFIX}{ref}:{version}" + (f"#{label}" if sep else ""), "ref": ref,
                  "version": version, "latest": latest or version,
                  "own": bool(login["signedIn"]) and login["handle"] == handle}
        return path, (label if sep else None), opened

    def _load_motion(self, spec: str, ask_latest: bool = False):
        """``(scene, motion, notes, opened)`` for a motion spec (a record file with ``#label``, ``lerobot:<dir>#<n>``
        or ``ps:<handle>/<name>[:<version>]``) on the one machine the scene places. ``opened`` is None except for a
        ``ps:`` spec (see ``_catalog_record``). A bad spec raises ``InputError`` (the page shows it)."""
        from .cli import _check_motion, _load_motion, _split_label

        site = self._decided_site()
        if self._placed_machine() is None:
            raise ValueError("ロボットを置いてから動きを読み込んでください")
        scene = load_scene(self.scene_path, site)
        opened = stem = None
        if spec.startswith(PS_PREFIX):
            path, label, opened = self._catalog_record(spec, ask_latest)
            spec = str(path) + (f"#{label}" if label is not None else "")
            stem = f"{opened['ref'].split('/', 1)[1]}:{opened['version']}"
        elif not spec.startswith("lerobot:"):
            path, label = _split_label(spec)
            if not Path(path).expanduser().is_absolute() and (self.work / path).is_file():
                spec = str(self.work / path) + (f"#{label}" if label is not None else "")
        if spec.startswith("lerobot:") and len(scene.machines) == 1:
            self._check_dataset_joints(spec, scene)
        notes: list[str] = []
        motion, lerobot = _load_motion(spec, scene, notes)
        if lerobot is not None:
            motion = dataclasses.replace(
                motion, label=f"{_dataset_name(Path(lerobot[0]).expanduser()).rsplit('/', 1)[-1]} ep{lerobot[1]}")
        else:
            # A record file's motion is named by the file (two records of one robot are both "SO-101" inside):
            # ``<stem>``, or ``<stem>-<label>`` when the file holds several motions.
            # A PS.com record is named by its name and version (its file is always record.json).
            path = Path(_split_label(spec)[0]).expanduser()
            several = len(load_record(path)) > 1
            own = motion.label or motion.machine
            stem = stem or _record_stem(path)
            motion = dataclasses.replace(motion, label=f"{stem}-{own}" if several else stem)
        placed = scene.machines.get(motion.machine)
        try:
            _check_motion(scene, motion)
        except InputError:
            if placed is None:  # not placed: the page is told as the check words it
                raise
            raise ValueError(f"記録の関節がロボットと合いません（記録: {', '.join(motion.joint_names)} ／ "
                             f"ロボット: {', '.join(placed.machine.joint_names)}）") from None
        return scene, motion, notes, opened

    @staticmethod
    def _check_dataset_joints(spec: str, scene) -> None:
        """A LeRobot dataset whose joints are not the placed machine's is refused with the names (in the page's
        words); a dataset that cannot be read is left to the loader, which says what is wrong."""
        from .lerobot import dataset_joint_names

        directory = spec[len("lerobot:"):].rpartition("#")[0]
        if not directory or not (Path(directory).expanduser() / "meta" / "info.json").is_file():
            return
        try:
            names = dataset_joint_names(directory)
        except (InputError, OSError, ValueError, KeyError, TypeError):
            return
        (_, placed), = scene.machines.items()
        if names != list(placed.machine.joint_names):
            raise ValueError(f"データセットの関節がロボットと合いません（データセット: {', '.join(names)} ／ "
                             f"ロボット: {', '.join(placed.machine.joint_names)}）")

    def _motion_open(self, args: dict) -> dict:
        """One motion on the placed machine: what the compare screen draws (``screens.screen_data`` with one record)."""
        slot, spec = args.get("slot"), args.get("spec")
        if slot not in ("a", "b"):
            raise ValueError("slot must be 'a' or 'b'")
        if not isinstance(spec, str) or not spec.strip():
            raise ValueError("spec must be a record file or lerobot:<dir>#<episode>")
        spec = spec.strip()
        scene, motion, notes, opened = self._load_motion(spec, ask_latest=slot == "a")
        if opened is not None:
            spec = opened.pop("spec")  # with the version read: a later start opens the same bytes
        with open_source(self.picture) as source:
            out = screens.screen_data(scene.site, scene, self.camera, [motion], source)
        out["note"] = notes[0] if notes else None
        out["spec"] = spec
        if slot == "a":
            out["aOpened"] = opened
            self._update_stage_json(**self._a_keys(spec, opened))
        else:
            self._update_stage_json(b=spec)
        return out

    def _grip(self, args: dict) -> float | None:
        """``args["gripThreshold"]``: None (or absent) for the automatic threshold, else a finite number (rad)."""
        if args.get("gripThreshold") is None:
            return None
        return _number(args, "gripThreshold")

    def _two_motions(self, args: dict, ask_latest: bool = False):
        """``(scene, a, b, specs, notes, opened_a)`` for the compare calls' ``a`` and ``b`` specs: both load on the
        placed machine and fit each other (an ``InputError`` or ``ValueError`` says why not). A ``ps:`` spec comes
        back with the version read; ``opened_a`` is what A opened from PS.com (None for another spec)."""
        from .cli import _check_motions

        specs = []
        for key in ("a", "b"):
            spec = args.get(key)
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError(f"{key} must be a record file or lerobot:<dir>#<episode>")
            specs.append(spec.strip())
        scene, a, notes_a, opened_a = self._load_motion(specs[0], ask_latest=ask_latest)
        _, b, notes_b, opened_b = self._load_motion(specs[1])
        _check_motions(scene, a, b)
        for i, opened in enumerate((opened_a, opened_b)):
            if opened is not None:
                specs[i] = opened.pop("spec")
        return scene, a, b, specs, notes_a + notes_b, opened_a

    def _compare(self, args: dict) -> dict:
        """Two motions side by side on the placed machine: the compare screen's data plus the read-over notes."""
        grip = self._grip(args)
        scene, a, b, specs, notes, opened_a = self._two_motions(args, ask_latest=True)
        with open_source(self.picture) as source:
            out = screens.screen_data(scene.site, scene, self.camera, [a, b], source, grip_threshold=grip)
        out["notes"] = notes
        self._update_stage_json(**self._a_keys(specs[0], opened_a), b=specs[1], gripThreshold=grip)
        return out

    # ---- saving A to PS.com, exporting it, and the copies PS.com did not get ----

    def _a_record(self) -> tuple[dict, dict | None, str]:
        """``(record, opened, label)`` for the motion in A on the placed machine where it stands now, on the decided
        site. A record opened from PS.com (``opened`` is ``aOpened``) is merged: only its device's place, its motion
        and the site frame change. Anything else is a new record of the placed machine. Reads no network."""
        spec = self._kept("a")
        if spec is None:
            raise ValueError(_NO_A)
        _, motion, _, opened = self._load_motion(spec)
        placed = self._placed_machine()  # the load needs exactly one
        placement = (placed["placed"]["x"], placed["placed"]["y"], placed["placed"]["yaw"])
        site_id = self._decided_site().id
        if opened is None:
            machine = self._machine_ref(placed)
            try:
                record = record_edit.new_record(motion, machine_ref=machine, placement=placement, site_id=site_id)
            except ValueError as e:
                raise ValueError(f"ロボット {machine!r} の名前は記録に使えません（{e}）") from None
            return record, None, motion.label
        folder = catalog.cached(f"{opened['ref']}:{opened['version']}")
        if folder is None:
            raise ValueError(f"記録 {opened['ref']}:{opened['version']} をもう一度読み込んでから保存してください")
        raw = json.loads((folder / "record.json").read_text())
        device = record_edit.device_of(raw, placed["id"]) or _joint_device(raw, spec)
        if device is None:
            raise ValueError("記録のどの機器に保存するか決められません")
        record = record_edit.merge_into(raw, device_id=device, motion=motion, placement=placement, site_id=site_id)
        return record, self._kept_opened() or opened, motion.label

    @staticmethod
    def _machine_ref(placed: dict) -> str:
        """What a new record names the placed machine by: ``handle/name:version`` for a machine of the catalog's
        cache, else its own name (push brings that machine along to PS.com)."""
        try:
            rel = Path(placed["urdf"]).resolve().relative_to((catalog.cache_root() / "catalog").resolve())
            ref = f"{rel.parts[0]}/{rel.parts[1]}:{rel.parts[2]}"
            catalog.parse_ref(ref)
            return ref
        except (ValueError, IndexError, InputError):
            return placed["id"]

    def _send(self, path: Path, ref: str, mode: str, base: str | None) -> str:
        """``unsent``'s sender: push the record at ``path`` to ``ref`` (a ref kept while signed out has no handle:
        the signed-in one's), private, never starting a login; the ref written. A record made here (new, not from a
        PS.com record) brings its local machine along; one made from a PS.com record keeps the products it names."""
        url = catalog._base()
        if "/" not in ref:
            handle = push.login_status(url)["handle"]
            if not handle:
                raise PushError("signin", 401)
            ref = f"{handle}/{ref}"
        try:
            made_here = mode == "new" and "basedOn" not in json.loads(Path(path).read_text())
        except (OSError, ValueError, TypeError):
            made_here = True
        placed = self._placed_machine()
        try:
            result = push.push(ref, path, private=True, machine_dir=Path(placed["urdf"]).parent if placed else None,
                               log=lambda _: None, interactive=False, over=mode == "over",
                               over_base=base if mode == "over" else None, companions=made_here)
        except InputError as e:
            if mode == "new" and str(e) == push._EXISTS.format(ref=ref):
                raise PushError("taken", 409, note=f"{ref} はもうあります") from e
            raise
        return result.ref

    def _now_a(self, ref: str, version: str, label: str | None) -> str:
        """A holds ``ref`` at ``version`` from now on (what was just saved there); the new spec."""
        spec = f"{PS_PREFIX}{ref}:{version}" + (f"#{label}" if label else "")
        self._update_stage_json(a=spec, aOpened={"ref": ref, "version": version, "latest": version, "own": True})
        return spec

    def _keep_unsent(self, record: dict, ref: str, mode: str, base: str | None, reason: str) -> dict:
        return {"saved": False, "unsent": unsent.keep(self.work, record, ref=ref, mode=mode, base=base),
                "reason": reason}

    def _record_save(self, args: dict) -> dict:
        """上書き保存 (``mode: over``: the person's own PS.com record in A, as its next version, made from the
        version they saw as the latest) or 新しい記録として保存 (``mode: new``: under ``name``, titled ``title``).
        Signed out or offline the record is kept in ``unsent`` (``saved: false``). A record saved over elsewhere
        meanwhile goes as a new one under ``<name>-2`` with a ``note``."""
        mode = args.get("mode")
        if mode not in ("over", "new"):
            raise ValueError("mode must be 'over' or 'new'")
        with self._save_lock:
            spec = self._kept("a")
            if spec is None:
                raise ValueError(_NO_A)
            if mode == "over":
                kept = self._kept_opened()
                if not (kept and kept.get("own") is True):
                    raise ValueError(_NOT_OWN)
            record, opened, label = self._a_record()
            url = catalog._base()
            if mode == "over":
                ref, base = opened["ref"], opened["latest"]
            else:
                name = args.get("name") or _catalog_name(label)
                if not isinstance(name, str) or not re.fullmatch(catalog._NAME, name):
                    raise ValueError(_NAME_WORDS)
                title = args.get("title")
                record["name"] = title.strip() if isinstance(title, str) and title.strip() else label
                if opened is not None:
                    record["basedOn"] = f"{opened['ref']}:{opened['version']}"
                handle = push.login_status(url)["handle"]
                ref, base = (f"{handle}/{name}" if handle else name), None
            login = push.login_status(url)
            if not login["signedIn"] or not login["handle"]:
                return self._keep_unsent(record, ref, mode, base, unsent.SIGN_IN)
            with tempfile.TemporaryDirectory(prefix="psbx-save-") as tmp:
                path = Path(tmp) / "record.json"
                path.write_text(json.dumps(record, ensure_ascii=False))
                try:
                    written, _, note = unsent.send_one(path, ref, mode, base, self._send)
                except PushError as e:
                    if e.code == "signin":
                        return self._keep_unsent(record, ref, mode, base, unsent.SIGN_IN)
                    if e.code == "taken":
                        raise ValueError(f"{ref} はもうあります。別の名前を指定してください") from None
                    raise ValueError(f"PS.com に保存できませんでした（{e.note or e}）") from None
                except CatalogError as e:  # not a PushError: PS.com could not be reached
                    return self._keep_unsent(record, ref, mode, base, f"{unsent.OFFLINE}（{e}）")
            ref, _, version = written.partition(":")
            label_part = spec.partition("#")[2] if opened is not None and "#" in spec else None
            out = {"saved": True, "ref": ref, "version": version, "spec": self._now_a(ref, version, label_part)}
            if note:
                out["note"] = note
            return out

    def _record_export(self) -> dict:
        """エクスポート: A as a record file in the work folder, built as a save builds it
        (``<label>.record.json``; beside an existing one ``<label>-2.record.json``...). Reads no network."""
        record, opened, label = self._a_record()
        if opened is None:
            record["name"] = label
        stem = _safe_name(label)
        path, k = self.work / f"{stem}.record.json", 1
        while path.exists():
            k += 1
            path = self.work / f"{stem}-{k}.record.json"
        write_text_atomic(path, json.dumps(record, ensure_ascii=False, indent=1) + "\n")
        return {"path": str(path)}

    def _unsent_send(self) -> dict:
        """Send the kept copies (signed out: none is tried); ``{sent, left, notes}``. A copy saved over A's record
        makes A that version."""
        with self._save_lock:
            waiting = unsent.pending(self.work)
            if not waiting:
                return {"sent": [], "left": [], "notes": []}
            login = push.login_status(catalog._base())
            if not login["signedIn"] or not login["handle"]:
                return {"sent": [], "left": [_shown(p) for p in waiting], "notes": [unsent.SIGN_IN]}
            sent, left, notes = unsent.send_all(self.work, self._send)
            opened, spec = self._kept_opened(), self._kept("a") or ""
            for s in sent:
                if opened is not None and s["mode"] == "over" and s["ref"] == opened["ref"]:
                    self._now_a(s["ref"], s["version"], spec.partition("#")[2] or None)
            return {"sent": sent, "left": [_shown(p) for p in left], "notes": notes}

    def _export(self, args: dict) -> dict:
        """Start writing a file set in the work folder (one export at a time); answers the job id. ``replay``: one
        motion over the photo as an mp4. ``compare``: the four report files of two motions, in their own folder."""
        what = args.get("what")
        if what not in ("replay", "compare"):
            raise ValueError("what must be 'replay' or 'compare'")
        if what == "replay":
            spec = args.get("spec")
            if not isinstance(spec, str) or not spec.strip():
                raise ValueError("spec must be a record file or lerobot:<dir>#<episode>")
            scene, motion, _, _ = self._load_motion(spec.strip())
            out = self.work / f"replay-{_safe_name(motion.label)}.mp4"

            def produce(progress) -> list[Path]:
                with open_source(self.picture) as source:
                    render_overlay(scene, self.camera, source, [motion], out, progress=progress)
                return [out]
        else:
            grip = self._grip(args)
            scene, a, b, _, _, _ = self._two_motions(args)
            site = self._decided_site()
            folder = self.work / "compare" / f"{_safe_name(a.label)}-vs-{_safe_name(b.label)}"

            def produce(progress) -> list[Path]:
                # Only this export's files stay: an old video must not sit next to new numbers
                for name in ("replay.mp4", "gaps.json", "gaps.csv", "gaps.png"):
                    (folder / name).unlink(missing_ok=True)
                written = report.write_report(compare(scene, a, b, grip_threshold=grip), folder, site=site)
                try:
                    with open_source(self.picture) as source:
                        video = render_overlay(scene, self.camera, source, [a, b], folder / "replay.mp4",
                                               progress=progress)
                except Exception as e:  # the numbers are written; say that the video is not
                    text = str(e).strip()
                    raise RuntimeError("動画を書き出せませんでした（数値のファイルは書きました）: "
                                       + (text.splitlines()[0] if text else type(e).__name__)) from e
                return [Path(p) for p in (*written.values(), video)]
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
                paths = [str(p) for p in produce(progress)]
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
