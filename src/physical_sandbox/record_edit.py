"""Records psbx writes: a new one with a placement, saving over one that keeps what psbx does not read."""
from __future__ import annotations

import copy
import math
import re

from .motion import Motion, _PRODUCT, _record_dict, joint_samples, record_floor

_REF = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*:[a-z0-9][a-z0-9.-]*$")


def _pose(placement) -> dict:
    x, y, yaw = (float(v) for v in placement)
    return {"xyz": [x, y, 0.0], "rpy": [0.0, 0.0, yaw]}


def new_record(motion: Motion, *, machine_ref: str, placement, site_id: str,
               title: str | None = None, based_on: str | None = None, floor=None) -> dict:
    """A recording of one motion placed at ``placement`` (x, y, yaw) in the site frame (x forward, y left,
    z up: the frame PS.com records use for ``pose``). ``machine_ref`` is a catalog ref or a bare name.
    ``floor``: the site's floor corners in that frame, carried as the recording's ``floor`` (``record_floor``)."""
    if not site_id:
        raise ValueError("site_id is empty")
    if not (_REF.match(machine_ref) or _PRODUCT.match(machine_ref)):
        raise ValueError(f"machine {machine_ref!r} is neither a catalog ref (handle/name:version) nor a bare name")
    rec = _record_dict([(motion, machine_ref, _pose(placement))], site_id, floor)
    if title:
        rec["name"] = title
    if based_on:
        rec["basedOn"] = based_on
    return rec


_SAME_FLOOR_M = 0.005  # edges closer than this are one floor measured twice


def floor_note(raw: dict, site_edges_m) -> str | None:
    """What to tell the person when a recording's own floor (``raw["floor"]``, edges 1→2 and 2→3) is not the
    size of this site's (``site_edges_m``): the recording was made on another floor, and its machines stand
    from that floor's corner 1. None when they agree, or when either floor is missing or cannot be read."""
    try:
        c = [(float(p["x"]), float(p.get("y", 0.0)), float(p["z"])) for p in raw["floor"]["corners"]]
        own = tuple(math.dist(c[i], c[i + 1]) for i in (0, 1))
        here = (float(site_edges_m[0]), float(site_edges_m[1]))
    except (KeyError, TypeError, ValueError, IndexError):
        return None
    if all(abs(a - b) <= _SAME_FLOOR_M for a, b in zip(own, here)):
        return None
    own_size, here_size = (f"{round(e[0], 3):g} × {round(e[1], 3):g} m" for e in (own, here))
    return (f"この記録は {own_size} の床で作られています（この場所の床は {here_size}）。"
            "ロボットの位置は、記録の床の角 1 からの位置です")


def _name_part(s: str) -> str:
    return s.split("/")[-1].split(":")[0]


def device_of(raw: dict, product_or_machine: str) -> str | None:
    """The id of the one device whose product is this machine (a bare name, or a ref whose name part
    matches); None when no device or several match."""
    try:
        want = _name_part(product_or_machine)
        found = [d.get("id") for d in raw["equipment"]["devices"]
                 if isinstance(d.get("product"), str) and _name_part(d["product"]) == want]
    except (KeyError, TypeError, AttributeError):
        return None
    return found[0] if len(found) == 1 and isinstance(found[0], str) else None


def _moved_pose(pose, placement) -> dict:
    """``pose`` moved to ``placement`` (x, y, yaw) on the floor: its z, roll and pitch stay (an arm standing on a
    table stays up there). A pose that is not ``{xyz: [3 numbers], rpy: [3 numbers]}`` is replaced by one on the floor."""
    def three(v) -> bool:
        return isinstance(v, list) and len(v) == 3 and all(
            isinstance(n, (int, float)) and not isinstance(n, bool) for n in v)
    if not (isinstance(pose, dict) and three(pose.get("xyz")) and three(pose.get("rpy"))):
        return _pose(placement)
    x, y, yaw = (float(v) for v in placement)
    return {**pose, "xyz": [x, y, pose["xyz"][2]], "rpy": [pose["rpy"][0], pose["rpy"][1], yaw]}


def merge_into(raw: dict, *, device_id: str, motion: Motion | None, placement, site_id: str | None,
               floor=None) -> dict:
    """A deep copy of ``raw`` where only the device's place (x, y and yaw of its pose), its trace and the site
    frame change; ``None`` leaves that part as it was. ``floor``: the corners of the floor of the site the
    recording now stands on (``record_floor``), which replace the recording's own."""
    out = copy.deepcopy(raw)
    devices = (out.get("equipment") or {}).get("devices") or []
    dev = next((d for d in devices if d.get("id") == device_id), None)
    if dev is None:
        raise ValueError(f"no device {device_id!r} in the record")
    if placement is not None:
        dev["pose"] = _moved_pose(dev.get("pose"), placement)
    if motion is not None:
        traces = out.setdefault("traces", [])
        tr = next((t for t in traces if t.get("deviceId") == device_id), None)
        if tr is None:
            traces.append({"deviceId": device_id, "samples": joint_samples(motion)})
        else:
            tr["samples"] = joint_samples(motion)
    if site_id is not None:
        out["siteFrameId"] = site_id
        out.setdefault("equipment", {}).setdefault("site", {})["id"] = site_id
    if floor is not None:
        out["floor"] = record_floor(floor)
    return out
