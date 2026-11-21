import json

import numpy as np
import pytest

from physical_sandbox.motion import Motion, load_record
from physical_sandbox.record_edit import device_of, merge_into, new_record


def _motion(machine="arm", v=0.1):
    return Motion(machine=machine, joint_names=["a", "b"], t=np.array([0.0, 1.0]),
                  q=np.array([[v, v], [v + 1, v + 1]]), source="numpy", label="L")


def _pscom():
    return {
        "schemaVersion": 3, "siteFrameId": "old", "name": "Demo", "basedOn": "h/n:1", "psbxOnly": {"k": [1]},
        "floor": {"corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}, "routes": [{"id": "r"}],
        "markers": [{"id": "1", "x": 0, "y": 0, "z": 0}], "issued": ["x"], "names": {"arm-1": "A", "car-1": "C"},
        "equipment": {"site": {"id": "old", "extra": 1}, "devices": [
            {"id": "arm-1", "product": "arm", "pose": {"xyz": [9, 9, 9], "rpy": [0, 0, 0]}, "extra": "e"},
            {"id": "car-1", "product": "car", "pose": {"xyz": [1, 2, 0], "rpy": [0, 0, 0.5]}}]},
        "traces": [
            {"deviceId": "arm-1", "samples": [{"t": 0, "joints": {"a": 0, "b": 0}, "position": [1, 1, 0]}]},
            {"deviceId": "car-1", "samples": [{"t": 0, "position": [1, 2, 0], "yaw": 0.5}]}],
    }


def _dump(d):
    return json.dumps(d, sort_keys=True)


def test_new_record_pose_and_based_on_with_ref():
    rec = new_record(_motion("arm"), machine_ref="acme/arm:2", placement=(1.0, 2.0, 0.3), site_id="s",
                     title="T", based_on="acme/x:3")
    dev = rec["equipment"]["devices"][0]
    assert dev["product"] == "acme/arm:2" and dev["id"] == "arm-1"
    assert dev["pose"] == {"xyz": [1.0, 2.0, 0.0], "rpy": [0.0, 0.0, 0.3]}
    assert rec["name"] == "T" and rec["basedOn"] == "acme/x:3" and rec["schemaVersion"] == 3
    assert rec["traces"][0]["deviceId"] == "arm-1"


def test_new_record_bare_name_without_optionals():
    rec = new_record(_motion(), machine_ref="arm", placement=(0, 0, 0), site_id="s")
    assert "name" not in rec and "basedOn" not in rec
    with pytest.raises(ValueError):
        new_record(_motion(), machine_ref="Bad Name", placement=(0, 0, 0), site_id="s")


def test_merge_changes_only_the_chosen_device():
    raw = _pscom()
    before = _dump(raw)
    out = merge_into(raw, device_id="arm-1", motion=_motion(v=0.5), placement=(3, 4, 1.0), site_id="new")
    assert _dump(raw) == before  # input untouched
    expect = _pscom()
    expect["siteFrameId"] = "new"
    expect["equipment"]["site"]["id"] = "new"
    expect["equipment"]["devices"][0]["pose"] = {"xyz": [3.0, 4.0, 9], "rpy": [0, 0, 1.0]}  # z, roll, pitch kept
    expect["traces"][0]["samples"] = [
        {"t": 0.0, "joints": {"a": 0.5, "b": 0.5}}, {"t": 1.0, "joints": {"a": 1.5, "b": 1.5}}]
    assert _dump(out) == _dump(expect)


def test_merge_with_nones_is_identity():
    raw = _pscom()
    assert _dump(merge_into(raw, device_id="arm-1", motion=None, placement=None, site_id=None)) == _dump(raw)


def test_merge_appends_trace_when_device_has_none():
    raw = _pscom()
    raw["traces"] = [t for t in raw["traces"] if t["deviceId"] != "arm-1"]
    out = merge_into(raw, device_id="arm-1", motion=_motion(), placement=None, site_id=None)
    assert out["traces"][-1]["deviceId"] == "arm-1" and len(out["traces"][-1]["samples"]) == 2
    with pytest.raises(ValueError):
        merge_into(raw, device_id="nope", motion=None, placement=None, site_id=None)


def test_device_of():
    raw = _pscom()
    assert device_of(raw, "arm") == "arm-1"
    assert device_of(raw, "h/arm:3") == "arm-1"
    assert device_of(raw, "nothing") is None
    raw["equipment"]["devices"].append({"id": "arm-2", "product": "h/arm:1"})
    assert device_of(raw, "arm") is None
    assert device_of({}, "arm") is None


def test_merged_record_still_loads(tmp_path):
    out = merge_into(_pscom(), device_id="arm-1", motion=_motion(v=0.2), placement=(1, 1, 0), site_id=None)
    p = tmp_path / "r.json"
    p.write_text(json.dumps(out))
    ms = load_record(p)
    assert len(ms) == 1 and ms[0].machine == "arm" and np.allclose(ms[0].q[0], [0.2, 0.2])


def test_merge_moves_an_existing_pose_only_in_x_y_and_yaw():
    raw = _pscom()
    raw["equipment"]["devices"][0]["pose"] = {"xyz": [9, 9, 0.7], "rpy": [0.1, 0.2, 0.3]}
    out = merge_into(raw, device_id="arm-1", motion=None, placement=(3, 4, 1.0), site_id=None)
    assert out["equipment"]["devices"][0]["pose"] == {"xyz": [3.0, 4.0, 0.7], "rpy": [0.1, 0.2, 1.0]}
    assert out["traces"] == raw["traces"]


def test_merge_gives_a_device_without_a_usable_pose_one_on_the_floor():
    for pose in (None, {"xyz": [1, 2]}, {"xyz": [1, 2, 3]}, "x"):
        raw = _pscom()
        if pose is None:
            del raw["equipment"]["devices"][0]["pose"]
        else:
            raw["equipment"]["devices"][0]["pose"] = pose
        out = merge_into(raw, device_id="arm-1", motion=None, placement=(3, 4, 1.0), site_id=None)
        assert out["equipment"]["devices"][0]["pose"] == {"xyz": [3.0, 4.0, 0.0], "rpy": [0.0, 0.0, 1.0]}, pose


# 2026-11-21: a recording carries the floor it was made on (PS.com's `floor`: four corners in its site
# frame, +Y up), so a headset lines it up on those corners. The floor here is x forward, y left, z up
A4 = np.array([[0, 0, 0], [0.297, 0, 0], [0.297, 0.21, 0], [0, 0.21, 0]], float)
A4_SITE = {"source": "corners", "corners": [{"x": 0.0, "y": 0.0, "z": 0.0}, {"x": 0.297, "y": 0.0, "z": 0.0},
                                            {"x": 0.297, "y": 0.0, "z": -0.21}, {"x": 0.0, "y": 0.0, "z": -0.21}]}


def test_new_record_carries_the_floor_in_the_records_frame():
    rec = new_record(_motion(), machine_ref="arm", placement=(0.1, 0.05, 0), site_id="s", floor=A4)
    assert rec["floor"] == A4_SITE
    assert "-0.0" not in json.dumps(rec["floor"])
    assert "floor" not in new_record(_motion(), machine_ref="arm", placement=(0, 0, 0), site_id="s")


def test_merge_onto_another_site_takes_that_sites_floor():
    raw = _pscom()
    raw["floor"] = {"source": "planes", "corners": [{"x": 0, "y": 0, "z": 0}, {"x": 4, "y": 0, "z": 0},
                                                    {"x": 4, "y": 0, "z": 3}, {"x": 0, "y": 0, "z": 3}]}
    dev = raw["equipment"]["devices"][0]["id"]
    out = merge_into(raw, device_id=dev, motion=None, placement=(0.1, 0.05, 0), site_id="here", floor=A4)
    assert out["floor"] == A4_SITE
    # No floor given: the recording's own stays, as every part not given does
    assert merge_into(raw, device_id=dev, motion=None, placement=None, site_id="here")["floor"] == raw["floor"]


# 2026-11-21: a recording made on another floor (a sheet of another size, another room) is placed from
# that floor's corner 1; opened on this site, the page says so. The same floor says nothing
def test_floor_note_only_when_the_records_floor_is_not_this_sites():
    from physical_sandbox.record_edit import floor_note
    assert floor_note({"floor": A4_SITE}, (0.297, 0.21)) is None
    assert floor_note({"floor": A4_SITE}, (0.2972, 0.2096)) is None  # within a few millimetres
    assert floor_note({}, (0.297, 0.21)) is None and floor_note({"floor": A4_SITE}, None) is None
    assert floor_note({"floor": {"corners": "junk"}}, (0.297, 0.21)) is None  # a floor not read is not judged
    note = floor_note({"floor": A4_SITE}, (0.42, 0.297))
    assert "0.297 × 0.21 m" in note and "0.42 × 0.297 m" in note
