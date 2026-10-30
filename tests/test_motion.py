import json
import os
from pathlib import Path

import numpy as np
import pytest

from physical_sandbox.motion import Motion, load_record, motion_from_numpy, save_record

JOINTS = ["a", "b"]
SCHEMA_DIR = Path(os.environ.get(
    "PS_RECORDING_SCHEMA_DIR", "~/work/physical-sandbox-proto/schema")).expanduser()
needs_schema = pytest.mark.skipif(
    not (SCHEMA_DIR / "recording.schema.json").exists(),
    reason="0.4 recording schema not found (set PS_RECORDING_SCHEMA_DIR)")


def _motion(**kw):
    args = dict(q=[[0.0, 1.0], [1.0, 3.0], [2.0, 2.0]], fps=2.0, joint_names=JOINTS,
                machine="so101", label="demo")
    args.update(kw)
    return motion_from_numpy(args.pop("q"), **args)


def test_numpy_with_fps_makes_time_from_zero():
    m = _motion()
    assert isinstance(m, Motion)
    assert np.allclose(m.t, [0.0, 0.5, 1.0])
    assert m.q.shape == (3, 2)
    assert m.source == "numpy" and m.label == "demo" and m.machine == "so101"
    assert m.joint_names == JOINTS


def test_numpy_with_t_starts_at_zero():
    m = _motion(fps=None, t=[10.0, 10.5, 12.0])
    assert np.allclose(m.t, [0.0, 0.5, 2.0])


def test_numpy_needs_exactly_one_of_fps_and_t():
    with pytest.raises(ValueError, match="fps.*t|t.*fps"):
        _motion(fps=None)
    with pytest.raises(ValueError, match="fps.*t|t.*fps"):
        _motion(t=[0.0, 1.0, 2.0])


def test_numpy_refuses_bad_shapes_and_times():
    with pytest.raises(ValueError, match="joint"):
        _motion(q=[[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match="rising"):
        _motion(fps=None, t=[0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="3"):
        _motion(fps=None, t=[0.0, 1.0])
    with pytest.raises(ValueError, match="fps"):
        _motion(fps=0.0)
    with pytest.raises(ValueError, match="finite"):
        _motion(q=[[0.0, np.nan], [0.0, 1.0], [0.0, 1.0]])


def test_at_interpolates_linearly_and_holds_the_ends():
    m = _motion()
    assert np.allclose(m.at(0.25), [0.5, 2.0])
    assert np.allclose(m.at(0.5), [1.0, 3.0])
    assert np.allclose(m.at(0.75), [1.5, 2.5])
    assert np.allclose(m.at(-5), [0.0, 1.0])
    assert np.allclose(m.at(99), [2.0, 2.0])
    assert m.at(0.25).shape == (2,)


def test_motion_is_frozen():
    m = _motion()
    with pytest.raises(Exception):
        m.label = "other"


def _two_motions():
    a = _motion(label="first")
    b = _motion(q=[[0.1, 0.2], [0.3, 0.4]], fps=None, t=[0.0, 0.7], label="second")
    return [a, b]


def test_record_round_trip(tmp_path):
    path = tmp_path / "r.json"
    ms = _two_motions()
    save_record(ms, "site-1", path)
    back = load_record(path)
    assert len(back) == 2
    for x, y in zip(ms, back):
        assert y.machine == x.machine and y.joint_names == x.joint_names
        assert y.label == x.label and y.source == x.source
        assert np.array_equal(y.t, x.t) and np.array_equal(y.q, x.q)


def test_record_names_the_site_frame_and_each_device_once(tmp_path):
    path = tmp_path / "r.json"
    save_record(_two_motions(), "site-1", path)
    rec = json.loads(path.read_text())
    assert rec["siteFrameId"] == "site-1"
    ids = [d["id"] for d in rec["equipment"]["devices"]]
    assert len(set(ids)) == 2
    assert [t["deviceId"] for t in rec["traces"]] == ids
    assert rec["traces"][0]["samples"][1]["joints"] == {"a": 1.0, "b": 3.0}
    assert rec["traces"][0]["samples"][1]["t"] == 0.5


@needs_schema
def test_record_validates_against_the_04_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    from referencing.jsonschema import DRAFT202012

    registry = referencing.Registry()
    for f in SCHEMA_DIR.glob("*.schema.json"):
        s = json.loads(f.read_text())
        registry = registry.with_resource(s["$id"], referencing.Resource(s, DRAFT202012))
    schema = json.loads((SCHEMA_DIR / "recording.schema.json").read_text())
    path = tmp_path / "r.json"
    save_record(_two_motions(), "site-1", path)
    jsonschema.Draft202012Validator(schema, registry=registry).validate(json.loads(path.read_text()))


@needs_schema
def test_the_validator_would_catch_a_bad_record(tmp_path):
    """Guards the test above: it must not pass for every document."""
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    from referencing.jsonschema import DRAFT202012

    registry = referencing.Registry()
    for f in SCHEMA_DIR.glob("*.schema.json"):
        s = json.loads(f.read_text())
        registry = registry.with_resource(s["$id"], referencing.Resource(s, DRAFT202012))
    schema = json.loads((SCHEMA_DIR / "recording.schema.json").read_text())
    path = tmp_path / "r.json"
    save_record(_two_motions(), "site-1", path)
    rec = json.loads(path.read_text())
    rec["traces"][0]["samples"][0]["joints"] = {"a": "x"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema, registry=registry).validate(rec)


def test_save_refuses_nothing_to_save_and_bad_machine_names(tmp_path):
    with pytest.raises(ValueError, match="no motions"):
        save_record([], "site-1", tmp_path / "r.json")
    with pytest.raises(ValueError, match="machine"):
        save_record([_motion(machine="SO 101")], "site-1", tmp_path / "r.json")


def test_load_refuses_a_record_without_traces_of_joints(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"schemaVersion": 3, "frame": {"frameId": "site"}, "markers": [],
                                "equipment": {"site": {"id": "s"}, "devices": []}, "names": {},
                                "traces": [{"deviceId": "x", "samples": [{"t": 0, "position": [0, 0, 0]}]}]}))
    with pytest.raises(ValueError, match="x"):
        load_record(path)
