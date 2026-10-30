import json
import math

import numpy as np
import pytest

from physical_sandbox.compare import Comparison, compare
from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_numpy
from physical_sandbox.scene import PlacedMachine, Scene
from physical_sandbox.site import Site

INERTIAL = '<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
URDF = f"""<?xml version="1.0"?>
<robot name="rail">
  <link name="base">{INERTIAL}</link>
  <link name="rail">{INERTIAL}</link>
  <link name="carriage">{INERTIAL}</link>
  <link name="jaw">{INERTIAL}</link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="rail"/><origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="side" type="prismatic">
    <parent link="rail"/><child link="carriage"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="gripper" type="revolute">
    <parent link="carriage"/><child link="jaw"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""
NAMES = ["slide", "side", "gripper"]
FPS = 30.0


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "rail.urdf"
    p.write_text(URDF)
    placed = PlacedMachine(machine=load_machine(p), urdf=str(p), x=0.0, y=0.0, yaw=0.0)
    return Scene(site=Site(id="s-1", cameras={}), machines={"rail": placed})


def _motion(slide, grip=None, side=None, *, label="", fps=FPS, machine="rail", names=NAMES):
    slide = np.asarray(slide, float)
    grip = np.zeros_like(slide) if grip is None else np.asarray(grip, float)
    side = np.zeros_like(slide) if side is None else np.asarray(side, float)
    return motion_from_numpy(np.column_stack([slide, side, grip]), fps=fps, joint_names=names,
                             machine=machine, label=label)


def _smooth(duration, slow=1.0, length=0.3):
    t = np.arange(0, duration * slow + 1e-9, 1 / FPS)
    return length * (1 - np.cos(np.pi * t / (duration * slow))) / 2


def test_sideways_offset_is_the_tool_deviation(scene):
    s = _smooth(2.0)
    a, b = _motion(s, label="demo"), _motion(s, side=np.full_like(s, 0.02), label="policy")
    c = compare(scene, a, b)
    assert isinstance(c, Comparison)
    assert c.tool_deviation_m["max"] == pytest.approx(0.02, abs=1e-6)
    assert c.tool_deviation_m["mean"] == pytest.approx(0.02, abs=1e-6)
    assert 0.0 <= c.tool_deviation_m["at_s"] <= 2.0
    assert c.joints["side"]["max_rad"] == pytest.approx(0.02, abs=1e-6)
    assert c.joints["slide"]["max_rad"] == pytest.approx(0.0, abs=1e-9)
    assert c.joints["gripper"]["max_rad"] == pytest.approx(0.0, abs=1e-9)
    assert c.shape_deviation_m == pytest.approx(0.02, abs=1e-6)  # a sideways shift cannot be warped away
    assert c.duration_s == {"a": pytest.approx(2.0), "b": pytest.approx(2.0), "diff": pytest.approx(0.0)}


def test_slower_copy_differs_in_time_but_not_in_shape(scene):
    a = _motion(_smooth(2.0))
    b = _motion(_smooth(2.0, slow=1.5))
    c = compare(scene, a, b)
    assert c.tool_deviation_m["max"] > 0.01
    assert c.shape_deviation_m < 0.002
    assert c.duration_s["a"] == pytest.approx(2.0)
    assert c.duration_s["b"] == pytest.approx(3.0)
    assert c.duration_s["diff"] == pytest.approx(1.0)  # b - a
    # the shorter motion holds its last pose, so the gap runs to the end of the longer one
    assert c.series["t"][-1] >= 3.0 - 1e-9
    assert len(c.series["t"]) == len(c.series["tool_gap_m"])
    assert c.series["tool_gap_m"][-1] == pytest.approx(0.0, abs=1e-9)


def test_constant_velocity_has_no_jerk_and_a_wiggle_has_some(scene):
    line = _motion(np.linspace(0.0, 0.3, 61))
    wiggle = _motion(0.15 + 0.05 * np.sin(np.arange(61) * 0.9))
    c = compare(scene, line, wiggle)
    assert c.jerk["a"] == pytest.approx(0.0, abs=1e-6)
    assert c.jerk["b"] > 1.0


def test_one_grasp_each_pairs_up_with_a_known_time_gap(scene):
    n = 91
    t = np.arange(n) / FPS
    slide = np.linspace(0.0, 0.3, n)

    def closing(at):  # gripper 1 -> 0 over 0.2 s starting at ``at``
        return np.clip(1.0 - (t - at) / 0.2, 0.0, 1.0)

    a = _motion(slide, closing(1.0))
    b = _motion(slide, closing(1.5))
    c = compare(scene, a, b)
    assert len(c.grasps) == 1
    g = c.grasps[0]
    # the default threshold is the middle of the gripper's limits (0..1 -> 0.5): halfway through the 0.2 s close
    assert g["a_s"] == pytest.approx(1.1, abs=1e-6)
    assert g["b_s"] == pytest.approx(1.6, abs=1e-6)
    assert g["tool_gap_m"] == pytest.approx(0.1 * 0.5, abs=1e-6)  # slide moves 0.1 m/s, 0.5 s apart
    assert "grasps" not in c.unknown
    # a threshold of 0.25 is crossed later in the close
    c2 = compare(scene, a, b, grip_threshold=0.25)
    assert c2.grasps[0]["a_s"] == pytest.approx(1.15, abs=1e-6)


def test_opening_is_not_a_grasp_and_extra_grasps_are_counted(scene):
    n = 91
    t = np.arange(n) / FPS
    opens = np.clip((t - 1.0) / 0.2, 0.0, 1.0)  # 0 -> 1: opening only
    closes_once = np.where(t < 1.0, 1.0, 0.0)
    twice = np.where((t < 0.8) | ((t >= 1.4) & (t < 2.2)), 1.0, 0.0)  # two closes
    slide = np.linspace(0.0, 0.3, n)
    c = compare(scene, _motion(slide, opens), _motion(slide, twice))
    assert c.grasps == []
    assert c.grasp_counts == {"a": 0, "b": 2}
    c2 = compare(scene, _motion(slide, closes_once), _motion(slide, twice))
    assert len(c2.grasps) == 1 and c2.grasp_counts == {"a": 1, "b": 2}


def test_no_gripper_joint_says_grasps_are_unknown(scene):
    import copy
    a = motion_from_numpy(np.zeros((5, 1)), fps=FPS, joint_names=["slide"], machine="rail")
    # a machine without a gripper: a copy of the placed rail with one joint (and a stubbed fk)
    placed = copy.copy(scene.machines["rail"])
    m = copy.copy(placed.machine)
    m.joint_names = ["slide"]
    m.limits = placed.machine.limits[:1]
    m.fk = lambda q: np.array([float(q[0]), 0.0, 0.1])
    scene2 = Scene(site=scene.site, machines={"rail": PlacedMachine(machine=m, urdf=placed.urdf, x=0, y=0, yaw=0)})
    c = compare(scene2, a, a)
    assert c.grasps == []
    assert "grasps" in c.unknown
    assert c.unknown_reasons["grasps"]


def test_to_dict_is_plain_json_with_unknowns_and_labels(scene):
    s = _smooth(1.0)
    c = compare(scene, _motion(s, label="demo"), _motion(s + 0.01, label="policy"), step_s=1 / 20)
    d = c.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["labels"] == {"a": "demo", "b": "policy"}
    assert d["step_s"] == 1 / 20
    assert d["unknown"] == ["clearance", "reach", "world"]
    assert d["unknown_reasons"]["clearance"] == "奥行きと対象物は段 3"
    for key in ("tool_deviation_m", "shape_deviation_m", "joints", "duration_s", "jerk", "grasps", "series"):
        assert key in d
    assert set(d["tool_deviation_m"]) == {"max", "mean", "at_s"}
    assert set(d["joints"]["slide"]) == {"mean_rad", "max_rad", "at_s"}
    assert type(d["tool_deviation_m"]["max"]) is float
    assert all(type(x) is float for x in d["series"]["t"])


def test_refuses_different_machines_or_joints(scene):
    a = _motion([0.0, 0.1])
    with pytest.raises(ValueError, match="machine"):
        compare(scene, a, _motion([0.0, 0.1], machine="other"))
    with pytest.raises(ValueError, match="joint"):
        compare(scene, a, _motion([0.0, 0.1], names=["slide", "side", "grip"]))
    with pytest.raises(ValueError, match="step_s"):
        compare(scene, a, a, step_s=0)
