"""The screens' data builder (no MCP SDK needed): the compare screen's data for one or two motions.

Uses the synthetic site, scene and records of tests/synthetic.py.
"""
import json

import numpy as np
import pytest

from physical_sandbox import comparison as compare_mod
from physical_sandbox import screens
from physical_sandbox.motion import motion_from_numpy
from physical_sandbox.scene import Scene
from physical_sandbox.sources import open_source
from synthetic import look_at, make_world, motion


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    site, placed, still = make_world(tmp_path)
    scene = Scene(site=site, machines={"blocks": placed})
    with open_source(still) as source:
        yield {"site": site, "scene": scene, "placed": placed, "source": source, "dir": tmp_path}


def parts_of(d):
    """What the screen fetches with its part tool, as one dict."""
    text = "".join(json.loads(screens.data_part(d["dataKey"], i, "test"))["data"] for i in range(d["dataParts"]))
    return json.loads(text)


def test_screen_data_sends_poses_in_the_machine_frame_with_the_base_apart(synthetic):
    s = synthetic
    a, b = motion(1.0, "slow"), motion(1.5, "wide")
    d = screens.screen_data(s["site"], s["scene"], "camera", [a, b], s["source"])
    part = parts_of(d)
    base = np.array(d["base"]).reshape(4, 4)
    assert len(d["base"]) == 16 and np.allclose(base, s["placed"].base_transform(), atol=1e-6)
    assert d["placed"] == {"id": "blocks", "x": 0.1, "y": 0.0, "yaw": 0.3}
    machine = s["placed"].machine
    names = [l["name"] for l in d["model"]["links"]]
    for r, m in enumerate((a, b)):
        for k in (0, 5, d["frames"] - 1):
            local = machine.link_poses(m.at(part["t"][k]))
            for i, name in enumerate(names):
                got = np.array(part["poses"][r][k][i]).reshape(4, 4)
                assert np.allclose(got, local[name], atol=2e-5)
    # the machine's own frame: the base link sits at the origin, not at the placement
    assert np.allclose(np.array(part["poses"][0][0][0]).reshape(4, 4)[:3, 3], 0, atol=1e-5)


def test_screen_data_for_one_motion_has_one_record_and_the_small_figures(synthetic):
    s = synthetic
    a = motion(1.0, "slow")
    d = screens.screen_data(s["site"], s["scene"], "camera", [a], s["source"])
    assert len(d["records"]) == 1 and d["records"][0]["label"] == "slow"
    assert set(d["figures"]) == {"duration_s", "grasp_counts", "grip_threshold"}
    assert d["figures"]["duration_s"] == {"a": pytest.approx(1.0)} and d["figures"]["grasp_counts"] == {"a": 0}
    part = parts_of(d)
    assert len(part["poses"]) == 1 and len(part["poses"][0]) == d["frames"] == len(part["t"])
    assert part["t"][-1] == pytest.approx(1.0)


def test_screen_data_takes_the_labels_over(synthetic):
    s = synthetic
    d = screens.screen_data(s["site"], s["scene"], "camera", [motion(1.0, "slow"), motion(1.5, "wide")],
                            s["source"], labels=["手本", None])
    assert [r["label"] for r in d["records"]] == ["手本", "wide"]
    assert d["figures"]["labels"] == {"a": "手本", "b": "wide"}


def test_grasps_of_a_machine_without_a_gripper_says_so(synthetic):
    got = compare_mod.grasps_of(motion(0.5, "flat"), synthetic["scene"])
    assert got["count"] == 0 and got["times"] == [] and got["threshold"] is None and "gripper" in got["note"]


def test_grasps_of_matches_compare_on_a_machine_with_a_gripper(tmp_path):
    from physical_sandbox.machine import load_machine
    from physical_sandbox.scene import PlacedMachine
    from physical_sandbox.site import Site
    urdf = tmp_path / "g.urdf"
    urdf.write_text("""<robot name="g"><link name="base"><inertial><mass value="1"/>
<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="jaw"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="gripper" type="revolute"><parent link="base"/><child link="jaw"/><axis xyz="0 0 1"/>
<limit lower="0" upper="1" effort="1" velocity="1"/></joint></robot>""")
    scene = Scene(site=Site(id="s", cameras={"camera": look_at((-2.5, 0.8, 1.2))}),
                  machines={"g": PlacedMachine(machine=load_machine(urdf), urdf=str(urdf), x=0, y=0, yaw=0)})
    # open, close, open, close: two closings
    q = np.array([1, 1, 0, 0, 1, 1, 0, 0], float).reshape(-1, 1)
    m = motion_from_numpy(q, fps=1.0, joint_names=["gripper"], machine="g", label="g")
    got = compare_mod.grasps_of(m, scene)
    both = compare_mod.compare(scene, m, m)
    assert got["count"] == 2 == both.grasp_counts["a"] and got["threshold"] == both.grip_threshold
    assert len(got["times"]) == 2 and got["note"] == ""
    assert [round(g["a_s"], 6) for g in both.grasps] == [round(t, 6) for t in got["times"]]
