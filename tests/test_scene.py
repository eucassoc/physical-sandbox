import json
import math

import numpy as np
import pytest

from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_numpy
from physical_sandbox.scene import PlacedMachine, Scene, load_scene, save_scene
from physical_sandbox.site import Site

URDF = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="tip"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <link name="end"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="turn" type="revolute">
    <parent link="base"/><child link="tip"/><origin xyz="0.5 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <joint name="fixed_end" type="fixed"><parent link="tip"/><child link="end"/><origin xyz="0.3 0 0"/></joint>
</robot>
"""


@pytest.fixture
def urdf(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "arm.urdf"
    p.write_text(URDF)
    return p


def _scene(urdf, with_motion=True):
    m = load_machine(urdf)
    placed = PlacedMachine(machine=m, urdf=str(urdf), x=0.2, y=0.1, yaw=math.pi / 2,
                           confirmed_by_person=True)
    motions = []
    if with_motion:
        motions.append(motion_from_numpy([[0.0], [math.pi / 2]], fps=1.0, joint_names=["turn"],
                                         machine="arm", label="demo"))
    return Scene(site=Site(id="s-1", cameras={}), machines={"arm": placed}, motions=motions)


def test_base_transform_is_x_y_and_yaw(urdf):
    T = _scene(urdf).machines["arm"].base_transform()
    assert np.allclose(T[:3, 3], [0.2, 0.1, 0.0])
    # yaw 90 deg: base +x points at site +y
    assert np.allclose(T[:3, :3] @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(T[:3, :3] @ [0, 0, 1], [0, 0, 1])


def test_tool_path_is_fk_turned_and_moved(urdf):
    scene = _scene(urdf)
    motion = scene.motions[0]
    path = scene.tool_path(motion)
    assert path.shape == (2, 3)
    # q=0: tool at base-frame (0.8, 0, 0.1); yawed 90 deg and moved by (0.2, 0.1) -> site (0.2, 0.9, 0.1)
    assert np.allclose(path[0], [0.2, 0.9, 0.1])
    # q=90 deg: base-frame (0.5, 0.3, 0.1) -> site (0.2 - 0.3, 0.1 + 0.5, 0.1)
    assert np.allclose(path[1], [-0.1, 0.6, 0.1])
    m = scene.machines["arm"]
    T = m.base_transform()
    for row, p in zip(motion.q, path):
        assert np.allclose(p, (T @ np.append(m.machine.fk(row), 1.0))[:3])


def test_link_poses_are_in_site_frame(urdf):
    scene = _scene(urdf)
    poses = scene.link_poses(scene.motions[0], 0)
    assert np.allclose(poses["tip"][:3, 3], [0.2, 0.6, 0.1])  # joint origin: base-frame (0.5, 0, 0.1)
    assert np.allclose(poses["end"][:3, 3], [0.2, 0.9, 0.1])
    assert np.allclose(poses["tip"][:3, :3] @ [1, 0, 0], [0, 1, 0])  # tip frame = base yawed 90 deg at q=0
    poses1 = scene.link_poses(scene.motions[0], 1)
    assert np.allclose(poses1["tip"][:3, :3] @ [1, 0, 0], [-1, 0, 0])  # joint turned another 90 deg


def test_tool_path_refuses_other_joint_names(urdf):
    scene = _scene(urdf)
    bad = motion_from_numpy([[0.0]], fps=1.0, joint_names=["other"], machine="arm")
    with pytest.raises(ValueError, match="other.*turn|turn.*other"):
        scene.tool_path(bad)


def test_tool_path_refuses_unplaced_machine(urdf):
    scene = _scene(urdf)
    bad = motion_from_numpy([[0.0]], fps=1.0, joint_names=["turn"], machine="ghost")
    with pytest.raises(ValueError, match="ghost"):
        scene.tool_path(bad)


def test_round_trip(urdf, tmp_path):
    scene = _scene(urdf)
    path = tmp_path / "out" / "scene.json"
    save_scene(scene, path)
    data = json.loads(path.read_text())
    assert data["format"] == "physical-sandbox.scene/1"
    assert data["site"] == "s-1"
    assert data["machines"]["arm"] == {"urdf": str(urdf), "x": 0.2, "y": 0.1, "yaw": math.pi / 2,
                                       "confirmedByPerson": True}
    assert len(data["records"]) == 1 and not data["records"][0].startswith("/")
    assert (path.parent / data["records"][0]).is_file()

    back = load_scene(path, scene.site)
    p = back.machines["arm"]
    assert (p.x, p.y, p.yaw, p.confirmed_by_person, p.urdf) == (0.2, 0.1, math.pi / 2, True, str(urdf))
    assert p.machine.joint_names == ["turn"]
    assert len(back.motions) == 1
    assert np.allclose(back.motions[0].q, scene.motions[0].q)
    assert back.motions[0].label == "demo"
    assert np.allclose(back.tool_path(back.motions[0]), scene.tool_path(scene.motions[0]))


def test_round_trip_without_motions(urdf, tmp_path):
    path = tmp_path / "scene.json"
    save_scene(_scene(urdf, with_motion=False), path)
    assert json.loads(path.read_text())["records"] == []
    assert load_scene(path, Site(id="s-1", cameras={})).motions == []


def test_load_refuses_other_site(urdf, tmp_path):
    path = tmp_path / "scene.json"
    save_scene(_scene(urdf), path)
    with pytest.raises(ValueError, match="s-1.*s-2|s-2.*s-1"):
        load_scene(path, Site(id="s-2", cameras={}))


def test_load_refuses_other_format(urdf, tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"format": "nope/9", "site": "s-1", "machines": {}, "records": []}))
    with pytest.raises(ValueError, match="format"):
        load_scene(path, Site(id="s-1", cameras={}))
