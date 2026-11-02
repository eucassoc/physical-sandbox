import json
import math
import shutil

import numpy as np
import pytest

from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_numpy
from physical_sandbox.scene import FORMAT, PlacedMachine, Scene, load_scene, save_placement, save_scene
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
    assert data["machines"]["arm"] == {"urdf": "../arm.urdf", "x": 0.2, "y": 0.1, "yaw": math.pi / 2,
                                       "confirmedByPerson": True}
    assert len(data["records"]) == 1 and not data["records"][0].startswith("/")
    assert (path.parent / data["records"][0]).is_file()

    back = load_scene(path, scene.site)
    p = back.machines["arm"]
    assert (p.x, p.y, p.yaw, p.confirmed_by_person) == (0.2, 0.1, math.pi / 2, True)
    assert p.urdf == str(urdf.resolve())
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


def test_scene_folder_is_portable(urdf, tmp_path):
    src = tmp_path / "a"
    save_scene(_scene(urdf), src / "scene.json")
    shutil.copytree(urdf.parent, tmp_path / "moved", ignore=shutil.ignore_patterns("a", "cache"))
    shutil.copytree(src, tmp_path / "moved" / "a")
    urdf.unlink()  # the original absolute location is gone
    back = load_scene(tmp_path / "moved" / "a" / "scene.json", Site(id="s-1", cameras={}))
    assert back.machines["arm"].machine.joint_names == ["turn"]
    assert len(back.motions) == 1


def test_scene_folder_wins_over_cwd(urdf, tmp_path, monkeypatch):
    folder = tmp_path / "scenes"
    folder.mkdir()
    (folder / "arm.urdf").write_text(URDF)
    other = tmp_path / "cwd"
    other.mkdir()
    (other / "arm.urdf").write_text(URDF.replace('name="turn"', 'name="cwd_turn"'))
    path = folder / "scene.json"
    path.write_text(json.dumps({"format": "physical-sandbox.scene/1", "site": "s-1", "records": [],
                                "machines": {"arm": {"urdf": "arm.urdf", "x": 0, "y": 0, "yaw": 0}}}))
    monkeypatch.chdir(other)
    back = load_scene(path, Site(id="s-1", cameras={}))
    assert back.machines["arm"].machine.joint_names == ["turn"]
    # not in the scene folder: falls back to the path as written (relative to the cwd)
    (folder / "arm.urdf").unlink()
    back = load_scene(path, Site(id="s-1", cameras={}))
    assert back.machines["arm"].machine.joint_names == ["cwd_turn"]


def test_missing_urdf_names_machine_and_candidates(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"format": "physical-sandbox.scene/1", "site": "s-1", "records": [],
                                "machines": {"arm": {"urdf": "gone.urdf", "x": 0, "y": 0, "yaw": 0}}}))
    with pytest.raises(ValueError, match=r"machine 'arm': urdf 'gone\.urdf' not found") as e:
        load_scene(path, Site(id="s-1", cameras={}))
    assert str(tmp_path / "gone.urdf") in str(e.value)


def test_save_refuses_to_overwrite_unrelated_record(urdf, tmp_path):
    other = tmp_path / "scene.record.json"
    other.write_text("precious")
    with pytest.raises(FileExistsError, match="scene.record.json"):
        save_scene(_scene(urdf), tmp_path / "scene.json")
    assert other.read_text() == "precious"
    assert not (tmp_path / "scene.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith(".")) == []


def test_save_again_overwrites_own_record(urdf, tmp_path):
    path = tmp_path / "scene.json"
    save_scene(_scene(urdf), path)
    save_scene(_scene(urdf), path)  # its own record: fine
    assert len(load_scene(path, Site(id="s-1", cameras={})).motions) == 1
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


def test_failed_save_leaves_no_half_written_pair(urdf, tmp_path):
    scene = _scene(urdf)
    scene.motions.append(motion_from_numpy([[0.0]], fps=1.0, joint_names=["turn"], machine="Bad Name"))
    with pytest.raises(ValueError):
        save_scene(scene, tmp_path / "scene.json")
    assert list(tmp_path.glob("*scene*")) == [] and list(tmp_path.glob(".*")) == []


@pytest.mark.parametrize("rec", ["../outside.json", "/etc/passwd"])
def test_load_refuses_records_outside_folder(tmp_path, rec):
    (tmp_path / "outside.json").write_text("{}")
    folder = tmp_path / "f"
    folder.mkdir()
    path = folder / "scene.json"
    path.write_text(json.dumps({"format": "physical-sandbox.scene/1", "site": "s-1",
                                "machines": {}, "records": [rec]}))
    with pytest.raises(ValueError, match="record"):
        load_scene(path, Site(id="s-1", cameras={}))


def test_load_names_missing_keys(tmp_path):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"format": "physical-sandbox.scene/1", "site": "s-1", "records": [],
                                "machines": {"arm": {"urdf": "x.urdf", "x": 0}}}))
    with pytest.raises(ValueError, match="machine 'arm': missing 'y'"):
        load_scene(path, Site(id="s-1", cameras={}))


def test_save_with_then_without_then_with_motions(urdf, tmp_path):
    path = tmp_path / "scene.json"
    site = Site(id="s-1", cameras={})
    save_scene(_scene(urdf), path)
    assert (tmp_path / "scene.record.json").exists()
    save_scene(_scene(urdf, with_motion=False), path)
    assert not (tmp_path / "scene.record.json").exists()  # its own record goes with the motions
    save_scene(_scene(urdf), path)  # no FileExistsError on the user's own file
    assert len(load_scene(path, site).motions) == 1


def test_save_without_motions_keeps_unrelated_record(urdf, tmp_path):
    other = tmp_path / "scene.record.json"
    other.write_text("precious")
    save_scene(_scene(urdf, with_motion=False), tmp_path / "scene.json")
    assert other.read_text() == "precious"


def test_save_placement_keeps_an_existing_records_list(tmp_path, urdf):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({"format": FORMAT, "site": "s-1", "machines": {}, "records": ["scene.record.json"]}))
    placed = _scene(urdf, with_motion=False).machines["arm"]
    save_placement(path, "s-1", {"arm": placed}, picture="/p/desk.jpg")
    d = json.loads(path.read_text())
    assert d["records"] == ["scene.record.json"] and d["picture"] == "/p/desk.jpg"
    assert d["machines"]["arm"]["confirmedByPerson"] and d["format"] == FORMAT and d["site"] == "s-1"


def test_save_placement_starts_an_empty_records_list_and_touches_no_record_file(tmp_path, urdf):
    path = tmp_path / "scene.json"
    (tmp_path / "scene.record.json").write_text("{}")
    placed = _scene(urdf, with_motion=False).machines["arm"]
    save_placement(path, "s-1", {"arm": placed}, picture=None)
    d = json.loads(path.read_text())
    assert d["records"] == [] and "picture" not in d
    assert (tmp_path / "scene.record.json").read_text() == "{}"
    assert load_scene(path, Site(id="s-1", cameras={})).machines["arm"].x == 0.2
