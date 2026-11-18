import math
import os
from pathlib import Path

import numpy as np
import pytest

from physical_sandbox.machine import Machine, load_machine

SO101 = Path(os.environ.get(
    "PS_SO101_URDF", "~/work/physical-sandbox-proto/robots/so101/so101.urdf")).expanduser()
needs_so101 = pytest.mark.skipif(not SO101.exists(), reason="SO-101 URDF not found (set PS_SO101_URDF)")

# a unit tetrahedron with material lines, as exported CAD tools write them
OBJ_WITH_MTL = """\
mtllib part.mtl
o part
v 0 0 0
v 0.05 0 0
v 0 0.05 0
v 0 0 0.05
usemtl red
f 1 3 2
f 1 2 4
usemtl blue
f 1 4 3
f 2 3 4
"""

# a tetrahedron as ASCII STL
STL = """\
solid part
facet normal 0 0 -1
 outer loop
  vertex 0 0 0
  vertex 0 0.05 0
  vertex 0.05 0 0
 endloop
endfacet
facet normal 0 -1 0
 outer loop
  vertex 0 0 0
  vertex 0.05 0 0
  vertex 0 0 0.05
 endloop
endfacet
facet normal -1 0 0
 outer loop
  vertex 0 0 0
  vertex 0 0 0.05
  vertex 0 0.05 0
 endloop
endfacet
facet normal 1 1 1
 outer loop
  vertex 0.05 0 0
  vertex 0 0.05 0
  vertex 0 0 0.05
 endloop
endfacet
endsolid part
"""

URDF = """\
<?xml version="1.0"?>
<robot name="tiny">
  <link name="base">
    <inertial><origin xyz="0 0 0"/><mass value="1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.05"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><origin xyz="0 0 0"/><mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <visual><geometry><mesh filename="{mesh}"/></geometry></visual>
  </link>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <origin xyz="0.1 0 0"/><parent link="base"/><child link="arm"/>
    <axis xyz="0 0 1"/><limit lower="-1.5" upper="2.0" effort="1" velocity="1"/>
  </joint>
  <joint name="tip_joint" type="fixed">
    <origin xyz="0.2 0 0"/><parent link="arm"/><child link="tip"/>
  </joint>
</robot>
"""


@pytest.fixture
def cache(tmp_path, monkeypatch):
    d = tmp_path / "cache"
    monkeypatch.setenv("PS_CACHE_DIR", str(d))
    return d


def write_tiny(tmp_path, mesh="part.obj", content=OBJ_WITH_MTL):
    (tmp_path / mesh).write_text(content)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh=mesh))
    return urdf


def test_tiny_machine_reads_joints_limits_and_tool(tmp_path, cache):
    m = load_machine(write_tiny(tmp_path))
    assert isinstance(m, Machine)
    assert m.joint_names == ["j1"]
    assert m.limits.shape == (1, 2)
    assert np.allclose(m.limits, [[-1.5, 2.0]])
    assert m.tool_link == "tip"  # no gripper_frame_joint: the deepest link


def test_tiny_fk_moves_with_the_joint(tmp_path, cache):
    m = load_machine(write_tiny(tmp_path))
    assert np.allclose(m.fk(np.array([0.0])), [0.3, 0.0, 0.0], atol=1e-9)
    # the joint sits at x = 0.1, so the 0.2 m of arm beyond it swings round to +y
    assert np.allclose(m.fk(np.array([math.pi / 2])), [0.1, 0.2, 0.0], atol=1e-9)
    # a second call does not depend on the first
    assert np.allclose(m.fk(np.array([0.0])), [0.3, 0.0, 0.0], atol=1e-9)


def test_tiny_link_poses_cover_every_body_but_the_world(tmp_path, cache):
    m = load_machine(write_tiny(tmp_path))
    poses = m.link_poses(np.array([math.pi / 2]))
    assert set(poses) == {"base", "arm", "tip"}
    assert poses["arm"].shape == (4, 4)
    assert np.allclose(poses["arm"][3], [0, 0, 0, 1])
    # arm is rotated a quarter turn about z at x = 0.1
    assert np.allclose(poses["arm"][:3, 3], [0.1, 0.0, 0.0], atol=1e-9)
    assert np.allclose(poses["arm"][:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-9)
    assert np.allclose(poses["tip"][:3, 3], m.fk(np.array([math.pi / 2])), atol=1e-9)


def test_obj_material_lines_are_stripped_in_the_cached_copy(tmp_path, cache):
    urdf = write_tiny(tmp_path)
    load_machine(urdf)
    copies = list((cache / "meshes").rglob("*.obj"))
    assert len(copies) == 1
    text = copies[0].read_text()
    assert "usemtl" not in text and "mtllib" not in text
    assert text.count("\nf ") + text.startswith("f ") == 4  # faces are kept
    assert "usemtl" in (tmp_path / "part.obj").read_text()  # the original is untouched


def test_ascii_stl_mesh_loads(tmp_path, cache):
    m = load_machine(write_tiny(tmp_path, "part.stl", STL))
    copies = list((cache / "meshes").rglob("*.stl"))
    assert len(copies) == 1
    assert m.model.nmesh == 1  # MuJoCo reads binary STL only: the copy of an ASCII file is converted
    assert (copies[0].stat().st_size - 84) / 50 == 4
    assert np.allclose(m.fk(np.array([0.0])), [0.3, 0.0, 0.0], atol=1e-9)


def test_binary_stl_is_copied_byte_for_byte(tmp_path, cache):
    import struct
    tris = [((0, 0, -1), (0, 0, 0), (0, .05, 0), (.05, 0, 0)),
            ((0, -1, 0), (0, 0, 0), (.05, 0, 0), (0, 0, .05)),
            ((-1, 0, 0), (0, 0, 0), (0, 0, .05), (0, .05, 0)),
            ((1, 1, 1), (.05, 0, 0), (0, .05, 0), (0, 0, .05))]
    blob = bytes(80) + struct.pack("<I", 4)
    for t in tris:
        blob += struct.pack("<12fH", *[c for v in t for c in v], 0)
    (tmp_path / "part.stl").write_bytes(blob)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="part.stl"))
    m = load_machine(urdf)
    assert m.model.nmesh == 1
    assert next((cache / "meshes").rglob("*.stl")).read_bytes() == blob


def test_stl_reference_falls_back_to_obj_and_meshes_subfolder(tmp_path, cache):
    # the URDF says assets/part.stl; only meshes/assets/part.obj exists
    (tmp_path / "meshes" / "assets").mkdir(parents=True)
    (tmp_path / "meshes" / "assets" / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="assets/part.stl"))
    m = load_machine(urdf)
    assert m.model.nmesh == 1
    copies = list((cache / "meshes").rglob("*.obj"))
    assert len(copies) == 1 and "usemtl" not in copies[0].read_text()


def test_stl_reference_falls_back_to_obj_next_to_it(tmp_path, cache):
    (tmp_path / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="part.stl"))
    assert load_machine(urdf).model.nmesh == 1


def test_missing_mesh_names_the_file(tmp_path, cache):
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="nope.stl"))
    with pytest.raises(FileNotFoundError, match="nope.stl"):
        load_machine(urdf)


def test_explicit_tool_link(tmp_path, cache):
    m = load_machine(write_tiny(tmp_path), tool_link="arm")
    assert m.tool_link == "arm"
    assert np.allclose(m.fk(np.array([0.0])), [0.1, 0.0, 0.0], atol=1e-9)
    with pytest.raises(ValueError, match="nope"):
        load_machine(write_tiny(tmp_path), tool_link="nope")


def test_continuous_joint_limits_are_minus_pi_to_pi(tmp_path, cache):
    (tmp_path / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="part.obj").replace(
        'type="revolute"', 'type="continuous"').replace(
        '<limit lower="-1.5" upper="2.0" effort="1" velocity="1"/>', ""))
    m = load_machine(urdf)
    assert np.allclose(m.limits, [[-math.pi, math.pi]])


def test_wrong_length_q_is_refused(tmp_path, cache):
    m = load_machine(write_tiny(tmp_path))
    with pytest.raises(ValueError, match="1"):
        m.fk(np.zeros(3))


@needs_so101
def test_so101_joints_limits_and_zero_pose_tip(cache):
    m = load_machine(SO101)
    assert m.joint_names == [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    assert m.tool_link == "gripper_frame_link"
    i = m.joint_names.index("wrist_flex")
    assert np.allclose(m.limits[i], (-1.65806, 1.65806))
    tip = m.fk(np.zeros(6))
    assert np.all(np.isfinite(tip))
    assert 0.1 < np.linalg.norm(tip) < 0.6
    poses = m.link_poses(np.zeros(6))
    assert "gripper_frame_link" in poses and "base_link" in poses
    # no material line survives in any cached copy
    for p in (cache / "meshes").rglob("*.obj"):
        assert "usemtl" not in p.read_text()


def cached_obj_texts(cache):
    return sorted(p.read_text() for p in (cache / "meshes").rglob("*.obj"))


def test_editing_only_a_mesh_file_uses_the_new_mesh(tmp_path, cache):
    urdf = write_tiny(tmp_path)
    first = load_machine(urdf)
    old_verts = first.model.mesh_vert.copy()
    # a bigger tetrahedron, the URDF itself untouched
    (tmp_path / "part.obj").write_text(OBJ_WITH_MTL.replace("0.05", "0.5"))
    second = load_machine(urdf)
    assert second.model.mesh_vert.shape == old_verts.shape
    assert np.abs(second.model.mesh_vert).max() > 5 * np.abs(old_verts).max()
    assert "0.5" in " ".join(cached_obj_texts(cache)) and len(cached_obj_texts(cache)) == 2


def test_repeated_load_reuses_the_cache_folder_without_rewriting(tmp_path, cache):
    urdf = write_tiny(tmp_path)
    load_machine(urdf)
    files = sorted(p for p in (cache / "meshes").rglob("*") if p.is_file())
    stamps = [p.stat().st_mtime_ns for p in files]
    assert len(files) == 2  # the URDF and one mesh
    load_machine(urdf)
    assert sorted(p for p in (cache / "meshes").rglob("*") if p.is_file()) == files
    assert [p.stat().st_mtime_ns for p in files] == stamps
    # no temporary folders are left behind
    assert not [p for p in (cache / "meshes").iterdir() if p.name.startswith(".tmp-")]


def test_same_stem_in_different_folders_gives_distinct_meshes(tmp_path, cache):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
    (tmp_path / "a" / "part.stl").write_text(STL)
    (tmp_path / "b" / "part.obj").write_text(OBJ_WITH_MTL.replace("0.05", "0.2"))
    urdf = tmp_path / "two.urdf"
    urdf.write_text(URDF.format(mesh="a/part.stl").replace(
        "</robot>",
        '<link name="b"><inertial><mass value="0.1"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>'
        '<visual><geometry><mesh filename="b/part.obj"/></geometry></visual></link>'
        '<joint name="jb" type="fixed"><parent link="base"/><child link="b"/></joint></robot>'))
    m = load_machine(urdf)
    assert m.model.nmesh == 2
    sizes = [np.abs(m.model.mesh_vert[m.model.mesh_vertadr[i]:][:m.model.mesh_vertnum[i]]).max()
             for i in range(2)]
    # MuJoCo recentres meshes, so compare the sizes: the two shapes are 0.05 and 0.2 tetrahedra
    assert max(sizes) / min(sizes) == pytest.approx(4.0, rel=1e-3)


def test_case_variants_of_a_stem_do_not_collide(tmp_path, cache):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "Part.obj").write_text(OBJ_WITH_MTL)
    (tmp_path / "b" / "part.obj").write_text(OBJ_WITH_MTL.replace("0.05", "0.2"))
    urdf = tmp_path / "two.urdf"
    urdf.write_text(URDF.format(mesh="a/Part.obj").replace(
        "</robot>",
        '<link name="b"><inertial><mass value="0.1"/>'
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>'
        '<visual><geometry><mesh filename="b/part.obj"/></geometry></visual></link>'
        '<joint name="jb" type="fixed"><parent link="base"/><child link="b"/></joint></robot>'))
    assert load_machine(urdf).model.nmesh == 2


def test_two_processes_loading_at_once_both_get_a_whole_folder(tmp_path, cache):
    import multiprocessing as mp
    urdf = write_tiny(tmp_path)
    ctx = mp.get_context("spawn")
    with ctx.Pool(3) as pool:
        out = pool.starmap(_load_in_child, [(str(urdf), str(cache))] * 6)
    assert all(abs(x - 0.3) < 1e-9 for x in out)
    assert not [p for p in (cache / "meshes").iterdir() if p.name.startswith(".tmp-")]
    assert len([p for p in (cache / "meshes").iterdir()]) == 1


def _load_in_child(urdf, cache_dir):
    os.environ["PS_CACHE_DIR"] = cache_dir
    return float(load_machine(urdf).fk(np.array([0.0]))[0])


# 2026-11-18: the public catalog's Panda, UR5, Go2 and Scout name .dae meshes; the catalog folder holds
# the same mesh as .obj under meshes/. Only .stl fell back to .obj, so these did not load
def test_dae_reference_falls_back_to_obj_under_meshes(tmp_path, cache):
    (tmp_path / "meshes" / "robots" / "visual").mkdir(parents=True)
    (tmp_path / "meshes" / "robots" / "visual" / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="package://some-robot-data/robots/visual/part.dae"))
    assert load_machine(urdf).model.nmesh == 1


# 2026-11-18: the catalog keeps the meshes that are drawn, not the collision ones (TIAGo++). Nothing here
# collides, so a collision mesh that is not in the folder is left out; a drawn one is still refused
def test_a_collision_mesh_that_is_not_in_the_folder_is_left_out(tmp_path, cache):
    (tmp_path / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="part.obj").replace(
        '<link name="tip"/>',
        '<link name="tip"><collision><geometry><mesh filename="package://x/collision/tip.stl"/></geometry></collision></link>'))
    m = load_machine(urdf)
    assert m.model.nmesh == 1 and m.joint_names == ["j1"]


# 2026-11-18: a URDF written only to be drawn has no <inertial> (the catalog's drone and gate). MuJoCo
# refuses a moving body with no mass; only the kinematics are used here, so a small mass is filled in
def test_a_moving_link_without_inertial_loads(tmp_path, cache):
    (tmp_path / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    text = URDF.format(mesh="part.obj")
    start = text.index("<inertial>", text.index('<link name="arm">'))
    end = text.index("</inertial>", start) + len("</inertial>")
    urdf.write_text(text[:start] + text[end:].replace("<visual><geometry><mesh", "<visual><geometry><mesh", 1))
    m = load_machine(urdf)
    assert m.joint_names == ["j1"]
    np.testing.assert_allclose(m.fk(np.array([0.0])), [0.3, 0.0, 0.0], atol=1e-9)


# 2026-11-18: the catalog's Go2 lists every material of a mesh in one <visual>; MuJoCo takes one
def test_a_visual_with_several_materials_loads(tmp_path, cache):
    (tmp_path / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="part.obj").replace(
        '<mesh filename="part.obj"/></geometry>',
        '<mesh filename="part.obj"/></geometry><material name="a"><color rgba="1 0 0 1"/></material>'
        '<material name="b"><color rgba="0 0 1 1"/></material>'))
    assert load_machine(urdf).model.nmesh == 1


# 2026-11-18: the catalog's TIAGo++ has a link whose inertia no rigid body can have (its off-diagonal
# terms outweigh the diagonal). MuJoCo refuses it; only the kinematics are used, so the terms are dropped
def test_an_inertia_no_body_can_have_loads(tmp_path, cache):
    (tmp_path / "part.obj").write_text(OBJ_WITH_MTL)
    urdf = tmp_path / "tiny.urdf"
    urdf.write_text(URDF.format(mesh="part.obj").replace(
        '<inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>',
        '<inertia ixx="0.00000002371" ixy="0.00000006119" ixz="0.00000001179" iyy="0.00000002833" '
        'iyz="0.00000000774" izz="0.00000003849"/>'))
    m = load_machine(urdf)
    np.testing.assert_allclose(m.fk(np.array([0.0])), [0.3, 0.0, 0.0], atol=1e-9)
