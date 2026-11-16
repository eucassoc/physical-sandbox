import math

import cv2
import numpy as np
import pytest

from physical_sandbox.camera import Camera
from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_numpy
from physical_sandbox.render import render_overlay
from physical_sandbox.scene import PlacedMachine, Scene
from physical_sandbox.site import Site
from physical_sandbox.sources import open_source

URDF = """<?xml version="1.0"?>
<robot name="blocks">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.15"/><geometry><box size="0.3 0.3 0.3"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.1 0.1"/></geometry></visual>
  </link>
  <joint name="turn" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>
"""
W, H = 320, 240
GREY = 128
CODEC = 8  # the video codec shifts a flat grey by a few levels (mp4v: to about (124, 127, 124))


def look_at(eye, target=(0, 0, 0.2), size=(W, H), f=300.0, dist=None, pp=None):
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    cx, cy = pp if pp is not None else (size[0] / 2, size[1] / 2)
    K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
    return Camera(size[0], size[1], K, dist if dist is not None else np.zeros(5), R, -R @ eye)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    urdf = tmp_path / "blocks.urdf"
    urdf.write_text(URDF)
    still = tmp_path / "grey.png"
    cv2.imwrite(str(still), np.full((H, W, 3), GREY, np.uint8))

    def make(cam=None, dist=None):
        cam = cam or look_at((-2.5, 0.8, 1.2), dist=dist)
        placed = PlacedMachine(machine=load_machine(urdf), urdf=str(urdf), x=0.0, y=0.0, yaw=0.0)
        site = Site(id="s", cameras={"camera": cam})
        return Scene(site=site, machines={"blocks": placed}), cam
    return make, still, tmp_path


def motion(end, label="a", fps=10.0, n=11):
    q = np.linspace(0, end, n).reshape(-1, 1)
    return motion_from_numpy(q, fps=fps, joint_names=["turn"], machine="blocks", label=label)


def read(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


BASE_BOX = ((0, 0, 0.15), (0.15, 0.15, 0.15))  # centre, half sizes (site frame)
ARM_BOX = ((0.2, 0, 0.4), (0.2, 0.05, 0.05))     # the arm at turn = 0


def edge_points(box, n=60):
    """Points along the 12 edges of an axis-aligned box (centre, half sizes)."""
    c, h = np.array(box[0], float), np.array(box[1], float)
    pts = []
    for axis in range(3):
        o = [a for a in range(3) if a != axis]
        for sa in (-1, 1):
            for sb in (-1, 1):
                for u in np.linspace(-1, 1, n):
                    d = np.zeros(3)
                    d[axis], d[o[0]], d[o[1]] = u, sa, sb
                    pts.append(c + d * h)
    return np.array(pts)


def projected_bbox(cam, *boxes):
    """(x0, y0, x1, y1) of the boxes' edges through the camera (with its distortion)."""
    px = cam.project(np.vstack([edge_points(b) for b in boxes]))
    return px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()


def shell_bbox(frame, threshold=28):
    """(x0, y0, x1, y1) of the pixels that moved toward blue."""
    b, g, r = (frame[..., i].astype(int) for i in range(3))
    ys, xs = np.nonzero(b - r > threshold)
    return xs.min(), ys.min(), xs.max(), ys.max()


def assert_bbox_close(got, want, tol, size=(W, H)):
    assert 3 < want[0] and 3 < want[1] and want[2] < size[0] - 4 and want[3] < size[1] - 4, f"test camera cuts the box: {want}"
    assert np.abs(np.array(got, float) - np.array(want)).max() <= tol, f"shell {got} vs projection {tuple(round(float(v), 1) for v in want)}"


def inside_point(cam):
    c = BASE_BOX[0]
    return tuple(int(v) for v in cam.project(np.array([c]))[0])


def bluish(pixel):
    b, g, r = (int(v) for v in pixel)
    return b - r


def test_shell_is_blue_inside_the_box_and_grey_outside(world):
    make, still, tmp = world
    scene, cam = make()
    out = render_overlay(scene, "camera", open_source(still), [motion(1.0)], tmp / "a.mp4")
    frames = read(out)
    assert frames and frames[0].shape == (H, W, 3)
    x, y = inside_point(cam)
    for f in (frames[0], frames[-1]):
        assert bluish(f[y, x]) > 25            # moved toward the record's blue
        assert np.abs(f[5, 5].astype(int) - GREY).max() < CODEC  # outside: the picture as it was
        assert np.abs(f[H - 6, W - 6].astype(int) - GREY).max() < CODEC


def test_frame_count_is_duration_times_fps(world):
    make, still, tmp = world
    scene, _ = make()
    out = render_overlay(scene, "camera", open_source(still), [motion(1.0, fps=10.0, n=11)], tmp / "a.mp4", fps=20.0)
    assert len(read(out)) == 20  # 1.0 s at 20 fps
    out = render_overlay(scene, "camera", open_source(still), [motion(1.0, n=6, fps=5.0)], tmp / "b.mp4", fps=30.0)
    assert len(read(out)) == 30
    # two motions: as long as the longer one
    out = render_overlay(scene, "camera", open_source(still), [motion(1.0, n=6, fps=5.0), motion(1.0, n=21, fps=10.0)],
                         tmp / "c.mp4", fps=10.0)
    assert len(read(out)) == 20


def test_progress_is_reported_after_each_written_frame(world):
    make, still, tmp = world
    scene, _ = make()
    seen = []
    render_overlay(scene, "camera", open_source(still), [motion(1.0, fps=10.0, n=11)], tmp / "p.mp4", fps=20.0,
                   progress=lambda k, n: seen.append((k, n)))
    assert seen == [(k, 20) for k in range(1, 21)]


def test_second_call_in_one_process_is_not_black(world):
    make, still, tmp = world
    scene, cam = make()
    x, y = inside_point(cam)
    for name in ("one.mp4", "two.mp4", "three.mp4"):
        out = render_overlay(scene, "camera", open_source(still), [motion(0.5)], tmp / name)
        f = read(out)[0]
        assert bluish(f[y, x]) > 25, name


def test_two_records_each_in_its_own_colour(world):
    make, still, tmp = world
    scene, cam = make()
    # B (orange) is the same machine, turned the other way, so the arms point in different directions
    out = render_overlay(scene, "camera", open_source(still), [motion(0.0, "a"), motion(math.pi, "b")], tmp / "ab.mp4",
                         show={"machine": True, "trail": False, "link": False})
    f = read(out)[-1]
    # the arm tip of A (at +x) and of B (at -x), well outside the base box
    theta = 0.9 * math.pi  # the last frame is at 0.9 s of a 1.0 s motion that ends at pi
    a_tip = tuple(int(v) for v in cam.project(np.array([[0.3, 0, 0.4]]))[0])
    b_tip = tuple(int(v) for v in cam.project(np.array([[0.3 * math.cos(theta), 0.3 * math.sin(theta), 0.4]]))[0])
    assert bluish(f[a_tip[1], a_tip[0]]) > 20       # A: blue
    assert bluish(f[b_tip[1], b_tip[0]]) < -20      # B: orange (red > blue)


def test_toggles_machine_trail_link(world):
    make, still, tmp = world
    scene, cam = make()
    x, y = inside_point(cam)
    src = open_source(still)
    off = render_overlay(scene, "camera", src, [motion(1.0, "a"), motion(2.0, "b")], tmp / "off.mp4",
                         show={"machine": False, "trail": False, "link": False})
    for f in read(off):
        assert np.abs(f.astype(int) - GREY).max() < CODEC   # nothing drawn: the picture as it was
    no_machine = render_overlay(scene, "camera", src, [motion(1.0, "a"), motion(2.0, "b")], tmp / "nm.mp4",
                                show={"machine": False, "trail": True, "link": True})
    last = read(no_machine)[-1]
    assert np.abs(last.astype(int) - GREY).max() > 40   # trail and link lines are there
    assert abs(bluish(last[y, x])) < CODEC                  # but no shell over the base box


def test_a_machine_not_in_the_scene_is_refused(world):
    make, still, tmp = world
    scene, _ = make()
    bad = motion_from_numpy([[0.0], [1.0]], fps=2.0, joint_names=["turn"], machine="nope")
    with pytest.raises(ValueError, match="nope"):
        render_overlay(scene, "camera", open_source(still), [bad], tmp / "x.mp4")


NO_OVERLAY = {"machine": True, "trail": False, "link": False}


def first_frame(path):
    return read(path)[0]


def test_the_shell_sits_on_the_projection_with_an_off_centre_principal_point(world):
    make, still, tmp = world
    cam = look_at((-2.5, 0.8, 1.2), pp=(215.0, 95.0))
    scene, cam = make(cam)
    out = render_overlay(scene, "camera", open_source(still), [motion(0.0)], tmp / "pp.mp4", show=NO_OVERLAY)
    assert_bbox_close(shell_bbox(first_frame(out)), projected_bbox(cam, BASE_BOX, ARM_BOX), 3)


def test_the_warp_puts_the_shell_on_the_projection_of_a_distorted_camera(world):
    make, still, tmp = world
    # wide angle, off-centre principal point, the machine well away from it, strong barrel distortion:
    # without the warp the shell lands about 16 pixels off
    look = dict(target=(0, -1.0, 0.2), f=150.0, pp=(250.0, 70.0))
    cam = look_at((-1.3, 0.55, 0.85), dist=[-0.4, 0.0, 0.0, 0.0, 0.0], **look)
    scene, cam = make(cam)
    out = render_overlay(scene, "camera", open_source(still), [motion(0.0)], tmp / "dist.mp4", show=NO_OVERLAY)
    want = projected_bbox(cam, BASE_BOX, ARM_BOX)
    assert_bbox_close(shell_bbox(first_frame(out)), want, 3)
    # and the same picture without distortion is somewhere else (the test can tell the warp from none)
    flat = look_at((-1.3, 0.55, 0.85), **look)
    assert np.abs(np.array(projected_bbox(flat, BASE_BOX, ARM_BOX)) - np.array(want)).max() > 10


COLLISION_URDF = """<?xml version="1.0"?>
<robot name="guarded">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.15"/><geometry><box size="0.3 0.3 0.3"/></geometry></visual>
    <collision><origin xyz="0 0 0.15"/><geometry><sphere radius="0.8"/></geometry></collision>
  </link>
  <link name="arm"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial></link>
  <joint name="turn" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def scene_for(tmp, text, name, cam, machine_id="blocks"):
    urdf = tmp / f"{name}.urdf"
    urdf.write_text(text)
    placed = PlacedMachine(machine=load_machine(urdf), urdf=str(urdf), x=0.0, y=0.0, yaw=0.0)
    return Scene(site=Site(id="s", cameras={"camera": cam}), machines={machine_id: placed})


def test_collision_shapes_are_not_drawn(world):
    make, still, tmp = world
    cam = look_at((-2.5, 0.8, 1.2))
    scene = scene_for(tmp, COLLISION_URDF, "guarded", cam)
    # the loader really did keep the collision sphere, in another group than the visual box
    assert set(int(g) for g in scene.machines["blocks"].machine.model.geom_group) == {0, 1}
    out = render_overlay(scene, "camera", open_source(still), [motion(0.0)], tmp / "g.mp4", show=NO_OVERLAY)
    f = first_frame(out)
    want = projected_bbox(cam, BASE_BOX)
    assert_bbox_close(shell_bbox(f), want, 3)
    # a point inside the big collision sphere but well outside the visual box stays the picture
    u, v = (int(c) for c in cam.project(np.array([[0.0, 0.5, 0.15]]))[0])
    assert want[0] - 8 > u or u > want[2] + 8 or v > want[3] + 8 or v < want[1] - 8
    assert abs(bluish(f[v, u])) < CODEC


def test_a_machine_with_only_collision_shapes_is_drawn_from_them(world):
    make, still, tmp = world
    cam = look_at((-2.5, 0.8, 1.2))
    text = COLLISION_URDF.replace('    <visual><origin xyz="0 0 0.15"/><geometry><box size="0.3 0.3 0.3"/></geometry></visual>\n', "")
    scene = scene_for(tmp, text, "bare", cam)
    out = render_overlay(scene, "camera", open_source(still), [motion(0.0)], tmp / "bare.mp4", show=NO_OVERLAY)
    x, y = inside_point(cam)
    assert bluish(first_frame(out)[y, x]) > 25


def cube_obj(sx, sy, sz):
    v = [(x * sx, y * sy, z * sz) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    f = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    lines = [f"v {a} {b} {c}" for a, b, c in v]
    for q in f:
        lines += [f"f {q[0]} {q[1]} {q[2]}", f"f {q[0]} {q[2]} {q[3]}"]
    return "\n".join(lines) + "\n"


MESH_URDF = """<?xml version="1.0"?>
<robot name="meshy">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.1 0.2 0.05" rpy="0.3 0.5 0.7"/>
      <geometry><mesh filename="block.obj" scale="0.1 0.15 0.2"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.3 -0.1 0.02" rpy="0.4 0 0"/>
      <geometry><mesh filename="slab.obj"/></geometry></visual>
  </link>
  <joint name="turn" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0.2 0.1 0.3" rpy="0.2 0 1.0"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def world_vertices(model, data, geoms):
    pts = []
    for g in geoms:
        mid = int(model.geom_dataid[g])
        v = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        pts.append(v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    return np.vstack(pts)


def test_mesh_vertices_land_where_the_machine_has_them(world):
    """Source and render model agree on every mesh vertex in the site frame (origin rpy, scale, rotated joint)."""
    import mujoco

    from physical_sandbox.render import _Shells

    make, still, tmp = world
    (tmp / "block.obj").write_text(cube_obj(1, 2, 3))
    (tmp / "slab.obj").write_text(cube_obj(0.5, 0.2, 0.05))
    cam = look_at((-2.5, 0.8, 1.2))
    scene = scene_for(tmp, MESH_URDF, "meshy", cam)
    placed = scene.machines["blocks"]
    placed.x, placed.y, placed.yaw = 0.7, -0.4, 0.9  # not at the origin, so the base transform is checked too
    m = motion_from_numpy([[0.6]], fps=1.0, joint_names=["turn"], machine="blocks")
    src = placed.machine.model
    assert (src.geom_type == mujoco.mjtGeom.mjGEOM_MESH).all() and src.ngeom == 2

    # the source: each body's pose in the site frame, then its mesh geoms
    poses = scene.link_poses(m, 0)
    want = []
    for g in range(src.ngeom):
        T = poses[src.body(int(src.geom_bodyid[g])).name]
        Rg = np.zeros(9)
        mujoco.mju_quat2Mat(Rg, src.geom_quat[g])
        mid = int(src.geom_dataid[g])
        v = src.mesh_vert[src.mesh_vertadr[mid]:src.mesh_vertadr[mid] + src.mesh_vertnum[mid]]
        want.append((v @ Rg.reshape(3, 3).T + src.geom_pos[g]) @ T[:3, :3].T + T[:3, 3])
    want = np.vstack(want)

    shells = _Shells(scene, [m], cam)
    try:
        shells.draw(0, poses, (255, 0, 0))
        got = world_vertices(shells.model, shells.data, shells.motion_geoms[0])
    finally:
        shells.close()
    d = np.linalg.norm(want[:, None, :] - got[None, :, :], axis=2)
    assert d.min(axis=1).max() < 1e-5 and d.min(axis=0).max() < 1e-5


def test_unknown_show_key_is_refused(world):
    make, still, tmp = world
    scene, _ = make()
    with pytest.raises(ValueError, match="trial"):
        render_overlay(scene, "camera", open_source(still), [motion(1.0)], tmp / "x.mp4", show={"trial": True})


def test_an_error_leaves_no_half_written_video(world):
    make, still, tmp = world
    scene, _ = make()
    src = open_source(still)
    calls = []

    def at(seconds):
        calls.append(seconds)
        return np.zeros((10, 10, 3), np.uint8) if len(calls) > 2 else np.full((H, W, 3), GREY, np.uint8)

    src.at = at
    out = tmp / "e.mp4"
    out.write_bytes(b"old")
    with pytest.raises(ValueError, match="changed size"):
        render_overlay(scene, "camera", src, [motion(1.0)], out)
    assert out.read_bytes() == b"old"                         # an earlier video is not overwritten by a failure
    assert not [p for p in tmp.iterdir() if "partial" in p.name]


def test_duration_cuts_the_video_short_of_the_longest_motion(world):
    make, still, tmp = world
    scene, _ = make()
    n = lambda **kw: len(read(render_overlay(scene, "camera", open_source(str(still)), [motion(1.0)],
                                              tmp / "d.mp4", fps=10.0, **kw)))
    assert n() == 10  # the motion is 1.0 s
    assert n(duration=0.5) == 5
    assert n(duration=5.0) == 10  # never longer than the motion
    with pytest.raises(ValueError, match="duration"):
        n(duration=0)


@pytest.mark.parametrize("h264", [True, False])
def test_the_video_is_written_with_h264_or_else_mpeg4(world, monkeypatch, h264):
    import physical_sandbox.render as render
    make, still, tmp = world
    scene, _ = make()
    tried = []
    real = cv2.VideoWriter

    def writer(path, fourcc, fps, size):
        name = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))
        tried.append(name)
        if name == "avc1" and not h264:  # as an OpenCV build without an H.264 encoder answers
            return real(path, 0, -1, (0, 0))
        return real(path, fourcc, fps, size)
    monkeypatch.setattr(render.cv2, "VideoWriter", writer)
    out = render_overlay(scene, "camera", open_source(still), [motion(0.5)], tmp / "c.mp4", fps=10.0)
    assert tried == (["avc1"] if h264 else ["avc1", "mp4v"])
    assert out.stat().st_size > 0 and len(read(out)) == 10  # 1.0 s at 10 fps


def test_the_near_clip_is_the_compare_screens(world):
    """The replay video and the in-chat screen cut a machine close to the camera at the same depth."""
    import re
    from pathlib import Path

    import physical_sandbox.render as render
    make, _, _ = world
    scene, cam = make()
    shells = render._Shells(scene, [motion(1.0)], cam)
    try:
        near = shells.model.vis.map.znear * shells.model.stat.extent
    finally:
        shells.close()
    assert near == pytest.approx(render.NEAR_M) and render.NEAR_M == 0.05
    html = (Path(render.__file__).parent / "mcp" / "widgets" / "compare.html").read_text()
    screen_near = float(re.search(r"new THREE\.PerspectiveCamera\(\s*[\d.]+,\s*[\d.]+,\s*([\d.]+),", html).group(1))
    assert screen_near == render.NEAR_M


def test_a_machine_between_the_old_and_the_new_near_clip_is_drawn(world, tmp_path):
    """The camera 7 cm from the box's face: the old near clip (0.1 m) cut the face away, 0.05 m keeps it."""
    make, still, tmp = world
    cam = look_at((-0.22, 0.0, 0.15), target=(0.0, 0.0, 0.15), f=150.0)  # the base box's face is at x = -0.15
    scene, _ = make(cam=cam)
    frame = read(render_overlay(scene, "camera", open_source(still), [motion(0.1, n=2)], tmp / "n.mp4",
                                fps=10.0, show={"trail": False, "link": False}))[0]
    assert bluish(frame[H // 2, W // 2])


def test_a_big_picture_gives_a_video_no_longer_than_1920_on_its_long_side(world):
    # a phone photo is 4032x3024; a video that size takes minutes to write and plays nowhere
    make, _, tmp = world
    scene, _ = make(cam=look_at((-2.5, 0.8, 1.2), size=(2560, 1920), f=2400.0))
    big = tmp / "big.png"
    cv2.imwrite(str(big), np.full((1920, 2560, 3), GREY, np.uint8))
    out = render_overlay(scene, "camera", open_source(big), [motion(1.0)], tmp / "big.mp4", fps=10.0, duration=0.1)
    assert read(out)[0].shape == (1440, 1920, 3)


def test_the_shell_of_a_shrunk_video_sits_on_the_shrunk_projection(world):
    make, still, tmp = world
    scene, cam = make()
    out = render_overlay(scene, "camera", open_source(still), [motion(1.0)], tmp / "s.mp4", max_side=160)
    frame = read(out)[0]
    assert frame.shape == (120, 160, 3)
    x, y = inside_point(cam)
    assert bluish(frame[y // 2, x // 2]) > 25
    assert np.abs(frame[3, 3].astype(int) - GREY).max() < CODEC


def test_max_side_none_keeps_the_pictures_own_size(world):
    make, _, tmp = world
    scene, _ = make(cam=look_at((-2.5, 0.8, 1.2), size=(2560, 1920), f=2400.0))
    big = tmp / "big.png"
    cv2.imwrite(str(big), np.full((1920, 2560, 3), GREY, np.uint8))
    out = render_overlay(scene, "camera", open_source(big), [motion(1.0)], tmp / "full.mp4", fps=10.0, duration=0.1,
                         max_side=None)
    assert read(out)[0].shape == (1920, 2560, 3)


def test_a_shrunk_video_has_even_sides_whatever_the_aspect(world):
    # H.264 wants even sides, and 321x240 has no smaller size of exactly its own aspect
    make, _, tmp = world
    scene, cam = make(cam=look_at((-2.5, 0.8, 1.2), size=(321, 240)))
    odd = tmp / "odd.png"
    cv2.imwrite(str(odd), np.full((240, 321, 3), GREY, np.uint8))
    out = render_overlay(scene, "camera", open_source(odd), [motion(1.0)], tmp / "odd.mp4", max_side=100)
    frame = read(out)[0]
    assert frame.shape == (74, 100, 3)
    x, y = inside_point(cam)
    assert bluish(frame[round(y * 74 / 240), round(x * 100 / 321)]) > 25
