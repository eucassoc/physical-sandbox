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
CODEC = 8  # mp4v shifts a flat grey to about (124, 127, 124)


def look_at(eye, target=(0, 0, 0.2), size=(W, H), f=300.0, dist=None):
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    K = [[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]]
    return Camera(size[0], size[1], K, dist if dist is not None else np.zeros(5), R, -R @ eye)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    urdf = tmp_path / "blocks.urdf"
    urdf.write_text(URDF)
    still = tmp_path / "grey.png"
    cv2.imwrite(str(still), np.full((H, W, 3), GREY, np.uint8))

    def make(dist=None):
        cam = look_at((-2.5, 0.8, 1.2), dist=dist)
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


def base_box_pixels(cam):
    corners = [(x, y, z) for x in (-0.15, 0.15) for y in (-0.15, 0.15) for z in (0.0, 0.3)]
    px = cam.project(np.array(corners))
    return px, cv2.convexHull(px.astype(np.float32))


def inside_point(cam):
    px, hull = base_box_pixels(cam)
    return tuple(int(v) for v in px.mean(axis=0))


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
    x, y = inside_point(cam)  # the base box of both records is here: blended twice, both colours present
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


def test_distortion_keeps_the_shell_on_the_projected_box(world):
    make, still, tmp = world
    scene, cam = make(dist=[-0.25, 0.08, 0.0, 0.0, 0.0])
    out = render_overlay(scene, "camera", open_source(still), [motion(0.3)], tmp / "d.mp4")
    f = read(out)[0]
    x, y = inside_point(cam)
    assert bluish(f[y, x]) > 25
    px, _ = base_box_pixels(cam)
    # a pixel just outside the projected hull of the box stays grey
    hull = cv2.convexHull(px.astype(np.float32))
    ox, oy = int(px[:, 0].max()) + 12, y
    assert cv2.pointPolygonTest(hull, (float(ox), float(oy)), False) < 0
    assert abs(bluish(f[oy, ox])) < CODEC


def test_a_machine_not_in_the_scene_is_refused(world):
    make, still, tmp = world
    scene, _ = make()
    bad = motion_from_numpy([[0.0], [1.0]], fps=2.0, joint_names=["turn"], machine="nope")
    with pytest.raises(ValueError, match="nope"):
        render_overlay(scene, "camera", open_source(still), [bad], tmp / "x.mp4")
