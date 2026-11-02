"""A synthetic world for the screen tests: a tiny URDF of boxes, a site with a known camera, a scene and records.

Shared by tests/test_mcp_compare.py (the chat screen, needs the [mcp] extra) and tests/test_screens.py (the data
builder, needs no SDK).
"""
import cv2
import numpy as np

from physical_sandbox.camera import Camera
from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_numpy, save_record
from physical_sandbox.scene import PlacedMachine, Scene, save_scene
from physical_sandbox.site import Floor, Site, save_site

URDF = """<?xml version="1.0"?>
<robot name="blocks">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.15"/><geometry><box size="0.3 0.3 0.3"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.1 0.1"/></geometry></visual>
    <visual><origin xyz="0.4 0 0"/><geometry><sphere radius="0.06"/></geometry></visual>
  </link>
  <joint name="turn" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <link name="tip">
    <inertial><mass value="0.1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="tip_fixed" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="0.4 0 0"/></joint>
</robot>
"""
W, H = 320, 240


def look_at(eye, target=(0, 0, 0.2), size=(W, H), f=300.0, dist=None):
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    K = [[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]]
    return Camera(size[0], size[1], K, np.zeros(5) if dist is None else dist, R, -R @ eye)


def motion(end, label, fps=10.0, n=11):
    q = np.linspace(0, end, n).reshape(-1, 1)
    return motion_from_numpy(q, fps=fps, joint_names=["turn"], machine="blocks", label=label)


def make_world(tmp_path, cam=None, floor=True):
    urdf = tmp_path / "blocks.urdf"
    urdf.write_text(URDF)
    corners_m = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    site = Site(id="yard", cameras={"camera": cam or look_at((-2.5, 0.8, 1.2))}, residual_px={"camera": 0.7},
                floor=Floor("camera", [(0, 0)] * 4, (2.0, 2.0), corners_m) if floor else None,
                confirmed_by_person=True, dimensions="estimated")
    save_site(site, tmp_path / "site.json")
    placed = PlacedMachine(machine=load_machine(urdf), urdf=str(urdf), x=0.1, y=0.0, yaw=0.3)
    save_scene(Scene(site=site, machines={"blocks": placed}), tmp_path / "scene.json")
    save_record([motion(1.0, "slow")], "yard", tmp_path / "a.record.json")
    save_record([motion(1.5, "wide")], "yard", tmp_path / "b.record.json")
    still = tmp_path / "grey.png"
    cv2.imwrite(str(still), np.full((H, W, 3), 128, np.uint8))
    return site, placed, still
