"""The replay video: one or two motions of a machine drawn over the site's picture, through the site camera.

Each record's machine is drawn as a see-through shell in that record's colour only (never the machine's own
colours): it is rendered opaque with MuJoCo through the photo camera's numbers, then blended half and half
into the picture where it covers it. Machines of different records are blended one after the other, so they
never hide each other. On top of that: the tool tip's path so far (a trail, the second record's dashed) and
a white link between the two tool tips.

The picture camera becomes a MuJoCo camera with ``focalpixel`` / ``principalpixel`` / ``sensorsize`` /
``resolution`` at ``-Rᵀt`` with ``xyaxes = R row 0, -R row 1``; it matches OpenCV's projection within a pixel.
MuJoCo draws a pinhole, so when the site camera has lens distortion the drawing is warped to the distorted
picture (the same warp ``Camera.project`` applies to the trail).

Rendering notes (they cost time to find): the background of an opaque drawing is black and a lit shell never
is, so ``rgb.sum(2) > 0`` is the coverage mask; only one ``mujoco.Renderer`` may be open at a time; the first
frame of a fresh renderer can come out black, so one frame is drawn and thrown away first. On macOS the default
OpenGL backend works offscreen; set ``MUJOCO_GL=egl`` or ``osmesa`` on a headless Linux machine.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .camera import Camera
from .motion import Motion
from .scene import Scene
from .sources import Source, camera_for

COLORS = ("#2f7fe0", "#f08c1a")  # record A blue, record B orange
SHOW = {"machine": True, "trail": True, "link": True}
_SHELL_ALPHA = 0.5
_VISUAL_GROUP = 1  # with discardvisual="false" MuJoCo's URDF loader puts visuals in group 1 and collisions in group 0
_PRIMITIVES = {  # geom type -> (MJCF name, how many size numbers MJCF wants)
    int(mujoco.mjtGeom.mjGEOM_SPHERE): ("sphere", 1),
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): ("capsule", 2),
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): ("cylinder", 2),
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): ("ellipsoid", 3),
    int(mujoco.mjtGeom.mjGEOM_BOX): ("box", 3),
}


def _bgr(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"colour must be '#rrggbb', got {color!r}")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def _nums(a) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.asarray(a).reshape(-1))


def _camera_xml(cam: Camera) -> str:
    W, H = cam.width, cam.height
    pos = cam.position
    xyaxes = np.concatenate([cam.R[0], -cam.R[1]])
    pitch = 1e-5
    return (f'<camera name="photo" pos="{_nums(pos)}" xyaxes="{_nums(xyaxes)}" resolution="{W} {H}" '
            f'sensorsize="{W * pitch:.9g} {H * pitch:.9g}" focalpixel="{cam.K[0, 0]:.9g} {cam.K[1, 1]:.9g}" '
            # MuJoCo takes the principal point offset the other way round: centre minus cx (checked against OpenCV)
            f'principalpixel="{W / 2 - cam.K[0, 2]:.9g} {H / 2 - cam.K[1, 2]:.9g}"/>')


class _Shells:
    """One MuJoCo model that holds every motion's machine as mocap bodies, and one renderer over it."""

    def __init__(self, scene: Scene, motions: list[Motion], cam: Camera):
        self.cam = cam
        self.motions = motions
        assets, bodies = [], []
        self._body_names: list[tuple[int, int]] = []  # (motion index, machine body id) of each mocap body
        meshes_done: set[tuple[str, int]] = set()
        for mi, motion in enumerate(motions):
            src = scene.machines[motion.machine].machine.model
            # draw the visuals only (group 1); a machine without any (collision shapes only) is drawn from all its geoms
            groups = (_VISUAL_GROUP,) if (src.geom_group == _VISUAL_GROUP).any() else tuple(range(6))
            for b in range(1, src.nbody):
                gs = [g for g in range(src.ngeom) if src.geom_bodyid[g] == b and int(src.geom_group[g]) in groups]
                gxml = []
                for g in gs:
                    t = int(src.geom_type[g])
                    common = (f'pos="{_nums(src.geom_pos[g])}" quat="{_nums(src.geom_quat[g])}" rgba="0.5 0.5 0.5 1" '
                              f'contype="0" conaffinity="0"')
                    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
                        mid = int(src.geom_dataid[g])
                        name = f"{motion.machine}_{mid}"
                        if (motion.machine, mid) not in meshes_done:
                            meshes_done.add((motion.machine, mid))
                            v0, vn = int(src.mesh_vertadr[mid]), int(src.mesh_vertnum[mid])
                            f0, fn = int(src.mesh_faceadr[mid]), int(src.mesh_facenum[mid])
                            assets.append(f'<mesh name="{name}" vertex="{_nums(src.mesh_vert[v0:v0 + vn])}" '
                                          f'face="{_nums(src.mesh_face[f0:f0 + fn])}"/>')
                        gxml.append(f'<geom type="mesh" mesh="{name}" {common}/>')
                    elif t in _PRIMITIVES:
                        kind, n = _PRIMITIVES[t]
                        gxml.append(f'<geom type="{kind}" size="{_nums(src.geom_size[g][:n])}" {common}/>')
                if gxml:
                    bodies.append(f'<body name="m{mi}_{b}" mocap="true">{"".join(gxml)}</body>')
                    self._body_names.append((mi, b))
        W, H = cam.width, cam.height
        # extent scales MuJoCo's clipping planes (znear/zfar are fractions of it): a big extent keeps a site
        # tens of metres deep inside the view whatever the size of the machine
        xml = (f'<mujoco><statistic extent="50"/>'
               f'<visual><global offwidth="{W}" offheight="{H}"/><map znear="0.002" zfar="20"/>'
               f'<headlight ambient="0.6 0.6 0.6" diffuse="0.5 0.5 0.5" specular="0 0 0"/></visual>'
               f'<asset>{"".join(assets)}</asset><worldbody>{_camera_xml(cam)}{"".join(bodies)}</worldbody></mujoco>')
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        # which render geoms and which mocap slot belong to which motion and machine body
        self.motion_geoms = [[] for _ in motions]
        self.mocap = {}  # (motion, machine body id) -> mocap index
        for mi, b in self._body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"m{mi}_{b}")
            self.mocap[(mi, b)] = int(self.model.body_mocapid[bid])
            for g in range(self.model.ngeom):
                if self.model.geom_bodyid[g] == bid:
                    self.motion_geoms[mi].append(g)
        self._src_body_ids = {mi: {scene.machines[m.machine].machine.model.body(b).name: b
                                   for (mj, b) in self._body_names if mj == mi}
                              for mi, m in enumerate(motions)}
        self._remap = _distortion_map(cam)
        self.renderer = mujoco.Renderer(self.model, H, W)
        try:
            self._warm()
        except BaseException:
            self.renderer.close()
            raise

    def _warm(self):
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="photo")
        self.renderer.render()  # the first frame of a fresh renderer can be black: draw once and drop it

    def close(self):
        self.renderer.close()

    def draw(self, mi: int, poses: dict[str, np.ndarray], bgr: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Shell of motion ``mi`` alone, opaque in ``bgr``: (BGR image, coverage mask)."""
        rgba = self.model.geom_rgba
        rgba[:, 3] = 0.0
        for g in self.motion_geoms[mi]:
            rgba[g] = (bgr[2] / 255, bgr[1] / 255, bgr[0] / 255, 1.0)
        for name, b in self._src_body_ids[mi].items():
            T = poses[name]
            k = self.mocap[(mi, b)]
            self.data.mocap_pos[k] = T[:3, 3]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(T[:3, :3]).reshape(-1))
            self.data.mocap_quat[k] = quat
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="photo")
        rgb = self.renderer.render()
        img = rgb[..., ::-1]
        if self._remap is not None:
            img = cv2.remap(np.ascontiguousarray(img), *self._remap, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = np.ascontiguousarray(img)
        return img, img.sum(axis=2) > 0


def _distortion_map(cam: Camera):
    """For each pixel of the distorted picture, where to look in the pinhole drawing; None without distortion."""
    if not np.any(cam.dist):
        return None
    u, v = np.meshgrid(np.arange(cam.width, dtype=np.float32), np.arange(cam.height, dtype=np.float32))
    pts = np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-9)  # the default 5 steps are off by pixels at strong k1
    und = cv2.undistortPoints(pts, cam.K, cam.dist, None, None, cam.K, crit).reshape(cam.height, cam.width, 2).astype(np.float32)
    return und[..., 0].copy(), und[..., 1].copy()


def _polyline(img, pts: np.ndarray, color, dashed: bool, thickness: int = 2):
    ok = np.isfinite(pts).all(axis=1)
    pts = np.round(np.where(ok[:, None], pts, 0)).astype(np.int32)
    run = 0.0
    for i in range(1, len(pts)):
        if not (ok[i] and ok[i - 1]):
            continue
        seg = float(np.hypot(*(pts[i] - pts[i - 1])))
        if not dashed or int(run // 8) % 2 == 0:  # dashes of about 8 pixels
            cv2.line(img, tuple(int(c) for c in pts[i - 1]), tuple(int(c) for c in pts[i]), color, thickness, cv2.LINE_AA)
        run += seg


def render_overlay(scene: Scene, camera_name: str, source: Source, motions: list[Motion], out: Path, *,
                   colors=COLORS, show=None, fps: float | None = None, start_s: float = 0.0) -> Path:
    """Draw ``motions`` (one or two) over ``source`` through the site camera ``camera_name``; write an mp4.

    Time runs from ``start_s`` for the longest motion's duration at ``fps`` (default: the source's, else 30);
    each output frame takes the source frame at that time (a still repeats) and shows every motion at the time
    since the start. The number of frames is ``round(duration * fps)``, at least 1: frame k is time k / fps.
    ``show`` toggles ``machine`` (the shells), ``trail`` (tool tip path so far, record B dashed) and ``link``
    (a white line between the two tool tips, when two motions drive the same machine); missing keys stay on.
    """
    if not motions:
        raise ValueError("no motions to draw")
    if len(motions) > len(colors):
        raise ValueError(f"{len(motions)} motions but only {len(colors)} colours")
    if camera_name not in scene.site.cameras:
        raise ValueError(f"the site has no camera {camera_name!r} (it has {sorted(scene.site.cameras)})")
    unknown = sorted(set(show or {}) - set(SHOW))
    if unknown:
        raise ValueError(f"unknown show keys {unknown} (use {sorted(SHOW)})")
    show = {**SHOW, **(show or {})}
    cam = camera_for(source, scene.site.cameras[camera_name])
    bgr = [_bgr(c) for c in colors]
    paths = [scene.tool_path(m) for m in motions]  # also checks each motion against its placed machine
    fps = float(fps or source.fps or 30.0)
    if not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}")
    duration = max(float(m.t[-1]) for m in motions)
    n_frames = max(1, int(round(duration * fps)))
    link = show["link"] and len(motions) == 2 and motions[0].machine == motions[1].machine

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(f".{out.stem}.partial{out.suffix}")  # an error never leaves a half-written video at ``out``
    shells = _Shells(scene, motions, cam) if show["machine"] else None
    writer = None
    try:
        writer = cv2.VideoWriter(str(partial), cv2.VideoWriter_fourcc(*"mp4v"), fps, (cam.width, cam.height))
        if not writer.isOpened():
            raise RuntimeError(f"cannot write the video {str(out)!r}")
        for k in range(n_frames):
            t = k / fps
            img = np.ascontiguousarray(source.at(start_s + t))
            if img.shape[:2] != (cam.height, cam.width):
                raise ValueError(f"the picture changed size: {img.shape[1]}x{img.shape[0]}, expected {cam.width}x{cam.height}")
            tips = []
            for mi, motion in enumerate(motions):
                placed = scene.machines[motion.machine]
                T = placed.base_transform()
                local = placed.machine.link_poses(motion.at(t))
                poses = {name: T @ P for name, P in local.items()}
                tips.append(poses[placed.machine.tool_link][:3, 3])
                if shells is not None:
                    shell, mask = shells.draw(mi, poses, bgr[mi])
                    img[mask] = (_SHELL_ALPHA * img[mask] + (1 - _SHELL_ALPHA) * shell[mask]).astype(np.uint8)
            if show["trail"]:
                for mi, motion in enumerate(motions):
                    upto = int(np.searchsorted(motion.t, t, side="right"))
                    pts = np.vstack([paths[mi][:upto], tips[mi][None]])
                    _polyline(img, cam.project(pts), bgr[mi], dashed=(mi == 1))
            if link:
                a, b = cam.project(np.array(tips))
                if np.isfinite(a).all() and np.isfinite(b).all():
                    cv2.line(img, tuple(int(round(c)) for c in a), tuple(int(round(c)) for c in b),
                             (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(img)
        writer.release()
        writer = None
        os.replace(partial, out)
    finally:
        if writer is not None:
            writer.release()
        partial.unlink(missing_ok=True)
        if shells is not None:
            shells.close()
    return out
