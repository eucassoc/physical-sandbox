"""Fixing a site: the camera from four corners of a rectangle on the ground.

The floor is laid in site coordinates (z = 0) with corner 1 at the origin and edge 1 -> 2 along +x.
Edge 2 -> 3 goes to +y, or to -y when the corners go round the other way and the camera would
otherwise end up below the floor (the floor is then mirrored across the line 1 -> 2).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera

MIN_FOV_DEG = 15.0
MAX_FOV_DEG = 120.0


@dataclass(frozen=True, eq=False)
class FloorSolution:
    camera: Camera | None
    corners_m: np.ndarray  # (4, 3) the floor corners used, z = 0
    residual_px: float | None  # the largest reprojection distance over the four corners
    mirrored: bool  # the floor had to be mirrored across the line 1 -> 2
    reason: str | None  # None, 'flat' or 'mirrored'


def _floor(edges_m, sign: float) -> np.ndarray:
    a, b = edges_m
    return np.array([[0, 0, 0], [a, 0, 0], [a, sign * b, 0], [0, sign * b, 0]], float)


def _solve(width, height, px, world):
    """One attempt on a given floor layout: (camera, residual) or (None, None) when the picture is flat."""
    cx, cy = width / 2, height / 2
    floor2d = world[:, :2]
    H, _ = cv2.findHomography(floor2d, px - [cx, cy])
    if H is None:
        return None, None
    h1, h2 = H[:, 0], H[:, 1]
    # Two conditions on w = 1/f^2, solved together (either alone breaks down when one edge's
    # vanishing point is at infinity, a floor seen straight down a road): the columns are
    # orthogonal, and equally long (both edges are in metres).
    A = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    B = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    aa = float(A @ A)
    w = -float(A @ B) / aa if aa > 0 else -1.0
    if not w > 0:
        return None, None  # seen square-on, or the corners do not make a rectangle in perspective
    f = float(1 / np.sqrt(w))
    # Then the focal length and the pose together, the closest fit to the four corners (OpenCV's own
    # calibration from one view: principal point at the centre, square pixels, no distortion). The two
    # conditions above rarely agree exactly on a real photo, so the estimate is only the start.
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    K0 = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    # (calibrateCamera's own error value is unused: the residual is recomputed below as a Euclidean maximum.)
    _, K, _, rvecs, tvecs = cv2.calibrateCamera(
        [world.astype(np.float32)], [px.astype(np.float32)], (int(width), int(height)), K0, np.zeros(5), flags=flags)
    fov = np.degrees(2 * np.arctan(width / (2 * K[0, 0])))
    if not (MIN_FOV_DEG <= fov <= MAX_FOV_DEG):
        return None, None
    R, _ = cv2.Rodrigues(rvecs[0])
    t = tvecs[0].reshape(3)
    # Every floor corner must be in front of the camera; a fit that puts one behind it is no solution.
    if not ((world @ R.T + t)[:, 2] > 0).all():
        return None, None
    camera = Camera(width, height, K, np.zeros(5), R, t)
    residual = float(np.linalg.norm(camera.project(world) - px, axis=1).max())
    if not np.isfinite(residual):
        return None, None
    return camera, residual


def solve_floor(width: int, height: int, corners_px: list[tuple[float, float]], edges_m: tuple[float, float],
                *, allow_mirror: bool = True) -> FloorSolution:
    """The camera from four floor corners going round (1 -> 2 is ``edges_m[0]`` long, 2 -> 3 ``edges_m[1]``).

    Square pixels, the principal point at the centre, no lens distortion. ``reason`` is ``'flat'`` when the
    floor is seen square-on (or the focal length is out of range: field of view outside 15-120 degrees), and
    ``'mirrored'`` when the corners go round the other way and ``allow_mirror`` is False.
    """
    px = np.array(corners_px, dtype=float).reshape(4, 2)
    world = _floor(edges_m, +1.0)
    camera, residual = _solve(width, height, px, world)
    if camera is None:
        return FloorSolution(None, world, None, False, "flat")
    if camera.position[2] >= 0:
        return FloorSolution(camera, world, residual, False, None)
    if not allow_mirror:
        return FloorSolution(None, world, None, True, "mirrored")
    world = _floor(edges_m, -1.0)
    camera, residual = _solve(width, height, px, world)
    if camera is None:
        return FloorSolution(None, world, None, True, "flat")
    return FloorSolution(camera, world, residual, True, None)
