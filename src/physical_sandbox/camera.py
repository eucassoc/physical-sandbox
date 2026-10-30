"""The camera model: a pinhole with lens distortion, placed in site coordinates.

Site coordinates follow ROS: x forward, y left, z up, metres, the floor at z = 0. The camera
follows OpenCV: ``p_cam = R @ p + t`` with x right, y down, z forward.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _array(value, shape, name):
    a = np.array(value, dtype=float)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class Camera:
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsics
    dist: np.ndarray  # (5,) OpenCV distortion: k1, k2, p1, p2, k3
    R: np.ndarray  # (3, 3) site -> camera
    t: np.ndarray  # (3,)

    def __post_init__(self):
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "K", _array(self.K, (3, 3), "K"))
        object.__setattr__(self, "dist", _array(self.dist, (5,), "dist"))
        object.__setattr__(self, "R", _array(self.R, (3, 3), "R"))
        object.__setattr__(self, "t", _array(self.t, (3,), "t"))

    @property
    def position(self) -> np.ndarray:
        """The camera's centre in site coordinates, ``-Rᵀt``."""
        return -self.R.T @ self.t

    def project(self, points: np.ndarray) -> np.ndarray:
        """Site points (N, 3) to pixels (N, 2).

        A point in front of the camera always gives finite pixels, even outside the picture.
        A point at or behind the camera plane (camera-space z <= 0) gives NaN.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.full((len(pts), 2), np.nan)
        front = (pts @ self.R.T + self.t)[:, 2] > 0
        if front.any():
            rvec, _ = cv2.Rodrigues(self.R)
            px, _ = cv2.projectPoints(pts[front].reshape(-1, 1, 3), rvec, self.t, self.K, self.dist)
            out[front] = px.reshape(-1, 2)
        return out

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """The viewing ray through pixel (u, v): the camera's centre and a unit direction, in site coordinates."""
        xy = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), self.K, self.dist).reshape(2)
        d_cam = np.array([xy[0], xy[1], 1.0])
        d = self.R.T @ d_cam
        return self.position, d / np.linalg.norm(d)

    def floor_point(self, u: float, v: float) -> np.ndarray | None:
        """Where the ray through pixel (u, v) meets the floor (z = 0), or None if it never does in front of the camera."""
        origin, d = self.ray(u, v)
        if abs(d[2]) < 1e-12:
            return None
        s = -origin[2] / d[2]
        if s <= 0:
            return None
        p = origin + s * d
        p[2] = 0.0
        return p

    def scaled(self, width: int, height: int) -> Camera:
        """The same camera for a picture of another size. Only the same aspect ratio is allowed."""
        if width * self.height != height * self.width:
            raise ValueError(
                f"cannot scale a {self.width}x{self.height} camera to {width}x{height}: the aspect ratio differs"
            )
        f = width / self.width
        K = self.K.copy()
        K[:2] *= f  # fx, fy, cx, cy (the bottom row stays 0 0 1)
        return Camera(width, height, K, self.dist, self.R, self.t)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        return cls(d["width"], d["height"], d["K"], d["dist"], d["R"], d["t"])
