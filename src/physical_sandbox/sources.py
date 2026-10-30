"""One way to read a picture source: a still, a recording or a live camera.

People try the tool with a still or a recording first and stream a fixed camera later; the rest of
the package only sees a :class:`Source`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .camera import Camera

STILL_EXTENSIONS = (".jpg", ".jpeg", ".png")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
STREAM_PREFIXES = ("rtsp://", "http://", "https://")


class Source:
    """Frames from one place. Frames are BGR ``uint8`` arrays (OpenCV order), height x width x 3.

    ``count`` and ``fps`` are None where they make no sense (stills, live cameras).
    """

    kind: Literal["still", "video", "stream"]

    def __init__(self, spec):
        self.spec = spec
        self._size: tuple[int, int] | None = None

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self._size

    @property
    def fps(self) -> float | None:
        return None

    @property
    def count(self) -> int | None:
        return None

    def frame(self, i: int = 0) -> np.ndarray:
        raise NotImplementedError

    def at(self, seconds: float) -> np.ndarray:
        """The frame shown ``seconds`` into the picture (stills and live cameras ignore the time)."""
        return self.frame(0)

    def close(self) -> None:
        pass

    def __enter__(self) -> Source:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self):
        return f"<{type(self).__name__} {self.spec!r} {self._size}>"


class StillSource(Source):
    kind = "still"

    def __init__(self, path: Path):
        super().__init__(str(path))
        # IMREAD_COLOR applies the EXIF orientation, so a phone photo comes back the way it is seen
        # (see test_exif_rotated_jpeg_is_returned_the_way_it_is_seen).
        img = cv2.imread(str(path), cv2.IMREAD_COLOR) if path.is_file() else None
        if img is None:
            raise ValueError(f"cannot read the still image {str(path)!r}")
        img.setflags(write=False)
        self._image = img
        self._size = (img.shape[1], img.shape[0])

    def frame(self, i: int = 0) -> np.ndarray:
        if i < 0:
            raise ValueError(f"frame index must be >= 0, got {i}")
        return self._image


class VideoSource(Source):
    kind = "video"

    def __init__(self, path: Path):
        super().__init__(str(path))
        self._cap = cv2.VideoCapture(str(path)) if path.is_file() else None
        if self._cap is None or not self._cap.isOpened():
            raise ValueError(f"cannot open the recording {str(path)!r}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps > 0 or count <= 0:
            self._cap.release()
            raise ValueError(f"cannot read the frame rate or length of the recording {str(path)!r}")
        self._fps, self._count = fps, count
        self._next = 0  # index of the frame the next read() returns
        first = self._read_at(0)
        self._size = (first.shape[1], first.shape[0])

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def count(self) -> int:
        return self._count

    def _read_at(self, i: int) -> np.ndarray:
        if i != self._next:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self._next = i
        ok, img = self._cap.read()
        if not ok:
            self._next = -1  # position unknown: force a seek on the next call
            raise ValueError(f"cannot read frame {i} of the recording {self.spec!r}")
        self._next = i + 1
        return img

    def frame(self, i: int = 0) -> np.ndarray:
        if i < 0:
            raise ValueError(f"frame index must be >= 0, got {i}")
        return self._read_at(min(int(i), self._count - 1))

    def at(self, seconds: float) -> np.ndarray:
        if seconds < 0:
            raise ValueError(f"time must be >= 0, got {seconds}")
        return self.frame(round(seconds * self._fps))

    def close(self) -> None:
        self._cap.release()


class StreamSource(Source):
    kind = "stream"

    def __init__(self, spec: str | int):
        super().__init__(spec)
        self._cap = cv2.VideoCapture(spec)
        if not self._cap.isOpened():
            raise ValueError(f"cannot open the live camera {spec!r}")
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort: keep the latest frame, not a queue
        first = self.frame()
        self._size = (first.shape[1], first.shape[0])

    def frame(self, i: int = 0) -> np.ndarray:
        """The latest frame (the index is ignored)."""
        ok, img = self._cap.read()
        if not ok:
            raise ValueError(f"cannot read a frame from the live camera {self.spec!r}")
        return img

    def close(self) -> None:
        self._cap.release()


def open_source(spec: str | int | os.PathLike) -> Source:
    """Open a still (.jpg .jpeg .png), a recording (.mp4 .mov .avi .mkv) or a live camera.

    A live camera is an ``rtsp://`` / ``http(s)://`` address or a device number (an int or digit string).
    Anything else raises ValueError naming the spec.
    """
    if isinstance(spec, bool):
        raise ValueError(f"not a picture source: {spec!r}")
    if isinstance(spec, int):
        return StreamSource(spec)
    text = os.fspath(spec)
    if text.isdigit():
        return StreamSource(int(text))
    if text.lower().startswith(STREAM_PREFIXES):
        return StreamSource(text)
    ext = Path(text).suffix.lower()
    if ext in STILL_EXTENSIONS:
        return StillSource(Path(text))
    if ext in VIDEO_EXTENSIONS:
        return VideoSource(Path(text))
    raise ValueError(
        f"not a picture source: {text!r} (use {', '.join(STILL_EXTENSIONS + VIDEO_EXTENSIONS)}, "
        "an rtsp:// or http(s):// address, or a camera number)"
    )


def camera_for(source: Source, camera: Camera) -> Camera:
    """The site camera, adjusted to the size of this source's frames.

    Returns the camera itself when the sizes match, a rescaled one when only the scale differs
    (a shrunk recording), and raises ValueError for a different aspect ratio.
    """
    w, h = source.size
    if (w, h) == (camera.width, camera.height):
        return camera
    if w * camera.height != h * camera.width:
        raise ValueError(
            f"the picture is {w}x{h} but the site camera is {camera.width}x{camera.height}: "
            "the aspect ratio differs, so the camera cannot be used for it"
        )
    return camera.scaled(w, h)
