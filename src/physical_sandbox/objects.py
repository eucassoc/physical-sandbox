"""Fixed things chosen whole: an outline and a name, cut out from a rough box.

The cutting is done by a ``Segmenter`` (SAM 2.1 by default, behind the ``[sam]`` extra); everything else here
is plain numpy/OpenCV, so a person or a test can swap the segmenter for anything with the same signature.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .site import ObjectView

WEIGHTS_NAME = "sam2.1_t.pt"
_NEED_SAM = "Choosing objects needs SAM: pip install 'physical-sandbox[sam]'"


class Segmenter(Protocol):
    def __call__(self, image: np.ndarray, boxes: dict[str, tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
        """image: (H, W, 3) BGR; boxes: name -> (x0, y0, x1, y1). Returns name -> (H, W) bool mask."""


def _cache_dir() -> Path:
    d = Path(os.environ.get("PS_CACHE_DIR", "~/.cache/physical-sandbox")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _import_sam():
    try:
        from ultralytics import SAM
    except ImportError as e:
        raise ImportError(_NEED_SAM) from e
    return SAM


def fetch_sam_weights(dest: Path | None = None) -> Path:
    """Download the SAM weights once, ahead of time (the only time the network is used). Returns the file."""
    _import_sam()
    dest = Path(dest).expanduser() if dest is not None else _cache_dir() / WEIGHTS_NAME
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset
    attempt_download_asset(str(dest))  # downloads the asset named like the file into that path
    if not dest.exists():
        raise RuntimeError(f"could not download the SAM weights to {dest}")
    return dest


def sam_segmenter(weights: str | None = None) -> Segmenter:
    """A Segmenter backed by SAM 2.1 (tiny). ``weights`` may point to a local file; by default the cached one is
    used (fetched on first use)."""
    SAM = _import_sam()
    path = Path(weights).expanduser() if weights else fetch_sam_weights()
    if not path.exists():
        raise FileNotFoundError(f"SAM weights not found: {path}")
    model = SAM(str(path))

    def run(image: np.ndarray, boxes: dict[str, tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
        names = list(boxes)
        if not names:
            return {}

        def cut(sub: list[str]) -> list[np.ndarray]:
            res = model(image, bboxes=[[float(v) for v in boxes[n]] for n in sub], verbose=False)[0]
            if res.masks is None:
                return []
            return [m.astype(bool) for m in res.masks.data.cpu().numpy()]

        masks = cut(names)
        if len(masks) == len(names):  # masks come back in prompt order: map by order only when counts match
            return dict(zip(names, masks))
        out: dict[str, np.ndarray] = {}
        for n in names:  # otherwise one box at a time
            one = cut([n])
            if len(one) == 1:
                out[n] = one[0]
        return out

    return run


def ground_px(mask: np.ndarray) -> list[tuple[float, float]]:
    """Where a thing touches the ground, as three points on its bottom band (rows within 3 px of the lowest
    mask row): left end, middle, right end. A big thing is held by its bottom edge, not by one point."""
    ys, xs = np.nonzero(np.asarray(mask, bool))
    if ys.size == 0:
        raise ValueError("the mask is empty: nothing to find the ground of")
    bottom = int(ys.max())
    row = xs[ys >= bottom - 3]
    left, right = float(row.min()), float(row.max())
    y = float(bottom)
    return [(left, y), ((left + right) / 2.0, y), (right, y)]


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two masks; two empty masks give 0.0."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / union


def _hs_hist(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"box {box} is outside the image")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # coarse bins: the same thing under a shifted white balance or exposure should still match
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256])
    return cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)


def looks_present(image: np.ndarray, view: ObjectView, reference_image: np.ndarray) -> float:
    """How much the thing still looks like it did, 0..1: the hue-saturation histogram inside the view's box,
    compared with the same box in the reference picture. SAM cuts out something even when the thing has left,
    so this is the check that it is still there."""
    a = _hs_hist(image, view.box)
    b = _hs_hist(reference_image, view.box)
    score = float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))
    if not np.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, score))


def camera_moved(before: dict[str, np.ndarray], after: dict[str, np.ndarray], *, threshold: float = 0.7) -> list[str]:
    """Names of the fixed things whose outline overlaps less than ``threshold`` between two pictures; only
    names present in both are compared."""
    return [n for n in before if n in after and overlap(before[n], after[n]) < threshold]
