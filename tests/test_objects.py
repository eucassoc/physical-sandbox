import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from physical_sandbox.objects import (
    camera_moved, fetch_sam_weights, ground_px, looks_present, overlap, sam_segmenter,
)
from physical_sandbox.site import ObjectView, mask_to_rle


def rect(h, w, x0, y0, x1, y1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_ground_px_of_a_rectangle_is_its_bottom_edge():
    m = rect(100, 200, 40, 20, 100, 80)  # rows 20..79, columns 40..99
    pts = ground_px(m)
    assert len(pts) == 3
    assert pts[0] == (40.0, 79.0)
    assert pts[1] == (69.5, 79.0)
    assert pts[2] == (99.0, 79.0)
    assert all(type(x) is float and type(y) is float for x, y in pts)


def test_ground_px_uses_only_the_bottom_band():
    # a wide base (3 rows) under a narrow neck: the band must not include the neck
    m = rect(100, 200, 80, 10, 90, 70)
    m[70:73, 50:130] = True
    pts = ground_px(m)
    assert pts[0][0] == 50.0 and pts[2][0] == 129.0
    assert all(y == 72.0 for _, y in pts)


def test_ground_px_of_an_empty_mask_is_refused():
    with pytest.raises(ValueError):
        ground_px(np.zeros((10, 10), bool))


def test_overlap_known_values():
    a = rect(50, 50, 0, 0, 10, 10)
    assert overlap(a, a) == 1.0
    b = rect(50, 50, 5, 0, 15, 10)  # shifted by half: inter 50, union 150
    assert overlap(a, b) == pytest.approx(1 / 3)
    assert overlap(a, rect(50, 50, 20, 20, 30, 30)) == 0.0
    empty = np.zeros((50, 50), bool)
    assert overlap(empty, empty) == 0.0
    assert overlap(a, empty) == 0.0


def test_camera_moved_names_only_the_things_below_threshold():
    s = (60, 60)
    before = {"tray": rect(*s, 0, 0, 10, 10), "box": rect(*s, 20, 20, 30, 30), "shelf": rect(*s, 40, 40, 50, 50)}
    after = {"tray": rect(*s, 0, 0, 10, 10), "box": rect(*s, 20, 20, 30, 30), "shelf": rect(*s, 40, 40, 50, 50)}
    assert camera_moved(before, after) == []
    # 0.5 overlap: 10x10 shifted so inter = 5*10... union = 150 -> 1/3; use 10x10 vs 10x5 -> inter 50 / union 100
    after["box"] = rect(*s, 20, 20, 30, 25)
    assert overlap(before["box"], after["box"]) == 0.5
    assert camera_moved(before, after) == ["box"]
    assert camera_moved(before, after, threshold=0.4) == []


def test_camera_moved_skips_names_in_only_one_side():
    s = (60, 60)
    before = {"tray": rect(*s, 0, 0, 10, 10), "gone": rect(*s, 20, 20, 30, 30)}
    after = {"tray": rect(*s, 0, 0, 10, 10), "new": rect(*s, 40, 40, 50, 50)}
    assert camera_moved(before, after) == []


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((120, 160, 3), 90, np.uint8)
    img[:] = (90, 120, 100)
    img += rng.integers(0, 6, img.shape, dtype=np.uint8)
    # a coloured object: red body with blue stripe
    img[30:90, 40:110] = (30, 40, 200)
    img[50:60, 40:110] = (200, 60, 30)
    return img


def _view(box):
    x0, y0, x1, y1 = box
    return ObjectView(box, mask_to_rle(rect(120, 160, x0, y0, x1, y1)), [(float(x0), float(y1))] * 3)


def test_looks_present_same_picture_is_near_one():
    img = _scene()
    v = _view((35, 25, 115, 95))
    assert looks_present(img, v, img) > 0.95


def test_looks_present_drops_when_the_object_is_gone():
    ref = _scene(0)
    gone = ref.copy()
    gone[30:90, 40:110] = (90, 120, 100)  # the object left: floor shows
    v = _view((35, 25, 115, 95))
    score = looks_present(gone, v, ref)
    assert 0.0 <= score < 0.5
    assert looks_present(_scene(1), v, ref) > 0.9  # only noise differs


def test_looks_present_is_clipped_to_unit_interval():
    ref = _scene()
    inv = 255 - ref
    s = looks_present(inv, _view((35, 25, 115, 95)), ref)
    assert 0.0 <= s <= 1.0


def test_fake_segmenter_fits_the_protocol():
    img = _scene()

    def fake(image, boxes):
        return {n: rect(image.shape[0], image.shape[1], *b) for n, b in boxes.items()}

    out = fake(img, {"a": (40, 30, 110, 90)})
    assert out["a"].dtype == bool and out["a"].shape == img.shape[:2]


def test_sam_segmenter_without_extra_says_how_to_install(monkeypatch):
    import builtins
    real = builtins.__import__

    def deny(name, *a, **k):
        if name.split(".")[0] == "ultralytics":
            raise ImportError("no ultralytics")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(ImportError, match=r"pip install 'physical-sandbox\[sam\]'"):
        sam_segmenter()


# ---- real data (needs the [sam] extra and the LeRobot dataset) ----

DATASET = Path(os.environ.get(
    "PS_LEROBOT_DATASET",
    "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/"
    "f641879e22172be7e8161d5e6c1503c2d2feb657")).expanduser()
SIDE = DATASET / "videos/observation.images.side/chunk-000/file-000.mp4"


def _frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    assert ok, f"could not read frame {index}"
    return img


@pytest.mark.sam
def test_real_side_view_tray_and_box_stay_put_between_frame_0_and_6000():
    pytest.importorskip("ultralytics")
    if not SIDE.exists():
        pytest.skip(f"dataset video not found: {SIDE}")
    seg = sam_segmenter()
    boxes = {"tray": (40, 215, 295, 370), "box": (450, 0, 640, 160)}
    a, b = _frame(SIDE, 0), _frame(SIDE, 6000)
    ma, mb = seg(a, boxes), seg(b, boxes)
    assert set(ma) == set(boxes) == set(mb)
    ov = {n: overlap(ma[n], mb[n]) for n in boxes}
    print("real-data overlap (frame 0 vs 6000):", ov)
    assert ov["tray"] >= 0.9
    assert ov["box"] >= 0.95
    assert camera_moved(ma, mb) == []
    # the box (cardboard) still looks like itself in the later frame (white balance differs a little), and
    # when its area is replaced by bare floor it no longer does
    view = ObjectView(boxes["box"], mask_to_rle(ma["box"]), ground_px(ma["box"]))
    still = looks_present(b, view, a)
    gone = b.copy()
    gone[0:160, 450:640] = cv2.resize(b[330:480, 300:490], (190, 160))
    print("real-data looks_present: still", still, "gone", looks_present(gone, view, a))
    assert still > 0.7
    assert looks_present(gone, view, a) < 0.5
