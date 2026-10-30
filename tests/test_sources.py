import cv2
import numpy as np
import pytest
from PIL import Image

from physical_sandbox.camera import Camera
from physical_sandbox.sources import camera_for, open_source


def _coded_frame(i, w=64, h=48):
    """A frame whose flat colour encodes the index i (survives lossy mp4 compression)."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (20 + 20 * i, 128, 255 - 20 * i)
    return img


def _index_of(frame):
    return int(round((float(frame[..., 0].mean()) - 20) / 20))


@pytest.fixture
def video_path(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    assert w.isOpened()
    for i in range(10):
        w.write(_coded_frame(i))
    w.release()
    return path


@pytest.fixture
def png_path(tmp_path):
    path = tmp_path / "still.png"
    img = np.zeros((48, 64, 3), np.uint8)
    img[:24] = (255, 0, 0)  # BGR: blue top half
    img[24:] = (0, 0, 255)  # red bottom half
    cv2.imwrite(str(path), img)
    return path


def _camera(w=640, h=480):
    K = np.array([[500.0, 0, w / 2], [0, 500.0, h / 2], [0, 0, 1]])
    return Camera(w, h, K, np.zeros(5), np.eye(3), np.zeros(3))


# --- stills

def test_png_still_properties(png_path):
    with open_source(png_path) as s:
        assert s.kind == "still"
        assert s.size == (64, 48)
        assert s.fps is None and s.count is None
        f = s.frame()
        assert f.shape == (48, 64, 3) and f.dtype == np.uint8
        assert tuple(f[0, 0]) == (255, 0, 0)  # BGR order, as OpenCV reads it
        assert tuple(f[-1, 0]) == (0, 0, 255)


def test_still_returns_the_same_image_for_any_index_or_time(png_path):
    with open_source(str(png_path)) as s:
        a = s.frame()
        np.testing.assert_array_equal(s.frame(7), a)
        np.testing.assert_array_equal(s.at(3.5), a)


def test_exif_rotated_jpeg_is_returned_the_way_it_is_seen(tmp_path):
    # A 64-wide, 32-high picture tagged "rotate 90 degrees clockwise to view" (Orientation = 6).
    arr = np.zeros((32, 64, 3), np.uint8)
    arr[:, :32] = (255, 0, 0)  # left half red (RGB)
    arr[:, 32:] = (0, 0, 255)  # right half blue
    path = tmp_path / "phone.jpg"
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[0x0112] = 6
    img.save(path, exif=exif, quality=95)
    with open_source(path) as s:
        assert s.size == (32, 64)  # width and height swapped: portrait, as displayed
        f = s.frame()
        assert f.shape == (64, 32, 3)
        # Rotated 90 degrees clockwise: the old left half (red) ends up on top.
        top, bottom = f[:16].reshape(-1, 3).mean(0), f[-16:].reshape(-1, 3).mean(0)
        assert top[2] > 200 and top[0] < 60  # red (BGR)
        assert bottom[0] > 200 and bottom[2] < 60  # blue


# --- recordings

def test_video_properties(video_path):
    with open_source(video_path) as s:
        assert s.kind == "video"
        assert s.size == (64, 48)
        assert s.fps == pytest.approx(10.0)
        assert s.count == 10


def test_video_frame_seeks_and_at_uses_fps(video_path):
    with open_source(video_path) as s:
        assert _index_of(s.frame(0)) == 0
        assert _index_of(s.frame(5)) == 5
        assert _index_of(s.frame(2)) == 2  # backwards seek
        assert _index_of(s.at(0.2)) == 2
        assert _index_of(s.at(0.0)) == 0
        assert s.frame(3).shape == (48, 64, 3)


def test_video_frame_clamps_to_the_last_frame_and_rejects_negative(video_path):
    with open_source(video_path) as s:
        assert _index_of(s.frame(99)) == 9
        assert _index_of(s.at(100.0)) == 9
        with pytest.raises(ValueError):
            s.frame(-1)
        with pytest.raises(ValueError):
            s.at(-0.1)


def test_default_frame_is_the_first(video_path):
    with open_source(str(video_path)) as s:
        assert _index_of(s.frame()) == 0


# --- errors

def test_unknown_spec_names_itself():
    with pytest.raises(ValueError, match="clip.gif"):
        open_source("clip.gif")
    with pytest.raises(ValueError, match="notes.txt"):
        open_source("notes.txt")


def test_missing_file_names_itself(tmp_path):
    for name in ("nope.png", "nope.mp4"):
        with pytest.raises(Exception, match=name):
            open_source(tmp_path / name)


def test_unreadable_file_names_itself(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a jpeg")
    with pytest.raises(Exception, match="broken.jpg"):
        open_source(bad)
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(Exception, match="broken.mp4"):
        open_source(bad)


# --- camera_for

def test_camera_for_returns_the_camera_when_sizes_match(png_path):
    cam = _camera(64, 48)
    with open_source(png_path) as s:
        assert camera_for(s, cam) is cam


def test_camera_for_scales_when_only_the_scale_differs(png_path):
    cam = _camera(640, 480)
    with open_source(png_path) as s:
        c = camera_for(s, cam)
    assert (c.width, c.height) == (64, 48)
    np.testing.assert_allclose(c.K[0, 0], 50.0)
    np.testing.assert_allclose(c.K[:2, 2], [32, 24])


def test_camera_for_refuses_a_different_aspect_ratio(png_path):
    cam = _camera(640, 360)
    with open_source(png_path) as s:
        with pytest.raises(ValueError, match=r"640x360.*64x48|64x48.*640x360"):
            camera_for(s, cam)
