import json
from pathlib import Path

import numpy as np
import pytest

from physical_sandbox.camera import Camera
from physical_sandbox.site import load_old_site

OLD_SITE = Path.home() / "work/physical-sandbox-proto/sites/s-1793271472612.json"

pytestmark = pytest.mark.skipif(not OLD_SITE.exists(), reason="the old site file is not on this machine")


@pytest.fixture
def cam():
    return load_old_site(OLD_SITE).cameras['camera']


def test_project_matches_the_old_ps_camera(cam):
    px = cam.project(np.array([[0, 0, 0], [3, 0, 0], [6, 12, 0]], float))
    np.testing.assert_allclose(px, [[239.7, 972.2], [884.3, 1071.5], [1453.7, 618.7]], atol=0.15)


def test_floor_point_inverts_project(cam):
    got = cam.floor_point(*cam.project(np.array([[1.0, 4.0, 0]]))[0])
    np.testing.assert_allclose(got, [1.0, 4.0, 0], atol=1e-6)


def test_floor_point_above_the_horizon_is_none(cam):
    assert cam.floor_point(1000, 10) is None


def test_behind_the_camera_is_nan(cam):
    forward = cam.R[2]  # the camera's z axis in site coordinates
    assert np.isnan(cam.project((cam.position - 2 * forward)[None])).all()


def test_in_front_but_off_the_picture_is_a_number(cam):
    px = cam.project(np.array([[1.75, -2, 1.0]]))
    assert np.isfinite(px).all()
    assert px[0, 0] < 0  # spike b: u = -405.9


def test_ray_starts_at_the_camera_and_is_a_unit_vector(cam):
    origin, direction = cam.ray(1010, 554)
    np.testing.assert_allclose(origin, cam.position)
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    # the principal point looks along the camera's z axis
    np.testing.assert_allclose(direction, cam.R[2], atol=1e-9)


def test_dict_round_trip(cam):
    d = json.loads(json.dumps(cam.to_dict()))
    back = Camera.from_dict(d)
    assert (back.width, back.height) == (cam.width, cam.height)
    for a in ("K", "dist", "R", "t"):
        np.testing.assert_array_equal(getattr(back, a), getattr(cam, a))


def test_scaled_keeps_the_aspect_or_refuses(cam):
    small = cam.scaled(1010, 554)
    assert small.K[0, 0] == pytest.approx(cam.K[0, 0] / 2)
    assert (small.width, small.height) == (1010, 554)
    np.testing.assert_allclose(small.project(np.array([[3.0, 0, 0]])), cam.project(np.array([[3.0, 0, 0]])) / 2, atol=1e-6)
    with pytest.raises(ValueError):
        cam.scaled(640, 480)
