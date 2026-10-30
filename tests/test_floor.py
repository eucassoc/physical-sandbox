import cv2
import numpy as np
import pytest

from physical_sandbox.floor import solve_floor


def project_rect(f, rvec, tvec, a=3.5, b=6.8, cx=1010, cy=554):
    """Project an a x b m floor rectangle (corner 1 at the origin) through a known pinhole pose."""
    world = np.array([[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]], float)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    px, _ = cv2.projectPoints(world, np.array(rvec, float), np.array(tvec, float), K, None)
    return [(float(u), float(v)) for u, v in px.reshape(-1, 2)]


def fov_deg(camera):
    """Horizontal field of view in degrees."""
    return float(np.degrees(2 * np.arctan(camera.width / (2 * camera.K[0, 0]))))


TRUE_CORNERS = project_rect(f=900, rvec=(1.9, 0.15, -0.1), tvec=(-1.5, 0.8, 9.0))


def test_a_true_rectangle_comes_back_exactly():
    # f=900 on 2020x1108, a 3.5 x 6.8 m floor projected by a known pose (spike c)
    sol = solve_floor(2020, 1108, TRUE_CORNERS, (3.5, 6.8))
    assert sol.reason is None and sol.residual_px < 0.5
    assert fov_deg(sol.camera) == pytest.approx(97, abs=0.5)
    assert not sol.mirrored
    assert sol.camera.position[2] > 0
    assert sol.corners_m.shape == (4, 3)
    assert np.allclose(sol.corners_m, [[0, 0, 0], [3.5, 0, 0], [3.5, 6.8, 0], [0, 6.8, 0]])


def test_the_solved_camera_projects_the_corners_back():
    sol = solve_floor(2020, 1108, TRUE_CORNERS, (3.5, 6.8))
    back = sol.camera.project(sol.corners_m)
    assert np.abs(back - np.array(TRUE_CORNERS)).max() < 0.5
    assert sol.camera.width == 2020 and sol.camera.height == 1108


def test_hand_placed_corners_leave_a_residual_like_the_old_solve():
    # the corners a person decided in the spike (decided.json), with estimated lengths
    corners = [(205.3, 1018), (1951.6, 1005.1), (1463.2, 633), (1091.5, 633)]
    sol = solve_floor(2020, 1108, corners, (3.5, 6.8))
    # the old JS solve left 26.8 px; the spike's OpenCV fit left 37 px as the largest per-axis error,
    # which is 40.8 px as the largest Euclidean distance (what residual_px reports)
    assert sol.reason is None and sol.residual_px == pytest.approx(40.8, abs=1.0)


def test_corners_going_round_the_other_way_are_mirrored():
    reversed_corners = list(reversed(TRUE_CORNERS))
    sol = solve_floor(2020, 1108, reversed_corners, (3.5, 6.8))
    assert sol.mirrored and sol.camera.position[2] > 0
    assert sol.residual_px < 0.5
    # the floor lies on the -y side of corner 1 -> 2
    assert np.allclose(sol.corners_m, [[0, 0, 0], [3.5, 0, 0], [3.5, -6.8, 0], [0, -6.8, 0]])
    refused = solve_floor(2020, 1108, reversed_corners, (3.5, 6.8), allow_mirror=False)
    assert refused.reason == 'mirrored' and refused.camera is None and refused.mirrored


def test_a_floor_seen_square_on_is_refused():
    sol = solve_floor(2020, 1108, [(500, 800), (1500, 800), (1500, 300), (500, 300)], (3.5, 6.8))
    assert sol.reason == 'flat' and sol.camera is None and sol.residual_px is None


def test_a_floor_that_would_lie_behind_the_camera_is_refused():
    # A convex quad found by a seeded random search: the fit is exact-looking but puts floor corners
    # behind the camera, where the pixels are undefined (this used to come back accepted with a NaN residual).
    corners = [(5.1, 76.0), (1683.4, 309.8), (1546.7, 534.8), (1233.7, 569.6)]
    sol = solve_floor(2020, 1108, corners, (3.5, 6.8))
    assert sol.reason == 'flat' and sol.camera is None and sol.residual_px is None


@pytest.mark.parametrize("f, tvec", [(9000, (-1.5, 0.8, 60.0)), (380, (-1.5, 0.8, 3.0))])
def test_a_field_of_view_outside_15_to_120_degrees_is_refused(f, tvec):
    # f=9000 is about 13 degrees wide, f=380 about 139 degrees
    sol = solve_floor(2020, 1108, project_rect(f=f, rvec=(1.9, 0.15, -0.1), tvec=tvec), (3.5, 6.8))
    assert sol.reason == 'flat' and sol.camera is None


def test_no_accepted_solution_has_a_nan_residual():
    rng = np.random.default_rng(0)
    for _ in range(300):
        q = rng.uniform([0, 0], [2020, 1108], (4, 2))
        sol = solve_floor(2020, 1108, [tuple(p) for p in q], (3.5, 6.8))
        if sol.camera is not None:
            assert np.isfinite(sol.residual_px)
            depth = (sol.corners_m @ sol.camera.R.T + sol.camera.t)[:, 2]
            assert (depth > 0).all()


def test_a_quad_whose_first_fit_fails_is_still_mirrored_when_the_mirrored_floor_fits():
    # Found by a seeded random search (seed 0, 2 hits in 30000 random quads, not convex): the +y floor puts
    # a corner behind the camera, the mirrored floor gives a camera above the floor.
    # Full precision on purpose: this quad is marginal (rounded to 0.1 px the mirrored fit fails); the
    # patched test below covers the same path without depending on the numbers.
    corners = [(865.6868601331811, 1098.3354388157422), (1205.8742979115336, 758.367920837139),
               (1844.841155049077, 21.271732703624863), (1203.4139343581128, 961.4691803223752)]
    sol = solve_floor(2020, 1108, corners, (3.5, 6.8))
    assert sol.reason is None and sol.mirrored
    assert np.isfinite(sol.residual_px) and sol.camera.position[2] >= 0
    assert sol.corners_m[2, 1] == pytest.approx(-6.8)
    assert solve_floor(2020, 1108, corners, (3.5, 6.8), allow_mirror=False).reason == 'flat'


def test_the_mirror_path_is_tried_when_the_first_fit_returns_no_camera(monkeypatch):
    import physical_sandbox.floor as floor

    real = floor._solve
    calls = []

    def fail_on_the_plus_y_floor(width, height, px, world):
        calls.append(world[2, 1])
        return (None, None) if world[2, 1] > 0 else real(width, height, px, world)

    monkeypatch.setattr(floor, "_solve", fail_on_the_plus_y_floor)
    sol = solve_floor(2020, 1108, list(reversed(TRUE_CORNERS)), (3.5, 6.8))
    assert calls == [6.8, -6.8] and sol.mirrored and sol.reason is None
    calls.clear()
    assert solve_floor(2020, 1108, list(reversed(TRUE_CORNERS)), (3.5, 6.8), allow_mirror=False).reason == 'flat'
    assert calls == [6.8]
