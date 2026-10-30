import dataclasses

import numpy as np
import pytest

from physical_sandbox.camera import Camera
from physical_sandbox.site import Floor, ObjectView, Site, SiteObject, mask_to_rle
from physical_sandbox.objects import ground_px
from physical_sandbox.site import add_camera, locate_objects


def look_at(position, target, width, height, f, dist=None):
    """A camera at `position` looking at `target`, z up in the site."""
    position, target = np.asarray(position, float), np.asarray(target, float)
    z = target - position
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]]
    return Camera(width, height, K, np.zeros(5) if dist is None else dist, R, -R @ position)


FLOOR_M = np.array([[0, 0, 0], [4, 0, 0], [4, 2.5, 0], [0, 2.5, 0]], float)
POINTS = {
    "pole": [0.5, 0.4, 0.0],
    "post": [3.4, 0.6, 0.0],
    "kerb": [3.0, 2.2, 0.0],
    "cone": [1.0, 1.9, 0.0],
    "box": [2.0, 1.2, 0.0],
}


def mask_with_ground(u, v, h=480, w=640):
    """A small mask whose ground_px middle point is exactly (u, v) (integers)."""
    m = np.zeros((h, w), bool)
    m[int(v) - 6:int(v) + 1, int(u) - 3:int(u) + 4] = True
    return m


def make_site(cam1, with_views=True):
    objects = []
    for name, p in POINTS.items():
        px = cam1.project(np.array([p]))[0]
        views = {}
        if with_views:
            m = mask_with_ground(round(px[0]), round(px[1]))
            views["a"] = ObjectView((0, 0, 1, 1), mask_to_rle(m), ground_px(m))
        objects.append(SiteObject(name, "fixed", views, np.array(p, float)))
    floor = Floor("a", [], (4.0, 2.5), FLOOR_M)
    return Site(id="s", cameras={"a": cam1}, floor=floor, objects=objects)


def test_add_camera_recovers_a_second_camera_from_shared_objects():
    cam1 = look_at([2.0, -4.0, 3.0], [2.0, 1.2, 0.0], 640, 480, 600)
    cam2 = look_at([5.5, 4.0, 2.5], [2.0, 1.2, 0.0], 800, 600, 700)
    site = make_site(cam1)
    seen = {n: tuple(cam2.project(np.array([p]))[0]) for n, p in POINTS.items()}
    new_site, residual = add_camera(site, "b", 800, 600, seen)
    assert residual < 1.0
    assert new_site.residual_px["b"] == residual
    cam = new_site.cameras["b"]
    assert np.linalg.norm(cam.position - cam2.position) < 0.15
    probe = np.array([[1.0, 1.0, 0.0], [3.0, 2.0, 0.0], [2.0, 0.3, 0.0]])
    assert np.abs(cam.project(probe) - cam2.project(probe)).max() < 1.0
    # the input is untouched
    assert "b" not in site.cameras and "b" not in site.residual_px
    assert "a" in new_site.cameras


def test_add_camera_uses_floor_corners_too():
    cam1 = look_at([2.0, -4.0, 3.0], [2.0, 1.2, 0.0], 640, 480, 600)
    cam2 = look_at([5.5, 4.0, 2.5], [2.0, 1.2, 0.0], 800, 600, 700)
    site = make_site(cam1)
    site = dataclasses.replace(site, objects=site.objects[:1])  # only the pole is known besides the corners
    seen = {"pole": tuple(cam2.project(np.array([POINTS["pole"]]))[0])}
    for i in range(4):
        seen[f"corner-{i + 1}"] = tuple(cam2.project(FLOOR_M[i:i + 1])[0])
    _, residual = add_camera(site, "b", 800, 600, seen)
    assert residual < 1.0


def test_add_camera_with_three_shared_points_is_refused():
    cam1 = look_at([2.0, -4.0, 3.0], [2.0, 1.2, 0.0], 640, 480, 600)
    cam2 = look_at([5.5, 4.0, 2.5], [2.0, 1.2, 0.0], 800, 600, 700)
    site = make_site(cam1)
    seen = {n: tuple(cam2.project(np.array([POINTS[n]]))[0]) for n in ["pole", "post", "kerb"]}
    with pytest.raises(ValueError, match="4 or more"):
        add_camera(site, "b", 800, 600, seen)


def test_add_camera_ignores_names_with_no_known_position_and_lists_them():
    cam1 = look_at([2.0, -4.0, 3.0], [2.0, 1.2, 0.0], 640, 480, 600)
    cam2 = look_at([5.5, 4.0, 2.5], [2.0, 1.2, 0.0], 800, 600, 700)
    site = make_site(cam1)
    seen = {n: tuple(cam2.project(np.array([POINTS[n]]))[0]) for n in ["pole", "post", "kerb"]}
    seen["ghost"] = (10.0, 10.0)
    seen["corner-1"] = (20.0, 20.0)
    site = dataclasses.replace(site, floor=None)  # corners are unknown now
    with pytest.raises(ValueError, match="4 or more") as e:
        add_camera(site, "b", 800, 600, seen)
    assert "ghost" in str(e.value) and "corner-1" in str(e.value)


def test_add_camera_accepts_a_list_of_ground_points_and_uses_the_middle():
    cam1 = look_at([2.0, -4.0, 3.0], [2.0, 1.2, 0.0], 640, 480, 600)
    cam2 = look_at([5.5, 4.0, 2.5], [2.0, 1.2, 0.0], 800, 600, 700)
    site = make_site(cam1)
    seen = {}
    for n, p in POINTS.items():
        u, v = cam2.project(np.array([p]))[0]
        seen[n] = [(u - 5, v), (u, v), (u + 5, v)]
    _, residual = add_camera(site, "b", 800, 600, seen)
    assert residual < 1.0


def test_add_camera_refuses_a_solution_below_the_floor():
    # Pixels of a camera that sees the floor from below: mirror the camera under the floor.
    cam1 = look_at([2.0, -4.0, 3.0], [2.0, 1.2, 0.0], 640, 480, 600)
    below = look_at([5.5, 4.0, -2.5], [2.0, 1.2, 0.0], 800, 600, 700)
    site = make_site(cam1)
    seen = {n: tuple(below.project(np.array([p]))[0]) for n, p in POINTS.items()}
    with pytest.raises(ValueError):
        add_camera(site, "b", 800, 600, seen)


def test_locate_objects_puts_objects_on_the_floor_from_one_camera():
    cam1 = look_at([2.0, -4.0, 3.0], [2.0, 1.2, 0.0], 640, 480, 600)
    site = make_site(cam1)
    # forget the metres of all but the pole; the pole keeps its own ground_m via no view in camera "zzz"
    objs = [dataclasses.replace(o, ground_m=None) for o in site.objects]
    loose = SiteObject("loose", "target", {}, np.array([9.0, 9.0, 0.0]))
    site2 = dataclasses.replace(site, objects=objs + [loose])
    located = locate_objects(site2, "a")
    for o, (name, p) in zip(located.objects, POINTS.items()):
        assert o.name == name
        assert np.abs(o.ground_m - np.array(p)).max() < 0.02  # pixels were rounded to whole numbers
        assert o.ground_m[2] == 0.0
    assert np.array_equal(located.objects[-1].ground_m, [9.0, 9.0, 0.0])  # no view in "a": kept
    assert all(o.ground_m is None for o in site2.objects[:-1])  # the input is not mutated
    assert located is not site2
