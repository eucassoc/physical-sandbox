import json
from pathlib import Path

import numpy as np
import pytest

from physical_sandbox.camera import Camera
from physical_sandbox.site import (
    Floor, ObjectView, Site, SiteObject, load_old_site, load_site, mask_to_rle, rle_to_mask, save_site,
)

OLD_SITE = Path.home() / "work/physical-sandbox-proto/sites/s-1793271472612.json"
needs_old_site = pytest.mark.skipif(not OLD_SITE.exists(), reason="the old site file is not on this machine")


def make_camera(width=640, height=480):
    K = [[500.0, 0, width / 2], [0, 500.0, height / 2], [0, 0, 1]]
    R = [[0, -1, 0], [-0.5, 0, -0.8660254037844386], [0.8660254037844386, 0, -0.5]]
    return Camera(width, height, K, [0.1, -0.02, 0.001, 0.002, 0.0], R, [0.3, 1.2, 5.5])


def make_site():
    mask = np.zeros((6, 8), bool)
    mask[1:4, 2:5] = True
    view = ObjectView(box=(2, 1, 5, 4), mask_rle=mask_to_rle(mask), ground_px=[(3.5, 4.0), (4.5, 4.25)])
    obj = SiteObject("pole", "fixed", {"side": view}, np.array([1.0, 2.0, 0.0]))
    loose = SiteObject("box", "target", {"side": view}, None)
    floor = Floor("side", [(10.0, 20.0), (300.0, 22.5), (310.0, 200.0), (5.0, 190.0)], (4.0, 2.5),
                  np.array([[0, 0, 0], [4, 0, 0], [4, 2.5, 0], [0, 2.5, 0]], float))
    return Site(
        id="s-test", cameras={"side": make_camera(), "up": make_camera(320, 240)}, floor=floor,
        objects=[obj, loose], residual_px={"side": 0.75, "up": 1.5}, confirmed_by_person=True,
        dimensions="measured", history=[{"at": "2026-10-30T01:02:03Z", "what": "decided", "note": "x"}],
    )


def test_save_and_load_give_back_every_value(tmp_path):
    site = make_site()
    path = tmp_path / "site.json"
    save_site(site, path)
    back = load_site(path)
    assert back.id == site.id
    assert set(back.cameras) == {"side", "up"}
    for name, cam in site.cameras.items():
        b = back.cameras[name]
        assert (b.width, b.height) == (cam.width, cam.height)
        for a in ("K", "dist", "R", "t"):
            np.testing.assert_array_equal(getattr(b, a), getattr(cam, a))
    assert back.floor.camera == "side"
    assert back.floor.corners_px == site.floor.corners_px
    assert back.floor.edges_m == (4.0, 2.5)
    np.testing.assert_array_equal(back.floor.corners_m, site.floor.corners_m)
    assert [o.name for o in back.objects] == ["pole", "box"]
    pole, box = back.objects
    assert (pole.kind, box.kind) == ("fixed", "target")
    np.testing.assert_array_equal(pole.ground_m, [1.0, 2.0, 0.0])
    assert box.ground_m is None
    v = pole.views["side"]
    assert v.box == (2, 1, 5, 4)
    assert v.ground_px == [(3.5, 4.0), (4.5, 4.25)]
    np.testing.assert_array_equal(rle_to_mask(v.mask_rle), rle_to_mask(site.objects[0].views["side"].mask_rle))
    assert back.residual_px == {"side": 0.75, "up": 1.5}
    assert back.confirmed_by_person is True
    assert back.dimensions == "measured"
    assert back.history == site.history


def test_a_site_without_a_floor_round_trips(tmp_path):
    site = Site("s-empty", {"side": make_camera()}, None, [], {}, False, "estimated", [])
    save_site(site, tmp_path / "s.json")
    back = load_site(tmp_path / "s.json")
    assert back.floor is None and back.objects == [] and back.confirmed_by_person is False


def test_the_file_names_its_format(tmp_path):
    save_site(make_site(), tmp_path / "s.json")
    assert json.loads((tmp_path / "s.json").read_text())["format"] == "physical-sandbox.site/1"


@pytest.mark.parametrize("fmt", ["physical-sandbox.site/2", "other", None])
def test_an_unknown_or_missing_format_is_refused(tmp_path, fmt):
    save_site(make_site(), tmp_path / "s.json")
    d = json.loads((tmp_path / "s.json").read_text())
    if fmt is None:
        del d["format"]
    else:
        d["format"] = fmt
    (tmp_path / "bad.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="format"):
        load_site(tmp_path / "bad.json")


def test_rle_round_trip_and_the_coco_layout():
    rng = np.random.default_rng(0)
    mask = rng.random((7, 11)) > 0.6
    rle = mask_to_rle(mask)
    assert rle["size"] == [7, 11]
    assert sum(rle["counts"]) == 77
    np.testing.assert_array_equal(rle_to_mask(rle), mask)
    assert rle_to_mask(rle).dtype == bool
    # column-major, starting with a run of False (here 0 long, as the first pixel is True)
    m = np.array([[1, 0], [1, 1]], bool)  # column-major: 1 1 0 1
    assert mask_to_rle(m) == {"size": [2, 2], "counts": [0, 2, 1, 1]}
    assert mask_to_rle(np.zeros((2, 3), bool))["counts"] == [6]
    assert mask_to_rle(np.ones((2, 3), bool))["counts"] == [0, 6]
    np.testing.assert_array_equal(rle_to_mask(mask_to_rle(np.ones((2, 3), bool))), np.ones((2, 3), bool))


def test_rle_survives_json():
    mask = np.zeros((5, 5), bool)
    mask[2, :] = True
    rle = json.loads(json.dumps(mask_to_rle(mask)))
    np.testing.assert_array_equal(rle_to_mask(rle), mask)


@needs_old_site
def test_old_site_gives_the_camera_and_nothing_else():
    site = load_old_site(OLD_SITE)
    assert site.id == "s-1793271472612"
    assert list(site.cameras) == ["camera"]
    assert site.dimensions == "estimated" and site.confirmed_by_person is True
    assert site.floor is None and site.objects == [] and site.residual_px == {}
    px = site.cameras["camera"].project(np.array([[0, 0, 0], [3, 0, 0], [6, 12, 0]], float))
    np.testing.assert_allclose(px, [[239.7, 972.2], [884.3, 1071.5], [1453.7, 618.7]], atol=0.15)


@needs_old_site
def test_old_site_id_falls_back_to_the_file_stem(tmp_path):
    d = json.loads(OLD_SITE.read_text())
    del d["id"]
    p = tmp_path / "my-site.json"
    p.write_text(json.dumps(d))
    assert load_old_site(p).id == "my-site"
