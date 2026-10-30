import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from physical_sandbox.camera import Camera
from physical_sandbox.cli import main
from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_numpy, save_record
from physical_sandbox.report import write_report
from physical_sandbox.compare import compare
from physical_sandbox.scene import PlacedMachine, Scene, save_scene
from physical_sandbox.site import Site, load_site, save_site

URDF = """<?xml version="1.0"?>
<robot name="blocks">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.15"/><geometry><box size="0.3 0.3 0.3"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.1 0.1"/></geometry></visual>
  </link>
  <joint name="turn" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.4"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  <link name="tip">
    <inertial><mass value="0.1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
  <joint name="tip_fixed" type="fixed"><parent link="arm"/><child link="tip"/><origin xyz="0.4 0 0"/></joint>
</robot>
"""
W, H = 320, 240


def look_at(eye, target=(0, 0, 0.2), size=(W, H), f=300.0):
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    K = [[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]]
    return Camera(size[0], size[1], K, np.zeros(5), R, -R @ eye)


def motion(end, label, fps=10.0, n=11):
    q = np.linspace(0, end, n).reshape(-1, 1)
    return motion_from_numpy(q, fps=fps, joint_names=["turn"], machine="blocks", label=label)


@pytest.fixture
def files(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    urdf = tmp_path / "blocks.urdf"
    urdf.write_text(URDF)
    site = Site(id="yard", cameras={"camera": look_at((-2.5, 0.8, 1.2))}, residual_px={"camera": 0.7},
                confirmed_by_person=True, dimensions="measured")
    save_site(site, tmp_path / "site.json")
    placed = PlacedMachine(machine=load_machine(urdf), urdf=str(urdf), x=0.0, y=0.0, yaw=0.0)
    save_scene(Scene(site=site, machines={"blocks": placed}), tmp_path / "scene.json")
    save_record([motion(1.0, "slow")], "yard", tmp_path / "a.record.json")
    save_record([motion(1.5, "wide")], "yard", tmp_path / "b.record.json")
    still = tmp_path / "grey.png"
    cv2.imwrite(str(still), np.full((H, W, 3), 128, np.uint8))
    return tmp_path, site, still


def run_compare(tmp_path, still, extra=()):
    out = tmp_path / "out"
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "camera", "--a", str(tmp_path / "a.record.json"),
                 "--b", str(tmp_path / "b.record.json"), "--out", str(out), "--video", str(still), *extra])
    return code, out


def test_compare_writes_the_four_files(files):
    tmp_path, site, still = files
    code, out = run_compare(tmp_path, still)
    assert code == 0
    for name in ("gaps.json", "gaps.csv", "gaps.png", "replay.mp4"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    data = json.loads((out / "gaps.json").read_text())
    assert data["header"] == {"site": "yard", "residualPx": {"camera": 0.7}, "dimensions": "measured",
                              "dimensionsNote": None, "confirmedByPerson": True}
    assert data["labels"] == {"a": "slow", "b": "wide"}
    assert data["tool_deviation_m"]["max"] > 0
    cap = cv2.VideoCapture(str(out / "replay.mp4"))
    assert cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 1
    assert cv2.imread(str(out / "gaps.png")) is not None


def test_csv_has_one_row_per_scalar_item_without_the_series(files):
    tmp_path, _, still = files
    _, out = run_compare(tmp_path, still)
    rows = list(csv.DictReader((out / "gaps.csv").open()))
    assert list(rows[0]) == ["item", "value", "unit", "at_s"]
    by = {r["item"]: r for r in rows}
    assert by["tool_deviation.max"]["unit"] == "m" and by["tool_deviation.max"]["at_s"] != ""
    assert by["tool_deviation.mean"]["at_s"] == ""
    assert by["shape_deviation"]["unit"] == "m"
    assert by["duration.diff"]["unit"] == "s"
    assert by["joints.turn.max"]["unit"] == "rad"
    assert float(by["joints.turn.max"]["value"]) > 0
    assert "jerk.a" in by and "grasp_count.a" in by
    assert not any(k.startswith("series") for k in by)
    assert len(by) == len(rows)


def test_header_says_when_dimensions_are_estimated(files, tmp_path):
    tmp, site, still = files
    site.dimensions, site.confirmed_by_person = "estimated", False
    placed = PlacedMachine(machine=load_machine(tmp / "blocks.urdf"), urdf=str(tmp / "blocks.urdf"),
                           x=0.0, y=0.0, yaw=0.0)
    scene = Scene(site=site, machines={"blocks": placed})
    c = compare(scene, motion(1.0, "a"), motion(1.5, "b"))
    out = tmp / "r"
    paths = write_report(c, out, site=site)
    assert set(paths) == {"gaps.json", "gaps.csv", "gaps.png"}
    header = json.loads(paths["gaps.json"].read_text())["header"]
    assert header["dimensions"] == "estimated"
    assert header["dimensionsNote"] == "寸法は推定（未確認）"
    assert header["confirmedByPerson"] is False


def test_label_selects_a_motion_in_a_record(files):
    tmp_path, site, still = files
    save_record([motion(0.5, "first"), motion(1.0, "second")], "yard", tmp_path / "two.record.json")
    out = tmp_path / "out"
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "camera", "--a", str(tmp_path / "two.record.json") + "#second",
                 "--b", str(tmp_path / "b.record.json"), "--out", str(out), "--video", str(still)])
    assert code == 0
    assert json.loads((out / "gaps.json").read_text())["labels"]["a"] == "second"


def test_unknown_label_is_a_bad_input(files, capsys):
    tmp_path, _, still = files
    code, out = run_compare(tmp_path, still)  # warm the scene
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "camera", "--a", str(tmp_path / "a.record.json") + "#nope",
                 "--b", str(tmp_path / "b.record.json"), "--out", str(tmp_path / "o2"), "--video", str(still)])
    err = capsys.readouterr().err
    assert code == 2
    assert "nope" in err and "slow" in err and err.strip().count("\n") == 0


def test_no_video_and_not_lerobot_says_so(files, capsys):
    tmp_path, _, _ = files
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "camera", "--a", str(tmp_path / "a.record.json"),
                 "--b", str(tmp_path / "b.record.json"), "--out", str(tmp_path / "o")])
    assert code == 2
    assert "--video" in capsys.readouterr().err


def test_missing_file_is_a_one_line_error(files, capsys):
    tmp_path, _, still = files
    code = main(["compare", "--site", str(tmp_path / "nosite.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "camera", "--a", "x", "--b", "y", "--out", str(tmp_path / "o"),
                 "--video", str(still)])
    err = capsys.readouterr().err
    assert code == 2 and err.strip().count("\n") == 0 and "nosite.json" in err


def test_lerobot_spec_needs_exactly_one_placed_machine(files, capsys):
    tmp_path, site, still = files
    urdf = tmp_path / "blocks.urdf"
    two = Scene(site=site, machines={
        "m1": PlacedMachine(load_machine(urdf), str(urdf), 0, 0, 0),
        "m2": PlacedMachine(load_machine(urdf), str(urdf), 1, 0, 0)})
    save_scene(two, tmp_path / "two-scene.json")
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "two-scene.json"),
                 "--camera", "camera", "--a", f"lerobot:{tmp_path}#0", "--b", str(tmp_path / "b.record.json"),
                 "--out", str(tmp_path / "o"), "--video", str(still)])
    err = capsys.readouterr().err
    assert code == 2 and "m1" in err and "m2" in err


def floor_image(tmp_path):
    cam = look_at((-3.0, 1.0, 2.0), target=(1.0, 0.0, 0.0), size=(640, 480), f=500.0)
    world = np.array([[0, 0, 0], [4, 0, 0], [4, 3, 0], [0, 3, 0]], float)
    px = cam.project(world)
    image = tmp_path / "floor.png"
    cv2.imwrite(str(image), np.full((480, 640, 3), 100, np.uint8))
    return image, px


def test_site_fit_writes_an_unconfirmed_estimated_site(tmp_path, capsys):
    image, px = floor_image(tmp_path)
    out = tmp_path / "made.json"
    code = main(["site", "fit", str(image), "--edges", "4", "3", "--corners",
                 *[f"{u:.3f},{v:.3f}" for u, v in px], "--out", str(out)])
    assert code == 0
    site = load_site(out)
    assert site.confirmed_by_person is False and site.dimensions == "estimated"
    assert "camera" in site.cameras and site.residual_px["camera"] < 1.0
    assert site.floor is not None and site.floor.edges_m == (4.0, 3.0)
    assert site.id == "made"
    assert "camera" in capsys.readouterr().out


def test_site_fit_measured_and_camera_name(tmp_path):
    image, px = floor_image(tmp_path)
    out = tmp_path / "s.json"
    code = main(["site", "fit", str(image), "--edges", "4", "3", "--corners",
                 *[f"{u},{v}" for u, v in px], "--out", str(out), "--camera", "front", "--measured"])
    assert code == 0
    site = load_site(out)
    assert site.dimensions == "measured" and "front" in site.cameras and site.confirmed_by_person is False


def test_site_fit_refuses_a_failed_solve(tmp_path, capsys):
    image, _ = floor_image(tmp_path)
    out = tmp_path / "bad.json"
    # a degenerate "floor": all four corners in one place
    code = main(["site", "fit", str(image), "--edges", "4", "3", "--corners",
                 "100,100", "100,100", "100,100", "100,100", "--out", str(out)])
    err = capsys.readouterr().err
    assert code != 0 and not out.exists()
    assert err.strip().count("\n") == 0 and err.strip()


def test_site_fit_bad_corner_text(tmp_path, capsys):
    image, _ = floor_image(tmp_path)
    code = main(["site", "fit", str(image), "--edges", "4", "3", "--corners", "1,2", "3,4", "5,6", "x",
                 "--out", str(tmp_path / "o.json")])
    assert code == 2 and "x" in capsys.readouterr().err


def test_fetch_sam_prints_the_path(tmp_path, monkeypatch, capsys):
    import physical_sandbox.objects as objects
    monkeypatch.setattr(objects, "fetch_sam_weights", lambda dest=None: tmp_path / "w.pt")
    assert main(["fetch-sam", "--dest", str(tmp_path / "w.pt")]) == 0
    assert str(tmp_path / "w.pt") in capsys.readouterr().out


def test_mcp_without_the_extra_is_a_clear_error(monkeypatch, capsys):
    import sys
    monkeypatch.setitem(sys.modules, "physical_sandbox.mcp.server", None)  # makes the import fail
    assert main(["mcp"]) == 1
    assert "mcp" in capsys.readouterr().err


def test_mcp_runs_the_server(monkeypatch):
    server = pytest.importorskip("physical_sandbox.mcp.server")
    called = []
    monkeypatch.setattr(server, "main", lambda: called.append(1))
    assert main(["mcp"]) == 0 and called == [1]


def test_lerobot_episodes_use_the_episodes_own_recording(tmp_path, monkeypatch):
    """Real data when present: no --video, so A's own `side` recording is the picture."""
    DATASET = Path(os.environ.get(
        "PS_LEROBOT_DATASET",
        "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/"
        "f641879e22172be7e8161d5e6c1503c2d2feb657")).expanduser()
    SO101 = Path(os.environ.get(
        "PS_SO101_URDF", "~/work/physical-sandbox-proto/robots/so101/so101.urdf")).expanduser()
    if not ((DATASET / "meta" / "info.json").exists() and SO101.exists()):
        pytest.skip("LeRobot dataset or SO-101 URDF not found (set PS_LEROBOT_DATASET, PS_SO101_URDF)")
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    site = Site(id="bench", cameras={"side": look_at((-0.6, 0.5, 0.5), target=(0, 0, 0.1), size=(640, 480), f=500.0)})
    save_site(site, tmp_path / "site.json")
    placed = PlacedMachine(load_machine(SO101), str(SO101), 0.0, 0.0, 0.0)
    save_scene(Scene(site=site, machines={"so101": placed}), tmp_path / "scene.json")
    out = tmp_path / "out"
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "side", "--a", f"lerobot:{DATASET}#0", "--b", f"lerobot:{DATASET}#1", "--out", str(out)])
    assert code == 0
    assert (out / "replay.mp4").stat().st_size > 0 and (out / "gaps.json").is_file()
