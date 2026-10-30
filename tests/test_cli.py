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
    assert list(rows[0]) == ["item", "value", "unit", "at_s", "note"]
    by = {r["item"]: r for r in rows}
    # the site's quality heads the table, and what is not told yet is named with its reason
    assert [r["item"] for r in rows[:4]] == ["site", "residual.camera", "dimensions", "confirmed_by_person"]
    assert by["site"]["value"] == "yard" and by["residual.camera"]["value"] == "0.7"
    assert by["dimensions"]["value"] == "measured" and by["dimensions"]["note"] == ""
    assert by["confirmed_by_person"]["value"] == "true"
    for name in ("clearance", "reach", "world"):
        assert by[f"unknown.{name}"]["value"] == "" and by[f"unknown.{name}"]["note"]
    assert by["unknown.grasps"]["note"]  # the test machine has no gripper
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
    rows = {r["item"]: r for r in csv.DictReader(paths["gaps.csv"].open(encoding="utf-8"))}
    assert rows["dimensions"]["value"] == "estimated" and rows["dimensions"]["note"] == "寸法は推定（未確認）"
    assert rows["confirmed_by_person"]["value"] == "false"
    assert cv2.imread(str(paths["gaps.png"])) is not None


def test_label_selects_a_motion_in_a_record(files):
    tmp_path, site, still = files
    save_record([motion(0.5, "first"), motion(1.0, "second")], "yard", tmp_path / "two.record.json")
    out = tmp_path / "out"
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "camera", "--a", str(tmp_path / "two.record.json") + "#second",
                 "--b", str(tmp_path / "b.record.json"), "--out", str(out), "--video", str(still)])
    assert code == 0
    assert json.loads((out / "gaps.json").read_text())["labels"]["a"] == "second"


def args_for(tmp_path, still, a, b, out="o", scene="scene.json"):
    return ["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / scene),
            "--camera", "camera", "--a", a, "--b", b, "--out", str(tmp_path / out), "--video", str(still)]


def test_unknown_label_is_a_bad_input(files, capsys):
    tmp_path, _, still = files
    code = main(args_for(tmp_path, still, str(tmp_path / "a.record.json") + "#nope", str(tmp_path / "b.record.json")))
    err = capsys.readouterr().err
    assert code == 2
    assert "nope" in err and "slow" in err and err.strip().count("\n") == 0


@pytest.mark.parametrize("spec", ["lerobot:nowhere", "lerobot:nowhere#x", "lerobot:nowhere#3", "gone.record.json"])
def test_unknown_motion_format_is_a_bad_input(files, capsys, spec):
    tmp_path, _, still = files
    code = main(args_for(tmp_path, still, spec, str(tmp_path / "b.record.json")))
    err = capsys.readouterr().err
    assert code == 2 and err.startswith("psbx: ") and err.strip().count("\n") == 0
    assert "nowhere" in err or "gone.record.json" in err


def test_a_file_that_is_no_recording_is_a_bad_input(files, capsys):
    tmp_path, _, still = files
    code = main(args_for(tmp_path, still, str(tmp_path / "site.json"), str(tmp_path / "b.record.json")))
    assert code == 2 and "site.json" in capsys.readouterr().err


def test_a_motion_of_a_machine_that_is_not_placed_is_a_bad_input(files, capsys):
    tmp_path, _, still = files
    other = motion_from_numpy(np.zeros((3, 1)), fps=10.0, joint_names=["turn"], machine="other", label="x")
    save_record([other], "yard", tmp_path / "x.record.json")
    code = main(args_for(tmp_path, still, str(tmp_path / "x.record.json"), str(tmp_path / "b.record.json")))
    err = capsys.readouterr().err
    assert code == 2 and "other" in err and "blocks" in err


def test_a_wrong_video_leaves_no_partial_report(files, capsys):
    tmp_path, _, _ = files
    wide = tmp_path / "wide.png"
    cv2.imwrite(str(wide), np.full((100, 300, 3), 128, np.uint8))  # not the site camera's aspect ratio
    for bad in (wide, tmp_path / "missing.mp4"):
        code = main(args_for(tmp_path, bad, str(tmp_path / "a.record.json"), str(tmp_path / "b.record.json")))
        assert code == 2 and "--video" in capsys.readouterr().err
        assert not (tmp_path / "o").exists()


def test_a_runtime_fault_is_exit_1_with_its_type(files, capsys, monkeypatch):
    tmp_path, _, still = files
    import physical_sandbox.compare as cmp

    def boom(*a, **k):
        raise ZeroDivisionError("kaboom")
    monkeypatch.setattr(cmp, "compare", boom)
    code = main(args_for(tmp_path, still, str(tmp_path / "a.record.json"), str(tmp_path / "b.record.json")))
    err = capsys.readouterr().err
    assert code == 1 and "ZeroDivisionError: kaboom" in err and "Traceback" not in err
    monkeypatch.setenv("PSBX_DEBUG", "1")
    code = main(args_for(tmp_path, still, str(tmp_path / "a.record.json"), str(tmp_path / "b.record.json")))
    assert code == 1 and "Traceback" in capsys.readouterr().err


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
    assert code == 2 and not out.exists()
    assert err.strip().count("\n") == 0 and "no camera fits these corners (flat)" in err


def test_site_fit_bad_corner_text(tmp_path, capsys):
    image, _ = floor_image(tmp_path)
    code = main(["site", "fit", str(image), "--edges", "4", "3", "--corners", "1,2", "3,4", "5,6", "x",
                 "--out", str(tmp_path / "o.json")])
    assert code == 2 and "x" in capsys.readouterr().err


@pytest.mark.parametrize("edges", [["nan", "3"], ["inf", "3"], ["-1", "3"], ["0", "3"]])
def test_site_fit_edges_must_be_finite_and_positive(tmp_path, capsys, edges):
    image, px = floor_image(tmp_path)
    code = main(["site", "fit", str(image), "--edges", *edges, "--corners",
                 *[f"{u},{v}" for u, v in px], "--out", str(tmp_path / "o.json")])
    assert code == 2 and not (tmp_path / "o.json").exists()


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


@pytest.fixture
def fake_lerobot(files, monkeypatch):
    """A dataset folder that only has meta/info.json, with the episode reader and the video lookup replaced."""
    tmp_path, site, still = files
    data = tmp_path / "ds"
    (data / "meta").mkdir(parents=True)
    (data / "meta" / "info.json").write_text("{}")
    lengths = {4: 0.6, 5: 1.0}  # episode -> seconds of motion
    import physical_sandbox.lerobot as lerobot
    import physical_sandbox.motion as motion_mod
    monkeypatch.setattr(motion_mod, "motion_from_lerobot",
                        lambda d, ep, *, machine, machine_name, **k: motion(1.0, f"episode {ep}", fps=10.0,
                                                                          n=int(lengths[ep] * 10) + 1))
    spans = {}
    monkeypatch.setattr(lerobot, "lerobot_video", lambda d, ep, cam: (spans["video"], *spans["span"]))
    return tmp_path, data, spans


def test_lerobot_replay_starts_where_the_episode_starts_and_stops_where_it_ends(fake_lerobot, monkeypatch, capsys):
    tmp_path, data, spans = fake_lerobot
    # a recording where frame k is grey 4k: a frame shown at the wrong time is a visibly different grey
    video = tmp_path / "ds.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for k in range(60):
        w.write(np.full((H, W, 3), 4 * k, np.uint8))
    w.release()
    spans["video"], spans["span"] = video, (1.0, 1.5)  # episode 4 is the second second of the file, 0.5 s long
    out = tmp_path / "out"
    code = main(["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
                 "--camera", "camera", "--a", f"lerobot:{data}#4", "--b", f"lerobot:{data}#5", "--out", str(out)])
    assert code == 0
    cap = cv2.VideoCapture(str(out / "replay.mp4"))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    assert len(frames) == 15  # 0.5 s at 30 fps, not B's 1.0 s
    assert abs(int(frames[0][2, 2, 0]) - 120) < 16  # file time 1.0 s = frame 30 (grey 120); start 0 would be grey 0
    assert abs(int(frames[-1][2, 2, 0]) - 4 * 43) < 16  # file time 1.0 + 14/30 s = frame 44
    assert "replay cut at 0.50 s" in capsys.readouterr().out


def test_lerobot_start_and_cap_are_passed_to_the_renderer(fake_lerobot, monkeypatch, capsys):
    tmp_path, data, spans = fake_lerobot
    import physical_sandbox.render as render
    calls = []

    def fake_render(scene, camera, source, motions, out, **kw):
        calls.append(kw)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return out
    monkeypatch.setattr(render, "render_overlay", fake_render)
    still = tmp_path / "grey.png"
    spans["video"], spans["span"] = still, (12.0, 12.6)
    base = ["compare", "--site", str(tmp_path / "site.json"), "--scene", str(tmp_path / "scene.json"),
            "--camera", "camera", "--out", str(tmp_path / "out")]
    # A (episode 4, 0.6 s) is as long as its span and B (episode 5) is longer: cut, and say so
    assert main(base + ["--a", f"lerobot:{data}#4", "--b", f"lerobot:{data}#5"]) == 0
    assert calls[-1]["start_s"] == 12.0 and calls[-1]["duration"] == pytest.approx(0.6)
    assert "replay cut at 0.60 s" in capsys.readouterr().out
    # the longer motion is A itself (its picture is shorter than A): cut too, and it names A
    assert main(base + ["--a", f"lerobot:{data}#5", "--b", f"lerobot:{data}#4"]) == 0
    assert "(A runs 1.00 s)" in capsys.readouterr().out
    # nothing runs past the episode's span: no cut message
    spans["span"] = (12.0, 13.0)
    assert main(base + ["--a", f"lerobot:{data}#4", "--b", f"lerobot:{data}#5"]) == 0
    assert "cut" not in capsys.readouterr().out
    # an explicit --video is the person's own picture: from its start, no cap
    assert main(base + ["--a", f"lerobot:{data}#4", "--b", f"lerobot:{data}#5", "--video", str(still)]) == 0
    assert calls[-1]["start_s"] == 0.0 and calls[-1]["duration"] is None
