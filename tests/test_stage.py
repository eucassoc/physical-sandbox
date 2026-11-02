"""The stage: one browser screen over a work folder (floor step), served on 127.0.0.1."""
import json
import shutil
import threading
import time
import urllib.request

import cv2
import numpy as np
import pytest

from physical_sandbox import push, stage
from physical_sandbox.cli import main
from physical_sandbox.errors import InputError
from physical_sandbox.motion import motion_from_numpy, save_record
from physical_sandbox.scene import load_scene
from physical_sandbox.site import load_site
from physical_sandbox.stage import Stage, StageServer

import test_fit_screen
from mini_lerobot import make_dataset
from synthetic import URDF
from test_fit_screen import CORNERS, H, W, get, status_of


def call(server, name, args):
    """The tool's answer as the page reads it: the JSON the stage returned, or {"error": ...}."""
    out = test_fit_screen.call(server, name, args)
    return json.loads(out["text"]) if "text" in out else out


@pytest.fixture
def still(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "still.png"
    cv2.imwrite(str(path), np.full((H, W, 3), 100, np.uint8))
    return path


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def served(still, tmp_path, home):
    servers = []

    def start():
        s = StageServer(Stage(picture=still, work=tmp_path / "w"))
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        return s
    yield start
    for s in servers:
        s.shutdown()


def test_the_work_folder_is_next_to_the_picture_named_after_it(tmp_path):
    assert stage.work_dir_for(tmp_path / "desk.jpg") == tmp_path / "desk"


def test_first_starts_from_a4_and_no_login_without_a_token(served, home):
    s = served()
    first = json.loads(get(s.url + "first")[2])
    assert first["floor"]["edgesM"] == [0.21, 0.297] and first["floor"]["decided"] is False
    assert first["login"] == {"signedIn": False, "handle": None} and first["machine"] is None
    assert len(first["floor"]["corners"]) == 4 and first["floor"]["measured"] is False
    assert (first["width"], first["height"], first["camera"]) == (W, H, "camera")
    assert first["motionA"] is None and first["motionB"] is None and first["gripThreshold"] is None
    assert first["parts"] >= 1 and first["workDir"].endswith("w")


def test_decide_writes_the_site_and_first_then_continues_from_it(served, still, tmp_path):
    s = served()
    out = call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0], "measured": True})
    assert out["ok"] and load_site(tmp_path / "w" / "site.json").confirmed_by_person
    site = load_site(tmp_path / "w" / "site.json")
    assert site.history[-1]["what"] == "floor fixed in the browser (psbx stage) and confirmed by a person"
    assert site.history[-1]["image"] == "still.png"
    stage_json = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert stage_json == {"format": "physical-sandbox.stage/1", "picture": str(still.resolve())}
    again = StageServer(Stage(picture=still, work=tmp_path / "w"))
    again.close()
    first = json.loads(again.first())
    assert first["floor"]["decided"] and first["floor"]["measured"] and len(first["floor"]["corners"]) == 4
    assert first["floor"]["edgesM"] == [4.0, 3.0]


def test_a_bad_decide_writes_nothing(served, tmp_path):
    s = served()
    out = call(s, "ps_site_decide", {"corners": CORNERS[:3], "edgesM": [4.0, 3.0]})
    assert out["ok"] is False and "bad-input" in out["reason"]
    assert not (tmp_path / "w" / "site.json").exists() and not (tmp_path / "w" / "stage.json").exists()


def test_the_on_decide_hook_is_called_with_the_site_and_its_path(still, tmp_path, home):
    seen = []
    s = StageServer(Stage(picture=still, work=tmp_path / "w"), on_decide=lambda site, path: seen.append(path))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0]})
    finally:
        s.shutdown()
    assert seen == [tmp_path / "w" / "site.json"]


def test_the_photo_and_the_floor_grid_are_served_to_the_page(served, still):
    s = served()
    first = json.loads(get(s.url + "first")[2])
    part = call(s, "ps_spike_image", {"imagePath": first["imagePath"], "part": 0})
    assert part["part"] == 0 and part["data"]
    grid = call(s, "ps_floor_solve", {"width": W, "height": H, "corners": CORNERS, "edgesM": [4.0, 3.0]})
    assert grid["ok"] and grid["lines"]


def test_a_tool_the_stage_does_not_have_is_an_error(served):
    s = served()
    out = call(s, "ps_record_new", {})
    assert "the stage has no tool" in out["error"]


def test_writes_only_as_json_posts_to_the_call_address(served):
    s = served()
    assert status_of(s.url + "call", "POST", b"{}", {"Content-Type": "text/plain"}) == 404
    assert status_of(s.url + "first") == 200 and status_of(s.url.replace(s.token, "wrong") + "first") == 404
    with urllib.request.urlopen(s.url, timeout=5) as r:
        assert r.headers["Cache-Control"] == "no-store"


def test_a_folder_fixed_on_another_picture_is_refused(served, still, tmp_path):
    s = served()
    call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0]})
    other = still.with_name("other.png")
    shutil.copy(still, other)
    with pytest.raises(InputError, match="other.png") as e:
        Stage(picture=other, work=tmp_path / "w")
    assert "still.png" in str(e.value) and "--dir" in str(e.value)


def test_a_work_folder_alone_finds_its_picture(served, still, tmp_path):
    s = served()
    call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0]})
    assert Stage(work=tmp_path / "w").picture == still.resolve()
    with pytest.raises(InputError, match="no picture"):
        Stage(work=tmp_path / "empty")


def test_a_work_folder_alone_finds_the_picture_of_its_scene_file(still, tmp_path):
    work = tmp_path / "w"
    work.mkdir()
    (work / "scene.json").write_text(json.dumps({"picture": "../still.png"}))
    assert Stage(work=work).picture == (work / "../still.png").resolve()


def test_a_picture_that_is_not_a_still_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()
    with pytest.raises(InputError, match="not a still"):
        Stage(picture=video, work=tmp_path / "w")


def test_neither_picture_nor_work_is_refused():
    with pytest.raises(InputError):
        Stage()


def test_the_stage_html_has_the_four_sections_and_no_markers(served):
    html = get(served().url)[2]
    assert all(f'id="{s}"' in html for s in ("floor", "machine", "motion", "compare"))
    # three's own source mentions "__THREE__": only the marker comment must be gone
    assert "/*__LIB:" not in html and "/*__THREE__*/" not in html and "__EXT_APPS_BUNDLE__" not in html


# ---- the command ----

class FakeServer:
    made = []

    def __init__(self, stage_, port=0, on_decide=None):
        self.stage, self.port, self.on_decide = stage_, port, on_decide
        self.url = "http://127.0.0.1:1/t/"
        FakeServer.made.append(self)

    def serve_forever(self):
        raise KeyboardInterrupt

    def close(self):
        self.closed = True


def test_stage_command_serves_until_ctrl_c_and_prints_the_address(still, tmp_path, monkeypatch, capsys, home):
    import webbrowser
    opened = []
    monkeypatch.setattr(stage, "StageServer", FakeServer)
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    FakeServer.made.clear()
    assert main(["stage", str(still), "--dir", str(tmp_path / "w"), "--camera", "side", "--port", "8123"]) == 0
    (server,) = FakeServer.made
    assert server.port == 8123 and server.stage.camera == "side" and server.stage.work == tmp_path / "w"
    assert opened == [server.url] and server.closed
    assert "Stage at http://127.0.0.1:" in capsys.readouterr().out
    assert main(["stage", str(still), "--dir", str(tmp_path / "w"), "--no-browser"]) == 0
    assert opened == [server.url]  # --no-browser: nothing more opened


def test_stage_command_defaults_the_work_folder_next_to_the_picture_and_takes_a_folder(still, tmp_path, monkeypatch,
                                                                                         capsys, home):
    monkeypatch.setattr(stage, "StageServer", FakeServer)
    FakeServer.made.clear()
    main(["stage", str(still), "--no-browser"])
    assert FakeServer.made[-1].stage.work == tmp_path / "still"
    work = tmp_path / "w"
    work.mkdir()
    (work / "stage.json").write_text(json.dumps({"format": "physical-sandbox.stage/1", "picture": str(still)}))
    main(["stage", str(work), "--no-browser"])
    assert FakeServer.made[-1].stage.work == work and FakeServer.made[-1].stage.picture == still.resolve()


def test_stage_command_prints_what_was_written_on_each_decision(still, tmp_path, monkeypatch, capsys, home):
    monkeypatch.setattr(stage, "StageServer", FakeServer)
    FakeServer.made.clear()
    main(["stage", str(still), "--dir", str(tmp_path / "w"), "--no-browser"])
    real = StageServer(Stage(picture=still, work=tmp_path / "w"))
    real.close()
    call_ = FakeServer.made[-1].on_decide
    capsys.readouterr()
    real.on_decide = call_
    real.call("ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0]})
    assert f"wrote {tmp_path / 'w' / 'site.json'}" in capsys.readouterr().out


def test_stage_command_names_a_missing_picture(tmp_path, home, capsys):
    assert main(["stage", str(tmp_path / "nothing.jpg"), "--no-browser"]) == 2
    assert "nothing.jpg" in capsys.readouterr().err


@pytest.fixture
def decided(served):
    s = served()
    assert call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0], "measured": True})["ok"]
    return s


@pytest.fixture
def cache_mini(still, tmp_path):
    folder = tmp_path / "cache" / "machines" / "mini"
    folder.mkdir(parents=True)
    (folder / "mini.urdf").write_text(URDF)
    return str(folder / "mini.urdf")


def test_machines_list_finds_the_cache_the_catalog_cache_and_the_dirs(decided, cache_mini, tmp_path):
    out = call(decided, "ps_machines_list", {})
    assert {m["id"] for m in out["machines"]} == {"mini"} and out["machines"][0]["source"] == "cache"
    pulled = tmp_path / "cache" / "catalog" / "t" / "arm" / "2"
    pulled.mkdir(parents=True)
    (pulled / "arm.urdf").write_text(URDF)
    mine = tmp_path / "mine"
    (mine / "own").mkdir(parents=True)
    (mine / "own" / "own.urdf").write_text(URDF)
    decided.stage.machine_dirs = (mine,)
    got = {m["id"]: m for m in call(decided, "ps_machines_list", {})["machines"]}
    assert set(got) == {"mini", "arm", "own"}
    assert got["arm"]["source"] == "catalog" and got["arm"]["label"] == "t/arm:2"
    assert got["own"]["source"] == "dir" and got["own"]["urdf"].endswith("own.urdf")


def test_machine_open_gives_the_zero_pose_the_camera_and_the_floor_centre(decided, cache_mini):
    out = call(decided, "ps_machine_open", {"urdf": cache_mini, "id": "mini"})
    assert len(out["zeroPoses"]) == len(out["model"]["links"]) and len(out["zeroPoses"][0]) == 16
    assert out["placed"] is None and abs(out["center"]["x"] - 2.0) < 1e-6 and abs(out["center"]["y"] - 1.5) < 1e-6
    assert len(out["floor"]) == 4 and set(out["camera"]) == {"intr", "extr"} and len(out["camera"]["extr"]["R"]) == 9


def test_machine_open_refuses_a_urdf_that_was_not_listed(decided, tmp_path):
    stray = tmp_path / "x.urdf"
    stray.write_text(URDF)
    assert "error" in call(decided, "ps_machine_open", {"urdf": str(stray), "id": "x"})
    assert "model" in call(decided, "ps_machine_open", {"urdf": str(stray), "id": "x", "typed": True})
    assert "error" in call(decided, "ps_machine_open", {"urdf": str(tmp_path / "none.urdf"), "id": "x", "typed": True})
    (tmp_path / "x.txt").write_text(URDF)
    assert "error" in call(decided, "ps_machine_open", {"urdf": str(tmp_path / "x.txt"), "id": "x", "typed": True})


def test_machine_open_needs_a_decided_floor(served, cache_mini):
    assert "error" in call(served(), "ps_machine_open", {"urdf": cache_mini, "id": "mini"})


def test_scene_place_writes_the_scene_and_first_shows_it(decided, cache_mini, tmp_path, still):
    out = call(decided, "ps_scene_place", {"id": "mini", "urdf": cache_mini, "x": 1.0, "y": 0.5, "yaw": 0.3})
    assert out["scenePath"].endswith("scene.json") and len(out["base"]) == 16
    scene = load_scene(tmp_path / "w" / "scene.json", load_site(tmp_path / "w" / "site.json"))
    assert scene.machines["mini"].confirmed_by_person and abs(scene.machines["mini"].yaw - 0.3) < 1e-9
    d = json.loads((tmp_path / "w" / "scene.json").read_text())
    assert d["picture"] == str(still.resolve()) and d["records"] == []
    kept = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert kept["picture"] == str(still.resolve()) and kept["machine"] == {"id": "mini", "urdf": cache_mini}
    first = json.loads(StageServer(Stage(work=tmp_path / "w")).first())
    assert first["machine"] == {"id": "mini", "urdf": str(tmp_path / "cache" / "machines" / "mini" / "mini.urdf"),
                                "placed": {"x": 1.0, "y": 0.5, "yaw": 0.3}}
    reopened = call(decided, "ps_machine_open", {"urdf": cache_mini, "id": "mini"})
    assert reopened["placed"] == {"x": 1.0, "y": 0.5, "yaw": 0.3}


def test_scene_place_refuses_numbers_that_are_not_finite(decided, cache_mini, tmp_path):
    for bad in (float("nan"), float("inf"), "a"):
        assert "error" in call(decided, "ps_scene_place", {"id": "mini", "urdf": cache_mini, "x": bad, "y": 0, "yaw": 0})
    assert not (tmp_path / "w" / "scene.json").exists()


def test_scene_place_keeps_the_records_of_the_scene(decided, cache_mini, tmp_path):
    (tmp_path / "w" / "scene.json").write_text(json.dumps(
        {"format": "physical-sandbox.scene/1", "site": "site", "machines": {}, "records": ["a.record.json"]}))
    call(decided, "ps_scene_place", {"id": "mini", "urdf": cache_mini, "x": 1.0, "y": 0.5, "yaw": 0.0})
    assert json.loads((tmp_path / "w" / "scene.json").read_text())["records"] == ["a.record.json"]


def test_catalog_search_and_pull_go_through_the_stage_with_the_token(decided, catalog_server, monkeypatch, home):
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    push._keep(catalog_server.url, "tok-1", 3600, handle="ada")
    items = call(decided, "ps_catalog_search", {"q": "mini"})["items"]
    assert items[0]["ref"] == "t/mini" and items[0]["private"] is False and items[0]["version"] == "1"
    assert items[0]["label"] == "ミニ"
    assert catalog_server.headers[-1]["Authorization"] == "Bearer tok-1"
    out = call(decided, "ps_catalog_pull", {"ref": "t/mini:1"})
    assert out["id"] == "mini" and out["urdf"].endswith("mini.urdf")
    listed = call(decided, "ps_machines_list", {})["machines"]
    assert [m["label"] for m in listed] == ["t/mini:1"]


def test_a_catalog_that_cannot_be_reached_is_an_error_answer_not_a_raise(decided, monkeypatch):
    monkeypatch.setenv("PS_CATALOG_URL", "http://127.0.0.1:9")
    assert "error" in call(decided, "ps_catalog_search", {"q": "x"})
    assert "error" in call(decided, "ps_catalog_pull", {"ref": "t/mini:1"})


def test_first_and_the_machine_list_make_no_request_to_the_catalog(decided, catalog_server, monkeypatch, cache_mini):
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    get(decided.url + "first")
    call(decided, "ps_machines_list", {})
    assert catalog_server.hits == 0


def test_the_stage_html_has_the_machine_controls(served):
    html = get(served().url)[2]
    for ident in ("machinePick", "catalogQ", "urdfPath", "px", "py", "yaw"):
        assert f'id="{ident}"' in html
    assert "floorPointJs" in html and "ps_machine_open" in html


def test_a_typed_urdf_outside_the_listed_roots_can_be_placed_and_reopened_by_a_fresh_stage(
        decided, still, tmp_path, home):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    typed, other = outside / "typed.urdf", outside / "other.urdf"
    typed.write_text(URDF)
    other.write_text(URDF)
    assert "model" in call(decided, "ps_machine_open", {"urdf": str(typed), "id": "typed", "typed": True})
    placed = call(decided, "ps_scene_place", {"id": "typed", "urdf": str(typed), "x": 1.0, "y": 0.5, "yaw": 0.0})
    assert placed["scenePath"].endswith("scene.json") and (tmp_path / "w" / "scene.json").is_file()
    fresh = StageServer(Stage(picture=still, work=tmp_path / "w"))
    threading.Thread(target=fresh.serve_forever, daemon=True).start()
    try:
        assert "model" in call(fresh, "ps_machine_open", {"urdf": str(typed), "id": "typed"})
        assert "error" in call(fresh, "ps_machine_open", {"urdf": str(other), "id": "other"})
    finally:
        fresh.shutdown()


def test_an_unlisted_urdf_that_was_never_opened_cannot_be_placed(
        decided, still, tmp_path, home):
    stray = tmp_path / "elsewhere" / "stray.urdf"
    stray.parent.mkdir()
    stray.write_text(URDF)
    assert "error" in call(decided, "ps_scene_place", {"id": "stray", "urdf": str(stray), "x": 0, "y": 0, "yaw": 0})
    assert not (tmp_path / "w" / "scene.json").exists()


# ---- the motion step ----

@pytest.fixture
def placed(decided, cache_mini):
    out = call(decided, "ps_scene_place", {"id": "mini", "urdf": cache_mini, "x": 0, "y": 0, "yaw": 0})
    assert "scenePath" in out
    return decided


def _record(path, label="demo", machine="mini", site_id="site"):
    q = np.linspace(0, 1, 6).reshape(-1, 1)
    save_record([motion_from_numpy(q, fps=10, joint_names=["turn"], machine=machine, label=label)], site_id, path)
    return path


def test_motions_list_finds_records_in_the_folder_and_lerobot_dirs(placed, tmp_path, monkeypatch):
    _record(tmp_path / "w" / "demo.record.json")
    hub = tmp_path / "hf" / "lerobot" / "hub" / "datasets--lerobot--tiny" / "snapshots" / "abc"
    make_dataset(hub, ["turn"], [np.zeros((4, 1))])
    own = make_dataset(tmp_path / "own", ["turn"], [np.zeros((4, 1)), np.zeros((4, 1))])
    for root in (hub, own):  # the real files say how many episodes they hold
        info = json.loads((root / "meta" / "info.json").read_text())
        info["total_episodes"] = 1 if root == hub else 2
        (root / "meta" / "info.json").write_text(json.dumps(info))
    (tmp_path / "hf" / "lerobot" / "hub" / "datasets--lerobot--empty" / "snapshots" / "x").mkdir(parents=True)
    plain = tmp_path / "hf" / "hub" / "datasets--x--y" / "snapshots" / "abc"  # what huggingface-cli download writes
    make_dataset(plain, ["turn"], [np.zeros((4, 1))])
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    placed.stage.lerobot_dirs = (own, hub)  # a dir that is also under the hub is listed once
    out = call(placed, "ps_motions_list", {})
    assert out["records"] == [{"spec": str(tmp_path / "w" / "demo.record.json"), "label": "demo"}]
    by_dir = {d["dir"]: d for d in out["lerobot"]}
    assert len(out["lerobot"]) == 3 and set(by_dir) == {str(hub), str(own), str(plain)}
    assert by_dir[str(plain)]["name"] == "x/y"
    assert by_dir[str(hub)] == {"dir": str(hub), "name": "lerobot/tiny", "episodes": 1}
    assert by_dir[str(own)]["episodes"] == 2 and by_dir[str(own)]["name"] == "own"


def test_motions_list_names_each_motion_of_a_record_and_skips_what_is_not_one(placed, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "nohf"))
    q = np.linspace(0, 1, 6).reshape(-1, 1)
    save_record([motion_from_numpy(q, fps=10, joint_names=["turn"], machine=m, label=m) for m in ("p1", "p2")],
                "site", tmp_path / "w" / "two.record.json")
    (tmp_path / "w" / "broken.record.json").write_text("{not json")
    out = call(placed, "ps_motions_list", {})
    path = str(tmp_path / "w" / "two.record.json")
    assert out["records"] == [{"spec": path + "#p1", "label": "p1"}, {"spec": path + "#p2", "label": "p2"}]
    assert out["lerobot"] == []


def test_motion_open_gives_one_record_and_keeps_the_spec(placed, tmp_path):
    spec = str(_record(tmp_path / "w" / "demo.record.json"))
    out = call(placed, "ps_motion_open", {"spec": spec, "slot": "a"})
    assert len(out["records"]) == 1 and out["dataParts"] >= 1 and out["note"] is None and out["spec"] == spec
    assert len(out["base"]) == 16 and out["model"]["links"] and out["camera"]["intr"]["width"] == W
    kept = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert kept["a"].endswith("demo.record.json") and kept["machine"]["id"] == "mini"
    part = call(placed, "ps_compare_part", {"key": out["dataKey"], "part": 0})
    data = json.loads("".join(call(placed, "ps_compare_part", {"key": out["dataKey"], "part": i})["data"]
                              for i in range(out["dataParts"])))
    assert part["part"] == 0 and len(data["poses"]) == 1 and len(data["poses"][0]) == out["frames"]
    first = json.loads(StageServer(Stage(work=tmp_path / "w")).first())
    assert first["motionA"] == spec and first["motionB"] is None


def test_motion_open_reads_a_record_for_another_machine_over_the_placed_one(placed, tmp_path):
    spec = str(_record(tmp_path / "w" / "other.record.json", machine="elsewhere"))
    out = call(placed, "ps_motion_open", {"spec": spec, "slot": "b"})
    assert "elsewhere" in out["note"] and "mini" in out["note"]
    assert json.loads((tmp_path / "w" / "stage.json").read_text())["b"] == spec


def test_motion_open_refuses_without_a_placed_machine_and_for_a_bad_spec(decided, placed, tmp_path):
    spec = str(_record(tmp_path / "w" / "demo.record.json"))
    assert "error" in call(placed, "ps_motion_open", {"spec": str(tmp_path / "w" / "none.record.json"), "slot": "a"})
    assert "error" in call(placed, "ps_motion_open", {"spec": spec, "slot": "c"})
    (tmp_path / "w" / "scene.json").unlink()
    out = call(placed, "ps_motion_open", {"spec": spec, "slot": "a"})
    assert "ロボットを置いてから" in out["error"]


def test_motion_open_refuses_a_record_whose_joints_are_not_the_machines(placed, tmp_path):
    q = np.linspace(0, 1, 6).reshape(-1, 1)
    save_record([motion_from_numpy(q, fps=10, joint_names=["other"], machine="elsewhere", label="x")],
                "site", tmp_path / "w" / "foreign.record.json")
    spec = str(tmp_path / "w" / "foreign.record.json")
    out = call(placed, "ps_motion_open", {"spec": spec, "slot": "a"})
    assert "記録の関節がロボットと合いません" in out["error"] and "other" in out["error"] and "turn" in out["error"]
    assert "a" not in json.loads((tmp_path / "w" / "stage.json").read_text())
    assert "error" in call(placed, "ps_export", {"what": "replay", "spec": spec})


def _export(server, spec):
    job = call(server, "ps_export", {"what": "replay", "spec": spec})["job"]
    st = {}
    for _ in range(400):
        st = call(server, "ps_export_status", {"job": job})
        if st["done"]:
            break
        time.sleep(0.05)
    return job, st


def test_export_replay_writes_the_video_and_reports_progress(placed, tmp_path):
    spec = str(_record(tmp_path / "w" / "demo.record.json"))
    job, st = _export(placed, spec)
    assert len(job) == 8 and st["error"] is None and st["done"]
    assert st["paths"] == [str(tmp_path / "w" / "replay-demo.mp4")]
    assert (tmp_path / "w" / "replay-demo.mp4").stat().st_size > 0 and st["progress"] == 1


def test_export_names_a_label_with_odd_characters_safely(placed, tmp_path):
    spec = str(_record(tmp_path / "w" / "demo.record.json", label="ロボ a/b"))
    _, st = _export(placed, spec)
    assert st["error"] is None and st["paths"] == [str(tmp_path / "w" / "replay-___a_b.mp4")]


def test_a_second_export_while_one_runs_is_refused(placed, tmp_path, monkeypatch):
    gate = threading.Event()
    real = stage.render_overlay

    def slow(*args, **kw):
        gate.wait(5)
        return real(*args, **kw)
    monkeypatch.setattr(stage, "render_overlay", slow)
    spec = str(_record(tmp_path / "w" / "demo.record.json"))
    first = call(placed, "ps_export", {"what": "replay", "spec": spec})
    try:
        assert call(placed, "ps_export", {"what": "replay", "spec": spec}) == {"error": "書き出し中です"}
        assert call(placed, "ps_export_status", {"job": first["job"]})["done"] is False
    finally:
        gate.set()
    for _ in range(400):
        if call(placed, "ps_export_status", {"job": first["job"]})["done"]:
            break
        time.sleep(0.05)
    assert call(placed, "ps_export", {"what": "replay", "spec": spec}).get("job")


def test_an_export_that_fails_reports_one_line_and_an_unknown_job_is_an_error(placed, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no codec\nsecond line")
    monkeypatch.setattr(stage, "render_overlay", boom)
    _, st = _export(placed, str(_record(tmp_path / "w" / "demo.record.json")))
    assert st["done"] and st["error"] == "no codec" and st["paths"] == []
    assert "error" in call(placed, "ps_export_status", {"job": "nosuchjob"})
    assert "error" in call(placed, "ps_export", {"what": "gif", "spec": "x"})


def test_the_stage_html_has_the_motion_controls(served):
    html = get(served().url)[2]
    for ident in ("motionPick", "motionSpec", "episode", "tphoto", "tgl", "tview", "tplay", "tslider", "tclock", "exportBtn"):
        assert f'id="{ident}"' in html
    for name in ("ps_motions_list", "ps_motion_open", "ps_compare_part", "ps_export", "ps_export_status"):
        assert name in html
