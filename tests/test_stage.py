"""The stage: one browser screen over a work folder (floor, machine, motion, compare, export), served on 127.0.0.1."""
import json
import shutil
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from physical_sandbox import push, stage, unsent
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


def test_deciding_the_floor_again_keeps_what_stage_json_holds(served, cache_mini, still):
    s = served()
    call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0]})
    assert "scenePath" in call(s, "ps_scene_place", {"id": "mini", "urdf": cache_mini, "x": 0, "y": 0, "yaw": 0})
    record = _record(s.stage.work / "demo.record.json")
    call(s, "ps_motion_open", {"slot": "a", "spec": str(record)})
    assert call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0]})["ok"]
    first = json.loads(s.first())
    assert first["machine"]["id"] == "mini" and first["motionA"] == str(record)
    assert json.loads(s.stage.stage_path.read_text())["picture"] == str(still.resolve())


def test_a_folder_decided_with_another_camera_keeps_it_unless_one_is_given(still, tmp_path, home, cache_mini):
    work = tmp_path / "w"
    s = StageServer(Stage(picture=still, work=work, camera="side"))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    try:
        assert call(s, "ps_site_decide", {"corners": CORNERS, "edgesM": [4.0, 3.0]})["ok"]
        again = StageServer(Stage(picture=still, work=work))
        threading.Thread(target=again.serve_forever, daemon=True).start()
        try:
            assert json.loads(again.first())["camera"] == "side"
            opened = call(again, "ps_machine_open", {"id": "mini", "urdf": cache_mini, "typed": True})
            assert opened["id"] == "mini" and "error" not in opened
        finally:
            again.shutdown()
        assert Stage(picture=still, work=work, camera="other").camera == "other"
    finally:
        s.shutdown()


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
        r.read()  # the whole page: the test itself must not reset the connection mid-answer


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


def _node_checks(html: str, tmp_path) -> None:
    """``node --check`` of the last module script of an assembled page (skipped without node)."""
    import re
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = re.findall(r'<script type="module">(.*?)</script>', html, re.S)[-1]
    (tmp_path / "page.mjs").write_text(script)
    done = subprocess.run([node, "--check", str(tmp_path / "page.mjs")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_stage_pages_script_is_valid_javascript(tmp_path):
    _node_checks(stage.fit_screen.widget_html("stage.html", stage.LOCAL_BRIDGE, three=True), tmp_path)


def test_the_floor_screens_script_is_valid_javascript(tmp_path):
    from physical_sandbox.mcp import server
    _node_checks(server._widget("fit-floor.html"), tmp_path)


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
    assert ("GET", "/api/catalog/search?q=mini&kind=machine") in catalog_server.calls  # machines only
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


def test_motion_open_refuses_a_dataset_of_another_robot_in_the_pages_words(placed, tmp_path):
    other = make_dataset(tmp_path / "other", ["right_joint1", "right_gripper"], [np.zeros((4, 2))])
    out = call(placed, "ps_motion_open", {"spec": f"lerobot:{other}#0", "slot": "b"})
    assert "データセットの関節がロボットと合いません" in out["error"]
    assert "right_joint1" in out["error"] and "turn" in out["error"]
    assert "b" not in json.loads((tmp_path / "w" / "stage.json").read_text())


# ---- PS.com records in the motion slot ----

def _seed_record(fake, tmp_path, handle, name, version, frames, visibility):
    """A record of ``frames`` frames for machine ``so101`` (not the placed ``mini``), stored in the fake catalog."""
    path = tmp_path / f"{handle}-{name}-{version}.record.json"
    q = np.linspace(0, 1, frames).reshape(-1, 1)
    save_record([motion_from_numpy(q, fps=10, joint_names=["turn"], machine="so101", label=name)], "site", path)
    fake.put_record(handle, name, version, path.read_bytes(), visibility)


def _seconds(out):
    """How long the opened motion lasts: which version was read (the seeded ones differ in length)."""
    return out["figures"]["duration_s"]["a"]


@pytest.fixture
def ps_com(placed, catalog_server, monkeypatch, tmp_path):
    """Ada's two records (dance in versions 1 and 2, wave) and bob's public walk; signed in as ada."""
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    _seed_record(catalog_server, tmp_path, "ada", "dance", "1", 6, "private")
    _seed_record(catalog_server, tmp_path, "ada", "dance", "2", 8, "private")
    _seed_record(catalog_server, tmp_path, "ada", "wave", "1", 5, "public")
    _seed_record(catalog_server, tmp_path, "bob", "walk", "1", 7, "public")
    push._keep(catalog_server.url, "tok-1", 3600, handle="ada")
    return catalog_server


def test_motions_list_carries_the_signed_in_persons_ps_com_records(ps_com, placed):
    out = call(placed, "ps_motions_list", {})
    assert sorted(out["catalog"], key=lambda r: r["spec"]) == [
        {"spec": "ps:ada/dance:2", "label": "ada/dance:2", "own": True},
        {"spec": "ps:ada/wave:1", "label": "ada/wave:1", "own": True}]
    assert ("GET", "/api/catalog/search?kind=record&owner=me&limit=500") in ps_com.calls  # the person's own only
    assert ps_com.headers[-1]["Authorization"] == "Bearer tok-1"


def test_the_own_records_are_listed_under_many_newer_public_records_of_others(ps_com, placed, tmp_path):
    for i in range(60):  # more than a search answers by default, all newer than ada's
        _seed_record(ps_com, tmp_path, "bob", f"run-{i}", "1", 4, "public")
    specs = sorted(r["spec"] for r in call(placed, "ps_motions_list", {})["catalog"])
    assert specs == ["ps:ada/dance:2", "ps:ada/wave:1"]


def test_a_server_that_does_not_answer_holds_the_list_and_the_open_only_briefly(ps_com, placed, monkeypatch):
    import socket

    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:1", "slot": "a"})  # cached while the server answers
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(8)  # connections wait in the backlog: nobody ever answers
    url = f"http://127.0.0.1:{silent.getsockname()[1]}"
    monkeypatch.setattr(stage, "_QUICK_S", 0.5)
    monkeypatch.setenv("PS_CATALOG_URL", url)
    push._keep(url, "tok-1", 3600, handle="ada")
    try:
        started = time.monotonic()
        assert call(placed, "ps_motions_list", {})["catalog"] == []
        out = call(placed, "ps_motion_open", {"spec": "ps:ada/dance:1", "slot": "a"})
        assert time.monotonic() - started < 5
        assert out["aOpened"] == {"ref": "ada/dance", "version": "1", "latest": "1", "own": True}
    finally:
        silent.close()


def test_motions_list_has_no_ps_com_records_signed_out_or_offline(ps_com, placed, monkeypatch):
    push._forget(ps_com.url)
    before = len(ps_com.calls)
    out = call(placed, "ps_motions_list", {})
    assert out["catalog"] == [] and "error" not in out and len(ps_com.calls) == before  # no request at all
    monkeypatch.setenv("PS_CATALOG_URL", "http://127.0.0.1:9")
    push._keep("http://127.0.0.1:9", "tok-1", 3600, handle="ada")
    out = call(placed, "ps_motions_list", {})
    assert out["catalog"] == [] and "error" not in out


def test_motion_open_reads_a_ps_com_record_into_a_and_keeps_what_was_opened(ps_com, placed, tmp_path):
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    assert _seconds(out) == pytest.approx(0.7) and out["spec"] == "ps:ada/dance:2"
    assert out["records"][0]["label"] == "dance:2"
    assert "so101" in out["note"] and "mini" in out["note"]  # one device read onto the placed machine
    opened = {"ref": "ada/dance", "version": "2", "latest": "2", "own": True}
    assert out["aOpened"] == opened
    kept = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert kept["a"] == "ps:ada/dance:2" and kept["aOpened"] == opened
    first = json.loads(placed.first())
    assert first["motionA"] == "ps:ada/dance:2" and first["aOpened"] == opened
    assert (tmp_path / "cache" / "catalog" / "ada" / "dance" / "2" / "record.json").is_file()


def test_an_older_version_says_the_latest_and_another_persons_record_is_not_own(ps_com, placed):
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/dance:1", "slot": "a"})
    assert _seconds(out) == pytest.approx(0.5)
    assert out["aOpened"] == {"ref": "ada/dance", "version": "1", "latest": "2", "own": True}
    out = call(placed, "ps_motion_open", {"spec": "ps:bob/walk:1", "slot": "a"})
    assert _seconds(out) == pytest.approx(0.6)
    assert out["aOpened"] == {"ref": "bob/walk", "version": "1", "latest": "1", "own": False}


def test_a_ps_com_spec_without_a_version_is_kept_with_the_latest(ps_com, placed, tmp_path):
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/dance", "slot": "a"})
    assert _seconds(out) == pytest.approx(0.7) and out["spec"] == "ps:ada/dance:2" and out["aOpened"]["version"] == "2"
    assert json.loads((tmp_path / "w" / "stage.json").read_text())["a"] == "ps:ada/dance:2"


def test_signed_out_a_public_record_opens_but_is_not_own(ps_com, placed):
    push._forget(ps_com.url)
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/wave:1", "slot": "a"})
    assert out["aOpened"] == {"ref": "ada/wave", "version": "1", "latest": "1", "own": False}


def test_opening_a_file_into_a_drops_what_was_opened_from_ps_com(ps_com, placed, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    spec = str(_record(tmp_path / "w" / "demo.record.json"))
    out = call(placed, "ps_motion_open", {"spec": spec, "slot": "a"})
    assert out["aOpened"] is None
    assert "aOpened" not in json.loads((tmp_path / "w" / "stage.json").read_text())
    assert json.loads(placed.first())["aOpened"] is None


def test_b_never_writes_what_a_opened(ps_com, placed, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    spec = str(_record(tmp_path / "w" / "demo.record.json"))
    call(placed, "ps_motion_open", {"spec": spec, "slot": "b"})
    assert json.loads((tmp_path / "w" / "stage.json").read_text())["aOpened"]["ref"] == "ada/dance"


def test_a_ps_com_record_in_a_compares_with_a_file_and_keeps_what_was_opened(ps_com, placed, tmp_path):
    spec = str(_record(tmp_path / "w" / "demo.record.json", machine="so101"))
    out = call(placed, "ps_compare", {"a": "ps:ada/dance:1", "b": spec})
    assert "error" not in out and out["records"][0]["label"] == "dance:1"
    kept = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert kept["a"] == "ps:ada/dance:1" and kept["aOpened"]["version"] == "1"
    call(placed, "ps_compare", {"a": spec, "b": "ps:ada/dance:1"})
    assert "aOpened" not in json.loads((tmp_path / "w" / "stage.json").read_text())


def test_a_missing_or_bad_ps_com_record_is_an_error_in_the_pages_words(ps_com, placed, tmp_path):
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/nothing:1", "slot": "a"})
    assert "PS.com に記録 ada/nothing:1 が見つかりません" in out["error"]
    assert "見つかりません" in call(placed, "ps_motion_open", {"spec": "ps:ada/dance:9", "slot": "a"})["error"]
    assert "handle/name:バージョン" in call(placed, "ps_motion_open", {"spec": "ps:Not A Ref", "slot": "a"})["error"]
    push._forget(ps_com.url)  # another person's private record is not found, with the hint to sign in
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    assert "見つかりません" in out["error"] and "psbx login" in out["error"]
    assert "a" not in json.loads((tmp_path / "w" / "stage.json").read_text())


def test_offline_a_ps_com_spec_without_a_version_and_nothing_cached_says_it_cannot_connect(ps_com, placed,
                                                                                          monkeypatch):
    monkeypatch.setenv("PS_CATALOG_URL", "http://127.0.0.1:9")
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/dance", "slot": "a"})
    assert "PS.com に接続できず、記録 ada/dance を取得できませんでした" in out["error"]


def test_a_record_the_ps_com_app_wrote_opens_into_a(ps_com, placed, tmp_path):
    """A schemaVersion 3 record as the PS.com app writes it (ps-com tests/api-recordings.test.js's PROJECT_V3, with
    an arm whose trace carries joints): the mover's position-only trace is skipped, the arm is read onto the
    placed machine."""
    body = (Path(__file__).parent / "fixtures" / "app-record-v3.json").read_bytes()
    ps_com.put_record("ada", "yard", "1", body, "private")
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/yard:1", "slot": "a"})
    assert "error" not in out and _seconds(out) == pytest.approx(1.2)
    assert "so101" in out["note"] and "mini" in out["note"] and out["records"][0]["label"] == "yard:1"
    assert out["aOpened"] == {"ref": "ada/yard", "version": "1", "latest": "1", "own": True}


def test_a_ps_com_item_that_is_not_a_record_is_refused(ps_com, placed):
    out = call(placed, "ps_motion_open", {"spec": "ps:t/mini:1", "slot": "a"})
    assert "記録ではありません" in out["error"]


def test_a_cached_ps_com_record_opens_offline_with_its_own_version_as_the_latest(ps_com, placed, monkeypatch):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:1", "slot": "a"})
    monkeypatch.setenv("PS_CATALOG_URL", "http://127.0.0.1:9")
    push._keep("http://127.0.0.1:9", "tok-1", 3600, handle="ada")
    out = call(placed, "ps_motion_open", {"spec": "ps:ada/dance:1", "slot": "a"})
    assert _seconds(out) == pytest.approx(0.5)
    assert out["aOpened"] == {"ref": "ada/dance", "version": "1", "latest": "1", "own": True}


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
    assert stage._safe_name("手本 a/b") == "手本_a_b" and stage._safe_name("..hidden") == "hidden"
    assert stage._safe_name("") == "motion" and stage._safe_name("...") == "motion"
    spec = str(_record(tmp_path / "w" / "手本 a.record.json", label="SO-101"))  # the file names the motion
    _, st = _export(placed, spec)
    assert st["error"] is None and st["paths"] == [str(tmp_path / "w" / "replay-手本_a.mp4")]


def test_two_records_of_one_robot_export_under_their_own_names(placed, tmp_path):
    a = str(_record(tmp_path / "w" / "demo.record.json", label="SO-101"))
    b = str(_record(tmp_path / "w" / "policy.record.json", label="SO-101"))
    _, st = _export(placed, a)
    assert st["paths"] == [str(tmp_path / "w" / "replay-demo.mp4")]
    _, st = _export_compare(placed, a, b)
    assert st["error"] is None and (tmp_path / "w" / "compare" / "demo-vs-policy" / "gaps.json").is_file()


def test_a_record_with_several_motions_is_named_by_its_file_and_the_motion(placed, tmp_path, monkeypatch):
    q = np.linspace(0, 1, 6).reshape(-1, 1)
    save_record([motion_from_numpy(q, fps=10, joint_names=["turn"], machine="mini", label=m) for m in ("p1", "p2")],
                "site", tmp_path / "w" / "two.record.json")
    path = str(tmp_path / "w" / "two.record.json")
    assert placed.stage._load_motion(path + "#p2")[1].label == "two-p2"
    assert placed.stage._load_motion(str(_record(tmp_path / "w" / "one.record.json")))[1].label == "one"
    assert stage._record_stem(Path("/x/plain.json")) == "plain" and stage._record_stem(Path("/x/a.b")) == "a"


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


@pytest.fixture
def two_records(placed, tmp_path):
    a = str(_record(tmp_path / "w" / "demo.record.json", label="demo"))
    b = str(_record(tmp_path / "w" / "policy.record.json", label="policy"))
    return a, b


def _export_compare(server, a, b, grip=None):
    args = {"what": "compare", "a": a, "b": b}
    if grip is not None:
        args["gripThreshold"] = grip
    job = call(server, "ps_export", args)["job"]
    st = {}
    for _ in range(400):
        st = call(server, "ps_export_status", {"job": job})
        if st["done"]:
            break
        time.sleep(0.05)
    return job, st


def test_compare_in_the_stage_gives_two_records_and_keeps_the_choice(placed, two_records, tmp_path):
    spec_a, spec_b = two_records
    out = call(placed, "ps_compare", {"a": spec_a, "b": spec_b, "gripThreshold": None})
    assert len(out["records"]) == 2 and "tool_deviation_m" in out["figures"] and out["notes"] == []
    assert [r["label"] for r in out["records"]] == ["demo", "policy"] and out["dataParts"] >= 1
    kept = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert kept["a"] == spec_a and kept["b"] == spec_b and kept["gripThreshold"] is None
    assert kept["machine"]["id"] == "mini"
    data = json.loads("".join(call(placed, "ps_compare_part", {"key": out["dataKey"], "part": i})["data"]
                              for i in range(out["dataParts"])))
    assert len(data["poses"]) == 2 and len(data["t"]) == out["frames"]


def test_compare_keeps_the_grip_threshold_and_first_carries_it(placed, two_records, tmp_path):
    spec_a, spec_b = two_records
    out = call(placed, "ps_compare", {"a": spec_a, "b": spec_b, "gripThreshold": 0.25})
    assert out["figures"]["grip_threshold"] == 0.25
    first = json.loads(StageServer(Stage(work=tmp_path / "w")).first())
    assert first["motionA"] == spec_a and first["motionB"] == spec_b and first["gripThreshold"] == 0.25
    call(placed, "ps_compare", {"a": spec_a, "b": spec_b, "gripThreshold": None})
    assert json.loads(StageServer(Stage(work=tmp_path / "w")).first())["gripThreshold"] is None


def test_compare_refuses_a_bad_threshold_and_bad_specs_and_keeps_nothing(placed, two_records, tmp_path):
    spec_a, spec_b = two_records
    for bad in (float("nan"), float("inf"), True, "0.2"):
        out = call(placed, "ps_compare", {"a": spec_a, "b": spec_b, "gripThreshold": bad})
        assert "gripThreshold" in out["error"], bad
    assert "error" in call(placed, "ps_compare", {"a": spec_a, "b": str(tmp_path / "w" / "none.record.json"), "gripThreshold": None})
    assert "error" in call(placed, "ps_compare", {"a": "", "b": spec_b, "gripThreshold": None})
    assert "a" not in json.loads((tmp_path / "w" / "stage.json").read_text())


def test_compare_refuses_motions_that_do_not_fit_each_other(placed, two_records, tmp_path):
    spec_a, _ = two_records
    q = np.linspace(0, 1, 6).reshape(-1, 2)
    save_record([motion_from_numpy(q, fps=10, joint_names=["turn", "lift"], machine="mini", label="wide")],
                "site", tmp_path / "w" / "wide.record.json")
    out = call(placed, "ps_compare", {"a": spec_a, "b": str(tmp_path / "w" / "wide.record.json"), "gripThreshold": None})
    assert "記録の関節がロボットと合いません" in out["error"]
    assert "error" in call(placed, "ps_export", {"what": "compare", "a": spec_a, "b": str(tmp_path / "w" / "wide.record.json")})


def test_export_compare_writes_the_four_files_in_their_own_folder(placed, two_records, tmp_path):
    spec_a, spec_b = two_records
    job, st = _export_compare(placed, spec_a, spec_b)
    assert len(job) == 8 and st["error"] is None and st["done"] and st["progress"] == 1
    folder = tmp_path / "w" / "compare" / "demo-vs-policy"
    assert {p.name for p in folder.iterdir()} >= {"gaps.json", "gaps.csv", "gaps.png", "replay.mp4"}
    assert sorted(Path(p).name for p in st["paths"]) == ["gaps.csv", "gaps.json", "gaps.png", "replay.mp4"]
    assert all(Path(p).parent == folder and Path(p).stat().st_size > 0 for p in st["paths"])


def test_export_compare_takes_the_grip_threshold(placed, two_records, tmp_path):
    spec_a, spec_b = two_records
    _, st = _export_compare(placed, spec_a, spec_b, grip=0.3)
    assert st["error"] is None
    report = json.loads((tmp_path / "w" / "compare" / "demo-vs-policy" / "gaps.json").read_text())
    assert report["grip_threshold"] == 0.3
    out = call(placed, "ps_export", {"what": "compare", "a": spec_a, "b": spec_b, "gripThreshold": float("inf")})
    assert "gripThreshold" in out["error"]


def test_the_grip_threshold_may_be_any_finite_number(placed, two_records, tmp_path):
    spec_a, spec_b = two_records
    out = call(placed, "ps_compare", {"a": spec_a, "b": spec_b, "gripThreshold": -0.2})
    assert out["figures"]["grip_threshold"] == -0.2
    assert json.loads(StageServer(Stage(work=tmp_path / "w")).first())["gripThreshold"] == -0.2


def test_a_second_export_while_a_compare_runs_is_refused(placed, two_records, monkeypatch):
    spec_a, spec_b = two_records
    gate = threading.Event()
    real = stage.render_overlay

    def slow(*args, **kw):
        gate.wait(5)
        return real(*args, **kw)
    monkeypatch.setattr(stage, "render_overlay", slow)
    first = call(placed, "ps_export", {"what": "compare", "a": spec_a, "b": spec_b})
    try:
        assert call(placed, "ps_export", {"what": "compare", "a": spec_a, "b": spec_b}) == {"error": "書き出し中です"}
        assert call(placed, "ps_export", {"what": "replay", "spec": spec_a}) == {"error": "書き出し中です"}
    finally:
        gate.set()
    for _ in range(400):
        if call(placed, "ps_export_status", {"job": first["job"]})["done"]:
            break
        time.sleep(0.05)


def test_a_failed_compare_render_leaves_the_numbers_and_no_stale_video(placed, two_records, tmp_path, monkeypatch):
    spec_a, spec_b = two_records
    folder = tmp_path / "w" / "compare" / "demo-vs-policy"
    folder.mkdir(parents=True)
    for name in ("replay.mp4", "gaps.json", "gaps.csv", "gaps.png"):
        (folder / name).write_bytes(b"stale")

    def boom(*a, **k):
        raise RuntimeError("no codec\nsecond line")
    monkeypatch.setattr(stage, "render_overlay", boom)
    _, st = _export_compare(placed, spec_a, spec_b)
    assert st["done"] and st["paths"] == []
    assert st["error"] == "動画を書き出せませんでした（数値のファイルは書きました）: no codec"
    assert not (folder / "replay.mp4").exists()
    assert {p.name for p in folder.iterdir()} == {"gaps.json", "gaps.csv", "gaps.png"}
    assert (folder / "gaps.json").read_bytes() != b"stale"  # this export's numbers, not the old ones


def test_a_compare_that_fails_before_the_numbers_keeps_the_plain_error(placed, two_records, tmp_path, monkeypatch):
    spec_a, spec_b = two_records

    def boom(*a, **k):
        raise RuntimeError("no report")
    monkeypatch.setattr(stage.report, "write_report", boom)
    _, st = _export_compare(placed, spec_a, spec_b)
    assert st["done"] and st["error"] == "no report"


def test_the_stage_html_has_the_compare_controls(served):
    html = get(served().url)[2]
    for ident in ("comparePick", "compareSpec", "grip", "compareGo", "cphoto", "cgl", "cview", "cplay", "cslider",
                  "cclock", "compareExport", "cfigures", "csite"):
        assert f'id="{ident}"' in html
    for name in ("ps_compare", "ps_export"):
        assert name in html


def test_the_stage_html_has_the_motion_controls(served):
    html = get(served().url)[2]
    for ident in ("motionPick", "motionSpec", "episode", "tphoto", "tgl", "tview", "tplay", "tslider", "tclock", "exportBtn"):
        assert f'id="{ident}"' in html
    for name in ("ps_motions_list", "ps_motion_open", "ps_compare_part", "ps_export", "ps_export_status"):
        assert name in html
    assert "PS.com の記録" in html and "'ps:'" in html  # the catalog group, and a typed handle/name is prefixed


# ---- saving to PS.com, exporting, and the copies PS.com did not get ----

def _stored(fake, handle, name, version):
    return json.loads(fake.items[(handle, name)]["versions"][version]["raw"])


def _record_puts(fake):
    return [p for m, p in fake.calls if m == "PUT" and p.split("?")[0].endswith("/record")]


def _site_id(tmp_path):
    return load_site(tmp_path / "w" / "site.json").id


@pytest.fixture
def dance_kept(ps_com, tmp_path):
    """ada/dance version 2 rewritten with a key psbx does not read, as PS.com would hold it."""
    raw = json.loads(ps_com.items[("ada", "dance")]["versions"]["2"]["raw"])
    raw["keepMe"] = {"by": "the app"}
    ps_com.put_record("ada", "dance", "2", json.dumps(raw).encode(), "private")
    return raw


@pytest.fixture
def mini_has_product(tmp_path):
    """The cached machine ``mini`` with the product file push needs to bring it along to PS.com."""
    shutil.copy(Path(__file__).parent / "fixtures" / "mini-robot" / "product.json",
                tmp_path / "cache" / "machines" / "mini" / "product.json")


def _place(server, cache_mini, x=0.5, y=0.2, yaw=0.3):
    assert "scenePath" in call(server, "ps_scene_place", {"id": "mini", "urdf": cache_mini, "x": x, "y": y, "yaw": yaw})


def test_save_over_writes_the_next_version_keeping_what_psbx_does_not_read(ps_com, placed, dance_kept, cache_mini,
                                                                           tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    _place(placed, cache_mini)
    out = call(placed, "ps_record_save", {"mode": "over"})
    assert out == {"saved": True, "ref": "ada/dance", "version": "3", "spec": "ps:ada/dance:3"}
    assert _record_puts(ps_com)[-1] == "/api/catalog/ada/dance/record?base=2&visibility=private"
    saved = _stored(ps_com, "ada", "dance", "3")
    assert saved["keepMe"] == {"by": "the app"} and "basedOn" not in saved
    (dev,) = saved["equipment"]["devices"]
    assert dev["product"] == "so101"  # what the record names stays as it is: nothing is uploaded with it
    assert dev["pose"] == {"xyz": [0.5, 0.2, 0.0], "rpy": [0.0, 0.0, 0.3]}
    assert saved["siteFrameId"] == _site_id(tmp_path) and saved["traces"] == dance_kept["traces"]
    kept = json.loads((tmp_path / "w" / "stage.json").read_text())
    assert kept["a"] == "ps:ada/dance:3"
    assert kept["aOpened"] == {"ref": "ada/dance", "version": "3", "latest": "3", "own": True}
    out = call(placed, "ps_record_save", {"mode": "over"})  # the second save goes over the version just written
    assert out["saved"] is True and out["ref"] == "ada/dance" and out["version"] == "3"  # the same bytes: no version
    _place(placed, cache_mini, x=0.6)
    assert call(placed, "ps_record_save", {"mode": "over"})["version"] == "4"


def test_save_over_of_a_ps_com_app_record_keeps_its_trace_and_its_height(ps_com, placed, cache_mini, tmp_path):
    """A record the PS.com app wrote: trace samples carry position and yaw beside the joints, and the arm stands
    0.7 m up, tilted. Moving the arm in x/y on the stage changes only its x, y and yaw."""
    raw = json.loads(ps_com.items[("ada", "dance")]["versions"]["2"]["raw"])
    (dev,) = raw["equipment"]["devices"]
    dev["pose"] = {"xyz": [0.1, 0.1, 0.7], "rpy": [0.05, -0.05, 0.0]}
    for k, sample in enumerate(raw["traces"][0]["samples"]):
        sample["position"], sample["yaw"] = [0.1 * k, 0.0, 0.7], 0.25
    ps_com.put_record("ada", "dance", "2", json.dumps(raw).encode(), "private")
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    _place(placed, cache_mini, x=0.5, y=0.2, yaw=0.3)
    out = call(placed, "ps_record_save", {"mode": "over"})
    assert out["saved"] is True and out["version"] == "3"
    saved = _stored(ps_com, "ada", "dance", "3")
    (dev,) = saved["equipment"]["devices"]
    assert dev["pose"] == {"xyz": [0.5, 0.2, 0.7], "rpy": [0.05, -0.05, 0.3]}
    assert saved["traces"] == raw["traces"]  # the samples' position and yaw are still there


def test_save_over_needs_the_persons_own_ps_com_record_in_a(ps_com, placed, tmp_path):
    assert "動きを読み込んでから" in call(placed, "ps_record_save", {"mode": "over"})["error"]
    assert "動きを読み込んでから" in call(placed, "ps_record_export", {})["error"]
    call(placed, "ps_motion_open", {"spec": str(_record(tmp_path / "w" / "demo.record.json")), "slot": "a"})
    assert "自分の記録だけ" in call(placed, "ps_record_save", {"mode": "over"})["error"]
    call(placed, "ps_motion_open", {"spec": "ps:bob/walk:1", "slot": "a"})
    assert "自分の記録だけ" in call(placed, "ps_record_save", {"mode": "over"})["error"]
    assert "mode" in call(placed, "ps_record_save", {"mode": "sideways"})["error"]
    assert _record_puts(ps_com) == []


def test_save_new_from_a_ps_com_record_merges_it_and_says_what_it_came_from(ps_com, placed, dance_kept, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    out = call(placed, "ps_record_save", {"mode": "new", "name": "dance-copy", "title": "ダンスの写し"})
    assert out == {"saved": True, "ref": "ada/dance-copy", "version": "1", "spec": "ps:ada/dance-copy:1"}
    assert _record_puts(ps_com)[-1] == "/api/catalog/ada/dance-copy/record?new=1&visibility=private"
    saved = _stored(ps_com, "ada", "dance-copy", "1")
    assert saved["basedOn"] == "ada/dance:2" and saved["name"] == "ダンスの写し"
    assert saved["keepMe"] == {"by": "the app"} and saved["equipment"]["devices"][0]["product"] == "so101"
    assert ps_com.items[("ada", "dance-copy")]["visibility"] == "private"
    assert json.loads((tmp_path / "w" / "stage.json").read_text())["aOpened"] == {
        "ref": "ada/dance-copy", "version": "1", "latest": "1", "own": True}


def test_save_new_from_a_file_brings_the_machine_and_its_place(ps_com, placed, mini_has_product, cache_mini,
                                                              tmp_path):
    call(placed, "ps_motion_open", {"spec": str(_record(tmp_path / "w" / "demo.record.json")), "slot": "a"})
    _place(placed, cache_mini, x=0.1, y=-0.2, yaw=0.0)
    out = call(placed, "ps_record_save", {"mode": "new"})  # the name and the title default to A's label
    assert out["saved"] is True and out["ref"] == "ada/demo" and out["version"] == "1"
    saved = _stored(ps_com, "ada", "demo", "1")
    (dev,) = saved["equipment"]["devices"]
    assert dev["product"] == "ada/mini:1" and ps_com.items[("ada", "mini")]["visibility"] == "private"
    assert dev["pose"] == {"xyz": [0.1, -0.2, 0.0], "rpy": [0.0, 0.0, 0.0]}
    assert saved["name"] == "demo" and "basedOn" not in saved and saved["siteFrameId"] == _site_id(tmp_path)
    assert [list(s["joints"]) for s in saved["traces"][0]["samples"]][0] == ["turn"]


def test_a_server_error_keeps_the_save_as_unsent_like_offline(ps_com, placed, cache_mini, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    _place(placed, cache_mini)
    ps_com.fail_record = (503, {})
    out = call(placed, "ps_record_save", {"mode": "over"})
    assert out == {"saved": False, "unsent": "dance", "reason": unsent.SERVER_DOWN}
    assert list(ps_com.items[("ada", "dance")]["versions"]) == ["1", "2"]
    ps_com.fail_record = None
    out = call(placed, "ps_unsent_send", {})
    assert [(s["ref"], s["version"]) for s in out["sent"]] == [("ada/dance", "3")] and out["left"] == []


@pytest.mark.parametrize("status,error,words", [
    (429, "limit", "記録の数が上限に達しています"),
    (409, "renaming", "ハンドルの変更中です"),
    (409, "not-a-record", "その名前は記録以外に使われています"),
])
def test_a_save_refused_by_ps_com_says_why_in_words(ps_com, placed, status, error, words):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    ps_com.fail_record = (status, {"error": error})
    out = call(placed, "ps_record_save", {"mode": "new", "name": "fresh"})
    assert "PS.com に保存できませんでした（" in out["error"] and words in out["error"]
    assert error not in out["error"]


def test_save_new_refuses_a_bad_name_and_a_name_that_is_there(ps_com, placed):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    assert "英小文字" in call(placed, "ps_record_save", {"mode": "new", "name": "Bad Name"})["error"]
    assert "英小文字" in call(placed, "ps_record_save", {"mode": "new", "name": "x"})["error"]
    out = call(placed, "ps_record_save", {"mode": "new", "name": "wave"})
    assert "ada/wave はもうあります" in out["error"] and "--over" not in out["error"]
    assert list(ps_com.items[("ada", "wave")]["versions"]) == ["1"]


def test_a_record_saved_over_elsewhere_meanwhile_is_saved_as_new_and_said(ps_com, placed, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    _seed_record(ps_com, tmp_path, "ada", "dance", "3", 4, "private")  # saved over in the PS.com app meanwhile
    out = call(placed, "ps_record_save", {"mode": "over"})
    assert out["saved"] is True and out["ref"] == "ada/dance-2" and out["version"] == "1"
    assert out["note"] == "ada/dance は別の場所で変わっていたため、ada/dance-2 として新しく保存しました"
    assert list(ps_com.items[("ada", "dance")]["versions"]) == ["1", "2", "3"]  # nothing was written over
    assert _stored(ps_com, "ada", "dance-2", "1")["basedOn"] == "ada/dance:2"
    assert json.loads((tmp_path / "w" / "stage.json").read_text())["a"] == "ps:ada/dance-2:1"


def test_offline_a_save_is_kept_shown_after_a_restart_and_sent_when_ps_com_is_back(ps_com, placed, cache_mini,
                                                                                   tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    _place(placed, cache_mini)
    ps_com.stop()
    out = call(placed, "ps_record_save", {"mode": "over"})
    assert out["saved"] is False and out["unsent"] == "dance" and "PS.com に接続できませんでした" in out["reason"]
    assert (tmp_path / "w" / "unsent" / "dance.record.json").is_file()
    again = Stage(work=tmp_path / "w")  # the stage was closed and started again
    assert again.first()["unsent"] == [{"name": "dance", "ref": "ada/dance", "mode": "over"}]
    out = json.loads(again.call("ps_unsent_send", {}))  # still offline: it stays
    assert out["sent"] == [] and [p["name"] for p in out["left"]] == ["dance"] and out["notes"]
    ps_com.start(ps_com.port)
    out = json.loads(again.call("ps_unsent_send", {}))
    assert out == {"sent": [{"name": "dance", "ref": "ada/dance", "version": "3", "mode": "over"}], "left": [],
                   "notes": []}
    saved = _stored(ps_com, "ada", "dance", "3")
    assert saved["equipment"]["devices"][0]["pose"] == {"xyz": [0.5, 0.2, 0.0], "rpy": [0.0, 0.0, 0.3]}
    assert again.first()["unsent"] == [] and not list((tmp_path / "w" / "unsent").iterdir())
    assert again.first()["aOpened"] == {"ref": "ada/dance", "version": "3", "latest": "3", "own": True}


def test_an_unsent_save_over_that_moved_meanwhile_is_sent_as_new_with_the_note(ps_com, placed, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    ps_com.stop()
    assert call(placed, "ps_record_save", {"mode": "over"})["unsent"] == "dance"
    _seed_record(ps_com, tmp_path, "ada", "dance", "3", 4, "private")
    ps_com.start(ps_com.port)
    out = call(placed, "ps_unsent_send", {})
    assert out["sent"] == [{"name": "dance", "ref": "ada/dance-2", "version": "1", "mode": "new"}]
    assert out["notes"] == ["ada/dance は別の場所で変わっていたため、ada/dance-2 として新しく保存しました"]
    assert list(ps_com.items[("ada", "dance")]["versions"]) == ["1", "2", "3"]


def test_signed_out_a_save_is_kept_and_sent_under_the_handle_after_psbx_login(ps_com, placed, mini_has_product,
                                                                              tmp_path):
    call(placed, "ps_motion_open", {"spec": str(_record(tmp_path / "w" / "demo.record.json")), "slot": "a"})
    push._forget(ps_com.url)
    before = len(ps_com.calls)
    out = call(placed, "ps_record_save", {"mode": "new", "name": "demo-run", "title": "デモ"})
    assert out == {"saved": False, "unsent": "demo-run", "reason": "psbx login でログインすると送れます"}
    out = call(placed, "ps_unsent_send", {})
    assert out["sent"] == [] and out["notes"] == ["psbx login でログインすると送れます"]
    assert out["left"] == [{"name": "demo-run", "ref": "demo-run", "mode": "new"}]
    assert len(ps_com.calls) == before  # signed out: no request at all
    push._keep(ps_com.url, "tok-1", 3600, handle="ada")
    out = call(placed, "ps_unsent_send", {})
    assert out["sent"] == [{"name": "demo-run", "ref": "ada/demo-run", "version": "1", "mode": "new"}]
    assert _stored(ps_com, "ada", "demo-run", "1")["name"] == "デモ"


def test_export_writes_the_record_as_a_save_builds_it_without_the_network(ps_com, placed, dance_kept, cache_mini,
                                                                         tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    _place(placed, cache_mini)
    before = len(ps_com.calls)
    out = call(placed, "ps_record_export", {})
    assert out == {"path": str(tmp_path / "w" / "dance_2.record.json")}
    written = json.loads(Path(out["path"]).read_text())
    assert written["keepMe"] == {"by": "the app"}
    assert written["equipment"]["devices"][0]["pose"] == {"xyz": [0.5, 0.2, 0.0], "rpy": [0.0, 0.0, 0.3]}
    assert call(placed, "ps_record_export", {})["path"] == str(tmp_path / "w" / "dance_2-2.record.json")
    spec = str(_record(tmp_path / "w" / "demo.record.json"))
    call(placed, "ps_motion_open", {"spec": spec, "slot": "a"})
    out = call(placed, "ps_record_export", {})
    assert out["path"] == str(tmp_path / "w" / "demo-2.record.json")
    (dev,) = json.loads(Path(out["path"]).read_text())["equipment"]["devices"]
    assert dev["product"] == "mini" and dev["pose"]["xyz"] == [0.5, 0.2, 0.0]
    assert len(ps_com.calls) == before


def test_the_stage_html_has_the_save_controls(served):
    html = get(served().url)[2]
    for ident in ("saveOver", "saveName", "saveTitle", "saveNew", "recordExport", "unsentLines", "unsentSend"):
        assert f'id="{ident}"' in html
    for name in ("ps_record_save", "ps_record_export", "ps_unsent_send"):
        assert name in html
    for words in ("上書き保存", "新しい記録として保存", "エクスポート", "PS.com 未保存", "再送信する", "psbx login"):
        assert words in html
    assert "aOpened?.own && state?.login?.signedIn" in html  # 上書き保存 is hidden while signed out
    assert "直す" not in html and "直し" not in html  # 再送信する, never 送り直す


def test_two_offline_saves_over_keep_one_copy_and_make_one_version(ps_com, placed, cache_mini, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    ps_com.stop()
    _place(placed, cache_mini, x=0.5)
    assert call(placed, "ps_record_save", {"mode": "over"})["unsent"] == "dance"
    _place(placed, cache_mini, x=0.7)
    assert call(placed, "ps_record_save", {"mode": "over"})["unsent"] == "dance"
    assert placed.stage.first()["unsent"] == [{"name": "dance", "ref": "ada/dance", "mode": "over"}]
    ps_com.start(ps_com.port)
    out = call(placed, "ps_unsent_send", {})
    assert out == {"sent": [{"name": "dance", "ref": "ada/dance", "version": "3", "mode": "over"}], "left": [],
                   "notes": []}
    assert list(ps_com.items[("ada", "dance")]["versions"]) == ["1", "2", "3"] and ("ada", "dance-2") not in ps_com.items
    assert _stored(ps_com, "ada", "dance", "3")["equipment"]["devices"][0]["pose"]["xyz"] == [0.7, 0.2, 0.0]


def test_a_kept_new_record_whose_name_was_taken_goes_under_a_free_name(ps_com, placed, mini_has_product, tmp_path):
    call(placed, "ps_motion_open", {"spec": str(_record(tmp_path / "w" / "demo.record.json")), "slot": "a"})
    push._forget(ps_com.url)
    assert call(placed, "ps_record_save", {"mode": "new", "name": "wave"})["unsent"] == "wave"
    push._keep(ps_com.url, "tok-1", 3600, handle="ada")
    out = call(placed, "ps_unsent_send", {})
    assert out["sent"] == [{"name": "wave", "ref": "ada/wave-2", "version": "1", "mode": "new"}] and out["left"] == []
    assert out["notes"] == ["ada/wave はもうあったため、ada/wave-2 として保存しました"]
    assert list(ps_com.items[("ada", "wave")]["versions"]) == ["1"]
    assert "basedOn" not in _stored(ps_com, "ada", "wave-2", "1")


def test_a_kept_copy_never_goes_up_with_another_placed_machine(ps_com, placed, mini_has_product, tmp_path):
    call(placed, "ps_motion_open", {"spec": str(_record(tmp_path / "w" / "demo.record.json")), "slot": "a"})
    push._forget(ps_com.url)
    assert call(placed, "ps_record_save", {"mode": "new", "name": "demo"})["unsent"] == "demo"
    arm = tmp_path / "arms" / "arm"  # another machine is placed before the copy goes
    arm.mkdir(parents=True)
    (arm / "arm.urdf").write_text(URDF)
    (arm / "product.json").write_text(json.dumps({**json.loads(
        (tmp_path / "cache" / "machines" / "mini" / "product.json").read_text()), "id": "arm"}))
    out = call(placed, "ps_scene_place", {"id": "arm", "urdf": str(arm / "arm.urdf"), "x": 0, "y": 0, "yaw": 0,
                                          "typed": True})
    assert "scenePath" in out
    push._keep(ps_com.url, "tok-1", 3600, handle="ada")
    mini = tmp_path / "cache" / "machines" / "mini"
    mini.rename(tmp_path / "mini-away")  # its own machine is nowhere: the copy stays, with a note
    out = call(placed, "ps_unsent_send", {})
    assert out["sent"] == [] and [p["name"] for p in out["left"]] == ["demo"] and "mini" in out["notes"][0]
    assert ("ada", "arm") not in ps_com.items and ("ada", "mini") not in ps_com.items
    (tmp_path / "mini-away").rename(mini)  # back in the cache: the copy goes with mini, not with arm
    out = call(placed, "ps_unsent_send", {})
    assert out["sent"] == [{"name": "demo", "ref": "ada/demo", "version": "1", "mode": "new"}]
    files = ps_com.items[("ada", "mini")]["versions"]["1"]["files"]
    assert "mini.urdf" in files and "arm.urdf" not in files and ("ada", "arm") not in ps_com.items
    assert _stored(ps_com, "ada", "demo", "1")["equipment"]["devices"][0]["product"] == "ada/mini:1"


def test_a_save_online_also_sends_the_waiting_copies(ps_com, placed, mini_has_product, tmp_path):
    call(placed, "ps_motion_open", {"spec": str(_record(tmp_path / "w" / "demo.record.json")), "slot": "a"})
    push._forget(ps_com.url)
    assert call(placed, "ps_record_save", {"mode": "new", "name": "demo-run"})["unsent"] == "demo-run"
    push._keep(ps_com.url, "tok-1", 3600, handle="ada")
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    out = call(placed, "ps_record_save", {"mode": "over"})
    assert out["saved"] is True and out["version"] == "3"
    assert out["unsentSend"] == {"sent": [{"name": "demo-run", "ref": "ada/demo-run", "version": "1",
                                           "mode": "new"}], "left": [], "notes": []}
    assert placed.stage.first()["unsent"] == []


def test_a_save_over_online_replaces_a_waiting_save_over_of_the_same_record(ps_com, placed, cache_mini, tmp_path):
    call(placed, "ps_motion_open", {"spec": "ps:ada/dance:2", "slot": "a"})
    ps_com.stop()
    _place(placed, cache_mini, x=0.5)
    assert call(placed, "ps_record_save", {"mode": "over"})["unsent"] == "dance"
    ps_com.start(ps_com.port)
    _place(placed, cache_mini, x=0.7)
    out = call(placed, "ps_record_save", {"mode": "over"})
    assert out["saved"] is True and out["version"] == "3" and "note" not in out and "unsentSend" not in out
    assert list(ps_com.items[("ada", "dance")]["versions"]) == ["1", "2", "3"] and ("ada", "dance-2") not in ps_com.items
    assert placed.stage.first()["unsent"] == []


def test_the_stage_has_one_word_for_signing_in():
    """One word for signing in on the stage, as its status line (ログイン中／未ログイン) and unsent.SIGN_IN say."""
    import physical_sandbox.stage as stage_module
    src = Path(stage_module.__file__)
    for path in (src, src.parent / "mcp" / "widgets" / "stage.html", src.parent / "unsent.py"):
        assert "サインイン" not in path.read_text(), path.name
