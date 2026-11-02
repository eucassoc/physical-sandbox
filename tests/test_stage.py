"""The stage: one browser screen over a work folder (floor step), served on 127.0.0.1."""
import json
import shutil
import threading
import urllib.request

import cv2
import numpy as np
import pytest

from physical_sandbox import push, stage
from physical_sandbox.cli import main
from physical_sandbox.errors import InputError
from physical_sandbox.scene import load_scene
from physical_sandbox.site import load_site
from physical_sandbox.stage import Stage, StageServer

import test_fit_screen
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
