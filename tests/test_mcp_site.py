"""The MCP server and its screen to fix a site. No SAM weights needed: a fake segmenter stands in."""
import asyncio
import base64
import json
import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from physical_sandbox.mcp import server  # noqa: E402
from physical_sandbox.site import load_site  # noqa: E402

W, H = 2020, 1108
EDGES = [3.5, 6.8]


def project_rect(f=900, rvec=(1.9, 0.15, -0.1), tvec=(-1.5, 0.8, 9.0), a=3.5, b=6.8, cx=1010, cy=554):
    """Corners of an a x b m floor rectangle seen through a known pinhole pose (as in test_floor.py)."""
    world = np.array([[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]], float)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], float)
    px, _ = cv2.projectPoints(world, np.array(rvec, float), np.array(tvec, float), K, None)
    return [{"u": float(u), "v": float(v)} for u, v in px.reshape(-1, 2)]


CORNERS = project_rect()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A synthetic still, a site directory, and the photo cache redirected into tmp_path."""
    cache = tmp_path / "cache"
    monkeypatch.setenv("PS_CACHE_DIR", str(cache))
    img = np.full((H, W, 3), (90, 120, 100), np.uint8)
    cv2.rectangle(img, (300, 500), (420, 700), (30, 40, 200), -1)
    path = tmp_path / "still.png"
    cv2.imwrite(str(path), img)
    return {"image": str(path), "site": str(tmp_path / "site"), "cache": cache}


def fake_segmenter(monkeypatch, found):
    """Swap SAM for a function giving a filled box for each name in ``found`` (others have no mask)."""
    def run(image, boxes):
        out = {}
        for name, (x0, y0, x1, y1) in boxes.items():
            if name in found:
                m = np.zeros(image.shape[:2], bool)
                m[y0:y1, x0:x1] = True
                out[name] = m
        return out
    monkeypatch.setattr(server, "make_segmenter", lambda: run)


def boxes(*names):
    spots = {"pole": (300, 500, 420, 700), "cone": (900, 600, 960, 700), "gate": (1500, 400, 1700, 800)}
    return [{"name": n, "kind": "fixed", **dict(zip(("x0", "y0", "x1", "y1"), spots[n]))} for n in names]


def shapes(out):
    """The outlines and ground points the screen fetches with ps_objects_part, by name."""
    if not out["objectsKey"]:
        return {}
    data = "".join(json.loads(server.ps_objects_part(out["objectsKey"], i))["data"] for i in range(out["objectsParts"]))
    return {o["name"]: o for o in json.loads(data)["objects"]}


def text(result):
    assert not result.is_error, result.content
    return result.content[0].text


@asynccontextmanager
async def started(env):
    """One start of the server over stdio, as Claude Desktop starts it."""
    params = StdioServerParameters(command=sys.executable, args=["-m", "physical_sandbox.mcp.server"],
                                   env={"PS_CACHE_DIR": str(env["cache"])})
    async with Client(params) as client:
        yield client


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 90))


# ---- what the host sees ----

def test_tool_list_marks_the_screen_and_hides_the_screen_only_tools(env):
    async def go():
        async with started(env) as client:
            return [t.model_dump(by_alias=True) for t in (await client.list_tools()).tools]
    tools = {t["name"]: t for t in run(go())}
    assert set(tools) == {"ps_fit_floor", "ps_spike_image", "ps_floor_solve", "ps_objects_propose",
                          "ps_site_decide", "ps_site_status", "ps_compare", "ps_model_part", "ps_compare_part",
                          "ps_objects_part", "ps_catalog_search", "ps_catalog_pull", "ps_catalog_push",
                          "ps_machine_check"}
    for name in ("ps_fit_floor", "ps_spike_image", "ps_floor_solve", "ps_objects_propose", "ps_site_decide",
                 "ps_objects_part"):
        assert tools[name]["_meta"]["ui"]["resourceUri"] == "ui://physical-sandbox/fit-floor.html"
    for name in ("ps_spike_image", "ps_floor_solve", "ps_site_decide", "ps_objects_part"):
        assert tools[name]["_meta"]["ui"]["visibility"] == ["app"]
    for name in ("ps_fit_floor", "ps_objects_propose"):
        assert "model" in (tools[name]["_meta"]["ui"].get("visibility") or ["model"])  # the agent can call it
    assert "ui" not in (tools["ps_site_status"].get("_meta") or {})


def test_machine_check_over_stdio_says_what_is_missing(env, tmp_path):
    import shutil
    from pathlib import Path

    mini = tmp_path / "mini"
    shutil.copytree(Path(__file__).parent / "fixtures" / "mini-robot", mini)

    async def go():
        async with started(env) as client:
            green = json.loads(text(await client.call_tool("ps_machine_check", {"path": str(mini)})))
            (mini / "product.json").unlink()
            red = json.loads(text(await client.call_tool("ps_machine_check", {"path": str(mini)})))
            return green, red
    green, red = run(go())
    assert green["ok"] is True and all(i["ok"] for i in green["items"])
    assert red["ok"] is False
    assert any(not i["ok"] and "product.json" in i["text"] for i in red["items"])


def test_no_tool_input_schema_has_ref_or_prefix_items(env):
    async def go():
        async with started(env) as client:
            return (await client.list_tools()).tools
    for tool in run(go()):
        schema = json.dumps(tool.input_schema)
        assert "$ref" not in schema and "prefixItems" not in schema, tool.name


def test_the_screen_is_served_as_an_app_with_the_bridge_inlined(env):
    async def go():
        async with started(env) as client:
            return await client.read_resource("ui://physical-sandbox/fit-floor.html")
    res = run(go())
    content = res.contents[0]
    assert content.mime_type == "text/html;profile=mcp-app"
    assert "globalThis.ExtApps=" in content.text
    assert server.BUNDLE_MARKER not in content.text
    assert "ps_floor_solve" in content.text and "ps_site_decide" in content.text
    assert "ps_objects_part" in content.text


def test_the_screen_and_bridge_ship_as_package_data():
    import tomllib
    from pathlib import Path
    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    patterns = data["tool"]["setuptools"]["package-data"]["physical_sandbox.mcp"]
    assert "widgets/*.html" in patterns and "widgets/vendor/*.js" in patterns


def test_the_vendored_bundles_licences_ship_with_them_and_are_named_in_the_notice():
    import fnmatch
    import tomllib
    from pathlib import Path
    root = Path(__file__).parent.parent
    patterns = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"][
        "physical_sandbox.mcp"]
    vendor = root / "src" / "physical_sandbox" / "mcp" / "widgets" / "vendor"
    shipped = {p.name for p in vendor.iterdir()
               if any(fnmatch.fnmatch(f"widgets/vendor/{p.name}", pat) for pat in patterns)}
    assert {"three.iife.js", "ext-apps-app-with-deps.js", "LICENSE-three.txt", "LICENSE-ext-apps.txt"} <= shipped
    three = (vendor / "LICENSE-three.txt").read_text()
    assert "three.js authors" in three and "Permission is hereby granted" in three
    ext = (vendor / "LICENSE-ext-apps.txt").read_text()
    for name in ("@modelcontextprotocol/ext-apps", "@modelcontextprotocol/client", "zod", "content-type",
                 "Colin McDonnell", "Douglas Christopher Wilson", "Model Context Protocol a Series of LF Projects"):
        assert name in ext, name
    notice = (root / "NOTICE").read_text()
    assert "LICENSE-three.txt" in notice and "LICENSE-ext-apps.txt" in notice


# ---- the floor ----

def test_floor_solve_on_a_synthetic_floor_leaves_under_half_a_pixel():
    out = json.loads(server.ps_floor_solve(W, H, CORNERS, EDGES))
    assert out["ok"] and out["residualPx"] < 0.5 and not out["mirrored"]
    assert out["fovDeg"] == pytest.approx(97, abs=1)
    # a 1 m grid: 4 lines across the 3.5 m edge (0..3) and 7 along the 6.8 m edge (0..6)
    assert len(out["lines"]) == 4 + 7
    assert all(len(line) == 5 for line in out["lines"])


def test_floor_solve_mirrors_corners_going_round_the_other_way():
    out = json.loads(server.ps_floor_solve(W, H, list(reversed(CORNERS)), EDGES))
    assert out["ok"] and out["mirrored"] and out["residualPx"] < 0.5


def test_floor_solve_says_why_when_it_cannot():
    flat = [{"u": 500, "v": 300}, {"u": 1500, "v": 300}, {"u": 1500, "v": 800}, {"u": 500, "v": 800}]
    out = json.loads(server.ps_floor_solve(W, H, flat, EDGES))
    assert out == {"ok": False, "reason": "flat"}
    bad = json.loads(server.ps_floor_solve(W, H, CORNERS[:3], EDGES))
    assert bad["ok"] is False and bad["reason"].startswith("bad-input")


def test_fit_floor_gives_the_screen_its_state(env):
    state = json.loads(server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS))
    assert (state["width"], state["height"]) == (W, H)
    assert state["edgesM"] == EDGES and state["corners"] == CORNERS
    assert state["siteDir"] == env["site"] and state["imagePath"] == env["image"]
    assert state["objects"] == [] and state["parts"] >= 1
    # without corners it proposes a default four inside the photo
    d = json.loads(server.ps_fit_floor(env["image"], EDGES, env["site"]))
    assert len(d["corners"]) == 4 and all(0 < c["u"] < W and 0 < c["v"] < H for c in d["corners"])


def test_photo_parts_are_chunks_that_join_into_the_display_copy(env):
    state = json.loads(server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS))
    parts = [json.loads(server.ps_spike_image(env["image"], i))["data"] for i in range(state["parts"])]
    assert all(len(p) <= server.CHUNK for p in parts[:-1]) and all(parts)
    assert json.loads(server.ps_spike_image(env["image"], state["parts"]))["data"] == ""
    jpg = cv2.imdecode(np.frombuffer(base64.b64decode("".join(parts)), np.uint8), cv2.IMREAD_COLOR)
    assert jpg.shape[1] == server.MAX_SIDE and abs(jpg.shape[0] - round(H * server.MAX_SIDE / W)) <= 1


def test_the_display_copy_goes_to_the_cache_dir_and_follows_exif_rotation(env, tmp_path):
    from PIL import Image
    path = tmp_path / "phone.jpg"
    im = Image.new("RGB", (200, 100), (10, 120, 200))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees: it is seen as 100 wide, 200 tall
    im.save(path, exif=exif)
    state = json.loads(server.ps_fit_floor(str(path), EDGES, env["site"]))
    assert (state["width"], state["height"]) == (100, 200)
    assert list((env["cache"] / "photos").glob("*.jpg"))


# ---- the objects ----

def test_propose_cuts_out_named_objects_and_shows_their_outlines(env, monkeypatch):
    fake_segmenter(monkeypatch, {"pole", "gate"})
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    out = json.loads(server.ps_objects_propose(env["image"], boxes("pole", "gate", "cone"), env["site"]))
    assert [o["name"] for o in out["objects"]] == ["pole", "gate"] and out["missing"] == ["cone"]
    pole = out["objects"][0]
    # the model reads names, kinds and boxes; the outlines and ground points go to the screen by key
    assert pole == {"name": "pole", "kind": "fixed", "box": [300, 500, 420, 700]}
    outline = shapes(out)["pole"]
    assert len(outline["outline"]) >= 3 and all(isinstance(x, float) for p in outline["outline"] for x in p)
    assert outline["ground_px"][1] == [359.5, 699.0]
    # the screen's state comes with it: the floor from the earlier ps_fit_floor
    assert out["corners"] == CORNERS and out["edgesM"] == EDGES and out["siteDir"] == env["site"]
    # kept on disk, so that ps_fit_floor (any start of the server) shows them
    assert (server._work(env["site"]) / "proposal.json").is_file()
    again = json.loads(server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS))
    assert [o["name"] for o in again["objects"]] == ["pole", "gate"]


def test_a_second_propose_adds_to_the_first(env, monkeypatch):
    fake_segmenter(monkeypatch, {"pole", "cone"})
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    server.ps_objects_propose(env["image"], boxes("pole"), env["site"])
    out = json.loads(server.ps_objects_propose(env["image"], boxes("cone"), env["site"]))
    assert [o["name"] for o in out["objects"]] == ["pole", "cone"]


def test_an_outline_is_simplified_to_at_most_200_points(env, monkeypatch):
    def run_(image, boxes_):
        m = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(m, (1000, 550), 500, 1, -1)
        return {"pond": m.astype(bool)}
    monkeypatch.setattr(server, "make_segmenter", lambda: run_)
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    out = json.loads(server.ps_objects_propose(
        env["image"], [{"name": "pond", "kind": "fixed", "x0": 500, "y0": 50, "x1": 1500, "y1": 1050}], env["site"]))
    assert 3 <= len(shapes(out)["pond"]["outline"]) <= server.MAX_OUTLINE_POINTS


def test_the_models_result_holds_no_outline_and_the_screens_part_tool_is_key_guarded(env, monkeypatch):
    fake_segmenter(monkeypatch, {"pole", "gate"})
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    text_ = server.ps_objects_propose(env["image"], boxes("pole", "gate"), env["site"])
    assert "outline" not in text_ and "ground_px" not in text_
    again = json.loads(server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS))
    assert "outline" not in json.dumps(again) and set(shapes(again)) == {"pole", "gate"}
    empty = json.loads(server.ps_fit_floor(env["image"], EDGES, str(env["site"]) + "-other", CORNERS))
    assert empty["objects"] == [] and empty["objectsKey"] is None and empty["objectsParts"] == 0
    for key in ("../proposal", "", "g" * 40):
        with pytest.raises(ValueError, match="bad data key"):
            server.ps_objects_part(key, 0)
    with pytest.raises(ValueError, match="part"):
        server.ps_objects_part(again["objectsKey"], -1)
    with pytest.raises(ValueError, match="no data"):
        server.ps_objects_part("0" * 40, 0)


def test_propose_without_a_floor_asks_for_the_lengths(env, monkeypatch):
    fake_segmenter(monkeypatch, {"pole"})
    with pytest.raises(ValueError, match="edge lengths"):
        server.ps_objects_propose(env["image"], boxes("pole"), env["site"])
    out = json.loads(server.ps_objects_propose(env["image"], boxes("pole"), env["site"], edgesM=EDGES))
    assert out["edgesM"] == EDGES and len(out["corners"]) == 4


# ---- deciding ----

def propose(env, monkeypatch, *names):
    fake_segmenter(monkeypatch, set(names))
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    server.ps_objects_propose(env["image"], boxes(*names), env["site"])


def test_decide_writes_a_site_a_person_confirmed(env, monkeypatch):
    propose(env, monkeypatch, "pole", "gate")
    out = json.loads(server.ps_site_decide(
        env["site"], env["image"], CORNERS, EDGES, [{"name": "pole", "kind": "fixed"}], False))
    assert out["ok"] and out["residualPx"] < 0.5 and out["mirrored"] is False
    assert out["sitePath"].endswith("site.json")
    site = load_site(out["sitePath"])
    assert site.confirmed_by_person and site.dimensions == "estimated"
    assert site.floor.camera == "camera" and site.floor.edges_m == (3.5, 6.8)
    assert site.residual_px["camera"] < 0.5
    assert [o.name for o in site.objects] == ["pole"]  # the person kept only the pole
    pole = site.objects[0]
    assert pole.kind == "fixed" and pole.views["camera"].box == (300, 500, 420, 700)
    assert pole.ground_m is not None and abs(pole.ground_m[2]) < 1e-9  # located on the floor
    assert site.history and "at" in site.history[0] and "what" in site.history[0]


def test_the_camera_name_given_to_fit_floor_is_the_one_the_site_gets(env, monkeypatch):
    fake_segmenter(monkeypatch, {"pole"})
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS, camera="side")
    server.ps_objects_propose(env["image"], boxes("pole"), env["site"])  # keeps the name it found
    out = json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, [{"name": "pole"}], False))
    assert out["ok"] and out["camera"] == "side"
    site = load_site(out["sitePath"])
    assert set(site.cameras) == {"side"} and site.floor.camera == "side" and set(site.residual_px) == {"side"}
    assert set(site.objects[0].views) == {"side"}
    assert json.loads(server.ps_site_status(env["site"]))["residualPx"] < 0.5
    # the decide call may name it too, and a bad name is refused with a reason
    out = json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, [], False, camera="front"))
    assert set(load_site(out["sitePath"]).cameras) == {"front"}
    bad = json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, [], False, camera=" "))
    assert bad["ok"] is False and "camera" in bad["reason"]
    with pytest.raises(ValueError, match="camera"):
        server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS, camera="")


def test_decide_marks_measured_lengths(env, monkeypatch):
    propose(env, monkeypatch, "pole")
    out = json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, [], True))
    assert load_site(out["sitePath"]).dimensions == "measured"


def test_decide_refuses_with_a_reason(env, monkeypatch):
    propose(env, monkeypatch, "pole")
    flat = [{"u": 500, "v": 300}, {"u": 1500, "v": 300}, {"u": 1500, "v": 800}, {"u": 500, "v": 800}]
    out = json.loads(server.ps_site_decide(env["site"], env["image"], flat, EDGES, [], False))
    assert out == {"ok": False, "reason": "flat"}
    out = json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, [{"name": "ghost"}], False))
    assert out["ok"] is False and "ghost" in out["reason"]
    assert not (server.Path(env["site"]) / "site.json").exists()


def test_decide_with_corners_going_round_the_other_way_says_so(env, monkeypatch):
    propose(env, monkeypatch, "pole")
    out = json.loads(server.ps_site_decide(env["site"], env["image"], list(reversed(CORNERS)), EDGES, [], False))
    assert out["ok"] and out["mirrored"] is True


def test_status_before_and_after(env, monkeypatch):
    before = json.loads(server.ps_site_status(env["site"]))
    assert before == {"decided": False, "residualPx": None, "objects": [], "dimensions": None,
                      "confirmedByPerson": False}
    propose(env, monkeypatch, "pole", "gate")
    server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES,
                          [{"name": "pole"}, {"name": "gate"}], True)
    after = json.loads(server.ps_site_status(env["site"]))
    assert after["decided"] and after["confirmedByPerson"] and after["objects"] == ["pole", "gate"]
    assert after["dimensions"] == "measured" and after["residualPx"] < 0.5


# ---- nothing kept in memory ----

def test_a_separate_start_answers_from_what_the_first_left_on_disk(env, monkeypatch):
    """The host may answer the screen's calls from another start of the server: start twice, separately."""
    propose(env, monkeypatch, "pole")  # in this process: the proposal goes to siteDir/.work

    async def first():
        async with started(env) as client:
            return json.loads(text(await client.call_tool(
                "ps_fit_floor", {"imagePath": env["image"], "edgesM": EDGES, "siteDir": env["site"],
                                 "corners": CORNERS})))

    async def second(parts):
        async with started(env) as client:  # a new process
            data = ""
            for i in range(parts):
                data += json.loads(text(await client.call_tool(
                    "ps_spike_image", {"imagePath": env["image"], "part": i})))["data"]
            outlines = json.loads(text(await client.call_tool(
                "ps_objects_part", {"key": state["objectsKey"], "part": 0})))["data"]
            assert "pole" in outlines and "outline" in outlines
            decided = json.loads(text(await client.call_tool("ps_site_decide", {
                "siteDir": env["site"], "imagePath": env["image"], "corners": CORNERS, "edgesM": EDGES,
                "objects": [{"name": "pole", "kind": "fixed"}], "measured": False})))
            return data, decided

    state = run(first())
    assert [o["name"] for o in state["objects"]] == ["pole"]
    data, decided = run(second(state["parts"]))
    assert cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR) is not None
    assert decided["ok"] and load_site(decided["sitePath"]).objects[0].name == "pole"


# ---- libraries that print to stdout must not corrupt the protocol ----

CHATTY_SERVER = '''
import logging, os, sys
import numpy as np
from physical_sandbox.mcp import server

# like ultralytics: a logger whose StreamHandler holds the original sys.stdout
log = logging.getLogger("chatty")
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))


def segmenter():
    print("Downloading weights ...")
    log.info("weights saved")
    os.write(1, b"raw write on fd 1\\n")
    sys.stdout.flush()

    def run(image, boxes):
        print("settings created")
        log.info("predicting")
        out = {}
        for name, (x0, y0, x1, y1) in boxes.items():
            m = np.zeros(image.shape[:2], bool)
            m[y0:y1, x0:x1] = True
            out[name] = m
        return out
    return run


server.make_segmenter = segmenter
server.main()
'''


def test_library_output_on_stdout_does_not_corrupt_the_protocol(env):
    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-c", CHATTY_SERVER],
                                       env={"PS_CACHE_DIR": str(env["cache"])})
        async with Client(params) as client:
            await client.call_tool("ps_fit_floor", {"imagePath": env["image"], "edgesM": EDGES,
                                                    "siteDir": env["site"], "corners": CORNERS})
            proposed = json.loads(text(await client.call_tool(
                "ps_objects_propose", {"imagePath": env["image"], "boxes": boxes("pole"), "siteDir": env["site"]})))
            status = json.loads(text(await client.call_tool("ps_site_status", {"siteDir": env["site"]})))
            decided = json.loads(text(await client.call_tool("ps_site_decide", {
                "siteDir": env["site"], "imagePath": env["image"], "corners": CORNERS, "edgesM": EDGES,
                "objects": [{"name": "pole"}], "measured": False})))
            return proposed, status, decided
    proposed, status, decided = run(go())
    assert [o["name"] for o in proposed["objects"]] == ["pole"]
    assert status["decided"] is False and decided["ok"]


def _raw_session(env, script, calls):
    """Speak JSON-RPC to a server over pipes, wait for every answer, and give back its raw stdout lines and its
    stderr (the Client would swallow a stray line, so this checks what a stricter host would see)."""
    import os
    import subprocess
    import threading
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ] + [{"jsonrpc": "2.0", "id": i + 2, "method": "tools/call", "params": {"name": n, "arguments": a}}
         for i, (n, a) in enumerate(calls)]
    proc = subprocess.Popen([sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env={**os.environ, "PS_CACHE_DIR": str(env["cache"])})
    timer = threading.Timer(60, proc.kill)  # never hang the suite
    timer.start()
    try:
        for m in messages:
            proc.stdin.write(json.dumps(m) + "\n")
        proc.stdin.flush()
        lines, answered = [], 0
        while answered < 1 + len(calls):  # read until every request has been answered
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line.rstrip("\n"))
            try:
                answered += "id" in json.loads(line)
            except ValueError:
                pass  # a stray line: the test below fails on it
        proc.stdin.close()
        proc.wait(timeout=30)
        return lines, proc.stderr.read()
    finally:
        timer.cancel()
        proc.kill()


def test_stdout_carries_only_protocol_lines_while_a_library_prints(env):
    lines, err = _raw_session(env, CHATTY_SERVER, [
        ("ps_objects_propose", {"imagePath": env["image"], "boxes": boxes("pole"), "siteDir": env["site"],
                                "edgesM": EDGES, "corners": CORNERS})])
    parsed = [json.loads(line) for line in lines]  # every line is JSON, or this raises
    assert all(m.get("jsonrpc") == "2.0" for m in parsed)
    answer = next(m for m in parsed if m.get("id") == 2)
    assert not answer["result"].get("isError")
    assert "predicting" in err and "raw write on fd 1" in err  # it went to stderr instead


# ---- hardening ----

def test_decide_answers_with_a_reason_for_malformed_input(env, monkeypatch):
    propose(env, monkeypatch, "pole")
    decide = lambda objs: json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, objs, False))
    assert decide(["pole"])["ok"] is False  # not dicts
    twice = decide([{"name": "pole"}, {"name": "pole"}])
    assert twice["ok"] is False and "twice" in twice["reason"]
    # a damaged proposal file loses its damaged entries instead of raising
    path = server._work(env["site"]) / "proposal.json"
    d = json.loads(path.read_text())
    d["objects"] = [{"name": "pole"}, "junk"]
    path.write_text(json.dumps(d))
    out = decide([{"name": "pole"}])
    assert out["ok"] is False and "pole" in out["reason"]
    path.write_text("not json")
    assert decide([{"name": "pole"}])["ok"] is False
    assert not (server.Path(env["site"]) / "site.json").exists()


def test_decide_needs_the_picture_to_have_been_opened_for_this_site(env):
    out = json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, [], False))
    assert out["ok"] is False and "ps_fit_floor" in out["reason"]
    assert not (server.Path(env["site"]) / "site.json").exists()


def test_a_negative_photo_part_is_refused(env):
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    with pytest.raises(ValueError, match="part"):
        server.ps_spike_image(env["image"], -1)


def test_the_screen_cannot_make_display_copies(env):
    with pytest.raises(ValueError, match="ps_fit_floor"):
        server.ps_spike_image(env["image"], 0)
    assert not list((env["cache"] / "photos").glob("*"))
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    assert json.loads(server.ps_spike_image(env["image"], 0))["data"]


def test_proposals_match_on_the_resolved_path(env, monkeypatch):
    from pathlib import Path
    other = f"{Path(env['image']).parent}/./{Path(env['image']).name}"  # the same file, spelled differently
    assert other != env["image"]
    fake_segmenter(monkeypatch, {"pole"})
    server.ps_fit_floor(env["image"], EDGES, env["site"], CORNERS)
    out = json.loads(server.ps_objects_propose(other, boxes("pole"), env["site"]))  # finds the floor of the first
    assert out["edgesM"] == EDGES and out["corners"] == CORNERS
    assert [o["name"] for o in json.loads(server.ps_fit_floor(env["image"], EDGES, env["site"]))["objects"]] == ["pole"]
    decided = json.loads(server.ps_site_decide(env["site"], other, CORNERS, EDGES, [{"name": "pole"}], False))
    assert decided["ok"]


def test_the_site_history_keeps_only_the_photos_file_name(env, monkeypatch):
    propose(env, monkeypatch, "pole")
    out = json.loads(server.ps_site_decide(env["site"], env["image"], CORNERS, EDGES, [{"name": "pole"}], False))
    entry = load_site(out["sitePath"]).history[0]
    assert entry["image"] == "still.png"
    assert env["image"] not in json.dumps(entry)


def test_propose_decodes_the_picture_once(env, monkeypatch):
    fake_segmenter(monkeypatch, {"pole"})
    opened = []
    real = server.open_source
    monkeypatch.setattr(server, "open_source", lambda spec: opened.append(spec) or real(spec))
    server.ps_objects_propose(env["image"], boxes("pole"), env["site"], edgesM=EDGES)
    assert len(opened) == 1  # the display copy is made from the frame the segmenter got
