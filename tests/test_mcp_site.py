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
                          "ps_site_decide", "ps_site_status"}
    for name in ("ps_fit_floor", "ps_spike_image", "ps_floor_solve", "ps_objects_propose", "ps_site_decide"):
        assert tools[name]["_meta"]["ui"]["resourceUri"] == "ui://physical-sandbox/fit-floor.html"
    for name in ("ps_spike_image", "ps_floor_solve", "ps_site_decide"):
        assert tools[name]["_meta"]["ui"]["visibility"] == ["app"]
    for name in ("ps_fit_floor", "ps_objects_propose"):
        assert "model" in (tools[name]["_meta"]["ui"].get("visibility") or ["model"])  # the agent can call it
    assert "ui" not in (tools["ps_site_status"].get("_meta") or {})


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


def test_the_screen_and_bridge_ship_as_package_data():
    import tomllib
    from pathlib import Path
    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    patterns = data["tool"]["setuptools"]["package-data"]["physical_sandbox.mcp"]
    assert "widgets/*.html" in patterns and "widgets/vendor/*.js" in patterns


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
    assert pole["kind"] == "fixed" and pole["box"] == [300, 500, 420, 700]
    assert len(pole["outline"]) >= 3 and all(isinstance(x, float) for p in pole["outline"] for x in p)
    assert pole["ground_px"][1] == [359.5, 699.0]
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
    assert 3 <= len(out["objects"][0]["outline"]) <= server.MAX_OUTLINE_POINTS


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
            decided = json.loads(text(await client.call_tool("ps_site_decide", {
                "siteDir": env["site"], "imagePath": env["image"], "corners": CORNERS, "edgesM": EDGES,
                "objects": [{"name": "pole", "kind": "fixed"}], "measured": False})))
            return data, decided

    state = run(first())
    assert [o["name"] for o in state["objects"]] == ["pole"]
    data, decided = run(second(state["parts"]))
    assert cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR) is not None
    assert decided["ok"] and load_site(decided["sitePath"]).objects[0].name == "pole"
