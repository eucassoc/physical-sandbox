"""The in-chat screen to compare two motions: ps_compare, its model parts, and the packing of the shapes.

A synthetic site, scene and records (a tiny URDF of boxes) and a still stand in for real data; the SO-101
test runs only when its URDF is on this machine.
"""
import asyncio
import base64
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.client.stdio import StdioServerParameters  # noqa: E402

from physical_sandbox.camera import Camera  # noqa: E402
from physical_sandbox.machine import load_machine  # noqa: E402
from physical_sandbox.mcp import pack, server  # noqa: E402
from physical_sandbox.motion import motion_from_numpy, save_record  # noqa: E402
from physical_sandbox.scene import PlacedMachine, Scene, save_scene  # noqa: E402
from physical_sandbox.site import Floor, Site, save_site  # noqa: E402

URDF = """<?xml version="1.0"?>
<robot name="blocks">
  <link name="base">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0 0 0.15"/><geometry><box size="0.3 0.3 0.3"/></geometry></visual>
  </link>
  <link name="arm">
    <inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.2 0 0"/><geometry><box size="0.4 0.1 0.1"/></geometry></visual>
    <visual><origin xyz="0.4 0 0"/><geometry><sphere radius="0.06"/></geometry></visual>
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
SO101 = Path(os.environ.get("PS_SO101_URDF", "~/work/physical-sandbox-proto/robots/so101/so101.urdf")).expanduser()


def look_at(eye, target=(0, 0, 0.2), size=(W, H), f=300.0, dist=None):
    eye, target = np.array(eye, float), np.array(target, float)
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.array([right, down, fwd])
    K = [[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]]
    return Camera(size[0], size[1], K, np.zeros(5) if dist is None else dist, R, -R @ eye)


def motion(end, label, fps=10.0, n=11):
    q = np.linspace(0, end, n).reshape(-1, 1)
    return motion_from_numpy(q, fps=fps, joint_names=["turn"], machine="blocks", label=label)


def make_world(tmp_path, cam=None, floor=True):
    urdf = tmp_path / "blocks.urdf"
    urdf.write_text(URDF)
    corners_m = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    site = Site(id="yard", cameras={"camera": cam or look_at((-2.5, 0.8, 1.2))}, residual_px={"camera": 0.7},
                floor=Floor("camera", [(0, 0)] * 4, (2.0, 2.0), corners_m) if floor else None,
                confirmed_by_person=True, dimensions="estimated")
    save_site(site, tmp_path / "site.json")
    placed = PlacedMachine(machine=load_machine(urdf), urdf=str(urdf), x=0.1, y=0.0, yaw=0.3)
    save_scene(Scene(site=site, machines={"blocks": placed}), tmp_path / "scene.json")
    save_record([motion(1.0, "slow")], "yard", tmp_path / "a.record.json")
    save_record([motion(1.5, "wide")], "yard", tmp_path / "b.record.json")
    still = tmp_path / "grey.png"
    cv2.imwrite(str(still), np.full((H, W, 3), 128, np.uint8))
    return site, placed, still


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    site, placed, still = make_world(tmp_path)
    return {"dir": tmp_path, "site": site, "placed": placed, "still": still, "cache": tmp_path / "cache"}


def args(w, **over):
    d = dict(siteDir=str(w["dir"]), scenePath=str(w["dir"] / "scene.json"), camera="camera",
             a=str(w["dir"] / "a.record.json"), b=str(w["dir"] / "b.record.json"), video=str(w["still"]))
    d.update(over)
    return d


def data_of(out):
    """What the screen fetches with ps_compare_part, as one dict."""
    text = "".join(json.loads(server.ps_compare_part(out["dataKey"], i))["data"] for i in range(out["dataParts"]))
    return json.loads(text)


def merged(out):
    """The answer with the screen's data put back in (as the screen does): poses, times, pixel paths, the series."""
    part = data_of(out)
    out = dict(out, poses=part["poses"], t=part["t"])
    out["figures"] = dict(out["figures"], series=part["series"])
    out["records"] = [{**r, "devices": [{**d, **pd} for d, pd in zip(r["devices"], p["devices"])]}
                      for r, p in zip(out["records"], part["records"])]
    return out


def compare(w, **over):
    return merged(json.loads(server.ps_compare(**args(w, **over))))


def raw_compare(w, **over):
    return server.ps_compare(**args(w, **over))


def text(result):
    assert not result.is_error, result.content
    return result.content[0].text


@asynccontextmanager
async def started(w):
    params = StdioServerParameters(command=sys.executable, args=["-m", "physical_sandbox.mcp.server"],
                                   env={"PS_CACHE_DIR": str(w["cache"])})
    async with Client(params) as client:
        yield client


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 120))


# ---- what the host sees ----

def test_the_compare_tool_opens_the_screen_and_the_model_part_tool_is_the_screens_own(world):
    async def go():
        async with started(world) as client:
            return {t.name: t.model_dump(by_alias=True) for t in (await client.list_tools()).tools}
    tools = run(go())
    assert tools["ps_compare"]["_meta"]["ui"]["resourceUri"] == "ui://physical-sandbox/compare.html"
    assert "model" in (tools["ps_compare"]["_meta"]["ui"].get("visibility") or ["model"])
    assert tools["ps_model_part"]["_meta"]["ui"]["resourceUri"] == "ui://physical-sandbox/compare.html"
    assert tools["ps_model_part"]["_meta"]["ui"]["visibility"] == ["app"]
    assert tools["ps_compare_part"]["_meta"]["ui"]["resourceUri"] == "ui://physical-sandbox/compare.html"
    assert tools["ps_compare_part"]["_meta"]["ui"]["visibility"] == ["app"]
    for name in ("ps_compare", "ps_model_part", "ps_compare_part"):
        schema = json.dumps(tools[name]["inputSchema"])
        assert "$ref" not in schema and "prefixItems" not in schema, name
    required = set(tools["ps_compare"]["inputSchema"]["required"])
    assert required == {"siteDir", "scenePath", "camera", "a", "b"}


def test_the_compare_screen_is_served_with_the_bridge_and_three_inlined(world):
    async def go():
        async with started(world) as client:
            return await client.read_resource("ui://physical-sandbox/compare.html")
    content = run(go()).contents[0]
    assert content.mime_type == "text/html;profile=mcp-app"
    assert "globalThis.ExtApps=" in content.text
    assert "WebGLRenderer" in content.text and "globalThis.THREE" in content.text
    assert server.BUNDLE_MARKER not in content.text and server.THREE_MARKER not in content.text
    assert all(n in content.text for n in ("ps_model_part", "ps_compare_part", "ps_spike_image"))


def test_three_ships_as_package_data():
    assert (Path(server.__file__).parent / "widgets" / "vendor" / "three.iife.js").is_file()
    assert (Path(server.__file__).parent / "widgets" / "compare.html").is_file()


# ---- what ps_compare answers ----

def test_the_answer_has_one_pose_set_per_frame_and_record(world):
    out = compare(world)
    n = len(out["figures"]["series"]["t"])
    assert out["frames"] == n and out["stride"] == 1
    assert out["stepS"] == pytest.approx(out["figures"]["step_s"])
    assert len(out["poses"]) == 2
    links = out["model"]["links"]
    assert [l["name"] for l in links] == ["base", "arm"]  # the tip has no shape: not sent
    for r in range(2):
        assert len(out["poses"][r]) == out["frames"]
        for frame in out["poses"][r]:
            assert len(frame) == len(links) and all(len(p) == 16 for p in frame)
    assert [r["label"] for r in out["records"]] == ["slow", "wide"]
    for rec in out["records"]:
        assert [d["id"] for d in rec["devices"]] == ["blocks"]
        dev = rec["devices"][0]
        assert len(dev["centers"]) == out["frames"] and len(dev["tool_trail"]) >= 2


def test_a_pose_is_the_link_in_the_site_frame(world):
    out = compare(world)
    placed, a = world["placed"], world["site"]
    motions = {m.label: m for m in (motion(1.0, "slow"), motion(1.5, "wide"))}
    T = placed.base_transform()
    for r, label in enumerate(("slow", "wide")):
        for k in (0, 7, out["frames"] - 1):
            t = out["figures"]["series"]["t"][k]
            want = T @ placed.machine.link_poses(motions[label].at(t))["arm"]
            got = np.array(out["poses"][r][k][1]).reshape(4, 4)
            assert np.allclose(got, want, atol=2e-5)
    # a machine placed at (0.1, 0) turned 0.3 rad: the base link sits there
    base = np.array(out["poses"][0][0][0]).reshape(4, 4)
    assert np.allclose(base[:3, 3], [0.1, 0.0, 0.0], atol=1e-5)
    assert np.allclose(base[:2, :2], [[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]], atol=1e-5)


def test_the_tool_tip_is_projected_through_the_pinhole_camera(world):
    out = compare(world)
    cam = world["site"].cameras["camera"]
    k = out["frames"] - 1
    t = out["figures"]["series"]["t"][k]
    placed = world["placed"]
    tip = (placed.base_transform() @ placed.machine.link_poses(motion(1.5, "wide").at(t))["tip"])[:3, 3]
    u, v = cam.project(tip[None])[0]
    assert out["records"][1]["devices"][0]["centers"][k] == pytest.approx([u, v], abs=0.01)
    cam_json = out["camera"]
    assert cam_json["intr"] == {"width": W, "height": H, "fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0}
    assert np.allclose(cam_json["extr"]["R"], cam.R.reshape(-1)) and np.allclose(cam_json["extr"]["t"], cam.t)


def test_the_floor_is_a_box_in_pixels_and_absent_without_a_floor(world, tmp_path, monkeypatch):
    out = compare(world)
    assert len(out["floor"]) == 4 and all(len(p) == 2 for p in out["floor"])
    other = tmp_path / "nofloor"
    other.mkdir()
    make_world(other, floor=False)
    w2 = {"dir": other, "still": other / "grey.png"}
    assert compare(w2)["floor"] is None


def test_figures_are_the_comparisons_numbers_and_the_site_quality_comes_with_them(world):
    out = compare(world)
    f = out["figures"]
    assert f["labels"] == {"a": "slow", "b": "wide"}
    assert f["tool_deviation_m"]["max"] > 0 and set(f["joints"]) == {"turn"}
    assert set(f["unknown"]) >= {"clearance", "reach", "world"} and all(f["unknown_reasons"][k] for k in f["unknown"])
    assert out["site"] == {"site": "yard", "residualPx": {"camera": 0.7}, "dimensions": "estimated",
                           "dimensionsNote": "寸法は推定（未確認）", "confirmedByPerson": True}


def test_grip_threshold_is_passed_on_and_must_be_finite(world, monkeypatch):
    import physical_sandbox.compare as cmp
    seen = []
    real = cmp.compare
    monkeypatch.setattr(cmp, "compare", lambda *a, **k: seen.append(k.get("grip_threshold")) or real(*a, **k))
    out = compare(world, gripThreshold=0.2)
    assert seen == [0.2] and "grip_threshold" in out["figures"]
    with pytest.raises(ValueError, match="gripThreshold"):
        compare(world, gripThreshold=float("nan"))


def test_labels_can_be_given(world):
    out = compare(world, labelA="手本", labelB="試し")
    assert [r["label"] for r in out["records"]] == ["手本", "試し"]
    assert out["figures"]["labels"] == {"a": "手本", "b": "試し"}


def test_the_still_is_a_display_copy_the_screen_can_fetch(world):
    out = compare(world)
    assert (out["width"], out["height"]) == (W, H) and Path(out["imagePath"]).is_file()
    assert Path(world["cache"]).resolve() in Path(out["imagePath"]).resolve().parents
    data = "".join(json.loads(server.ps_spike_image(out["imagePath"], i))["data"] for i in range(out["parts"]))
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (H, W) and abs(int(img[5, 5, 0]) - 128) < 6


def test_without_a_picture_it_asks_for_one(world):
    with pytest.raises(ValueError, match="video"):
        compare(world, video=None)


def test_the_scene_may_name_its_picture(world):
    p = world["dir"] / "scene.json"
    d = json.loads(p.read_text())
    d["picture"] = "grey.png"
    p.write_text(json.dumps(d))
    out = compare(world, video=None)
    assert (out["width"], out["height"]) == (W, H)


def test_bad_input_is_named(world):
    with pytest.raises(ValueError, match="camera 'ghost'"):
        compare(world, camera="ghost")
    with pytest.raises(ValueError, match="gone.record.json"):
        compare(world, a=str(world["dir"] / "gone.record.json"))
    with pytest.raises(ValueError, match="site"):
        compare(world, siteDir=str(world["dir"] / "nowhere"))


def test_a_lens_with_distortion_gives_an_undistorted_picture_and_a_pinhole_camera(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    dist = np.array([-0.3, 0.1, 0, 0, 0])
    make_world(tmp_path, cam=look_at((-2.5, 0.8, 1.2), dist=dist))
    img = np.zeros((H, W, 3), np.uint8)
    img[::8] = 255
    cv2.imwrite(str(tmp_path / "grid.png"), img)
    w = {"dir": tmp_path, "still": tmp_path / "grid.png"}
    out = compare(w)
    saved = cv2.imread(out["imagePath"])
    assert saved.shape == img.shape and np.abs(saved.astype(int) - img.astype(int)).mean() > 3  # it was warped
    assert "dist" not in json.dumps(out["camera"])
    want = look_at((-2.5, 0.8, 1.2))  # the same pose without the lens
    tip_site = world_tip(tmp_path, out)
    assert out["records"][0]["devices"][0]["centers"][-1] == pytest.approx(
        want.project(tip_site[None])[0].tolist(), abs=0.01)


def world_tip(tmp_path, out):
    placed = PlacedMachine(machine=load_machine(tmp_path / "blocks.urdf"), urdf=str(tmp_path / "blocks.urdf"),
                           x=0.1, y=0.0, yaw=0.3)
    t = out["figures"]["series"]["t"][-1]
    return (placed.base_transform() @ placed.machine.link_poses(motion(1.0, "slow").at(t))["tip"])[:3, 3]


def test_a_picture_of_another_shape_is_refused(world, tmp_path):
    odd = tmp_path / "odd.png"
    cv2.imwrite(str(odd), np.zeros((100, 300, 3), np.uint8))
    with pytest.raises(ValueError, match="aspect"):
        compare(world, video=str(odd))


# ---- LeRobot A without a video: the episode's own recording at its start ----

def test_a_lerobot_episode_gives_its_own_recording_frame_at_the_episodes_start(world, monkeypatch):
    tmp_path = world["dir"]
    data = tmp_path / "ds"
    (data / "meta").mkdir(parents=True)
    (data / "meta" / "info.json").write_text("{}")
    import physical_sandbox.lerobot as lerobot
    import physical_sandbox.motion as motion_mod
    monkeypatch.setattr(motion_mod, "motion_from_lerobot",
                        lambda d, ep, *, machine, machine_name, **k: motion(1.0, f"episode {ep}"))
    video = tmp_path / "ds.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (W, H))
    for k in range(60):
        w.write(np.full((H, W, 3), 4 * k, np.uint8))  # frame k is grey 4k
    w.release()
    monkeypatch.setattr(lerobot, "lerobot_video", lambda d, ep, cam: (video, 1.0, 1.5))
    out = compare(world, a=f"lerobot:{data}#4", video=None)
    assert out["records"][0]["label"] == "episode 4"
    saved = cv2.imread(out["imagePath"])
    assert abs(int(saved[2, 2, 0]) - 120) < 16  # file time 1.0 s = frame 30 = grey 120; frame 0 would be grey 0


def test_video_camera_names_the_datasets_camera_apart_from_the_site_camera(world, monkeypatch):
    """A site fixed in the chat screen calls its camera 'camera'; the dataset's camera is 'side'."""
    import physical_sandbox.lerobot as lerobot
    from mini_lerobot import make_dataset
    tmp_path = world["dir"]
    monkeypatch.setitem(lerobot.JOINT_MAP, "turn", (-100.0, 100.0, 1))
    ds = make_dataset(tmp_path / "ds", ["turn"], [np.linspace(0, 50, 9).reshape(-1, 1),
                                                  np.linspace(0, -50, 12).reshape(-1, 1)], size=(W, H))
    with pytest.raises(ValueError, match="videoCamera"):  # the dataset has no camera 'camera'
        compare(world, a=f"lerobot:{ds}#1", b=f"lerobot:{ds}#0", video=None)
    out = compare(world, a=f"lerobot:{ds}#1", b=f"lerobot:{ds}#0", video=None, videoCamera="side")
    assert abs(int(cv2.imread(out["imagePath"])[2, 2, 0]) - 36) < 12  # episode 1 starts at frame 9 (grey 36)
    # the episode's own file named as the video: the still is still at the episode's start
    own = ds / "videos" / "observation.images.side" / "chunk-000" / "file-000.mp4"
    out = compare(world, a=f"lerobot:{ds}#1", b=f"lerobot:{ds}#0", video=str(own))
    assert abs(int(cv2.imread(out["imagePath"])[2, 2, 0]) - 36) < 12


# ---- two separate starts of the server ----

def test_a_separate_start_answers_model_and_picture_parts_from_the_disk(world):
    async def first():
        async with started(world) as client:
            return json.loads(text(await client.call_tool("ps_compare", args(world))))

    async def second(out):
        async with started(world) as client:  # a new process: nothing in its memory
            model = ""
            for i in range(out["model"]["binParts"]):
                part = json.loads(text(await client.call_tool("ps_model_part", {"key": out["model"]["key"], "part": i})))
                assert part["part"] == i
                model += part["data"]
            past = json.loads(text(await client.call_tool(
                "ps_model_part", {"key": out["model"]["key"], "part": out["model"]["binParts"]})))
            data = ""
            for i in range(out["dataParts"]):
                data += json.loads(text(await client.call_tool(
                    "ps_compare_part", {"key": out["dataKey"], "part": i})))["data"]
            image = ""
            for i in range(out["parts"]):
                image += json.loads(text(await client.call_tool(
                    "ps_spike_image", {"imagePath": out["imagePath"], "part": i})))["data"]
            return model, past, image, data

    out = run(first())
    model, past, image, data = run(second(out))
    assert past["data"] == ""
    part = json.loads(data)  # the second start served the same text the first one wrote
    assert len(part["poses"]) == 2 and len(part["poses"][0]) == out["frames"] and len(part["t"]) == out["frames"]
    assert all(len(r["devices"][0]["centers"]) == out["frames"] for r in part["records"])
    blob = base64.b64decode(model)
    links = out["model"]["links"]
    end = links[-1]["offset"] + 12 * links[-1]["vCount"] + 4 * links[-1]["iCount"]
    assert len(blob) == end
    for l in links:
        pos = np.frombuffer(blob, "<f4", l["vCount"] * 3, l["offset"]).reshape(-1, 3)
        idx = np.frombuffer(blob, "<u4", l["iCount"], l["offset"] + 12 * l["vCount"])
        assert idx.max() < l["vCount"] and np.isfinite(pos).all()
    assert cv2.imdecode(np.frombuffer(base64.b64decode(image), np.uint8), cv2.IMREAD_COLOR) is not None


def test_model_part_refuses_a_key_that_is_not_a_pack(world):
    for key in ("../../etc/passwd", "x" * 40, "0" * 40, ""):
        with pytest.raises(ValueError):
            server.ps_model_part(key, 0)
    out = json.loads(raw_compare(world))
    with pytest.raises(ValueError, match="part"):
        server.ps_model_part(out["model"]["key"], -1)
    for key in ("../../etc/passwd", "x" * 40, "0" * 40, "", out["model"]["key"][:39]):
        with pytest.raises(ValueError):
            server.ps_compare_part(key, 0)
    with pytest.raises(ValueError, match="part"):
        server.ps_compare_part(out["dataKey"], -1)
    assert json.loads(server.ps_compare_part(out["dataKey"], out["dataParts"]))["data"] == ""
    # a pack's key is not a comparison's: no file of that name there
    with pytest.raises(ValueError, match="no data"):
        server.ps_compare_part(out["model"]["key"], 0)


# ---- the size of the answer ----

def long_record(world, name="long.record.json"):
    d = world["dir"]
    long = motion_from_numpy(np.linspace(0, 2.0, 601).reshape(-1, 1), fps=30.0, joint_names=["turn"],
                             machine="blocks", label="long")
    save_record([long], "yard", d / name)
    return str(d / name)


def frames_for(n, stride):
    """The frames sent: every stride-th and, always, the last."""
    keep = list(range(0, n, stride))
    return len(keep) + (0 if keep[-1] == n - 1 else 1)


def test_the_text_the_model_reads_is_short_and_the_data_goes_through_parts(world):
    path = long_record(world)
    raw = raw_compare(world, a=path, b=path)
    assert len(raw) < 20_000  # a 20 s motion at 30 fps: no poses, no pixel paths, no series in it
    out = json.loads(raw)
    assert not {"poses", "t", "series"} & set(out) and "series" not in out["figures"]
    assert all("centers" not in d and "tool_trail" not in d for r in out["records"] for d in r["devices"])
    assert out["figures"]["unknown_reasons"] and out["site"]["dimensionsNote"] and out["dataParts"] > 1
    full = data_of(out)
    assert len(json.dumps(full)) < 2_000_000 and len(full["series"]["t"]) == 601  # the numbers' series is in the data
    assert out["stride"] == 1 and out["frames"] == len(full["poses"][0]) == len(full["t"]) == 601
    assert full["t"][-1] == pytest.approx(20.0) and full["t"][1] == pytest.approx(out["stepS"])
    # the same inputs are the same key, and the file is not written twice
    assert json.loads(raw_compare(world, a=path, b=path))["dataKey"] == out["dataKey"]


def test_a_big_answer_is_sampled_every_nth_frame_and_ends_on_the_last(world, monkeypatch):
    monkeypatch.setattr(server, "POSES_BUDGET", 20_000)
    path = long_record(world)
    out = merged(json.loads(raw_compare(world, a=path, b=path)))
    n = len(out["figures"]["series"]["t"])
    assert out["stride"] > 1 and n % out["stride"] != 0  # so the last frame is not on the stride
    assert out["frames"] == frames_for(n, out["stride"]) == len(out["poses"][0]) == len(out["t"])
    assert len(out["records"][0]["devices"][0]["centers"]) == out["frames"]
    assert out["t"][-1] == pytest.approx(out["figures"]["series"]["t"][-1])  # the end of the motion is there
    assert out["t"][1] == pytest.approx(out["stepS"])
    assert len(json.dumps(out["poses"])) < 40_000


# ---- the packing of shapes ----

def volume(v, f):
    """The enclosed volume by the divergence theorem: positive when the faces point outward."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6)


@pytest.mark.parametrize("name,t,size,want", [
    ("box", mujoco.mjtGeom.mjGEOM_BOX, [0.1, 0.2, 0.3], 8 * 0.1 * 0.2 * 0.3),
    ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE, [0.5, 0, 0], 4 / 3 * np.pi * 0.5 ** 3),
    ("cylinder", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.3, 0.5, 0], np.pi * 0.09 * 1.0),
    ("capsule", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.2, 0.4, 0], np.pi * 0.04 * 0.8 + 4 / 3 * np.pi * 0.2 ** 3),
    ("ellipsoid", mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.1, 0.2, 0.3], 4 / 3 * np.pi * 0.1 * 0.2 * 0.3),
])
def test_primitives_are_closed_shapes_facing_outward(name, t, size, want):
    v, f = pack._primitive(int(t), np.array(size, float))
    assert f.min() == 0 and f.max() == len(v) - 1
    assert 0.8 * want < volume(v, f) < 1.05 * want  # a tessellation is slightly inside the true shape


def write_sphere_obj(path, rings=60, segs=60):
    """A UV sphere whose every triangle has its own three vertices (as a mesh split at texture seams has)."""
    pts = lambda i, j: (np.sin(np.pi * i / rings) * np.cos(2 * np.pi * j / segs),  # noqa: E731
                        np.sin(np.pi * i / rings) * np.sin(2 * np.pi * j / segs), np.cos(np.pi * i / rings))
    lines, n = ["mtllib x.mtl", "usemtl m"], 0
    for i in range(rings):
        for j in range(segs):
            quad = [pts(i, j), pts(i + 1, j), pts(i + 1, j + 1), pts(i, j + 1)]
            for tri in ((0, 1, 2), (0, 2, 3)):
                for k in tri:
                    lines.append("v %.6f %.6f %.6f" % tuple(0.1 * c for c in quad[k]))
                lines.append(f"f {n + 1} {n + 2} {n + 3}")
                n += 3
    path.write_text("\n".join(lines) + "\n")
    return 2 * rings * segs


def test_a_dense_mesh_is_welded_and_thinned_and_the_pack_is_kept_on_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    faces = write_sphere_obj(tmp_path / "ball.obj")
    urdf = tmp_path / "ball.urdf"
    urdf.write_text('<robot name="ball"><link name="base"><inertial><mass value="1"/>'
                    '<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
                    '<visual><geometry><mesh filename="ball.obj"/></geometry></visual></link></robot>')
    machine = load_machine(urdf)
    made = pack.pack_model(machine, 20_000)
    (link,) = made["links"]
    assert link["name"] == "base"
    assert link["iCount"] // 3 <= faces / 4 and link["iCount"] // 3 >= 300  # about a ninth, never under 300
    assert link["vCount"] < faces  # welded: nowhere near three vertices a face
    blob = base64.b64decode("".join(pack.model_part(made["key"], i, 20_000) for i in range(made["binParts"])))
    pos = np.frombuffer(blob, "<f4", link["vCount"] * 3).reshape(-1, 3)
    r = np.linalg.norm(pos, axis=1)
    assert 0.09 < r.min() and r.max() < 0.11  # still a ball of radius 0.1
    files = sorted((tmp_path / "cache" / "models").iterdir())
    stamps = [f.stat().st_mtime_ns for f in files]
    again = pack.pack_model(load_machine(urdf), 20_000)
    assert again == made and [f.stat().st_mtime_ns for f in files] == stamps  # found, not made again
    # a different mesh is a different key
    write_sphere_obj(tmp_path / "ball.obj", rings=40, segs=40)
    assert pack.pack_model(load_machine(urdf), 20_000)["key"] != made["key"]


@pytest.mark.skipif(not SO101.exists(), reason="the SO-101 URDF is not on this machine (set PS_SO101_URDF)")
def test_so101_packs_into_a_screen_sized_model_and_a_twenty_second_pair_stays_short(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    machine = load_machine(SO101)
    made = pack.pack_model(machine, 20_000)
    assert len(made["links"]) >= 6
    total_faces = sum(l["iCount"] for l in made["links"]) // 3
    src = machine.model
    visual = [g for g in range(src.ngeom) if src.geom_group[g] == 1 and src.geom_type[g] == int(mujoco.mjtGeom.mjGEOM_MESH)]
    original = sum(int(src.mesh_facenum[src.geom_dataid[g]]) for g in visual)
    # Measured: 0.147 of MuJoCo's faces (about 1/7; 1/5 of the welded ones). The simplifier stalls there on the
    # motors' non-manifold meshes whatever its aggressiveness (7 to 12, up to 30 rounds, asking for less)
    print(f"SO-101 pack: {total_faces} of {original} faces = {total_faces / original:.3f}, {made['binParts']} parts")
    assert total_faces < 0.2 * original
    assert made["binParts"] < 120
    # a 20 s pair on a desk: the text the model reads stays short, whatever the robot
    urdf = SO101
    cam = look_at((-0.6, 0.5, 0.5), target=(0.1, 0, 0.15), size=(640, 480), f=500)
    site = Site(id="desk", cameras={"camera": cam}, residual_px={"camera": 0.5}, confirmed_by_person=False,
                dimensions="estimated")
    save_site(site, tmp_path / "site.json")
    placed = PlacedMachine(machine=machine, urdf=str(urdf), x=0.0, y=0.0, yaw=0.0)
    save_scene(Scene(site=site, machines={"so101": placed}), tmp_path / "scene.json")
    lo, hi = machine.limits[:, 0], machine.limits[:, 1]
    t = np.linspace(0, 2 * np.pi, 601)[:, None]
    for name, ph in (("a", 0.0), ("b", 0.25)):
        u = 0.5 + 0.35 * np.sin(t * np.arange(1, 7)[None] * 0.5 + ph)
        save_record([motion_from_numpy(lo + u * (hi - lo), fps=30.0, joint_names=machine.joint_names,
                                       machine="so101", label=name)], "desk", tmp_path / f"{name}.record.json")
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((480, 640, 3), 120, np.uint8))
    raw = server.ps_compare(siteDir=str(tmp_path), scenePath=str(tmp_path / "scene.json"), camera="camera",
                            a=str(tmp_path / "a.record.json"), b=str(tmp_path / "b.record.json"), video=str(still))
    out = json.loads(raw)
    print(f"SO-101 20 s pair: model-facing text {len(raw)} chars, data {len(json.dumps(data_of(out)))} chars "
          f"in {out['dataParts']} parts")
    assert len(raw) < 20_000 and out["frames"] == 601 and out["stride"] == 1
