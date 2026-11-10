"""The floor-fitting screen in the browser: psbx site fit without --corners serves the widget on 127.0.0.1."""
import base64
import json
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from physical_sandbox import fit_screen
from physical_sandbox.cli import main
from physical_sandbox.fit_screen import BUNDLE_MARKER, FitServer
from physical_sandbox.site import load_site

W, H = 640, 480
EDGES = (4.0, 3.0)


def floor_corners(a=4.0, b=3.0, f=500.0, rvec=(1.9, 0.15, -0.1), tvec=(-1.5, 0.8, 9.0)):
    """Corners of an a x b m floor seen through a known pinhole pose, as the screen sends them."""
    world = np.array([[0, 0, 0], [a, 0, 0], [a, b, 0], [0, b, 0]], float)
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], float)
    px, _ = cv2.projectPoints(world, np.array(rvec, float), np.array(tvec, float), K, None)
    return [{"u": float(u), "v": float(v)} for u, v in px.reshape(-1, 2)]


CORNERS = floor_corners()


@pytest.fixture
def still(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    path = tmp_path / "still.png"
    cv2.imwrite(str(path), np.full((H, W, 3), 100, np.uint8))
    return path


@pytest.fixture
def served(still, tmp_path):
    """A FitServer answering in a thread; yields it, then stops it."""
    servers = []

    def start(**kw):
        s = FitServer(still, tmp_path / "site.json", **kw)
        threading.Thread(target=s.serve_forever, daemon=True).start()
        servers.append(s)
        return s
    yield start
    for s in servers:
        s.shutdown()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return r.status, r.headers.get("Content-Type", ""), r.read().decode()


def call(server, name, args, ctype="application/json"):
    req = urllib.request.Request(server.url + "call", data=json.dumps({"name": name, "args": args}).encode(),
                                 headers={"Content-Type": ctype}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def status_of(url, method="GET", data=None, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers or {}, method=method),
                                    timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_the_screen_is_served_with_its_own_bridge_at_a_token_address(served):
    s = served()
    assert s.url.startswith("http://127.0.0.1:") and s.url.endswith(f"/{s.token}/") and len(s.token) >= 16
    code, ctype, html = get(s.url)
    assert code == 200 and ctype.startswith("text/html")
    assert "kind = 'local'" in html and BUNDLE_MARKER not in html and "globalThis.ExtApps" in html
    assert 'id="edgeA"' in html and 'id="preset"' in html and "A4" in html  # the lengths are typed in the screen
    # without the token, nothing
    root = s.url[:s.url.index("/", len("http://"))] + "/"
    assert status_of(root) == 404 and status_of(root + "first") == 404


def test_the_first_state_starts_from_an_a4_sheet_unless_lengths_are_given(served):
    s = served()
    first = json.loads(get(s.url + "first")[2])
    assert first["edgesM"] == [0.21, 0.297] and first["measured"] is False
    assert (first["width"], first["height"]) == (W, H) and first["parts"] >= 1
    assert len(first["corners"]) == 4 and first["objects"] == [] and first["objectsKey"] is None
    given = json.loads(get(served(edges=(1.2, 0.8), measured=True).url + "first")[2])
    assert given["edgesM"] == [1.2, 0.8] and given["measured"] is True


def test_the_photo_comes_in_parts_and_the_floor_solves_with_a_grid_step_to_match(served, still):
    s = served()
    first = json.loads(get(s.url + "first")[2])
    b64 = "".join(json.loads(call(s, "ps_spike_image", {"imagePath": str(still), "part": p})["text"])["data"]
                  for p in range(first["parts"]))
    assert base64.b64decode(b64)[:2] == b"\xff\xd8"  # a JPEG
    out = json.loads(call(s, "ps_floor_solve", {"width": W, "height": H, "corners": CORNERS, "edgesM": EDGES})["text"])
    assert out["ok"] and out["residualPx"] < 0.5 and out["stepM"] == 1.0
    assert len(out["lines"]) == 5 + 4  # 0..4 m across, 0..3 m along
    small = json.loads(call(s, "ps_floor_solve", {"width": W, "height": H, "corners": floor_corners(0.21, 0.297),
                                                  "edgesM": [0.21, 0.297]})["text"])
    assert small["ok"] and small["stepM"] == 0.1 and len(small["lines"]) == 3 + 3  # 0, 0.1, 0.2 each way


def test_decide_writes_a_site_confirmed_by_a_person(served, tmp_path):
    seen = []
    s = served(camera="front", on_decide=lambda site, path: seen.append((site.id, path)))
    out = json.loads(call(s, "ps_site_decide", {"siteDir": "", "imagePath": "ignored", "corners": CORNERS,
                                                "edgesM": EDGES, "objects": [], "measured": True})["text"])
    assert out["ok"] and out["camera"] == "front" and out["sitePath"] == str(tmp_path / "site.json")
    site = load_site(tmp_path / "site.json")
    assert site.confirmed_by_person is True and site.dimensions == "measured" and site.id == "site"
    assert "front" in site.cameras and site.residual_px["front"] < 0.5 and site.floor.edges_m == EDGES
    assert site.objects == [] and "browser" in site.history[0]["what"]
    assert seen == [("site", tmp_path / "site.json")] and s.decided == 1
    # 決定 again, estimated this time, overwrites it
    call(s, "ps_site_decide", {"siteDir": "", "imagePath": "ignored", "corners": CORNERS, "edgesM": EDGES,
                               "objects": []})
    assert load_site(tmp_path / "site.json").dimensions == "estimated" and s.decided == 2


def test_decide_refuses_objects_bad_corners_and_a_flat_floor(served, tmp_path):
    s = served()
    flat = [{"u": 100, "v": 100}, {"u": 500, "v": 100}, {"u": 500, "v": 400}, {"u": 100, "v": 400}]
    for args, word in [({"corners": CORNERS, "edgesM": EDGES, "objects": [{"name": "pole", "kind": "fixed"}]}, "objects"),
                       ({"corners": CORNERS[:3], "edgesM": EDGES, "objects": []}, "bad-input"),
                       ({"corners": CORNERS, "edgesM": [0, 3], "objects": []}, "bad-input"),
                       ({"corners": flat, "edgesM": EDGES, "objects": []}, "flat")]:
        out = json.loads(call(s, "ps_site_decide", {"siteDir": "", "imagePath": "x", **args})["text"])
        assert out["ok"] is False and word in out["reason"]
    assert not (tmp_path / "site.json").exists()
    assert "error" in call(s, "ps_objects_part", {"key": "k", "part": 0})  # the chat's other tools are not here


def test_a_post_that_is_not_json_is_refused(served):
    s = served()
    body = json.dumps({"name": "ps_floor_solve", "args": {}}).encode()
    assert status_of(s.url + "call", "POST", body, {"Content-Type": "text/plain"}) == 404
    assert status_of(s.url + "other", "POST", body, {"Content-Type": "application/json"}) == 404


def test_a_bad_picture_is_refused_before_anything_is_served(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises((ValueError, FileNotFoundError)):
        FitServer(tmp_path / "missing.png", tmp_path / "site.json")
    with pytest.raises(ValueError):
        FitServer(tmp_path / "missing.png", tmp_path / "site.json", edges=(0, 1))


# ---- the command ----

class FakeServer:
    made = []

    def __init__(self, *args):
        self.args = args
        self.url = "http://127.0.0.1:1/t/"
        FakeServer.made.append(self)

    def serve_forever(self):
        raise KeyboardInterrupt

    def close(self):
        self.closed = True


def test_site_fit_without_corners_serves_the_screen_until_ctrl_c(still, tmp_path, monkeypatch, capsys):
    import webbrowser
    opened = []
    monkeypatch.setattr(fit_screen, "FitServer", FakeServer)
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url))
    FakeServer.made.clear()
    out = tmp_path / "s.json"
    assert main(["site", "fit", str(still), "--out", str(out), "--camera", "side", "--edges", "1", "2",
                 "--measured", "--port", "8123"]) == 0
    (server,) = FakeServer.made
    image, path, camera, edges, measured, port, wrote = server.args
    assert (image, path, camera, edges, measured, port) == (str(still), str(out), "side", (1.0, 2.0), True, 8123)
    assert opened == [server.url] and server.closed
    assert server.url in capsys.readouterr().out
    # --no-browser only prints the address; the lengths default to None (the screen starts from A4)
    assert main(["site", "fit", str(still), "--no-browser"]) == 0
    assert len(FakeServer.made) == 2 and FakeServer.made[1].args[1] == "site.json" and FakeServer.made[1].args[3] is None
    assert opened == [server.url]


def test_site_fit_prints_what_was_written_on_each_decide(still, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fit_screen, "FitServer", FakeServer)
    FakeServer.made.clear()
    main(["site", "fit", str(still), "--out", str(tmp_path / "s.json"), "--no-browser"])
    wrote = FakeServer.made[-1].args[-1]
    real = FitServer(still, tmp_path / "s.json")
    real.close()
    real._decide({"corners": CORNERS, "edgesM": EDGES, "objects": [], "measured": True})
    wrote(load_site(tmp_path / "s.json"), tmp_path / "s.json")
    out = capsys.readouterr().out
    assert "wrote" in out and "measured" in out and "confirmed by a person" in out


def test_site_fit_with_corners_needs_edges(still, tmp_path, capsys):
    code = main(["site", "fit", str(still), "--corners", "1,2", "3,4", "5,6", "7,8", "--out", str(tmp_path / "o.json")])
    assert code == 2 and "--edges" in capsys.readouterr().err and not (tmp_path / "o.json").exists()


def test_widget_html_inlines_the_libs_it_names(tmp_path, monkeypatch):
    monkeypatch.setattr(fit_screen, "WIDGETS", tmp_path)
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "one.js").write_text("function one() { return 1; }\n")
    (tmp_path / "w.html").write_text("<script type=\"module\">/*__EXT_APPS_BUNDLE__*/\n/*__LIB:one__*/\n</script>")
    html = fit_screen.widget_html("w.html", "/*bridge*/")
    assert "function one()" in html and "__LIB:one__" not in html
    (tmp_path / "w.html").write_text("/*__LIB:none__*/")
    with pytest.raises(FileNotFoundError):
        fit_screen.widget_html("w.html", "")


def test_the_served_screens_carry_no_marker():
    for name, three in (("fit-floor.html", False), ("compare.html", True)):
        html = fit_screen.widget_html(name, "", three)
        assert "/*__LIB:" not in html and "/*__THREE__*/" not in html


def test_a_call_larger_than_the_screen_ever_sends_is_refused_unread(served):
    import http.client
    from urllib.parse import urlsplit

    from physical_sandbox.fit_screen import MAX_CALL_BYTES

    s = served()
    url = urlsplit(s.url)

    def answer(length: str) -> int:
        # only the headers are sent: an answer at all shows the body was never waited for
        conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
        conn.putrequest("POST", url.path + "call")
        conn.putheader("Content-Type", "application/json")
        conn.putheader("Content-Length", length)
        conn.endheaders()
        try:
            return conn.getresponse().status
        finally:
            conn.close()
    assert answer(str(MAX_CALL_BYTES + 1)) == 413
    assert answer("-5") == 400 and answer("lots") == 400
    assert call(s, "ps_floor_solve", {"width": 640, "height": 480, "corners": [], "edgesM": [0.297, 0.21]})  # still serving
