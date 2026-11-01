"""Shared fixtures: a fake PS.com catalog server."""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

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


FILES = {"mini.urdf": URDF.encode(), "notes.txt": b"two links, one joint\n"}


class FakeCatalog:
    """Serves the catalog API for ``t/mini`` (version 1). ``bad`` corrupts one sha256 in the manifest."""

    def __init__(self, bad=False):
        self.latest = "1"
        self.extra_files = None
        self.upper = False
        self.bad = bad
        self.port = 0
        self.hits = 0
        self.server = None

    def manifest(self):
        files = [{"path": p, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in FILES.items()]
        if self.extra_files is not None:
            files = self.extra_files
        if self.upper:
            files[0]["sha256"] = files[0]["sha256"].upper()
        if self.bad:
            files[1]["sha256"] = "0" * 64
        return {"format": "physical-sandbox-catalog/1", "name": "t/mini", "version": "1", "kind": "machine",
                "category": "arm", "license": "Apache-2.0", "display": {"ja": "ミニ", "en": "Mini arm"},
                "bundle": "bundle.json", "product": "mini", "profile": "default", "files": files}

    def route(self, path):
        item = {"id": "i1", "handle": "t", "name": "mini", "kind": "machine", "category": "arm",
                "visibility": "public", "display": {"ja": "ミニ", "en": "Mini arm"}, "tags": ["arm"],
                "license": "Apache-2.0", "maker": None, "latest_version": self.latest, "updated_at": "2026-11-01"}
        path = path.split("?")[0]
        if path == "/api/catalog/search":
            return 200, "application/json", json.dumps({"items": [item]}).encode()
        if path == "/api/catalog/t/mini":
            body = {"item": {**item, "description": "", "source_url": None},
                    "versions": [{"version": "1", "bytes": 10, "finished_at": "2026-11-01"}]}
            return 200, "application/json", json.dumps(body).encode()
        if path in ("/api/catalog/t/mini/1/manifest", "/api/catalog/t/mini/latest/manifest"):
            return 200, "application/json", json.dumps(self.manifest()).encode()
        prefix = "/api/catalog/t/mini/1/files/"
        if path.startswith(prefix) and path[len(prefix):] in FILES:
            return 200, "application/octet-stream", FILES[path[len(prefix):]]
        return 404, "application/json", b'{"error": "not-found"}'

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.hits += 1
                code, ctype, body = outer.route(self.path)
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


@pytest.fixture
def catalog_server():
    fake = FakeCatalog().start()
    yield fake
    fake.stop()


@pytest.fixture
def bad_catalog_server():
    fake = FakeCatalog(bad=True).start()
    yield fake
    fake.stop()
