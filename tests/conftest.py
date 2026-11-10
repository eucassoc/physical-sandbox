"""Shared fixtures: a fake PS.com catalog server."""
import hashlib
import json
import re
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


def _version_key(v):
    return (0, int(v), "") if v.isdigit() else (1, 0, v)


def _next_version(names):
    numbers = [int(v) for v in names if v.isdigit()]
    return str(max(numbers) + 1) if numbers else "1"


_VERSION = re.compile(r"^[a-z0-9][a-z0-9.-]{0,31}$")


FILES = {"mini.urdf": URDF.encode(), "notes.txt": b"two links, one joint\n"}


class FakeCatalog:
    """Serves the catalog API for ``t/mini`` (version 1). ``bad`` corrupts one sha256 in the manifest.

    It also takes uploads: the device-code login (``/api/vr/*``, approved at the first poll), ``/api/handle``,
    machine/object versions (POST manifest, PUT files, POST finish) and records, kept in ``items``. Records
    follow PS.com: a version is never written over (``PUT .../{v}/record`` on a stored one is 409 ``taken``);
    ``PUT .../{name}/record`` writes the next version, takes ``base`` (409 ``moved`` when the latest is another)
    and ``new=1`` (409 ``taken`` when the item exists), and answers ``unchanged`` without writing when the bytes
    equal the latest version's. Every request is logged in ``calls`` as (method, path with query).
    """

    def __init__(self, bad=False):
        self.latest = "1"
        self.extra_files = None
        self.upper = False
        self.bad = bad
        self.port = 0
        self.hits = 0
        self.headers = []
        self.paths = []
        self.server = None
        self.handle = "ada"
        self.token = "tok-1"
        self.vr_starts = 0
        self.items = {}  # (handle, name) -> {kind, visibility, versions: {version: {...}}}
        self.calls = []
        self.fail_put = None  # an error code every PUT of a file answers with (400)

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
        parts = path.split("/")
        if path == "/api/handle":
            return self.need_auth() or self.reply(200, {"handle": self.handle})
        if len(parts) >= 5 and parts[:3] == ["", "api", "catalog"] and (parts[3], parts[4]) in self.items:
            return self.stored_get(parts[3], parts[4], parts[5:])
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

    # ---- the write side ----

    @staticmethod
    def reply(code, body):
        return code, "application/json", json.dumps(body).encode()

    def need_auth(self):
        if self.auth != f"Bearer {self.token}":
            return self.reply(401, {"error": "signin"})
        return None

    def stored_get(self, handle, name, rest):
        item = self.items[(handle, name)]
        finished = {v: x for v, x in item["versions"].items() if x["finished"]}
        if not rest:
            latest = item.get("latest") or (max(finished, key=_version_key) if finished else None)
            body = {"item": {"handle": handle, "name": name, "kind": item["kind"], "visibility": item["visibility"],
                             "latest_version": latest},
                    "versions": [{"version": v, "bytes": x["bytes"], "finished_at": "2026-11-01"}
                                 for v, x in finished.items()]}
            return self.reply(200, body)
        if len(rest) == 2 and rest[1] == "manifest" and rest[0] in finished:
            return self.reply(200, finished[rest[0]]["manifest"])
        if rest[1:] == ["files", "record.json"] and "raw" in finished.get(rest[0], {}):
            return 200, "application/json", finished[rest[0]]["raw"]
        return self.reply(404, {"error": "not-found"})

    def put_record(self, handle, name, version, body, visibility):
        """Store ``body`` as ``version`` of the record ``handle/name`` (the item made when it is new)."""
        try:
            record = json.loads(body)
        except ValueError:
            return self.reply(400, {"error": "bad-record"})
        created = (handle, name) not in self.items
        item = self.items.setdefault((handle, name), {"kind": "record", "visibility": visibility, "versions": {}})
        manifest = {"format": "physical-sandbox-catalog/1", "name": f"{handle}/{name}", "version": version,
                    "kind": "record", "files": [{"path": "record.json", "bytes": len(body),
                                                 "sha256": hashlib.sha256(body).hexdigest()}]}
        item["versions"][version] = {"record": record, "raw": body, "finished": True, "bytes": len(body),
                                     "manifest": manifest}
        item["latest"] = version
        return self.reply(201 if created else 200, {"name": f"{handle}/{name}", "version": version,
                                                    "bytes": len(body), "unchanged": False})

    def put_next_record(self, handle, name, query, body):
        """``PUT .../{name}/record``: the next version, with ``base``, ``new`` and ``unchanged`` as PS.com."""
        from urllib.parse import parse_qs

        q = {k: v[-1] for k, v in parse_qs(query).items()}
        base = q.get("base")
        if base is not None and not _VERSION.match(base):
            return self.reply(400, {"error": "base"})
        item = self.items.get((handle, name))
        visibility = "public" if q.get("visibility") == "public" else "private"
        if item is None:
            return self.put_record(handle, name, "1", body, visibility)
        if item["kind"] != "record":
            return self.reply(409, {"error": "not-a-record"})
        if q.get("new") == "1":
            return self.reply(409, {"error": "taken"})
        latest = item.get("latest")
        if latest and item["versions"][latest].get("raw") == body:
            return self.reply(200, {"name": f"{handle}/{name}", "version": latest, "bytes": len(body),
                                    "unchanged": True})
        if base is not None and latest != base:
            return self.reply(409, {"error": "moved", "latest": latest})
        return self.put_record(handle, name, _next_version(list(item["versions"])), body, visibility)

    def write(self, method, raw_path, body):
        path, _, query = raw_path.partition("?")
        if path == "/api/vr/start" and method == "POST":
            self.vr_starts += 1
            return self.reply(200, {"deviceCode": "dev-1", "userCode": "123456"})
        if path == "/api/vr/poll" and method == "POST":
            ok = json.loads(body or b"{}").get("deviceCode") == "dev-1"
            return self.reply(200, {"status": "approved", "token": self.token, "expiresIn": 3600} if ok
                              else {"status": "expired"})
        denied = self.need_auth()
        if denied:
            return denied
        parts = path.split("/")
        if len(parts) < 6 or parts[:3] != ["", "api", "catalog"]:
            return self.reply(404, {"error": "not-found"})
        if len(parts) == 6 and parts[5] == "record" and method == "PUT":
            if parts[3] != self.handle:
                return self.reply(404, {"error": "not-found"})
            return self.put_next_record(parts[3], parts[4], query, body)
        handle, name, version, rest = parts[3], parts[4], parts[5], parts[6:]
        if handle != self.handle:
            return self.reply(404, {"error": "not-found"})
        visibility = "public" if "visibility=public" in query else "private"
        item = self.items.get((handle, name))
        if method == "POST" and not rest:
            manifest = json.loads(body)
            if item is not None and item["kind"] != manifest["kind"]:
                return self.reply(409, {"error": "taken"})
            if item is not None and item["versions"].get(version, {}).get("finished"):
                return self.reply(409, {"error": "taken"})
            item = self.items.setdefault((handle, name), {"kind": manifest["kind"], "visibility": visibility,
                                                          "versions": {}})
            item["versions"][version] = {"manifest": manifest, "files": {}, "finished": False,
                                         "bytes": sum(f["bytes"] for f in manifest["files"])}
            return self.reply(201, {"name": manifest["name"], "version": version,
                                    "upload": [f["path"] for f in manifest["files"]]})
        if method == "PUT" and rest == ["record"]:
            if item is not None and item["kind"] != "record":
                return self.reply(409, {"error": "not-a-record"})
            if item is not None and item["versions"].get(version, {}).get("finished"):
                return self.reply(409, {"error": "taken"})
            return self.put_record(handle, name, version, body, visibility)
        upload = (item or {}).get("versions", {}).get(version)
        if upload is None:
            return self.reply(404, {"error": "not-found"})
        if method == "PUT" and rest[:1] == ["files"]:
            if self.fail_put:
                return self.reply(400, {"error": self.fail_put})
            from urllib.parse import unquote
            rel = unquote("/".join(rest[1:]))
            want = {f["path"]: f for f in upload["manifest"]["files"]}.get(rel)
            if want is None:
                return self.reply(400, {"error": "bad-path"})
            if hashlib.sha256(body).hexdigest() != want["sha256"]:
                return self.reply(400, {"error": "bad-hash"})
            upload["files"][rel] = body
            return self.reply(200, {"ok": True})
        if method == "POST" and rest == ["finish"]:
            if set(upload["files"]) != {f["path"] for f in upload["manifest"]["files"]}:
                return self.reply(400, {"error": "incomplete"})
            upload["finished"] = True
            return self.reply(200, {"ok": True})
        return self.reply(404, {"error": "not-found"})

    def start(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def answer(self, code, ctype, body):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                outer.hits += 1
                outer.headers.append(dict(self.headers))
                outer.paths.append(self.path)
                outer.calls.append(("GET", self.path))
                outer.auth = self.headers.get("Authorization")
                self.answer(*outer.route(self.path))

            def _write(self, method):
                outer.headers.append(dict(self.headers))
                outer.calls.append((method, self.path))
                outer.auth = self.headers.get("Authorization")
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.answer(*outer.write(method, self.path, body))

            def do_POST(self):
                self._write("POST")

            def do_PUT(self):
                self._write("PUT")

            def log_message(self, *a):
                pass

        self.auth = None
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
