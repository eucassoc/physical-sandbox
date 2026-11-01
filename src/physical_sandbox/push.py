"""``psbx push``: put a machine, an object or a record into the PS.com catalog.

A machine folder (URDF, meshes, ``product.json``, ``profile.json``) or an object (one mesh and ``object.json``)
is checked with ``machine check``, turned into a display bundle, and uploaded with its manifest: POST the
manifest, PUT every file, POST finish. A record (schemaVersion 3) is PUT as one JSON body; the machines it
names by a bare local name (``so101``) go up with it, privately, as ``<handle>/<name>:<version>`` when the
catalog does not have them yet, and the record's ``devices[].product`` is rewritten to those names.

Signing in uses the headset's device-code flow (``/api/vr/start`` and ``/api/vr/poll``) and keeps the token in
``~/.config/physical-sandbox/ps-com-token.json``, the same file ps-com's ``tools/catalog-login.mjs`` keeps.
Standard library only.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import catalog
from .errors import CatalogError, InputError

FORMAT = "physical-sandbox-catalog/1"
CATEGORIES = ("arm", "mobile", "humanoid", "quadruped", "drone", "end-effector", "vehicle", "fixture", "figure")
POLL_S = 3.0
GIVE_UP_S = 10 * 60
_TIMEOUT = 120
_BUNDLE = "bundle.json"


@dataclass
class PushResult:
    ref: str  # handle/name:version as uploaded
    bytes: int
    companions: list[str] = field(default_factory=list)  # machines uploaded along with a record


class PushError(CatalogError):
    """The server said no: ``code`` is the API's ``error`` (``taken``, ``bad-manifest``, ...), ``status`` the HTTP code."""

    def __init__(self, code: str, status: int = 0, problems=()):
        self.code, self.status, self.problems = code, status, list(problems)
        super().__init__(code + (": " + "; ".join(self.problems) if self.problems else ""))


# ---- HTTP ----

def _call(method: str, url: str, *, token: str | None = None, body: bytes | None = None,
          content_type: str | None = None):
    headers = {"User-Agent": catalog._user_agent()}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    if body is not None:
        headers["Content-Length"] = str(len(body))
    req = urllib.request.Request(url, data=body, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        info = {}
        try:
            info = json.loads(e.read().decode())
        except Exception:  # noqa: BLE001 - the body is only a hint
            pass
        code = info.get("error") if isinstance(info, dict) else None
        raise PushError(str(code or f"HTTP {e.code}"), e.code, (info or {}).get("problems") or ()) from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise CatalogError(f"{url}: {getattr(e, 'reason', None) or e}") from e
    try:
        return json.loads(raw.decode()) if raw else {}
    except ValueError as e:
        raise CatalogError(f"{url}: not JSON") from e


# ---- login ----

def token_file() -> Path:
    return Path.home() / ".config" / "physical-sandbox" / "ps-com-token.json"


def _parse_time(text: str) -> datetime | None:
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def kept_token(base: str) -> str | None:
    """The kept token, while it is for ``base`` and has not expired."""
    try:
        kept = json.loads(token_file().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(kept, dict) or kept.get("base") != base or not kept.get("token"):
        return None
    expires = _parse_time(kept.get("expiresAt", ""))
    return kept["token"] if expires and expires > datetime.now(timezone.utc) else None


def _forget(base: str) -> None:
    try:
        if json.loads(token_file().read_text()).get("base") == base:
            token_file().unlink()
    except (OSError, ValueError, AttributeError):
        pass


def _keep(base: str, token: str, expires_in) -> None:
    path = token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    expires = datetime.now(timezone.utc) + timedelta(seconds=float(expires_in or 3600))
    text = json.dumps({"base": base, "token": token,
                       "expiresAt": expires.isoformat(timespec="milliseconds").replace("+00:00", "Z")})
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def login(base: str, *, log=print) -> str:
    """base -> token. Reuses the kept token while it is valid; otherwise shows a code to enter at ``{base}/vr``."""
    base = base.rstrip("/")
    token = kept_token(base)
    if token:
        return token
    start = _call("POST", f"{base}/api/vr/start")
    try:
        device_code, user_code = start["deviceCode"], start["userCode"]
    except (KeyError, TypeError) as e:
        raise CatalogError(f"{base}/api/vr/start: unexpected answer") from e
    log(f"Open {base}/vr and enter the code {user_code}")
    deadline = time.monotonic() + GIVE_UP_S
    while time.monotonic() < deadline:
        time.sleep(POLL_S)
        body = _call("POST", f"{base}/api/vr/poll", body=json.dumps({"deviceCode": device_code}).encode(),
                     content_type="application/json")
        status = body.get("status") if isinstance(body, dict) else None
        if status == "approved" and body.get("token"):
            _keep(base, body["token"], body.get("expiresIn"))
            return body["token"]
        if status == "expired":
            raise CatalogError("the code expired before it was approved; run again")
    raise CatalogError("gave up waiting for approval after 10 minutes; run again")


def _signed_in(base: str, log) -> tuple[str, str]:
    """(token, the account's handle). A kept token the server rejects is dropped and the login runs again."""
    for attempt in (0, 1):
        token = login(base, log=log)
        try:
            handle = _call("GET", f"{base}/api/handle", token=token).get("handle")
        except PushError as e:
            if e.status == 401 and attempt == 0:
                _forget(base)
                continue
            raise
        if not handle:
            raise InputError("decide your handle on PS.com first (/catalog/me)")
        return token, handle
    raise PushError("signin", 401)


# ---- what is already in the catalog ----

def _item(base: str, token: str, handle: str, name: str) -> dict | None:
    try:
        return _call("GET", f"{base}/api/catalog/{handle}/{name}", token=token)
    except PushError as e:
        if e.status == 404:
            return None
        raise


def _numbers(item: dict | None) -> list[int]:
    versions = (item or {}).get("versions") or []
    return [int(v["version"]) for v in versions if isinstance(v, dict) and str(v.get("version", "")).isdigit()]


def _next_version(item: dict | None) -> str:
    numbers = _numbers(item)
    return str(max(numbers) + 1) if numbers else "1"


def _latest(item: dict) -> str | None:
    v = (item.get("item") or {}).get("latest_version")
    if v:
        return str(v)
    numbers = _numbers(item)
    return str(max(numbers)) if numbers else None


# ---- building an upload ----

@dataclass
class _Upload:
    manifest: dict
    sources: dict  # published path -> local file

    @property
    def bytes(self) -> int:
        return sum(f["bytes"] for f in self.manifest["files"])

    def content(self) -> set:
        """(path, sha256) of every file but the bundle, whose ``version`` stamp changes on every build."""
        return {(f["path"], f["sha256"]) for f in self.manifest["files"] if f["path"] != self.manifest["bundle"]}


def _file_entry(path: str, source: Path) -> dict:
    data = source.read_bytes()
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _walk(folder: Path) -> dict:
    """Every file under ``folder`` by its POSIX path, leaving out hidden files and folders."""
    out = {}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder)
        if p.is_file() and not any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            out[rel.as_posix()] = p
    return out


def _add(sources: dict, more: dict) -> None:
    for path, source in more.items():
        if path in sources:
            raise InputError(f"{source}: the path {path!r} is also a file of the display bundle; rename it")
        sources[path] = source


def _bundle_files(result, out_dir: Path) -> dict:
    files = {_BUNDLE: result.bundle_path}
    for p in result.mesh_paths:
        files[p.relative_to(out_dir).as_posix()] = p
    return files


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InputError(f"{path}: does not read as JSON ({e})") from e
    if not isinstance(data, dict):
        raise InputError(f"{path}: is not a JSON object")
    return data


def _category(given: str | None, own) -> str:
    category = given or (own if isinstance(own, str) and own and own != "<fill>" else None)
    if not category:
        raise InputError(f"category: give --category ({', '.join(CATEGORIES)}) or set it in the folder's file")
    if category not in CATEGORIES:
        raise InputError(f"category {category!r} is not one of {', '.join(CATEGORIES)}")
    return category


def _tags(category: str, tags) -> list[str]:
    return list(dict.fromkeys([category, *tags]))


def _kind_of(path: Path) -> str:
    from .machine_tools import OBJECT_SUFFIXES, _find_model, _meshes_in

    if path.is_file() and path.suffix.lower() == ".json":
        return "record"
    if path.is_file():
        return "object" if path.suffix.lower() in OBJECT_SUFFIXES else "machine"
    if path.is_dir() and _find_model(path) is None and ((path / "object.json").is_file()
                                                        or len(_meshes_in(path)) == 1):
        return "object"
    return "machine"


def _checked(path: Path) -> None:
    from .machine_tools import check

    result = check(path)
    if not result.ok:
        raise InputError("; ".join(text for ok, text in result.items if not ok))


def _machine_upload(path: Path, handle: str, name: str, version: str, *, category, tags, work: Path,
                    log, checked: bool = False) -> _Upload:
    from .bundle import make_bundle
    from .machine_tools import _find_model

    if not checked:
        _checked(path)
    folder, model = (path.parent, path) if path.is_file() else (path, _find_model(path))
    product = _read(folder / "product.json")
    profile = _read(folder / "profile.json") if (folder / "profile.json").is_file() else None
    category = _category(category, product.get("category"))
    out = work / "bundle"
    result = make_bundle(model, product=product, profile=profile, name=name, out_dir=out)
    for w in result.warnings:
        log(f"warning: {w}")
    sources = _bundle_files(result, out)
    _add(sources, _walk(folder))
    manifest = {
        "format": FORMAT, "name": f"{handle}/{name}", "version": version, "kind": "machine",
        "category": category, "license": product.get("licenseId") or "unknown",
        "display": product["displayName"], "tags": _tags(category, tags),
        "origin": "C" if product.get("generic") is True else "A",
        "product": product, "bundle": _BUNDLE,
    }
    for key, value in (("description", product.get("description")), ("maker", product.get("maker")),
                       ("sourceUrl", (profile or {}).get("urdfSourceUrl"))):
        if isinstance(value, str) and value.strip():
            manifest[key] = value
    if profile is not None:
        manifest["profile"] = profile
    manifest["files"] = [_file_entry(p, s) for p, s in sources.items()]
    return _Upload(manifest, sources)


def _object_upload(path: Path, handle: str, name: str, version: str, *, category, tags, work: Path,
                   log, checked: bool = False) -> _Upload:
    from .bundle import make_bundle
    from .machine_tools import _meshes_in, wrap_object

    if not checked:
        _checked(path)
    folder = path.parent if path.is_file() else path
    mesh = path if path.is_file() else _meshes_in(folder)[0]
    obj = _read(folder / "object.json")
    category = _category(category, obj.get("category"))
    src = work / "object"
    urdf = wrap_object(mesh, unit=obj["unit"], out_dir=src)
    out = work / "bundle"
    result = make_bundle(urdf, product={"displayName": obj["display"], "licenseId": obj["license"]},
                         profile=None, name=name, out_dir=out)
    for w in result.warnings:
        log(f"warning: {w}")
    sources = _bundle_files(result, out)
    _add(sources, _walk(src))
    _add(sources, {"object.json": folder / "object.json"})
    manifest = {
        "format": FORMAT, "name": f"{handle}/{name}", "version": version, "kind": "object",
        "category": category, "license": obj["license"], "display": obj["display"],
        "tags": _tags(category, tags), "unit": obj["unit"], "product": {}, "bundle": _BUNDLE,
    }
    for key in ("description", "sourceUrl", "maker", "origin"):
        if isinstance(obj.get(key), str) and obj[key].strip():
            manifest[key] = obj[key]
    manifest["files"] = [_file_entry(p, s) for p, s in sources.items()]
    return _Upload(manifest, sources)


def _send(base: str, token: str, upload: _Upload, *, private: bool) -> None:
    handle, name = upload.manifest["name"].split("/")
    root = f"{base}/api/catalog/{handle}/{name}/{upload.manifest['version']}"
    visibility = "private" if private else "public"
    answer = _call("POST", f"{root}?visibility={visibility}", token=token,
                   body=json.dumps(upload.manifest, ensure_ascii=False).encode(), content_type="application/json")
    wanted = answer.get("upload") if isinstance(answer, dict) else None
    for path in (wanted if isinstance(wanted, list) else list(upload.sources)):
        if path not in upload.sources:
            raise CatalogError(f"the server asked for {path!r}, which the manifest does not list")
        _call("PUT", f"{root}/files/{urllib.parse.quote(path)}", token=token,
              body=upload.sources[path].read_bytes(), content_type="application/octet-stream")
    _call("POST", f"{root}/finish", token=token)


# ---- records ----

def _machine_folder(product: str, machine_dir: Path | None, only_one: bool) -> Path:
    candidates = []
    if machine_dir is not None:
        candidates.append(machine_dir / product)
        if only_one:
            candidates.append(machine_dir)
    candidates.append(catalog.cache_root() / "machines" / product)
    from .machine_tools import _find_model

    for c in candidates:
        if c.is_dir() and _find_model(c) is not None:
            return c
    raise InputError(f"no machine {product!r} here ({', '.join(str(c) for c in candidates)}): "
                     f"--machine-dir で機体の URDF の場所を指定してください")


def _companion(base, token, handle, product, folder, *, work: Path, log) -> tuple[str, bool]:
    """(the catalog name for ``product``, whether it was uploaded now)."""
    item = _item(base, token, handle, product)
    if item is not None:
        latest = _latest(item)
        if latest:
            mine = _machine_upload(folder, handle, product, latest, category=None, tags=(), work=work, log=log)
            there = _call("GET", f"{base}/api/catalog/{handle}/{product}/{latest}/manifest", token=token)
            theirs = {(f.get("path"), str(f.get("sha256", "")).lower()) for f in there.get("files") or []
                      if f.get("path") != there.get("bundle", _BUNDLE)}
            if theirs == mine.content():
                return f"{handle}/{product}:{latest}", False
            work = work / "next"
    version = _next_version(item)
    ref = f"{handle}/{product}:{version}"
    log(f"{ref} として非公開で一緒に上げます")
    upload = _machine_upload(folder, handle, product, version, category=None, tags=(), work=work, log=log)
    _send(base, token, upload, private=True)
    return ref, True


def _push_record(base, token, handle, name, version, path: Path, *, private, machine_dir, work, log):
    from .motion import load_record

    try:
        load_record(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise InputError(f"{path}: not a record psbx can read ({e})") from e
    record = _read(path)
    if record.get("schemaVersion") != 3:
        raise InputError(f"{path}: schemaVersion must be 3, not {record.get('schemaVersion')!r}")
    devices = (record.get("equipment") or {}).get("devices") or []
    local = list(dict.fromkeys(d["product"] for d in devices
                               if isinstance(d, dict) and isinstance(d.get("product"), str)
                               and d["product"] and "/" not in d["product"]))
    names, companions = {}, []
    for i, product in enumerate(local):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,37}[a-z0-9]", product):
            raise InputError(f"the machine {product!r} cannot be a catalog name (a-z 0-9 -, 2 to 39 characters)")
        folder = _machine_folder(product, machine_dir, len(local) == 1)
        names[product], uploaded = _companion(base, token, handle, product, folder, work=work / f"m{i}", log=log)
        if uploaded:
            companions.append(names[product])
    for d in devices:
        if isinstance(d, dict) and d.get("product") in names:
            d["product"] = names[d["product"]]
    body = json.dumps(record, ensure_ascii=False).encode()
    visibility = "private" if private else "public"
    _call("PUT", f"{base}/api/catalog/{handle}/{name}/{version}/record?visibility={visibility}", token=token,
          body=body, content_type="application/json")
    return len(body), companions


# ---- push ----

def push(ref: str, path: Path, *, private: bool = True, category: str | None = None, tags=(),
         machine_dir: Path | None = None, base: str | None = None, log=print) -> PushResult:
    """Upload a machine folder, an object (mesh + object.json) or a record to ``ref`` (handle/name[:version]).

    Without a version it is ``1``, or one more than the highest number already there.
    """
    base = (base or catalog._base()).rstrip("/")
    path = Path(path).expanduser()
    if not path.exists():
        raise InputError(f"{path}: no such file or folder")
    handle, name, version = catalog.parse_ref(ref)
    kind = _kind_of(path)
    if kind != "record":
        _checked(path)  # before signing in: a folder that cannot go up should not ask for a code
    token, mine = _signed_in(base, log)
    if handle != mine:
        raise InputError(f"you can push only under your handle {mine!r}, not {handle!r}")
    if version == "latest":
        version = _next_version(_item(base, token, handle, name))
    tags = list(tags or ())
    with tempfile.TemporaryDirectory(prefix="psbx-push-") as tmp:
        work = Path(tmp)
        if kind == "record":
            size, companions = _push_record(base, token, handle, name, version, path, private=private,
                                            machine_dir=Path(machine_dir).expanduser() if machine_dir else None,
                                            work=work, log=log)
            return PushResult(f"{handle}/{name}:{version}", size, companions)
        build = _object_upload if kind == "object" else _machine_upload
        upload = build(path, handle, name, version, category=category, tags=tags, work=work, log=log,
                       checked=True)
        _send(base, token, upload, private=private)
        return PushResult(f"{handle}/{name}:{version}", upload.bytes, [])
