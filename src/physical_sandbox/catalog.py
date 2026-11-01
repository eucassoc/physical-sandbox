"""Fetch machines and other bundles from the PS.com catalog into a local cache.

``pull("handle/name[:version]")`` downloads the files of one version into
``<PS_CACHE_DIR>/catalog/<handle>/<name>/<version>/`` and checks every sha256 before the folder appears, so the
cache never holds a half-written version. Standard library only.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .errors import CatalogError, InputError

BASE = os.environ.get("PS_CATALOG_URL", "https://physicalsandbox.com").rstrip("/")
_SLUG = r"[a-z0-9](?:[a-z0-9-]{1,37}[a-z0-9])?"
_VERSION = r"[a-z0-9][a-z0-9.-]{0,31}"
_REF = re.compile(rf"^({_SLUG})/({_SLUG})(?::({_VERSION}))?$")
_TIMEOUT = 30


class _Offline(CatalogError):
    """The server could not be reached or the transfer broke (as opposed to the server saying no)."""


def cache_root() -> Path:
    return Path(os.environ.get("PS_CACHE_DIR", "~/.cache/physical-sandbox")).expanduser()


def parse_ref(ref: str) -> tuple[str, str, str]:
    m = _REF.match(ref)
    if not m:
        raise InputError(f"{ref!r} is not a catalog name: use handle/name or handle/name:version")
    return m.group(1), m.group(2), m.group(3) or "latest"


def _name_dir(handle: str, name: str) -> Path:
    return cache_root() / "catalog" / handle / name


def _get(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        code = ""
        try:
            code = json.loads(e.read().decode()).get("error", "")
        except Exception:  # noqa: BLE001 - the body is only a hint
            pass
        raise CatalogError(f"{url}: HTTP {e.code} {code}".strip()) from e
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise _Offline(f"{url}: {getattr(e, 'reason', None) or e}") from e


def _get_json(url: str):
    try:
        return json.loads(_get(url).decode())
    except ValueError as e:
        raise CatalogError(f"{url}: not JSON") from e


def search(q: str = "", *, category=None, tag=None, kind=None, base: str | None = None) -> list[dict]:
    base = (base or BASE).rstrip("/")
    params = {k: v for k, v in (("q", q), ("category", category), ("tag", tag), ("kind", kind)) if v}
    url = f"{base}/api/catalog/search?{urllib.parse.urlencode(params)}"
    try:
        return _get_json(url)["items"]
    except (KeyError, TypeError) as e:
        raise CatalogError(f"{url}: no items in the answer") from e


def _complete(d: Path) -> bool:
    mf = d / "manifest.json"
    if not mf.is_file():
        return False
    try:
        files = json.loads(mf.read_text())["files"]
        return all((d / _safe_rel(f["path"])).is_file() for f in files)
    except (ValueError, KeyError, TypeError, OSError, CatalogError):
        return False


def cached(ref: str) -> Path | None:
    """The folder of a version already pulled (``latest`` means the last one resolved), without the network."""
    handle, name, version = parse_ref(ref)
    base = _name_dir(handle, name)
    if version == "latest":
        try:
            version = (base / "latest").read_text().strip()
        except OSError:
            return None
        if not re.fullmatch(_VERSION, version):
            return None
    d = base / version
    return d if _complete(d) else None


def _safe_rel(path: str) -> str:
    parts = Path(path).parts
    if not path or Path(path).is_absolute() or ".." in parts or "\\" in path:
        raise CatalogError(f"the manifest has an unsafe file path {path!r}")
    return path


def _resolve_latest(handle: str, name: str, base: str) -> str:
    try:
        v = _get_json(f"{base}/api/catalog/{handle}/{name}")["item"]["latest_version"]
    except _Offline as e:
        c = cached(f"{handle}/{name}")
        if c is not None:
            return c.name
        raise CatalogError(f"{e}; nothing is cached for {handle}/{name}") from e
    except (KeyError, TypeError) as e:
        raise CatalogError(f"{handle}/{name}: unexpected answer") from e
    if not v:
        raise CatalogError(f"{handle}/{name}: no finished version")
    if not isinstance(v, str) or not re.fullmatch(_VERSION, v):
        raise CatalogError(f"{handle}/{name}: the server gave an invalid version {v!r}")
    return v


def pull(ref: str, *, base: str | None = None, force: bool = False) -> Path:
    base = (base or BASE).rstrip("/")  # read at call time so PS_CATALOG_URL-style overrides and tests can change it
    handle, name, version = parse_ref(ref)
    parent = _name_dir(handle, name)
    if version == "latest":
        # latest moves, so ask the server; the cached marker is only the offline fallback
        version = _resolve_latest(handle, name, base)
        resolved_latest = True
    else:
        resolved_latest = False
    dest = parent / version
    if not force and _complete(dest):
        if resolved_latest:
            _write_latest(parent, version)
        return dest
    partial = parent / f".{version}.partial-{os.getpid()}"
    try:
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(parents=True)
        root = f"{base}/api/catalog/{handle}/{name}/{version}"
        manifest_bytes = _get(f"{root}/manifest")
        try:
            files = json.loads(manifest_bytes.decode())["files"]
        except (ValueError, KeyError, TypeError) as e:
            raise CatalogError(f"{root}/manifest: not a manifest") from e
        for f in files:
            rel = _safe_rel(f["path"])
            data = _get(f"{root}/files/{urllib.parse.quote(rel)}")
            if hashlib.sha256(data).hexdigest() != str(f["sha256"]).lower():
                raise CatalogError(f"{handle}/{name}:{version}: sha256 of {rel} does not match the manifest")
            out = partial / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
        (partial / "manifest.json").write_bytes(manifest_bytes)
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(partial, dest)
    except CatalogError:
        raise
    except (OSError, KeyError, TypeError) as e:
        raise CatalogError(f"{handle}/{name}:{version}: {type(e).__name__}: {e}") from e
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    if resolved_latest:
        _write_latest(parent, version)
    return dest


def _write_latest(parent: Path, version: str) -> None:
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "latest").write_text(version + "\n")


def urdf_path(d: Path) -> Path:
    """The first ``.urdf`` file of the manifest: what ``load_machine`` takes."""
    files = json.loads((Path(d) / "manifest.json").read_text())["files"]
    for f in files:
        if f["path"].lower().endswith(".urdf"):
            return Path(d) / f["path"]
    raise CatalogError(f"{d}: the manifest lists no .urdf file")
