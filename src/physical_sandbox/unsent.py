"""Records PS.com did not get yet: kept in the work folder's ``unsent/`` and sent again later.

A save that cannot reach PS.com (offline, or not signed in) leaves two files: ``<name>.record.json``, the record
exactly as it would have gone, and ``<name>.send.json``, ``{ref, mode, base}`` saying where it goes (``mode``
``over`` with the version it was made from, or ``new``). ``send_all`` sends them through the ``sender`` it is given
and removes what went; this module never reaches the network itself.

A save over a record that was saved over elsewhere meanwhile is neither lost nor written over: it goes as a new
record under the first free ``<name>-2``, ``<name>-3``... and the note says so.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ._atomic import write_text_atomic
from .errors import CatalogError, InputError
from .push import PushError

FOLDER = "unsent"
SIGN_IN = "psbx login でログインすると送れます"
OFFLINE = "PS.com に接続できませんでした。接続できるようになったら送り直してください"
MOVED = "{ref} は別の場所で変わっていたため、{new} として新しく保存しました"
_NAME_MAX = 39  # a catalog name is at most 39 characters
_TRIES = 50  # suffixes tried for a free name


def _name_of(ref: str) -> str:
    """The name part of ``handle/name[:version]`` (or of a bare ``name``)."""
    return ref.split("/")[-1].split(":")[0]


def _suffixed(ref: str, k: int) -> str:
    """``ref`` with ``-k`` added to its name, the name cut so that it stays a catalog name."""
    handle, _, name = ref.rpartition("/")
    tail = f"-{k}"
    name = name[: _NAME_MAX - len(tail)].rstrip("-") + tail
    return f"{handle}/{name}" if handle else name


def keep(work: Path, record: dict, *, ref: str, mode: str, base: str | None) -> str:
    """Keep ``record`` for ``ref`` in ``<work>/unsent/``; the name it is kept under (the ref's name, or the first
    free ``<name>-2``, ``<name>-3``... beside an earlier copy)."""
    if mode not in ("over", "new"):
        raise ValueError(f"mode must be 'over' or 'new', not {mode!r}")
    folder = Path(work) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    stem = _name_of(ref)
    name, k = stem, 1
    while (folder / f"{name}.record.json").exists() or (folder / f"{name}.send.json").exists():
        k += 1
        name = f"{stem}-{k}"
    # The record first: a copy counts as pending only once its send file is there too
    write_text_atomic(folder / f"{name}.record.json", json.dumps(record, ensure_ascii=False, indent=1) + "\n")
    write_text_atomic(folder / f"{name}.send.json", json.dumps({"ref": ref, "mode": mode, "base": base}) + "\n")
    return name


def pending(work: Path) -> list[dict]:
    """The kept copies, by name: ``[{name, ref, mode, base}]`` (a pair that is not whole or not readable is left out)."""
    folder = Path(work) / FOLDER
    out = []
    names = sorted(s.name[: -len(".send.json")] for s in folder.glob("*.send.json")) if folder.is_dir() else []
    for name in names:
        send = folder / f"{name}.send.json"
        if not (folder / f"{name}.record.json").is_file():
            continue
        try:
            d = json.loads(send.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict) or not isinstance(d.get("ref"), str) or d.get("mode") not in ("over", "new"):
            continue
        base = d.get("base")
        out.append({"name": name, "ref": d["ref"], "mode": d["mode"], "base": base if isinstance(base, str) else None})
    return out


def send_one(path: Path, ref: str, mode: str, base: str | None, sender) -> tuple[str, str, str | None]:
    """Send the record at ``path`` with ``sender(path, ref, mode, base)``, which answers the ref written
    (``handle/name:version``). A save over that answers PushError ``moved`` goes as a new record under the first
    free suffix, carrying ``basedOn`` (the file at ``path`` stays as it was). Returns (the ref written, the mode
    it went as, the note or None); other failures are raised as they come."""
    try:
        return sender(path, ref, mode, base), mode, None
    except PushError as e:
        if e.code != "moved":
            raise
    record = json.loads(Path(path).read_text())
    if base:
        record["basedOn"] = f"{ref}:{base}"
    with tempfile.TemporaryDirectory(prefix="psbx-unsent-") as tmp:
        copy = Path(tmp) / "record.json"
        copy.write_text(json.dumps(record, ensure_ascii=False))
        for k in range(2, 2 + _TRIES):
            new = _suffixed(ref, k)
            try:
                written = sender(copy, new, "new", None)
            except PushError as e:
                if e.code == "taken":
                    continue
                raise
            return written, "new", MOVED.format(ref=ref, new=new)
    raise PushError("taken", 409, note=f"{ref} の新しい名前が見つかりませんでした")


def send_all(work: Path, sender) -> tuple[list[dict], list[dict], list[str]]:
    """Send every kept copy (see ``send_one``); ``(sent, left, notes)``. ``sent`` is ``[{name, ref, version,
    mode}]`` as written; ``left`` is what stays kept (``pending``'s entries). Offline (CatalogError) or signed out
    on the server, the rest is not tried: every copy stays. Another refusal keeps that copy with the reason."""
    folder = Path(work) / FOLDER
    sent, left, notes = [], [], []
    waiting = pending(work)
    for i, p in enumerate(waiting):
        try:
            written, mode, note = send_one(folder / f"{p['name']}.record.json", p["ref"], p["mode"], p["base"],
                                           sender)
        except PushError as e:
            if e.code == "signin":
                left += waiting[i:]
                notes.append(SIGN_IN)
                break
            left.append(p)
            notes.append(f"{p['name']} を送れませんでした（{e.note or e}）")
            continue
        except CatalogError:  # not a PushError: PS.com could not be reached
            left += waiting[i:]
            notes.append(OFFLINE)
            break
        except InputError as e:
            left.append(p)
            notes.append(f"{p['name']} を送れませんでした（{e}）")
            continue
        (folder / f"{p['name']}.send.json").unlink(missing_ok=True)  # first: a copy without it is never sent again
        (folder / f"{p['name']}.record.json").unlink(missing_ok=True)
        ref, _, version = written.partition(":")
        sent.append({"name": p["name"], "ref": ref, "version": version, "mode": mode})
        if note:
            notes.append(note)
    return sent, left, notes
