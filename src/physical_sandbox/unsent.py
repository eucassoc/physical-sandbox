"""Records PS.com did not get yet: kept in the work folder's ``unsent/`` and sent again later.

A save that cannot reach PS.com (offline, or not signed in) leaves two files: ``<name>.record.json``, the record
exactly as it would have gone, and ``<name>.send.json``, ``{ref, mode, base}`` saying where it goes (``mode``
``over`` with the version it was made from, or ``new``). ``send_all`` sends them through the ``sender`` it is given
and removes what went; this module never reaches the network itself.

A save over a record that was saved over elsewhere meanwhile is neither lost nor written over: it goes as a new
record under the first free ``<name>-2``, ``<name>-3``... and the note says so. A kept new record whose name was
taken meanwhile goes the same way. Saving over the same record again while a copy of it waits replaces that copy
(its ``base`` stays): one save over makes one version.
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
OFFLINE = "PS.com に接続できませんでした。接続できるようになったら「再送信する」を押してください"
SERVER_DOWN = "PS.com が応答できませんでした。しばらくしてから「再送信する」を押してください"
MOVED = "{ref} は別の場所で変わっていたため、{new} として新しく保存しました"
TAKEN = "{ref} はもうあったため、{new} として保存しました"
_NAME_MAX = 39  # a catalog name is at most 39 characters
_TRIES = 50  # suffixes tried for a free name


# What PS.com's refusals mean, by (error, HTTP status) and else by error alone
_REFUSALS = {
    ("limit", 429): "記録の数が上限に達しています",
    ("limit", 413): "保存できる容量の上限に達しています",
    ("too-large", 413): "記録が大きすぎます",
    ("renaming", 409): "ハンドルの変更中です。しばらくしてからもう一度お試しください",
    ("not-a-record", 409): "その名前は記録以外に使われています。別の名前を指定してください",
}


def server_down(e: PushError) -> bool:
    """Whether PS.com failed on its side (HTTP 5xx): nothing was refused, the copy is kept as when offline."""
    return e.status >= 500


def refusal_words(e: PushError) -> str:
    """A refusal said in words for the person: a known ``error`` in a sentence (with the note on what was left
    behind, when there is one), anything else as the server or push said it."""
    words = _REFUSALS.get((e.code, e.status)) or next(
        (w for (code, _), w in _REFUSALS.items() if code == e.code), None)
    if words is None:
        return e.note or str(e)
    return f"{words}。{e.note}" if e.note else words


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
    free ``<name>-2``, ``<name>-3``... beside an earlier copy). A save over ``ref`` while a save over it waits
    replaces that copy's record and keeps its ``base``."""
    if mode not in ("over", "new"):
        raise ValueError(f"mode must be 'over' or 'new', not {mode!r}")
    folder = Path(work) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    if mode == "over":  # the same record saved over again: the newer content, from the base the first one had
        for p in pending(work):
            if p["ref"] == ref and p["mode"] == "over":
                write_text_atomic(folder / f"{p['name']}.record.json",
                                  json.dumps(record, ensure_ascii=False, indent=1) + "\n")
                return p["name"]
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


def _under_free_name(record: dict, ref: str, sender, words: str) -> tuple[str, str, str]:
    """Send ``record`` as new under the first free ``<ref>-2``, ``<ref>-3``... (a ``taken`` answer tries the
    next); (the ref written, ``new``, the note in ``words``)."""
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
            shown_new = written.partition(":")[0]
            handle = shown_new.rpartition("/")[0]
            shown_ref = f"{handle}/{ref}" if handle and "/" not in ref else ref  # kept signed out: no handle
            return written, "new", words.format(ref=shown_ref, new=shown_new)
    raise PushError("taken", 409, note=f"{ref} の新しい名前が見つかりませんでした")


def send_one(path: Path, ref: str, mode: str, base: str | None, sender, *,
             taken_too: bool = False) -> tuple[str, str, str | None]:
    """Send the record at ``path`` with ``sender(path, ref, mode, base)``, which answers the ref written
    (``handle/name:version``; PushError ``taken`` for a new name that is there). A save over that answers PushError
    ``moved`` goes as a new record under the first free suffix, carrying ``basedOn``; with ``taken_too`` a new
    record whose name is taken goes the same way as it is. The file at ``path`` stays as it was. Returns (the ref
    written, the mode it went as, the note or None); other failures are raised as they come."""
    try:
        return sender(path, ref, mode, base), mode, None
    except PushError as e:
        if e.code == "moved":
            record = json.loads(Path(path).read_text())
            if base:
                record["basedOn"] = f"{ref}:{base}"
            return _under_free_name(record, ref, sender, MOVED)
        if e.code == "taken" and mode == "new" and taken_too:
            return _under_free_name(json.loads(Path(path).read_text()), ref, sender, TAKEN)
        raise


def drop_over(work: Path, ref: str) -> None:
    """Forget the kept saves over ``ref``: a newer save over it reached PS.com."""
    folder = Path(work) / FOLDER
    for p in pending(work):
        if p["ref"] == ref and p["mode"] == "over":
            (folder / f"{p['name']}.send.json").unlink(missing_ok=True)
            (folder / f"{p['name']}.record.json").unlink(missing_ok=True)


def send_all(work: Path, sender) -> tuple[list[dict], list[dict], list[str]]:
    """Send every kept copy (see ``send_one``; a taken new name also goes under a free suffix); ``(sent, left,
    notes)``. ``sent`` is ``[{name, ref, version, mode}]`` as written; ``left`` is what stays kept (``pending``'s
    entries). Offline (CatalogError), a server error (HTTP 5xx) or signed out on the server, the rest is not tried:
    every copy stays. Another refusal keeps that copy with the reason in words."""
    folder = Path(work) / FOLDER
    sent, left, notes = [], [], []
    waiting = pending(work)
    for i, p in enumerate(waiting):
        try:
            written, mode, note = send_one(folder / f"{p['name']}.record.json", p["ref"], p["mode"], p["base"],
                                           sender, taken_too=True)
        except PushError as e:
            if e.code == "signin":
                left += waiting[i:]
                notes.append(SIGN_IN)
                break
            if server_down(e):  # PS.com failed on its side: as when offline, the rest is not tried
                left += waiting[i:]
                notes.append(SERVER_DOWN)
                break
            left.append(p)
            notes.append(f"{p['name']} を送れませんでした（{refusal_words(e)}）")
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
