"""psbx push: machines, objects and records to the PS.com catalog, with the login and the companion machine."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from physical_sandbox import push as push_mod
from physical_sandbox.cli import main
from physical_sandbox.errors import CatalogError, InputError
from physical_sandbox.motion import motion_from_numpy, save_record
from physical_sandbox.push import PushError, login, push

MINI = Path(__file__).parent / "fixtures" / "mini-robot"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(push_mod, "POLL_S", 0)
    return tmp_path / "home"


@pytest.fixture
def mini(tmp_path):
    shutil.copytree(MINI, tmp_path / "mini")
    return tmp_path / "mini"


def quiet(*a, **k):
    pass


def writes(fake):
    return [(m, p.split("?")[0]) for m, p in fake.calls if m != "GET" and not p.startswith("/api/vr/")]


def test_push_refuses_a_urdf_whose_mesh_is_missing(mini, catalog_server):
    (mini / "meshes" / "box.stl").unlink()
    with pytest.raises(InputError, match="box.stl"):
        push("ada/mini:1", mini, base=catalog_server.url, log=quiet)
    assert writes(catalog_server) == []


def test_a_machine_is_posted_then_every_file_put_then_finished(mini, catalog_server):
    result = push("ada/mini:1", mini, base=catalog_server.url, log=quiet)
    assert result.ref == "ada/mini:1" and result.companions == []
    calls = writes(catalog_server)
    root = "/api/catalog/ada/mini/1"
    assert calls[0] == ("POST", root) and calls[-1] == ("POST", root + "/finish")
    puts = [p for m, p in calls[1:-1]]
    assert all(m == "PUT" for m, _ in calls[1:-1]) and puts
    stored = catalog_server.items[("ada", "mini")]
    assert stored["visibility"] == "private"
    manifest = stored["versions"]["1"]["manifest"]
    assert manifest["format"] == "physical-sandbox-catalog/1" and manifest["name"] == "ada/mini"
    assert manifest["kind"] == "machine" and manifest["category"] == "arm" and manifest["license"] == "Apache-2.0"
    assert manifest["display"] == {"ja": "小さな腕", "en": "Mini arm"} and manifest["bundle"] == "bundle.json"
    assert manifest["product"]["class"] == "mover" and manifest["profile"]["totalMassKg"] == 0.6
    paths = {f["path"] for f in manifest["files"]}
    assert {"bundle.json", "mini.urdf", "product.json", "profile.json", "meshes/box.stl"} <= paths
    assert any(p.startswith("assets/") and p.endswith(".bin") for p in paths)
    assert sorted(puts) == sorted(f"{root}/files/{p}" for p in paths)
    for f in manifest["files"]:
        data = stored["versions"]["1"]["files"][f["path"]]
        assert f["sha256"] == hashlib.sha256(data).hexdigest() and f["bytes"] == len(data)
    assert result.bytes == sum(f["bytes"] for f in manifest["files"])
    assert catalog_server.headers[-1]["Authorization"] == "Bearer tok-1"


def test_public_and_the_category_argument_and_tags(mini, catalog_server):
    product = json.loads((mini / "product.json").read_text())
    del product["category"]
    (mini / "product.json").write_text(json.dumps(product))
    with pytest.raises(InputError, match="category"):
        push("ada/mini:1", mini, base=catalog_server.url, log=quiet)
    push("ada/mini:1", mini, private=False, category="mobile", tags=["demo"], base=catalog_server.url, log=quiet)
    stored = catalog_server.items[("ada", "mini")]
    assert stored["visibility"] == "public"
    assert ("POST", "/api/catalog/ada/mini/1?visibility=public") in catalog_server.calls
    manifest = stored["versions"]["1"]["manifest"]
    assert manifest["category"] == "mobile" and "demo" in manifest["tags"]


def test_without_a_version_it_is_1_then_the_highest_plus_one(mini, catalog_server):
    assert push("ada/mini", mini, base=catalog_server.url, log=quiet).ref == "ada/mini:1"
    assert push("ada/mini", mini, base=catalog_server.url, log=quiet).ref == "ada/mini:2"


def test_taken_is_a_catalog_error_with_the_code(mini, catalog_server):
    push("ada/mini:1", mini, base=catalog_server.url, log=quiet)
    with pytest.raises(CatalogError) as e:
        push("ada/mini:1", mini, base=catalog_server.url, log=quiet)
    assert str(e.value) == "taken" and e.value.code == "taken"


def test_the_cli_says_to_change_the_version_on_taken(mini, catalog_server, monkeypatch, capsys):
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    assert main(["push", "ada/mini:1", str(mini)]) == 0
    assert main(["push", "ada/mini:1", str(mini)]) == 1
    assert "そのバージョンはもうあります。バージョンを変えてください" in capsys.readouterr().err


def test_the_cli_pushes_privately_unless_public(mini, catalog_server, monkeypatch):
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    assert main(["push", "ada/mini:1", str(mini)]) == 0
    assert catalog_server.items[("ada", "mini")]["visibility"] == "private"
    assert ("POST", "/api/catalog/ada/mini/1?visibility=private") in catalog_server.calls
    assert main(["push", "ada/mini:2", str(mini), "--private"]) == 0  # accepted, the default anyway
    assert main(["push", "ada/pub:1", str(mini), "--public"]) == 0
    assert ("POST", "/api/catalog/ada/pub/1?visibility=public") in catalog_server.calls


def test_pushing_under_another_handle_is_refused(mini, catalog_server):
    with pytest.raises(InputError, match="ada"):
        push("someone/mini:1", mini, base=catalog_server.url, log=quiet)


def test_no_handle_yet_says_to_decide_one(mini, catalog_server):
    catalog_server.handle = None
    with pytest.raises(InputError, match=r"decide your handle on PS.com first \(/catalog/me\)"):
        push("ada/mini:1", mini, base=catalog_server.url, log=quiet)


def test_an_object_without_unit_is_refused(tmp_path, catalog_server):
    folder = tmp_path / "statue"
    folder.mkdir()
    shutil.copy(MINI / "meshes" / "box.stl", folder / "statue.stl")
    obj = {"name": "statue", "display": {"ja": "像", "en": "Statue"}, "license": "CC0-1.0", "category": "figure"}
    (folder / "object.json").write_text(json.dumps(obj))
    with pytest.raises(InputError, match="unit"):
        push("ada/statue:1", folder, base=catalog_server.url, log=quiet)
    obj["unit"] = "mm"
    (folder / "object.json").write_text(json.dumps(obj))
    push("ada/statue:1", folder / "statue.stl", base=catalog_server.url, log=quiet)
    manifest = catalog_server.items[("ada", "statue")]["versions"]["1"]["manifest"]
    assert manifest["kind"] == "object" and manifest["unit"] == "mm" and manifest["category"] == "figure"
    assert "class" not in manifest["product"] and manifest["license"] == "CC0-1.0"
    paths = {f["path"] for f in manifest["files"]}
    assert {"bundle.json", "statue.urdf", "statue.stl", "object.json"} <= paths


def _machine_dir(root: Path, product_id: str) -> Path:
    d = root / product_id
    shutil.copytree(MINI, d)
    product = json.loads((d / "product.json").read_text())
    product["id"] = product_id
    (d / "product.json").write_text(json.dumps(product))
    return d


def _record(path: Path, products) -> Path:
    motions = [motion_from_numpy(np.zeros((3, 1)), fps=10, joint_names=["pan"], machine=p) for p in products]
    save_record(motions, "site-1", path)
    return path


def test_a_record_brings_its_machines_along_privately(tmp_path, catalog_server):
    machines = tmp_path / "machines"
    _machine_dir(machines, "mini")
    _machine_dir(machines, "other")
    rec = _record(tmp_path / "run.json", ["mini", "other", "mini"])
    said = []
    result = push("ada/run:1", rec, private=False, machine_dir=machines, base=catalog_server.url, log=said.append)
    assert sorted(result.companions) == ["ada/mini:1", "ada/other:1"]
    assert any("ada/mini:1 として非公開で一緒に上げます" in s for s in said)
    for name in ("mini", "other"):
        assert catalog_server.items[("ada", name)]["visibility"] == "private"
        assert catalog_server.items[("ada", name)]["versions"]["1"]["finished"]
    stored = catalog_server.items[("ada", "run")]
    assert stored["kind"] == "record" and stored["visibility"] == "public"
    products = [d["product"] for d in stored["versions"]["1"]["record"]["equipment"]["devices"]]
    assert products == ["ada/mini:1", "ada/other:1", "ada/mini:1"]
    assert ("PUT", "/api/catalog/ada/run/1/record?visibility=public") in catalog_server.calls
    assert json.loads(rec.read_text())["equipment"]["devices"][0]["product"] == "mini"  # the file is left alone


def test_the_same_machine_already_there_is_used(tmp_path, catalog_server):
    machines = tmp_path / "machines"
    d = _machine_dir(machines, "mini")
    push("ada/mini:1", d, base=catalog_server.url, log=quiet)
    rec = _record(tmp_path / "run.json", ["mini"])
    result = push("ada/run:1", rec, machine_dir=machines, base=catalog_server.url, log=quiet)
    assert result.companions == []
    record = catalog_server.items[("ada", "run")]["versions"]["1"]["record"]
    assert record["equipment"]["devices"][0]["product"] == "ada/mini:1"


def test_a_changed_machine_goes_up_as_the_next_version(tmp_path, catalog_server):
    machines = tmp_path / "machines"
    d = _machine_dir(machines, "mini")
    push("ada/mini:1", d, base=catalog_server.url, log=quiet)
    (d / "notes.txt").write_text("changed\n")
    rec = _record(tmp_path / "run.json", ["mini"])
    result = push("ada/run:1", rec, machine_dir=machines, base=catalog_server.url, log=quiet)
    assert result.companions == ["ada/mini:2"]
    record = catalog_server.items[("ada", "run")]["versions"]["1"]["record"]
    assert record["equipment"]["devices"][0]["product"] == "ada/mini:2"


def test_the_machine_cache_is_searched_and_a_missing_machine_says_machine_dir(tmp_path, catalog_server):
    rec = _record(tmp_path / "run.json", ["mini"])
    with pytest.raises(InputError, match="--machine-dir で機体の URDF の場所を指定してください"):
        push("ada/run:1", rec, base=catalog_server.url, log=quiet)
    _machine_dir(tmp_path / "cache" / "machines", "mini")
    assert push("ada/run:1", rec, base=catalog_server.url, log=quiet).companions == ["ada/mini:1"]


def test_a_record_naming_catalog_machines_needs_no_companion(tmp_path, catalog_server):
    rec = _record(tmp_path / "run.json", ["mini"])
    data = json.loads(rec.read_text())
    data["equipment"]["devices"][0]["product"] = "physical-sandbox/so101:1"
    rec.write_text(json.dumps(data))
    result = push("ada/run:1", rec, base=catalog_server.url, log=quiet)
    assert result.companions == []
    assert [m for m, p in writes(catalog_server)] == ["PUT"]


def test_without_companions_a_records_bare_names_go_as_they_are(tmp_path, catalog_server):
    rec = _record(tmp_path / "run.json", ["so101"])  # PS.com's own product: no local machine of that name
    result = push("ada/run", rec, base=catalog_server.url, log=quiet, companions=False)
    assert result.companions == [] and [m for m, p in writes(catalog_server)] == ["PUT"]
    record = catalog_server.items[("ada", "run")]["versions"]["1"]["record"]
    assert record["equipment"]["devices"][0]["product"] == "so101"


def test_a_json_that_is_not_a_record_is_refused(tmp_path, catalog_server):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"schemaVersion": 3, "traces": []}))
    with pytest.raises(InputError):
        push("ada/run:1", p, base=catalog_server.url, log=quiet)


def test_login_keeps_the_token_and_reuses_it(home, catalog_server):
    said = []
    assert login(catalog_server.url, log=said.append) == "tok-1"
    assert any(f"Open {catalog_server.url}/vr and enter the code 123456" in s for s in said)
    kept = home / ".config" / "physical-sandbox" / "ps-com-token.json"
    data = json.loads(kept.read_text())
    assert data["base"] == catalog_server.url and data["token"] == "tok-1"
    assert datetime.fromisoformat(data["expiresAt"]) > datetime.now(timezone.utc)
    assert os.stat(kept).st_mode & 0o777 == 0o600
    assert login(catalog_server.url, log=said.append) == "tok-1"
    assert catalog_server.vr_starts == 1


def test_login_starts_again_when_the_kept_token_is_expired_or_for_another_base(home, catalog_server):
    kept = home / ".config" / "physical-sandbox" / "ps-com-token.json"
    kept.parent.mkdir(parents=True)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    kept.write_text(json.dumps({"base": catalog_server.url, "token": "old", "expiresAt": past}))
    assert login(catalog_server.url, log=quiet) == "tok-1"
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    kept.write_text(json.dumps({"base": "https://elsewhere.example", "token": "old", "expiresAt": future}))
    assert login(catalog_server.url, log=quiet) == "tok-1"
    assert catalog_server.vr_starts == 2


def test_a_rejected_kept_token_is_dropped_and_login_runs_again(home, mini, catalog_server):
    kept = home / ".config" / "physical-sandbox" / "ps-com-token.json"
    kept.parent.mkdir(parents=True)
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    kept.write_text(json.dumps({"base": catalog_server.url, "token": "revoked", "expiresAt": future}))
    push("ada/mini:1", mini, base=catalog_server.url, log=quiet)
    assert catalog_server.vr_starts == 1
    assert json.loads(kept.read_text())["token"] == "tok-1"


def test_the_mcp_tool_asks_for_psbx_login_first_then_pushes_privately(mini, catalog_server, monkeypatch):
    from physical_sandbox.mcp.server import ps_catalog_push

    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    assert json.loads(ps_catalog_push("ada/mini:1", str(mini)))["error"] == "signin"
    assert catalog_server.vr_starts == 0
    login(catalog_server.url, log=quiet)
    out = json.loads(ps_catalog_push("ada/mini:1", str(mini)))
    assert out["ref"] == "ada/mini:1" and out["companions"] == [] and out["private"] is True
    assert catalog_server.items[("ada", "mini")]["visibility"] == "private"


def _keep_token(home, base, token):
    kept = home / ".config" / "physical-sandbox" / "ps-com-token.json"
    kept.parent.mkdir(parents=True, exist_ok=True)
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    kept.write_text(json.dumps({"base": base, "token": token, "expiresAt": future}))
    return kept


def test_the_mcp_tool_answers_signin_when_the_kept_token_is_rejected(home, mini, catalog_server, monkeypatch):
    from physical_sandbox.mcp.server import ps_catalog_push

    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    kept = _keep_token(home, catalog_server.url, "revoked")
    out = json.loads(ps_catalog_push("ada/mini:1", str(mini)))
    assert out["error"] == "signin" and "psbx login" in out["message"]
    assert catalog_server.vr_starts == 0
    assert not any(p.startswith("/api/vr/") for _, p in catalog_server.calls)
    assert not kept.exists()  # the rejected token is dropped


def test_a_failed_file_says_the_version_is_left_unfinished(mini, catalog_server):
    catalog_server.fail_put = "bad-hash"
    with pytest.raises(CatalogError) as e:
        push("ada/mini:3", mini, base=catalog_server.url, log=quiet)
    assert e.value.code == "bad-hash"
    assert "version 3 of ada/mini is left unfinished; push again with the next version" in str(e.value)


def test_a_failed_companion_says_what_finished_and_that_the_record_did_not_go_up(tmp_path, catalog_server):
    machines = tmp_path / "machines"
    _machine_dir(machines, "aaa")
    _machine_dir(machines, "bbb")
    rec = _record(tmp_path / "run.json", ["aaa", "bbb"])
    real = catalog_server.write

    def failing(method, raw_path, body):  # the second machine's files fail
        if method == "PUT" and "/bbb/" in raw_path and "/files/" in raw_path:
            return catalog_server.reply(400, {"error": "bad-size"})
        return real(method, raw_path, body)

    catalog_server.write = failing
    with pytest.raises(CatalogError) as e:
        push("ada/run:1", rec, machine_dir=machines, base=catalog_server.url, log=quiet)
    text = str(e.value)
    assert "ada/bbb failed" in text and "companions finished: ada/aaa:1" in text
    assert "the record was not uploaded" in text and ("ada", "run") not in catalog_server.items


def test_a_link_to_a_file_outside_the_folder_is_left_out(tmp_path, mini, catalog_server):
    secret = tmp_path / "secret.txt"
    secret.write_text("not for the catalog\n")
    (mini / "secret.txt").symlink_to(secret)
    (mini / "notes-link.txt").symlink_to(mini / "product.json")  # a link inside the folder stays
    said = []
    push("ada/mini:1", mini, base=catalog_server.url, log=said.append)
    paths = {f["path"] for f in catalog_server.items[("ada", "mini")]["versions"]["1"]["manifest"]["files"]}
    assert "secret.txt" not in paths and "notes-link.txt" in paths
    assert any("secret.txt" in s and "outside" in s for s in said)


def test_a_new_version_of_a_public_item_says_public_and_warns(mini, catalog_server, capsys, monkeypatch):
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    login(catalog_server.url, log=quiet)
    push("ada/mini:1", mini, private=False, base=catalog_server.url, log=quiet)
    said = []
    result = push("ada/mini:2", mini, base=catalog_server.url, log=said.append)  # the private default
    assert result.visibility == "public"
    assert any("already public" in s for s in said)
    assert main(["push", "ada/mini:3", str(mini)]) == 0
    out = capsys.readouterr().out
    assert "already public" in out and "(public," in out and "(private," not in out


def test_a_new_item_keeps_the_flag_without_a_warning(mini, catalog_server):
    said = []
    result = push("ada/fresh:1", mini, base=catalog_server.url, log=said.append)
    assert result.visibility == "private" and not any("already" in s for s in said)
    assert push("ada/fresh2:1", mini, private=False, base=catalog_server.url, log=quiet).visibility == "public"


def test_a_record_version_of_a_public_item_reports_public(tmp_path, catalog_server):
    machines = tmp_path / "machines"
    _machine_dir(machines, "mini")
    rec = _record(tmp_path / "run.json", ["mini"])
    push("ada/run:1", rec, private=False, machine_dir=machines, base=catalog_server.url, log=quiet)
    said = []
    result = push("ada/run:2", rec, machine_dir=machines, base=catalog_server.url, log=said.append)
    assert result.visibility == "public" and any("already public" in s for s in said)
    assert result.companion_visibility == {}


def test_login_status_reads_the_token_file_only(catalog_server, home):
    assert push_mod.login_status(catalog_server.url) == {"signedIn": False, "handle": None}
    push_mod._keep(catalog_server.url, "tok-1", 3600, handle="ada")
    assert push_mod.login_status(catalog_server.url) == {"signedIn": True, "handle": "ada"}
    assert catalog_server.hits == 0


def test_an_expired_token_is_not_signed_in(catalog_server, home):
    push_mod._keep(catalog_server.url, "tok-1", -1, handle="ada")
    assert push_mod.login_status(catalog_server.url) == {"signedIn": False, "handle": None}


def test_a_token_for_another_base_is_not_signed_in(catalog_server, home):
    push_mod._keep("https://elsewhere.example", "tok-1", 3600, handle="ada")
    assert push_mod.login_status(catalog_server.url) == {"signedIn": False, "handle": None}


def test_signing_in_remembers_the_handle(catalog_server, home):
    push_mod._signed_in(catalog_server.url, log=lambda s: None)
    kept = json.loads(push_mod.token_file().read_text())
    assert kept["handle"] == "ada" and kept["token"] == "tok-1"
    assert os.stat(push_mod.token_file()).st_mode & 0o777 == 0o600
    assert push_mod.login_status(catalog_server.url) == {"signedIn": True, "handle": "ada"}


def test_a_token_kept_without_a_handle_is_signed_in_with_no_handle(catalog_server, home):
    push_mod._keep(catalog_server.url, "tok-1", 3600)
    assert push_mod.login_status(catalog_server.url) == {"signedIn": True, "handle": None}


def test_the_cli_login_says_the_handle_and_keeps_it(catalog_server, monkeypatch, capsys):
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    assert main(["login"]) == 0
    assert f"signed in as ada; the token is kept in {push_mod.token_file()}" in capsys.readouterr().out
    assert push_mod.login_status(catalog_server.url) == {"signedIn": True, "handle": "ada"}


# ---- a record is saved as new or over, never in place ----

def _catalog_record(path: Path) -> Path:
    """A record whose only machine is already a catalog name, so nothing goes up with it."""
    rec = _record(path, ["mini"])
    data = json.loads(rec.read_text())
    data["equipment"]["devices"][0]["product"] = "physical-sandbox/so101:1"
    rec.write_text(json.dumps(data))
    return rec


def _change(rec: Path, title: str) -> None:
    data = json.loads(rec.read_text())
    data["note"] = title
    rec.write_text(json.dumps(data))


def record_puts(fake):
    return [p for m, p in fake.calls if m == "PUT" and "/record" in p]


def test_a_new_record_without_a_version_asks_for_a_new_item(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    result = push("ada/run", rec, base=catalog_server.url, log=quiet)
    assert record_puts(catalog_server) == ["/api/catalog/ada/run/record?new=1&visibility=private"]
    assert result.ref == "ada/run:1" and result.version == "1" and result.unchanged is False
    assert result.bytes == len(catalog_server.items[("ada", "run")]["versions"]["1"]["raw"])


def test_a_record_that_is_there_is_not_saved_without_over(tmp_path, catalog_server):
    machines = tmp_path / "machines"
    _machine_dir(machines, "mini")
    rec = _record(tmp_path / "run.json", ["mini"])
    push("ada/run", rec, machine_dir=machines, base=catalog_server.url, log=quiet)
    before = list(catalog_server.calls)
    _change(rec, "second")
    with pytest.raises(push_mod.RecordExists) as e:  # an InputError the stage can tell apart by its type
        push("ada/run", rec, machine_dir=machines, base=catalog_server.url, log=quiet)
    assert isinstance(e.value, InputError)
    assert str(e.value) == ("ada/run はもうあります。上書き保存するときは --over を付けてください。"
                            "新しい記録にするときは別の名前にしてください")
    assert [c for c in catalog_server.calls[len(before):] if c[0] != "GET"] == []  # no machine went up either
    assert list(catalog_server.items[("ada", "run")]["versions"]) == ["1"]


def test_over_saves_the_next_version_and_keeps_the_one_before(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    push("ada/run", rec, base=catalog_server.url, log=quiet)
    first = catalog_server.items[("ada", "run")]["versions"]["1"]["raw"]
    _change(rec, "second")
    result = push("ada/run", rec, over=True, base=catalog_server.url, log=quiet)
    assert result.ref == "ada/run:2" and result.version == "2" and result.unchanged is False
    assert record_puts(catalog_server)[-1] == "/api/catalog/ada/run/record?visibility=private"
    versions = catalog_server.items[("ada", "run")]["versions"]
    assert versions["1"]["raw"] == first and versions["2"]["record"]["note"] == "second"


def test_over_with_the_same_bytes_makes_no_version(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    push("ada/run", rec, base=catalog_server.url, log=quiet)
    result = push("ada/run", rec, over=True, base=catalog_server.url, log=quiet)
    assert result.unchanged is True and result.version == "1" and result.ref == "ada/run:1"
    assert list(catalog_server.items[("ada", "run")]["versions"]) == ["1"]


def test_over_a_base_that_is_still_the_latest_saves_the_next_version(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    push("ada/run", rec, base=catalog_server.url, log=quiet)
    _change(rec, "second")
    result = push("ada/run", rec, over=True, over_base="1", base=catalog_server.url, log=quiet)
    assert result.version == "2"
    assert record_puts(catalog_server)[-1] == "/api/catalog/ada/run/record?base=1&visibility=private"


def test_over_a_base_that_moved_is_refused_and_names_the_latest(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    push("ada/run", rec, base=catalog_server.url, log=quiet)
    _change(rec, "elsewhere")
    push("ada/run", rec, over=True, base=catalog_server.url, log=quiet)  # saved over somewhere else: 2
    _change(rec, "mine")
    with pytest.raises(PushError) as e:
        push("ada/run", rec, over=True, over_base="1", base=catalog_server.url, log=quiet)
    assert e.value.code == "moved" and e.value.status == 409
    assert "バージョン 2" in e.value.note and "--over 2" in e.value.note
    assert list(catalog_server.items[("ada", "run")]["versions"]) == ["1", "2"]


def test_over_base_without_over_is_refused_before_anything_is_sent(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    with pytest.raises(InputError, match="--over"):
        push("ada/run", rec, over_base="1", base=catalog_server.url, log=quiet)
    assert catalog_server.calls == []


def test_over_a_bad_base_is_refused_before_anything_is_sent(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    with pytest.raises(InputError, match="Bad Base"):
        push("ada/run", rec, over=True, over_base="Bad Base", base=catalog_server.url, log=quiet)
    assert catalog_server.calls == []


def test_over_with_a_version_or_for_a_machine_is_refused(tmp_path, mini, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    with pytest.raises(InputError, match="--over"):
        push("ada/run:3", rec, over=True, base=catalog_server.url, log=quiet)
    with pytest.raises(InputError, match="--over"):
        push("ada/mini", mini, over=True, base=catalog_server.url, log=quiet)
    assert catalog_server.calls == []


def test_over_a_record_that_is_not_there_makes_it(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    result = push("ada/run", rec, over=True, base=catalog_server.url, log=quiet)
    assert result.version == "1"
    assert record_puts(catalog_server) == ["/api/catalog/ada/run/record?new=1&visibility=private"]


def test_a_record_version_that_is_there_is_never_written_over(tmp_path, catalog_server):
    rec = _catalog_record(tmp_path / "run.json")
    push("ada/run:1", rec, base=catalog_server.url, log=quiet)
    first = catalog_server.items[("ada", "run")]["versions"]["1"]["raw"]
    _change(rec, "second")
    with pytest.raises(PushError) as e:
        push("ada/run:1", rec, base=catalog_server.url, log=quiet)
    assert e.value.code == "taken" and "そのバージョンはもうあります。バージョンを変えてください" in e.value.note
    assert catalog_server.items[("ada", "run")]["versions"]["1"]["raw"] == first


def test_over_brings_new_machines_along_as_before(tmp_path, catalog_server):
    machines = tmp_path / "machines"
    _machine_dir(machines, "mini")
    rec = _record(tmp_path / "run.json", ["mini"])
    push("ada/run", rec, machine_dir=machines, base=catalog_server.url, log=quiet)
    _change(rec, "second")
    result = push("ada/run", rec, over=True, machine_dir=machines, base=catalog_server.url, log=quiet)
    assert result.version == "2" and result.companions == []  # the same machine is used again
    record = catalog_server.items[("ada", "run")]["versions"]["2"]["record"]
    assert record["equipment"]["devices"][0]["product"] == "ada/mini:1"


def test_a_saved_record_can_be_pulled_back(tmp_path, catalog_server):
    from physical_sandbox import catalog

    rec = _catalog_record(tmp_path / "run.json")
    push("ada/run", rec, base=catalog_server.url, log=quiet)
    d = catalog.pull("ada/run:1", base=catalog_server.url, token="tok-1")
    manifest = json.loads((d / "manifest.json").read_text())
    assert manifest["kind"] == "record" and [f["path"] for f in manifest["files"]] == ["record.json"]
    assert (d / "record.json").read_bytes() == catalog_server.items[("ada", "run")]["versions"]["1"]["raw"]


def test_a_machine_push_says_its_version(mini, catalog_server):
    result = push("ada/mini", mini, base=catalog_server.url, log=quiet)
    assert result.version == "1" and result.unchanged is False


def test_the_mcp_tool_saves_a_record_over_a_base(tmp_path, catalog_server, monkeypatch):
    from physical_sandbox.mcp.server import ps_catalog_push

    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url)
    login(catalog_server.url, log=quiet)
    rec = _catalog_record(tmp_path / "run.json")
    assert json.loads(ps_catalog_push("ada/run", str(rec)))["version"] == "1"
    _change(rec, "second")
    out = json.loads(ps_catalog_push("ada/run", str(rec), over=True, base="1"))
    assert out["ref"] == "ada/run:2" and out["version"] == "2" and out["unchanged"] is False
    assert record_puts(catalog_server)[-1] == "/api/catalog/ada/run/record?base=1&visibility=private"
    assert json.loads(ps_catalog_push("ada/run", str(rec), over=True))["unchanged"] is True
