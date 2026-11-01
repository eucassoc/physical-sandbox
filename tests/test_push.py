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
from physical_sandbox.push import login, push

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
    assert "その版はあります。版を変えてください" in capsys.readouterr().err


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
