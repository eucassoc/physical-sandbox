"""psbx machine check / init, and a mesh-only object wrapped as a one-link URDF."""
import json
import shutil
from pathlib import Path

import pytest

from physical_sandbox.bundle import make_bundle
from physical_sandbox.cli import main
from physical_sandbox.errors import InputError
from physical_sandbox.machine_tools import check, init, wrap_object

MINI = Path(__file__).parent / "fixtures" / "mini-robot"


@pytest.fixture
def mini(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    shutil.copytree(MINI, tmp_path / "mini")
    return tmp_path / "mini"


def red(result):
    return [text for ok, text in result.items if not ok]


def test_a_complete_machine_folder_is_green(mini):
    result = check(mini)
    assert result.ok, red(result)
    assert all(ok for ok, _ in result.items) and len(result.items) >= 5


def test_a_missing_mesh_is_red_and_named(mini):
    (mini / "meshes" / "box.stl").unlink()
    result = check(mini)
    assert not result.ok
    assert any("box.stl" in t for t in red(result))


def test_no_product_json_is_red(mini):
    (mini / "product.json").unlink()
    result = check(mini)
    assert not result.ok
    assert any("product.json" in t for t in red(result))


def test_product_fields_left_to_fill_are_red(mini):
    product = json.loads((mini / "product.json").read_text())
    product["displayName"]["ja"] = "<fill>"
    del product["class"]
    (mini / "product.json").write_text(json.dumps(product))
    reds = " ".join(red(check(mini)))
    assert "displayName.ja" in reds and "class" in reds


def test_a_license_that_is_not_spdx_is_red(mini):
    product = json.loads((mini / "product.json").read_text())
    product["licenseId"] = "free for research"
    (mini / "product.json").write_text(json.dumps(product))
    assert any("SPDX" in t for t in red(check(mini)))
    product["licenseId"] = "MIT OR Apache-2.0"
    (mini / "product.json").write_text(json.dumps(product))
    assert check(mini).ok


def test_wheels_must_be_continuous_joints_of_the_urdf(mini):
    profile = json.loads((mini / "profile.json").read_text())
    profile["wheels"] = {"radius": 0.05, "halfTrack": 0.1, "joints": [{"name": "pan", "spin": 1},
                                                                        {"name": "nowhere", "spin": 1}]}
    (mini / "profile.json").write_text(json.dumps(profile))
    reds = " ".join(red(check(mini)))
    assert "pan" in reds and "continuous" in reds and "nowhere" in reds
    urdf = (mini / "mini.urdf").read_text().replace('type="revolute"', 'type="continuous"')
    (mini / "mini.urdf").write_text(urdf)
    profile["wheels"]["joints"].pop()
    (mini / "profile.json").write_text(json.dumps(profile))
    assert check(mini).ok, red(check(mini))


def test_a_folder_without_a_urdf_is_red(tmp_path):
    result = check(tmp_path)
    assert not result.ok and any("URDF" in t for t in red(result))


def test_init_writes_templates_for_a_urdf_folder(mini):
    (mini / "product.json").unlink()
    (mini / "profile.json").unlink()
    written = init(mini)
    assert sorted(p.name for p in written) == ["product.json", "profile.json"]
    product = json.loads((mini / "product.json").read_text())
    assert product["displayName"] == {"ja": "<fill>", "en": "<fill>"}
    assert product["licenseId"] == "<fill>" and product["class"] == "<fill>"
    profile = json.loads((mini / "profile.json").read_text())
    assert profile["totalMassKg"] == pytest.approx(0.6)  # the URDF's masses summed
    assert profile["chains"][0]["jointNames"] == ["pan"]
    assert not check(mini).ok  # the template still has to be filled in
    assert init(mini) == []  # never overwrites


def test_init_on_an_stl_writes_object_json(tmp_path):
    stl = tmp_path / "statue.stl"
    shutil.copy(MINI / "meshes" / "box.stl", stl)
    written = init(stl)
    assert written == [tmp_path / "object.json"]
    obj = json.loads(written[0].read_text())
    assert obj == {"name": "statue", "display": {"ja": "<fill>", "en": "<fill>"}, "license": "<fill>",
                   "category": "figure", "unit": "<m|mm>"}


def test_wrap_object_in_mm_scales_by_a_thousandth(tmp_path):
    stl = tmp_path / "src" / "statue.stl"
    stl.parent.mkdir()
    shutil.copy(MINI / "meshes" / "box.stl", stl)
    urdf = wrap_object(stl, unit="mm", out_dir=tmp_path / "wrapped")
    text = urdf.read_text()
    assert 'scale="0.001 0.001 0.001"' in text
    result = make_bundle(urdf, product={}, profile=None, name="statue", out_dir=tmp_path / "bundle")
    assert result.links == 1 and result.joints == [] and len(result.mesh_paths) == 1
    assert "scale" not in wrap_object(stl, unit="m", out_dir=tmp_path / "wrapped-m").read_text()
    with pytest.raises(InputError, match="unit"):
        wrap_object(stl, unit="cm", out_dir=tmp_path / "x")


def test_cli_machine_check_and_init(mini, tmp_path, capsys):
    assert main(["machine", "check", str(mini)]) == 0
    assert "product.json" in capsys.readouterr().out
    (mini / "meshes" / "box.stl").unlink()
    assert main(["machine", "check", str(mini)]) == 1
    assert "box.stl" in capsys.readouterr().out
    stl = tmp_path / "thing.stl"
    shutil.copy(MINI / "meshes" / "box.stl", stl)
    assert main(["machine", "init", str(stl)]) == 0
    assert "object.json" in capsys.readouterr().out
