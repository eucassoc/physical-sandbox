import pytest

from physical_sandbox import catalog
from physical_sandbox.errors import CatalogError, InputError
from physical_sandbox.machine import load_machine


@pytest.fixture(autouse=True)
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_parse_ref_splits_handle_name_and_version():
    assert catalog.parse_ref("t/mini:1") == ("t", "mini", "1")
    assert catalog.parse_ref("t/mini") == ("t", "mini", "latest")
    for bad in ("mini", "T/mini", "t/mini:", "t/-m", "a/b/c"):
        with pytest.raises(InputError):
            catalog.parse_ref(bad)


def test_search_returns_the_items(catalog_server):
    items = catalog.search("arm", base=catalog_server.url)
    assert [i["name"] for i in items] == ["mini"]


def test_pull_puts_the_files_in_the_cache_without_partials(catalog_server, cache):
    d = catalog.pull("t/mini:1", base=catalog_server.url)
    assert d == cache / "catalog" / "t" / "mini" / "1"
    assert (d / "manifest.json").is_file() and (d / "mini.urdf").is_file() and (d / "notes.txt").is_file()
    assert not [p for p in d.parent.iterdir() if ".partial-" in p.name]


def test_pull_latest_resolves_and_remembers_the_version(catalog_server, cache):
    d = catalog.pull("t/mini", base=catalog_server.url)
    assert d.name == "1"
    assert (cache / "catalog" / "t" / "mini" / "latest").read_text().strip() == "1"


def test_pull_with_a_wrong_sha256_leaves_nothing(bad_catalog_server, cache):
    with pytest.raises(CatalogError):
        catalog.pull("t/mini:1", base=bad_catalog_server.url)
    parent = cache / "catalog" / "t" / "mini"
    assert not (parent / "1").exists()
    assert not parent.exists() or not [p for p in parent.iterdir() if ".partial-" in p.name]


def test_second_pull_stays_off_the_network_and_latest_works_offline(catalog_server, cache):
    first = catalog.pull("t/mini", base=catalog_server.url)
    hits = catalog_server.hits
    assert catalog.pull("t/mini:1", base=catalog_server.url) == first
    assert catalog_server.hits == hits
    catalog_server.stop()
    assert catalog.pull("t/mini", base=catalog_server.url) == first


def test_urdf_path_points_at_the_urdf_and_it_loads(catalog_server):
    d = catalog.pull("t/mini:1", base=catalog_server.url)
    assert catalog.urdf_path(d).name == "mini.urdf"
    assert load_machine(catalog.urdf_path(d)).joint_names == ["turn"]


@pytest.mark.parametrize("hostile", ["../../../x", 7, "A/b"])
def test_pull_rejects_a_hostile_latest_version(catalog_server, cache, hostile):
    catalog_server.latest = hostile
    with pytest.raises(CatalogError):
        catalog.pull("t/mini", base=catalog_server.url)
    assert not [p for p in cache.rglob("*") if p.is_file()]
    assert not (cache.parent / "x").exists()


def test_a_hostile_latest_marker_is_ignored(cache):
    (cache / "catalog" / "t" / "mini").mkdir(parents=True)
    (cache / "catalog" / "t" / "mini" / "latest").write_text("../../x\n")
    assert catalog.cached("t/mini") is None


@pytest.mark.parametrize("path", ["../evil.txt", "/etc/evil.txt"])
def test_pull_rejects_unsafe_manifest_paths(catalog_server, cache, path):
    catalog_server.extra_files = [{"path": path, "bytes": 1, "sha256": "0" * 64}]
    with pytest.raises(CatalogError):
        catalog.pull("t/mini:1", base=catalog_server.url)
    parent = cache / "catalog" / "t" / "mini"
    assert not (parent / "1").exists()
    assert not parent.exists() or not list(parent.iterdir())


def test_an_uppercase_sha256_in_the_manifest_still_verifies(catalog_server):
    catalog_server.upper = True
    assert catalog.pull("t/mini:1", base=catalog_server.url).is_dir()


def test_a_broken_transfer_is_a_catalog_error(catalog_server):
    catalog_server.stop()
    with pytest.raises(CatalogError):
        catalog.search("arm", base=catalog_server.url)
