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


def test_parse_ref_accepts_two_character_names_but_not_one_or_short_handles():
    assert catalog.parse_ref("physical-sandbox/g1") == ("physical-sandbox", "g1", "latest")
    for bad in ("physical-sandbox/g", "ab/g1"):
        with pytest.raises(InputError):
            catalog.parse_ref(bad)


def test_requests_send_a_psbx_user_agent(catalog_server):
    from physical_sandbox import __version__
    catalog.search("arm", base=catalog_server.url)
    assert catalog_server.headers[-1]["User-Agent"] == f"psbx/{__version__}"


def test_base_url_is_read_at_call_time(catalog_server, monkeypatch):
    monkeypatch.setenv("PS_CATALOG_URL", catalog_server.url + "/")
    assert [i["name"] for i in catalog.search("arm")] == ["mini"]
    assert catalog_server.paths[-1].startswith("/api/catalog/search")


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


def test_search_and_pull_send_the_token_when_given(catalog_server, cache):
    catalog.search("mini", base=catalog_server.url, token="tok-1")
    assert catalog_server.headers[-1].get("Authorization") == "Bearer tok-1"
    catalog.pull("t/mini:1", base=catalog_server.url, token="tok-1")
    assert all(h.get("Authorization") == "Bearer tok-1" for h in catalog_server.headers[-3:])


def test_search_and_pull_send_no_token_without_one(catalog_server, cache):
    catalog.search("mini", base=catalog_server.url)
    assert "Authorization" not in catalog_server.headers[-1]


def test_pull_latest_sends_the_token_to_the_version_lookup_too(catalog_server, cache):
    catalog.pull("t/mini", base=catalog_server.url, token="tok-1")
    assert catalog_server.headers and all(h.get("Authorization") == "Bearer tok-1" for h in catalog_server.headers)


def test_search_marks_private_items(catalog_server):
    items = catalog.search("mini", base=catalog_server.url)
    assert all(isinstance(i["private"], bool) for i in items)
    assert items[0]["private"] == (items[0].get("visibility") == "private")


def test_search_sends_owner_and_limit_and_owner_me_needs_the_token(catalog_server):
    catalog.search(kind="record", owner="me", limit=500, base=catalog_server.url, token="tok-1")
    assert catalog_server.paths[-1] == "/api/catalog/search?kind=record&owner=me&limit=500"
    with pytest.raises(CatalogError, match="401"):
        catalog.search(kind="record", owner="me", base=catalog_server.url)


def test_the_timeout_reaches_every_request(catalog_server, cache, monkeypatch):
    seen = []
    real = catalog.urllib.request.urlopen
    monkeypatch.setattr(catalog.urllib.request, "urlopen", lambda req, timeout: seen.append(timeout) or real(req))
    catalog.search("mini", base=catalog_server.url, timeout=2)
    catalog.pull("t/mini", base=catalog_server.url, timeout=3)
    assert seen[0] == 2 and len(seen) > 2 and set(seen[1:]) == {3}
