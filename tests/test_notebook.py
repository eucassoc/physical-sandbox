"""The notebook surface: ``physical_sandbox.compare`` on a work folder, and what its result shows in a cell."""
import json
import math

import numpy as np
import pytest

import physical_sandbox
from physical_sandbox.errors import InputError
from physical_sandbox.motion import motion_from_numpy, save_record

from mini_lerobot import make_dataset
from test_stage import call, cache_mini, decided, home, served, still  # noqa: F401  (fixtures)


@pytest.fixture
def folder(decided, cache_mini, tmp_path):
    """A work folder the stage screen left: a decided floor and a placed machine (joint ``turn``)."""
    assert "scenePath" in call(decided, "ps_scene_place", {"id": "mini", "urdf": cache_mini, "x": 0, "y": 0, "yaw": 0})
    return tmp_path / "w"


def _record(path, label, q):
    save_record([motion_from_numpy(q, fps=10, joint_names=["turn"], machine="mini", label=label)], "site", path)
    return str(path)


@pytest.fixture
def two(folder):
    a = _record(folder / "demo.record.json", "demo", np.linspace(0, 1, 6).reshape(-1, 1))
    b = _record(folder / "policy.record.json", "policy", np.linspace(0, 0.5, 6).reshape(-1, 1))
    return a, b


def test_compare_on_the_work_folder_gives_the_report_numbers_and_the_site(folder, two):
    gaps = physical_sandbox.compare(folder, *two)
    d = gaps.to_dict()
    assert d["labels"] == {"a": "demo", "b": "policy"} and d["header"]["site"] == "site"
    assert d["header"]["dimensions"] == "measured" and d["tool_deviation_m"]["max"] > 0
    assert gaps.notes == [] and gaps.site.id == "site"
    assert json.dumps(d)  # plain JSON, as gaps.json would hold


def test_the_rows_are_the_csv_rows_and_row_is_one_flat_line(folder, two):
    gaps = physical_sandbox.compare(folder, *two)
    rows = gaps.rows()
    items = [r["item"] for r in rows]
    assert items[:3] == ["site", "residual.camera", "dimensions"] and "tool_deviation.max" in items
    assert items[-1].startswith("unknown.") and rows[-1]["value"] is None
    line = gaps.row()
    assert line["a"] == "demo" and line["b"] == "policy"
    assert line["tool_deviation.max"] == gaps.comparison.tool_deviation_m["max"]
    assert "unknown.world" not in line and "site" in line


def test_the_dataframe_has_one_row_per_item_when_pandas_is_there(folder, two):
    pd = pytest.importorskip("pandas")
    df = physical_sandbox.compare(folder, *two).to_dataframe()
    assert list(df.columns) == ["item", "value", "unit", "at_s", "note"]
    assert df.set_index("item").loc["tool_deviation.max", "unit"] == "m"


def test_the_cell_shows_the_screens_table_in_japanese(folder, two):
    html = physical_sandbox.compare(folder, *two)._repr_html_()
    assert "手先のずれ（同じ時刻）" in html and "関節 turn のずれ" in html
    assert "場所 site" in html and "実測" in html
    assert "置き場所の差（シミュレーターと現実）（まだ分かりません）" in html
    assert "<script" not in html


def test_the_figure_is_the_tool_gap_over_time(folder, two):
    fig = physical_sandbox.compare(folder, *two).figure()
    ax = fig.axes[0]
    assert ax.get_title().startswith("Tool gap over time: demo vs policy")
    (line,) = ax.get_lines()
    assert len(line.get_xdata()) > 0 and math.isclose(float(line.get_xdata()[0]), 0.0)


def test_a_lerobot_episode_and_a_grip_threshold_go_through(folder, monkeypatch):
    from physical_sandbox import lerobot

    monkeypatch.setitem(lerobot.JOINT_MAP, "turn", (-100.0, 100.0, 1))
    ds = make_dataset(folder.parent / "ds", ["turn"], [np.linspace(0, 1, 8).reshape(-1, 1),
                                                       np.linspace(0, 0.8, 8).reshape(-1, 1)])
    gaps = physical_sandbox.compare(folder, f"lerobot:{ds}#0", f"lerobot:{ds}#1", grip_threshold=0.3)
    assert gaps.to_dict()["labels"]["a"].endswith(" ep0") and gaps.comparison.grip_threshold == 0.3


def test_a_folder_without_a_placed_machine_or_a_bad_spec_says_so(decided, tmp_path, two):
    with pytest.raises((InputError, ValueError)):
        physical_sandbox.compare(tmp_path / "w", two[0], str(tmp_path / "w" / "none.record.json"))
    with pytest.raises((InputError, ValueError)) as e:
        physical_sandbox.compare(tmp_path / "nowhere", *two)
    assert "nowhere" in str(e.value)
