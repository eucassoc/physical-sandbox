"""The report of one comparison: the numbers (JSON, CSV) and a graph (PNG), headed by how good the site is."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .comparison import Comparison
from .site import Site

ESTIMATED_NOTE = "寸法は推定（未確認）"  # the dimensions are estimated (not confirmed)


def _header(site: Site) -> dict:
    return {
        "site": site.id,
        "residualPx": {k: float(v) for k, v in site.residual_px.items()},
        "dimensions": site.dimensions,
        "dimensionsNote": ESTIMATED_NOTE if site.dimensions == "estimated" else None,
        "confirmedByPerson": bool(site.confirmed_by_person),
    }


def _rows(d: dict, header: dict) -> list[tuple]:
    """(item, value, unit, at_s, note) for the site's quality, every scalar of ``Comparison.to_dict()`` (the
    series left out) and each thing the comparison cannot tell yet (``unknown.<name>``: no value, the reason
    in ``note``)."""
    rows: list[tuple] = [("site", header["site"], "", None, "")]
    for cam, px in header["residualPx"].items():
        rows.append((f"residual.{cam}", px, "px", None, ""))
    rows.append(("dimensions", header["dimensions"], "", None, header["dimensionsNote"] or ""))
    rows.append(("confirmed_by_person", str(header["confirmedByPerson"]).lower(), "", None, ""))
    tool = d["tool_deviation_m"]
    rows.append(("step", d["step_s"], "s", None, ""))
    rows.append(("tool_deviation.max", tool["max"], "m", tool["at_s"], ""))
    rows.append(("tool_deviation.mean", tool["mean"], "m", None, ""))
    rows.append(("shape_deviation", d["shape_deviation_m"], "m", None, ""))
    for name, j in d["joints"].items():
        rows.append((f"joints.{name}.max", j["max_rad"], "rad", j["at_s"], ""))
        rows.append((f"joints.{name}.mean", j["mean_rad"], "rad", None, ""))
    for k, v in d["duration_s"].items():
        rows.append((f"duration.{k}", v, "s", None, ""))
    for k, v in d["jerk"].items():
        rows.append((f"jerk.{k}", v, "m/s^3", None, ""))
    if d.get("grip_threshold") is not None:
        rows.append(("grip_threshold", d["grip_threshold"], "rad", None, "グリッパーがこの値を下回ったときを、つかむ瞬間とする"))
    found_none = "grasps" in d["unknown"]
    for k, v in d["grasp_counts"].items():
        rows.append((f"grasp_count.{k}", v, "count", None, d["unknown_reasons"].get("grasps", "") if found_none else ""))
    for i, g in enumerate(d["grasps"]):
        rows.append((f"grasps.{i}.tool_gap", g["tool_gap_m"], "m", g["a_s"], ""))
        rows.append((f"grasps.{i}.b_s", g["b_s"], "s", None, ""))
    for name in d["unknown"]:
        rows.append((f"unknown.{name}", None, "", None, d["unknown_reasons"].get(name, "")))
    return rows


_CJK_FONTS = ["Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", "Noto Sans JP", "IPAexGothic",
              "IPAGothic", "Yu Gothic", "Meiryo", "TakaoGothic", "VL Gothic"]


def _cjk_font() -> str | None:
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((n for n in _CJK_FONTS if n in have), None)


def _font_and_note(note: str | None) -> tuple[str | None, str | None]:
    """The CJK font to draw with and the note to draw: without such a font the Japanese note is said in English
    rather than drawn as empty boxes."""
    font = _cjk_font()
    if note and font is None:
        note = "dimensions estimated (not confirmed)"
    return font, note


def _plot(d: dict, path: Path, note: str | None) -> None:
    import matplotlib

    font, note = _font_and_note(note)
    with matplotlib.rc_context({"font.sans-serif": [font] if font else ["DejaVu Sans"]}):
        _draw(d, path, note)


def figure(d: dict, note: str | None):
    """The tool gap over time of ``Comparison.to_dict()`` as a matplotlib ``Figure`` (its own Agg canvas, so it
    draws the same whatever backend pyplot would pick; a notebook still shows it inline)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 4), dpi=120)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d["series"]["t"], d["series"]["tool_gap_m"], color="#444444", linewidth=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tool gap (m)")
    title = f"Tool gap over time: {d['labels']['a']} vs {d['labels']['b']}"
    ax.set_title(title + (f"\n{note}" if note else ""), fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _draw(d: dict, path: Path, note: str | None) -> None:
    figure(d, note).savefig(str(path))


def write_report(comparison: Comparison, out_dir, *, site: Site) -> dict[str, Path]:
    """Write ``gaps.json`` (``header`` + the comparison), ``gaps.csv`` (one row per scalar item: item, value,
    unit, at_s, note; the site's quality first, ``unknown.<name>`` rows last with an empty value and the reason in
    note) and ``gaps.png`` (tool gap over time) into ``out_dir``. Returns name -> path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = comparison.to_dict()
    header = _header(site)
    paths = {name: out_dir / name for name in ("gaps.json", "gaps.csv", "gaps.png")}
    paths["gaps.json"].write_text(json.dumps({"header": header, **d}, indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    with paths["gaps.csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "unit", "at_s", "note"])
        for item, value, unit, at_s, note in _rows(d, header):
            empty = value is None or (isinstance(value, float) and not math.isfinite(value))
            w.writerow([item, "" if empty else value, unit, "" if at_s is None else at_s, note])
    _plot(d, paths["gaps.png"], header["dimensionsNote"])
    return paths
