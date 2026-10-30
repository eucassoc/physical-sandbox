"""The report of one comparison: the numbers (JSON, CSV) and a graph (PNG), headed by how good the site is."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .compare import Comparison
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


def _rows(d: dict) -> list[tuple]:
    """(item, value, unit, at_s) for every scalar of ``Comparison.to_dict()``, the series left out."""
    rows: list[tuple] = []
    tool = d["tool_deviation_m"]
    rows.append(("step", d["step_s"], "s", None))
    rows.append(("tool_deviation.max", tool["max"], "m", tool["at_s"]))
    rows.append(("tool_deviation.mean", tool["mean"], "m", None))
    rows.append(("shape_deviation", d["shape_deviation_m"], "m", None))
    for name, j in d["joints"].items():
        rows.append((f"joints.{name}.max", j["max_rad"], "rad", j["at_s"]))
        rows.append((f"joints.{name}.mean", j["mean_rad"], "rad", None))
    for k, v in d["duration_s"].items():
        rows.append((f"duration.{k}", v, "s", None))
    for k, v in d["jerk"].items():
        rows.append((f"jerk.{k}", v, "m/s^3", None))
    for k, v in d["grasp_counts"].items():
        rows.append((f"grasp_count.{k}", v, "count", None))
    for i, g in enumerate(d["grasps"]):
        rows.append((f"grasps.{i}.tool_gap", g["tool_gap_m"], "m", g["a_s"]))
    return rows


def _plot(d: dict, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 4), dpi=120)
    FigureCanvasAgg(fig)  # the non-interactive Agg canvas, whatever backend pyplot would pick
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(d["series"]["t"], d["series"]["tool_gap_m"], color="#444444", linewidth=1.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tool gap (m)")
    ax.set_title(f"Tool gap over time: {d['labels']['a']} vs {d['labels']['b']}")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path))


def write_report(comparison: Comparison, out_dir, *, site: Site) -> dict[str, Path]:
    """Write ``gaps.json`` (``header`` + the comparison), ``gaps.csv`` (one row per scalar item: item, value,
    unit, at_s) and ``gaps.png`` (tool gap over time) into ``out_dir``. Returns name -> path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = comparison.to_dict()
    paths = {name: out_dir / name for name in ("gaps.json", "gaps.csv", "gaps.png")}
    paths["gaps.json"].write_text(json.dumps({"header": _header(site), **d}, indent=1, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
    with paths["gaps.csv"].open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "value", "unit", "at_s"])
        for item, value, unit, at_s in _rows(d):
            empty = value is None or (isinstance(value, float) and not math.isfinite(value))
            w.writerow([item, "" if empty else value, unit, "" if at_s is None else at_s])
    _plot(d, paths["gaps.png"])
    return paths
