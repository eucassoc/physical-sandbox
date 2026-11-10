"""The notebook surface: ``compare(folder, a, b)`` on a work folder the stage screen left, and ``Gaps``, what a
cell shows of the result (the screen's table, the tool-gap figure, the report's rows, one flat line for a table
over many comparisons). The screen is for what needs eyes (the floor, the placement); this is for the numbers a
researcher loops over in their own code."""
from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import report
from .comparison import Comparison
from .errors import InputError
from .site import Site

# What the numbers cannot tell yet, in the words the screen uses (``figures.js`` holds the same list)
_UNKNOWN_NAMES = {"clearance": "周りとのすき間", "reach": "届いたか", "world": "置き場所の差（シミュレーターと現実）",
                  "grasps": "つかむ瞬間"}


def _mm(m: float) -> str:
    return f"{m * 1000:.1f} mm"


def _deg(r: float) -> str:
    return f"{r * 180 / math.pi:.1f}°"


def _sec(s: float) -> str:
    return f"{s:.2f} s"


def _jerk(v) -> str:
    return "—" if v is None else f"{v:.3g} m/s³"


@dataclass
class Gaps:
    """One comparison with the site it was made on. ``to_dict()`` is what ``gaps.json`` holds, ``rows()`` what
    ``gaps.csv`` holds, ``row()`` one flat line (for a table of many comparisons), ``figure()`` the tool gap over
    time, ``to_dataframe()`` the rows as pandas; a notebook cell shows the screen's table."""
    comparison: Comparison
    site: Site
    notes: list[str] = field(default_factory=list)  # what was assumed when reading the motions (read-overs)

    def to_dict(self) -> dict:
        """``header`` (the site's quality) and the comparison, plain JSON types: the content of ``gaps.json``."""
        return {"header": report._header(self.site), **self.comparison.to_dict()}

    def rows(self) -> list[dict]:
        """The rows of ``gaps.csv``: ``{item, value, unit, at_s, note}``, the site's quality first, the
        ``unknown.<name>`` rows last (``value`` None, the reason in ``note``)."""
        d = self.comparison.to_dict()
        out = []
        for item, value, unit, at_s, note in report._rows(d, report._header(self.site)):
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            out.append({"item": item, "value": value, "unit": unit, "at_s": at_s, "note": note})
        return out

    def row(self) -> dict:
        """One flat line: ``a``, ``b``, then every item of ``rows()`` that has a value (``item -> value``), so a
        list of these from many comparisons is one table (``pandas.DataFrame(lines)``)."""
        line: dict = {"a": self.comparison.labels["a"], "b": self.comparison.labels["b"]}
        for r in self.rows():
            if r["value"] is not None:
                line[r["item"]] = r["value"]
        return line

    def to_dataframe(self):
        """``rows()`` as a ``pandas.DataFrame`` (columns item, value, unit, at_s, note). Needs pandas."""
        try:
            import pandas
        except ImportError as e:
            raise ImportError("to_dataframe() needs pandas: pip install pandas") from e
        return pandas.DataFrame(self.rows(), columns=["item", "value", "unit", "at_s", "note"])

    def figure(self):
        """The tool gap over time as a matplotlib ``Figure`` (the ``gaps.png`` drawing)."""
        import matplotlib

        font, note = report._font_and_note(report._header(self.site)["dimensionsNote"])
        with matplotlib.rc_context({"font.sans-serif": [font] if font else ["DejaVu Sans"]}):
            return report.figure(self.comparison.to_dict(), note)

    def _repr_html_(self) -> str:
        """The screen's table, in its words and units (mm, degrees, seconds), with the site line above it."""
        d = self.comparison.to_dict()
        h = report._header(self.site)
        esc = html.escape
        px = "、".join(f"{esc(cam)} {v:.1f} px" for cam, v in h["residualPx"].items()) or "—"
        head = (f"場所 {esc(h['site'])}　ずれ {px}　寸法 {'実測' if h['dimensions'] == 'measured' else '推定'}　"
                f"{'人が確認済み' if h['confirmedByPerson'] else '人の確認はまだ'}")
        if h["dimensionsNote"]:
            head += f"　<span style=\"color:#b00\">{esc(h['dimensionsNote'])}</span>"
        rows: list[tuple[str, str]] = []
        tool = d["tool_deviation_m"]
        rows.append(("手先のずれ（同じ時刻）", f"最大 {_mm(tool['max'])}（{_sec(tool['at_s'])}）、平均 {_mm(tool['mean'])}"))
        rows.append(("通り道の形のずれ", _mm(d["shape_deviation_m"])))
        du = d["duration_s"]
        rows.append(("動いた時間", f"A {_sec(du['a'])}、B {_sec(du['b'])}（B − A = {du['diff']:+.2f} s）"))
        rows.append(("なめらかさ（ジャーク）", f"A {_jerk(d['jerk']['a'])}、B {_jerk(d['jerk']['b'])}"))
        for name, j in d["joints"].items():
            rows.append((f"関節 {name} のずれ", f"最大 {_deg(j['max_rad'])}（{_sec(j['at_s'])}）、平均 {_deg(j['mean_rad'])}"))
        unknown = d["unknown"]
        if "grasps" not in unknown:
            n = d["grasp_counts"]
            at = "" if d["grip_threshold"] is None else f"（しきい値 {d['grip_threshold']:.3g}）"
            rows.append(("つかんだ回数", f"A {n['a']} 回、B {n['b']} 回{at}"))
            for i, g in enumerate(d["grasps"]):
                rows.append((f"{i + 1} 回目につかんだとき",
                             f"手先のずれ {_mm(g['tool_gap_m'])}（A の {_sec(g['a_s'])}、B の {_sec(g['b_s'])}）"))
        for name in unknown:
            rows.append((f"{_UNKNOWN_NAMES.get(name, name)}（まだ分かりません）",
                         f"<span style=\"color:#888\">{esc(d['unknown_reasons'].get(name, ''))}</span>"))
        body = "".join(f"<tr><th style=\"text-align:left;padding:2px 12px 2px 0\">{esc(k)}</th><td>{v}</td></tr>"
                       for k, v in rows)
        notes = "".join(f"<div style=\"color:#888\">{esc(n)}</div>" for n in self.notes)
        return (f"<div><div>{esc(d['labels']['a'])} vs {esc(d['labels']['b'])}</div>"
                f"<div style=\"color:#555\">{head}</div><table>{body}</table>{notes}</div>")


def compare(folder, a: str, b: str, *, grip_threshold: float | None = None, step_s: float = 1 / 30) -> Gaps:
    """The gaps between motions ``a`` and ``b`` on the work folder ``folder`` (the folder ``psbx stage`` left:
    its decided floor and placed robot). A motion is a record file (``path`` or ``path#label``, relative to the
    folder or absolute) or a LeRobot episode (``lerobot:<dataset_dir>#<episode>``). ``grip_threshold`` is the
    gripper value a grasp closes through (None: the middle of the range the gripper went through). Bad input
    raises ``InputError`` or ``ValueError`` saying what to fix."""
    from .comparison import compare as _compare
    from .stage import Stage

    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise InputError(f"{str(folder)!r}: no such work folder")
    stage = Stage(work=folder)
    scene, ma, mb, _, notes, _ = stage._two_motions({"a": a, "b": b})
    return Gaps(_compare(scene, ma, mb, step_s=step_s, grip_threshold=grip_threshold), stage._decided_site(), notes)
