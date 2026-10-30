"""The ``psbx`` command: compare two motions on a site in one go, fit a site, fetch weights, serve MCP."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path


class UsageError(Exception):
    """Bad input: exit code 2."""


def _spec_error(spec: str, why: str) -> UsageError:
    return UsageError(f"{spec!r}: {why}")


def _split_label(spec: str) -> tuple[str, str | None]:
    """``path#label``; a path that exists as written (a ``#`` in its name) is taken whole."""
    if Path(spec).expanduser().is_file() or "#" not in spec:
        return spec, None
    path, label = spec.rsplit("#", 1)
    return path, label


def _load_motion(spec: str, scene):
    """One motion from a ``--a``/``--b`` spec. Returns (motion, lerobot) with lerobot = (dir, episode) or None."""
    from .motion import load_record, motion_from_lerobot

    if spec.startswith("lerobot:"):
        body = spec[len("lerobot:"):]
        directory, sep, episode = body.rpartition("#")
        if not sep or not episode.strip().lstrip("-").isdigit():
            raise _spec_error(spec, "use lerobot:<dataset_dir>#<episode number>")
        if len(scene.machines) != 1:
            names = ", ".join(sorted(scene.machines)) or "none"
            raise _spec_error(spec, f"a LeRobot motion needs a scene with exactly one placed machine; "
                                    f"this scene has {len(scene.machines)} ({names})")
        (mid, placed), = scene.machines.items()
        motion = motion_from_lerobot(directory, int(episode), machine=placed.machine, machine_name=mid)
        return motion, (directory, int(episode))
    path, label = _split_label(spec)
    motions = load_record(path)
    if not motions:
        raise _spec_error(spec, "the recording has no motion with joint angles")
    if label is None:
        return motions[0], None
    for m in motions:
        if m.label == label:
            return m, None
    raise _spec_error(spec, f"no motion labelled {label!r} (it has: {', '.join(repr(m.label) for m in motions)})")


def _cmd_compare(args) -> int:
    from .compare import compare
    from .lerobot import lerobot_video
    from .render import render_overlay
    from .report import write_report
    from .scene import load_scene
    from .site import load_site
    from .sources import open_source

    site = load_site(args.site)
    if args.camera not in site.cameras:
        raise UsageError(f"the site has no camera {args.camera!r} (it has {', '.join(sorted(site.cameras))})")
    scene = load_scene(args.scene, site)
    a, a_lerobot = _load_motion(args.a, scene)
    b, _ = _load_motion(args.b, scene)
    start_s = 0.0
    if args.video:
        video = args.video
    elif a_lerobot is not None:
        video, start_s, _ = lerobot_video(*a_lerobot, args.camera)
    else:
        raise UsageError("give --video: the site file keeps no picture, and A is not a LeRobot episode "
                         "whose recording could be used")
    out = Path(args.out)
    result = compare(scene, a, b)
    paths = write_report(result, out, site=site)
    with open_source(video) as source:
        paths["replay.mp4"] = render_overlay(scene, args.camera, source, [a, b], out / "replay.mp4",
                                             start_s=start_s)
    t = result.tool_deviation_m
    print(f"tool gap: max {t['max'] * 1000:.1f} mm at {t['at_s']:.2f} s, mean {t['mean'] * 1000:.1f} mm; "
          f"path shape {result.shape_deviation_m * 1000:.1f} mm")
    for name in ("gaps.json", "gaps.csv", "gaps.png", "replay.mp4"):
        print(paths[name])
    return 0


def _corner(text: str) -> tuple[float, float]:
    try:
        u, v = text.split(",")
        return float(u), float(v)
    except ValueError:
        raise UsageError(f"corner {text!r} must be u,v in pixels (for example 120.5,300)") from None


def _cmd_site_fit(args) -> int:
    from .floor import solve_floor
    from .site import Floor, Site, save_site
    from .sources import open_source

    corners = [_corner(c) for c in args.corners]
    edges = (float(args.edges[0]), float(args.edges[1]))
    if not (edges[0] > 0 and edges[1] > 0):
        raise UsageError(f"--edges must be positive lengths in metres, got {edges[0]} {edges[1]}")
    with open_source(args.image) as source:
        if source.kind != "still":
            raise UsageError(f"{args.image!r} is not a still image (.jpg .jpeg .png)")
        width, height = source.size
    sol = solve_floor(width, height, corners, edges)
    if sol.camera is None:
        raise UsageError(f"no camera fits these corners ({sol.reason}): the floor may be seen square-on, "
                         f"the corners may not be a rectangle in perspective, or they are in the wrong order")
    out = Path(args.out)
    site = Site(
        id=out.stem, cameras={args.camera: sol.camera},
        floor=Floor(args.camera, corners, edges, sol.corners_m),
        residual_px={args.camera: float(sol.residual_px)},
        confirmed_by_person=False, dimensions="measured" if args.measured else "estimated",
        history=[{"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "what": "fitted from four floor corners by psbx site fit; not confirmed by a person"
                          + ("; floor mirrored across edge 1-2" if sol.mirrored else "")}])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_site(site, out)
    print(f"wrote {out}: camera {args.camera!r}, residual {sol.residual_px:.2f} px, "
          f"dimensions {site.dimensions}, not confirmed by a person")
    return 0


def _cmd_fetch_sam(args) -> int:
    from . import objects

    print(objects.fetch_sam_weights(Path(args.dest) if args.dest else None))
    return 0


def _cmd_mcp(args) -> int:
    try:
        from .mcp.server import main as serve
    except ImportError as e:
        print(f"psbx: the MCP server needs the [mcp] extra (pip install 'physical-sandbox[mcp]'): {e}",
              file=sys.stderr)
        return 1
    serve()
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="psbx", description="Compare two robot motions on a real site.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compare", help="gap numbers, a graph and a replay video for two motions")
    c.add_argument("--site", required=True, help="the site file")
    c.add_argument("--scene", required=True, help="the scene file (where each machine stands)")
    c.add_argument("--camera", required=True, help="the site camera's name (also the LeRobot camera name)")
    c.add_argument("--a", required=True, help="motion A: a recording file[#label] or lerobot:<dir>#<episode>")
    c.add_argument("--b", required=True, help="motion B: a recording file[#label] or lerobot:<dir>#<episode>")
    c.add_argument("--out", required=True, help="the output folder")
    c.add_argument("--video", help="the picture to draw on (default: A's own recording when A is LeRobot)")
    c.set_defaults(run=_cmd_compare)

    s = sub.add_parser("site", help="site tools")
    ssub = s.add_subparsers(dest="site_command", required=True)
    f = ssub.add_parser("fit", help="fix a site from four floor corners in a still (no person confirms it)")
    f.add_argument("image")
    f.add_argument("--edges", nargs=2, required=True, metavar=("A", "B"),
                   help="lengths in metres of edge 1-2 and edge 2-3")
    f.add_argument("--corners", nargs=4, required=True, metavar="u,v", help="the four corners, going round")
    f.add_argument("--out", required=True, help="the site file to write")
    f.add_argument("--camera", default="camera", help="the camera's name in the site (default: camera)")
    f.add_argument("--measured", action="store_true", help="the edge lengths were measured, not estimated")
    f.set_defaults(run=_cmd_site_fit)

    w = sub.add_parser("fetch-sam", help="download the SAM weights ahead of time")
    w.add_argument("--dest", help="where to put the weights file")
    w.set_defaults(run=_cmd_fetch_sam)

    m = sub.add_parser("mcp", help="serve the MCP tools on stdio")
    m.set_defaults(run=_cmd_mcp)
    return p


def _one_line(e: BaseException) -> str:
    text = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
    return " ".join(str(text).split()) or type(e).__name__


def main(argv=None) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:  # argparse: --help is 0, bad usage is 2
        return e.code if isinstance(e.code, int) else 2
    try:
        return args.run(args)
    except (UsageError, FileNotFoundError, ValueError, KeyError, IndexError) as e:
        print(f"psbx: {_one_line(e)}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 - one line, exit 1, whatever went wrong while running
        print(f"psbx: {type(e).__name__}: {_one_line(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
