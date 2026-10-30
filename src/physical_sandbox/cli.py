"""The ``psbx`` command: compare two motions on a site in one go, fit a site, fetch weights, serve MCP."""
from __future__ import annotations

import argparse
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .errors import InputError


def _spec_error(spec: str, why: str) -> InputError:
    return InputError(f"{spec!r}: {why}")


def _reading(what: str, fn, *args):
    """Run a step that reads the user's own file; a file that is missing or not what it should be is bad input
    (named by ``what``), not a fault of the tool."""
    try:
        return fn(*args)
    except InputError:
        raise
    except FileNotFoundError as e:
        raise InputError(f"{what}: no such file {e.filename or ''}".rstrip()) from e
    except (ValueError, KeyError, TypeError, AttributeError) as e:  # JSON that is not the expected shape, etc.
        raise InputError(f"{what}: {_one_line(e)}") from e


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
        if not (Path(directory).expanduser() / "meta" / "info.json").is_file():
            raise _spec_error(spec, f"{directory!r} is not a LeRobot dataset folder (no meta/info.json)")
        (mid, placed), = scene.machines.items()
        motion = motion_from_lerobot(directory, int(episode), machine=placed.machine, machine_name=mid)
        return motion, (directory, int(episode))
    path, label = _split_label(spec)
    motions = _reading(f"--a/--b {path!r}", load_record, path)
    if not motions:
        raise _spec_error(spec, "the recording has no motion with joint angles")
    if label is None:
        return motions[0], None
    for m in motions:
        if m.label == label:
            return m, None
    raise _spec_error(spec, f"no motion labelled {label!r} (it has: {', '.join(repr(m.label) for m in motions)})")


def _check_motions(scene, a, b) -> None:
    """The motions must fit the scene and each other: bad input otherwise (with the names to fix it)."""
    for m in (a, b):
        placed = scene.machines.get(m.machine)
        if placed is None:
            raise InputError(f"the motion {m.label!r} drives machine {m.machine!r}, which is not placed in the "
                             f"scene (placed: {', '.join(sorted(scene.machines)) or 'none'})")
        if list(m.joint_names) != list(placed.machine.joint_names):
            raise InputError(f"the motion {m.label!r} has joints {list(m.joint_names)}, the machine {m.machine!r} "
                             f"has {list(placed.machine.joint_names)}")
    if a.machine != b.machine:
        raise InputError(f"the motions drive different machines: {a.machine!r} and {b.machine!r}")
    if list(a.joint_names) != list(b.joint_names):
        raise InputError(f"the motions have different joints: {list(a.joint_names)} and {list(b.joint_names)}")


def _lerobot_picture(a_lerobot, camera: str, flag: str, given: bool) -> tuple:
    """(mp4, start_s, cap_s) of A's own recording through the dataset camera ``camera``. ``flag`` names the option
    that picks the dataset camera, for the message when the dataset has no camera of that name (``given`` says
    whether the caller named it, or it fell back to the site camera's name)."""
    from .errors import InputKeyError
    from .lerobot import lerobot_video

    try:
        video, start_s, to_s = lerobot_video(*a_lerobot, camera)
    except InputKeyError as e:
        hint = "" if given else f" (the site camera's name was used: name the dataset's camera with {flag})"
        raise InputKeyError(f"{e}{hint}") from e
    return video, start_s, to_s - start_s  # the file goes on into the next episode: stop where A's episode ends


def _own_episode_span(a_lerobot, video, camera: str | None):
    """When an explicit picture is A's own episode file: (start_s, cap_s) of that episode in it, else None.
    ``camera`` names the dataset camera when given; otherwise each of the dataset's cameras is tried."""
    from .lerobot import lerobot_cameras, lerobot_video

    given = Path(str(video)).expanduser()
    if not given.is_file():
        return None
    try:
        for name in [camera] if camera else lerobot_cameras(a_lerobot[0]):
            path, start_s, to_s = lerobot_video(*a_lerobot, name)
            if Path(path).resolve() == given.resolve():
                return start_s, to_s - start_s
    except (InputError, KeyError, OSError, ValueError):  # not a dataset this can read: just the picture
        return None
    return None


def _cmd_compare(args) -> int:
    from .compare import compare
    from .render import render_overlay
    from .report import write_report
    from .scene import load_scene
    from .site import load_site
    from .sources import camera_for, open_source

    site = _reading(f"--site {args.site!r}", load_site, args.site)
    if args.camera not in site.cameras:
        raise InputError(f"the site has no camera {args.camera!r} (it has {', '.join(sorted(site.cameras))})")
    scene = _reading(f"--scene {args.scene!r}", load_scene, args.scene, site)
    a, a_lerobot = _load_motion(args.a, scene)
    b, _ = _load_motion(args.b, scene)
    _check_motions(scene, a, b)
    start_s, cap_s = 0.0, None
    if args.video:
        video = args.video
        if a_lerobot is not None:  # A's own episode file given by name: still only A's episode of it
            start_s, cap_s = _own_episode_span(a_lerobot, video, args.lerobot_camera) or (0.0, None)
    elif a_lerobot is not None:
        video, start_s, cap_s = _lerobot_picture(a_lerobot, args.lerobot_camera or args.camera,
                                                 "--lerobot-camera", bool(args.lerobot_camera))
    else:
        raise InputError("give --video: the site file keeps no picture, and A is not a LeRobot episode "
                         "whose recording could be used")
    out = Path(args.out)
    # open the picture first, so a bad --video leaves no half a report
    source = _reading(f"--video {str(video)!r}", open_source, video)
    with source:
        try:
            camera_for(source, site.cameras[args.camera])
        except ValueError as e:
            raise InputError(f"--video {str(video)!r}: {_one_line(e)}") from e
        result = compare(scene, a, b, grip_threshold=args.grip_threshold)
        paths = write_report(result, out, site=site)
        paths["replay.mp4"] = render_overlay(scene, args.camera, source, [a, b], out / "replay.mp4",
                                             start_s=start_s, duration=cap_s)
    t = result.tool_deviation_m
    print(f"tool gap: max {t['max'] * 1000:.1f} mm at {t['at_s']:.2f} s, mean {t['mean'] * 1000:.1f} mm; "
          f"path shape {result.shape_deviation_m * 1000:.1f} mm")
    longest = max(float(a.t[-1]), float(b.t[-1]))
    if cap_s is not None and longest > cap_s + 1e-6:
        longer = "B" if b.t[-1] >= a.t[-1] else "A"
        print(f"replay cut at {cap_s:.2f} s: the recording of A's episode ends there ({longer} runs "
              f"{longest:.2f} s); the numbers cover the whole of both motions")
    if result.grip_threshold is not None and "grasps" not in result.unknown:
        print(f"grasps: A {result.grasp_counts['a']}, B {result.grasp_counts['b']} "
              f"(gripper closing through {result.grip_threshold:.3g})")
    if site.dimensions == "estimated":
        print("note: 寸法は推定（未確認）")
    if result.unknown:
        print("not told yet: " + ", ".join(result.unknown))
    for name in ("gaps.json", "gaps.csv", "gaps.png", "replay.mp4"):
        print(paths[name])
    return 0


def _finite(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number")
    return value


def _corner(text: str) -> tuple[float, float]:
    try:
        u, v = (float(x) for x in text.split(","))
    except ValueError:
        raise InputError(f"corner {text!r} must be u,v in pixels (for example 120.5,300)") from None
    if not (math.isfinite(u) and math.isfinite(v)):
        raise InputError(f"corner {text!r} must be finite numbers")
    return u, v


def _cmd_site_fit(args) -> int:
    from .floor import solve_floor
    from .site import Floor, Site, save_site
    from .sources import open_source

    corners = [_corner(c) for c in args.corners]
    edges = (float(args.edges[0]), float(args.edges[1]))
    if not (math.isfinite(edges[0]) and math.isfinite(edges[1]) and edges[0] > 0 and edges[1] > 0):
        raise InputError(f"--edges must be finite, positive lengths in metres, got {edges[0]} {edges[1]}")
    with _reading(f"image {args.image!r}", open_source, args.image) as source:
        if source.kind != "still":
            raise InputError(f"{args.image!r} is not a still image (.jpg .jpeg .png)")
        width, height = source.size
    sol = solve_floor(width, height, corners, edges)
    if sol.camera is None:
        raise InputError(f"no camera fits these corners ({sol.reason}): the floor may be seen square-on, "
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
    c.add_argument("--camera", required=True, help="the site camera's name")
    c.add_argument("--lerobot-camera", help="the camera of A's LeRobot dataset whose recording is drawn on "
                                            "(default: the same name as --camera)")
    c.add_argument("--a", required=True, help="motion A: a recording file[#label] or lerobot:<dir>#<episode>")
    c.add_argument("--b", required=True, help="motion B: a recording file[#label] or lerobot:<dir>#<episode>")
    c.add_argument("--out", required=True, help="the output folder")
    c.add_argument("--video", help="the picture to draw on (default: A's own recording when A is LeRobot)")
    c.add_argument("--grip-threshold", type=_finite, metavar="VALUE",
                   help="the gripper joint value (radians) a grasp closes through (default: the middle of the "
                        "range the gripper went through in the two motions)")
    c.set_defaults(run=_cmd_compare)

    s = sub.add_parser("site", help="site tools")
    ssub = s.add_subparsers(dest="site_command", required=True)
    f = ssub.add_parser("fit", help="fix a site from four floor corners in a still (no person confirms it)")
    f.add_argument("image")
    f.add_argument("--edges", nargs=2, type=float, required=True, metavar=("A", "B"),
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
    except InputError as e:  # the only exit 2: something the caller gave cannot be used
        return _fail(e, f"psbx: {_one_line(e)}", 2)
    except Exception as e:  # noqa: BLE001 - whatever else went wrong while running: exit 1, with its type
        return _fail(e, f"psbx: {type(e).__name__}: {_one_line(e)}", 1)


def _fail(e: BaseException, line: str, code: int) -> int:
    print(line, file=sys.stderr)
    if os.environ.get("PSBX_DEBUG") == "1":
        traceback.print_exception(type(e), e, e.__traceback__)
    return code


if __name__ == "__main__":
    sys.exit(main())
