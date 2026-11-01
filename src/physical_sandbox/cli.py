"""The ``psbx`` command: compare two motions on a site in one go, fit a site, fetch weights, serve MCP."""
from __future__ import annotations

import argparse
import json
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


def _corners(text: str) -> list[tuple[float, float]]:
    """Four corners from ``u,v;u,v;u,v;u,v`` (``_glue_corners`` makes that from four separate words)."""
    parts = [p for p in text.replace(";", " ").split() if p]
    if len(parts) != 4:
        raise InputError(f"--corners needs four corners u,v going round, got {len(parts)}: {text!r}")
    return [_corner(p) for p in parts]


def _glue_corners(argv: list[str]) -> list[str]:
    """``--corners A B C D`` -> ``--corners=A;B;C;D``, so that a corner left of the picture (``-12,340``, a
    negative u) is not taken for an option. Up to four words are taken, stopping at one that starts with --."""
    out, i = [], 0
    while i < len(argv):
        word = argv[i]
        if word == "--corners":
            taken = []
            while i + 1 < len(argv) and len(taken) < 4 and not argv[i + 1].startswith("--"):
                i += 1
                taken.append(argv[i])
            out.append("--corners=" + ";".join(taken))
        else:
            out.append(word)
        i += 1
    return out


def _edges_m(args) -> tuple[float, float] | None:
    if args.edges is None:
        return None
    edges = (float(args.edges[0]), float(args.edges[1]))
    if not (math.isfinite(edges[0]) and math.isfinite(edges[1]) and edges[0] > 0 and edges[1] > 0):
        raise InputError(f"--edges must be finite, positive lengths in metres, got {edges[0]} {edges[1]}")
    return edges


def _cmd_site_fit(args) -> int:
    if args.corners is None:
        return _fit_in_browser(args)
    if args.edges is None:
        raise InputError("--corners needs --edges too (the lengths of edge 1-2 and edge 2-3 in metres); "
                         "without --corners the floor is fitted in the browser")
    from .floor import solve_floor
    from .site import Floor, Site, save_site
    from .sources import open_source

    corners = _corners(args.corners)
    edges = _edges_m(args)
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


def _fit_in_browser(args) -> int:
    """The screen in the person's own browser: they drag the four corners and type the lengths there, and 決定
    writes the site file; the command runs until Ctrl-C."""
    import webbrowser
    from .fit_screen import FitServer

    def wrote(site, path):
        print(f"wrote {path}: camera {args.camera!r}, residual {site.residual_px[args.camera]:.2f} px, "
              f"dimensions {site.dimensions}, confirmed by a person (決定 again overwrites it; Ctrl-C to finish)",
              flush=True)

    server = _reading(f"image {args.image!r}", FitServer, args.image, args.out, args.camera, _edges_m(args),
                      args.measured, args.port, wrote)
    print(f"Fit the floor at {server.url}: drag the four corners onto a rectangle on the ground, type its "
          f"two edge lengths, press 決定. Ctrl-C here when done.", flush=True)  # flushed: the command then waits
    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.close()
    return 0


def _cmd_fetch_sam(args) -> int:
    from . import objects

    print(objects.fetch_sam_weights(Path(args.dest) if args.dest else None))
    return 0


def _cmd_pull(args) -> int:
    from . import catalog

    print(catalog.pull(args.ref, force=args.force))
    return 0


def _cmd_catalog_search(args) -> int:
    from . import catalog

    items = catalog.search(args.q, category=args.category, tag=args.tag, kind=args.kind)
    if args.json:
        print(json.dumps({"items": items}, ensure_ascii=False))
        return 0
    for i in items:
        print(f"{i['handle']}/{i['name']}  {i.get('latest_version') or '-'}  {i.get('category') or '-'}  "
              f"{(i.get('display') or {}).get('en', '')}")
    return 0


def _cmd_machine_check(args) -> int:
    from .machine_tools import check

    result = check(Path(args.path))
    for ok, text in result.items:
        print(f"{'ok ' if ok else 'NG '} {text}")
    print("ready to push" if result.ok else "not ready: fix the NG lines")
    return 0 if result.ok else 1


def _cmd_machine_init(args) -> int:
    from .machine_tools import init

    written = init(Path(args.path))
    for path in written:
        print(f"wrote {path}: fill in every \"<fill>\"")
    if not written:
        print("nothing written: the templates are there already")
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
    f = ssub.add_parser("fit", help="fix a site from four floor corners in a still: in the browser, or from "
                                    "--corners and --edges given here (then no person confirms it)")
    f.add_argument("image")
    f.add_argument("--edges", nargs=2, type=float, metavar=("A", "B"),
                   help="lengths in metres of edge 1-2 and edge 2-3 (in the browser: the lengths to start from, "
                        "default an A4 sheet)")
    f.add_argument("--corners", metavar="u,v",
                   help="the four corners in pixels, going round: four words u,v (a negative u is fine), or one "
                        "u,v;u,v;u,v;u,v; without it the screen opens in the browser")
    f.add_argument("--out", default="site.json", help="the site file to write (default: site.json)")
    f.add_argument("--camera", default="camera", help="the camera's name in the site (default: camera)")
    f.add_argument("--measured", action="store_true", help="the edge lengths were measured, not estimated")
    f.add_argument("--port", type=int, default=0, help="the browser screen's port on 127.0.0.1 (default: any free)")
    f.add_argument("--no-browser", action="store_true", help="print the screen's address instead of opening it")
    f.set_defaults(run=_cmd_site_fit)

    w = sub.add_parser("fetch-sam", help="download the SAM weights ahead of time")
    w.add_argument("--dest", help="where to put the weights file")
    w.set_defaults(run=_cmd_fetch_sam)

    pl = sub.add_parser("pull", help="download a catalog item (handle/name[:version]) into the cache")
    pl.add_argument("ref")
    pl.add_argument("--force", action="store_true", help="download again even if it is cached")
    pl.set_defaults(run=_cmd_pull)

    cg = sub.add_parser("catalog", help="catalog tools")
    csub = cg.add_subparsers(dest="catalog_command", required=True)
    cs = csub.add_parser("search", help="search the PS.com catalog")
    cs.add_argument("q", nargs="?", default="")
    cs.add_argument("--category")
    cs.add_argument("--tag")
    cs.add_argument("--kind")
    cs.add_argument("--json", action="store_true", help="print the items as received")
    cs.set_defaults(run=_cmd_catalog_search)

    mc = sub.add_parser("machine", help="get a machine (URDF folder) or an object (one mesh) ready for the catalog")
    msub = mc.add_subparsers(dest="machine_command", required=True)
    mk = msub.add_parser("check", help="say line by line whether the folder can be pushed (exit 1 when not)")
    mk.add_argument("path", help="the machine folder (or its URDF), or an object's folder or mesh file")
    mk.set_defaults(run=_cmd_machine_check)
    mi = msub.add_parser("init", help="write product.json and profile.json (or object.json) templates")
    mi.add_argument("path", help="the machine folder (or its URDF), or an object's mesh file")
    mi.set_defaults(run=_cmd_machine_init)

    m = sub.add_parser("mcp", help="serve the MCP tools on stdio")
    m.set_defaults(run=_cmd_mcp)
    return p


def _one_line(e: BaseException) -> str:
    text = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
    return " ".join(str(text).split()) or type(e).__name__


def main(argv=None) -> int:
    try:
        args = _parser().parse_args(_glue_corners(list(sys.argv[1:] if argv is None else argv)))
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
