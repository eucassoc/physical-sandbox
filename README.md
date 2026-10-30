# physical-sandbox

Overlay robots on photos and videos of a real site, at real scale. Fix a site from a single picture (the
floor's four corners), then draw robot motions through that camera and compare two motions with numbers and a
replay video. The command is `psbx`.

Everything runs on your machine: photos, videos and cut-outs never leave it.

## Install

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e .                  # the core: psbx site fit, psbx compare
.venv/bin/pip install -e '.[sam,mcp]'       # plus SAM 2.1 (cut out objects) and the MCP server (screens in the chat)
.venv/bin/pip install -e '.[sam,mcp,dev]'   # plus what the tests need
```

Python 3.10 or newer, macOS or Linux. The extras: `[sam]` cuts out fixed objects with SAM 2.1 (ultralytics),
`[mcp]` serves the in-chat screens (`psbx mcp`), `[dev]` runs the tests. The replay video is drawn with MuJoCo
offscreen: on macOS that works as it is; on a Linux machine without a display set `MUJOCO_GL=egl` (or
`osmesa`).

## Fix a site

A site is a JSON file with the camera that took the picture (its lens and where it stands) and the floor, in
metres: x forward, y left, z up, the floor at z = 0. The first site comes from four corners of a rectangle on
the ground in one picture (a lane, a marked area, tape on a table) and the lengths of two of its edges. Further
cameras can then be tied to that site through fixed objects both pictures show
(`physical_sandbox.site.add_camera` in Python).

### From the command line

```sh
psbx site fit photo.jpg --corners 42,330 120,360 290,272 219.6,257.3 --edges 0.061 0.129 --out site.json
```

The corners are pixels `u,v`, going round the rectangle (a corner outside the picture, with a negative `u`, is
fine); `--edges` are the lengths of edge 1-2 and edge 2-3 in metres. Add `--measured` when you measured them
(otherwise every report says the dimensions are estimated) and `--camera side` to name the camera (default
`camera`). Nobody has confirmed a site made this way, and the file says so.

### From Claude Desktop

An AI agent looks at a photo, proposes the floor's four corners and the fixed objects, and a screen inside the
chat shows them over the photo. Drag a corner if it is off, press 決定 (Decide), and the site file is written as
`site.json` in the folder the agent was given.

Add this to `claude_desktop_config.json` (Settings > Developer > Edit Config), with your own paths, and quit
and reopen Claude Desktop with Cmd+Q:

```json
{
  "mcpServers": {
    "physical-sandbox": {
      "command": "/path/to/physical-sandbox/.venv/bin/psbx",
      "args": ["mcp"]
    }
  }
}
```

Needs the `[mcp]` extra, and `[sam]` to cut out objects. A site fixed in the chat calls its camera `camera`
unless the agent names another. The display copy of a photo is kept in `~/.cache/physical-sandbox/` (set
`PS_CACHE_DIR` to move it).

SAM's weights (about 80 MB) are downloaded the first time an object is cut out; that is the only time the
network is used. To work offline, fetch them ahead of time with `psbx fetch-sam` (into
`~/.cache/physical-sandbox/`, or `$PS_CACHE_DIR`; `--dest` puts the file elsewhere, for
`physical_sandbox.objects.sam_segmenter(weights=...)`).

## Compare two motions

```sh
psbx compare --site site.json --scene scene.json --camera camera \
  --a demo.record.json --b policy.record.json --video site.mp4 --out out/
```

writes, into `out/`:

- `gaps.json` and `gaps.csv`: the site's quality first (its residual in pixels, whether its dimensions are
  measured or estimated, whether a person confirmed it), then the gaps: the tool tips' distance at the same
  time (maximum, when, mean), the difference in path shape (time ignored), each joint's gap, the durations, the
  smoothness (jerk) and the grasps (where the gripper closes; the threshold used is in the files, and
  `--grip-threshold` sets another). What the numbers cannot tell yet is listed with the reason.
- `gaps.png`: the tool gap over time.
- `replay.mp4`: both machines over the picture as see-through shells, A blue and B orange, with the tool tips'
  paths.

A motion (`--a`, `--b`) is either a recording file, `file.record.json` or `file.record.json#label` to pick one
motion in it, or one episode of a LeRobot v3 dataset, `lerobot:<dataset folder>#<episode>`. With a LeRobot A
and no `--video`, the replay is drawn on A's own recording, cut to A's episode; `--lerobot-camera side` names
the dataset's camera when it is not called like the site camera (a site fixed in the chat calls its camera
`camera`).

The scene says where each machine stands on the site. It is a small JSON file next to the site:

```json
{
  "format": "physical-sandbox.scene/1",
  "site": "site",
  "machines": {
    "so101": {"urdf": "robots/so101.urdf", "x": -0.025, "y": 0.251, "yaw": -0.242, "confirmedByPerson": false}
  },
  "records": []
}
```

`site` is the site's `id`, `x` and `y` are metres on the site and `yaw` radians counter-clockwise; the urdf path
is relative to the scene's folder. The machine's id (`so101`) is what a motion drives; a LeRobot motion drives
the scene's only machine. To write one from Python:

```python
from physical_sandbox.machine import load_machine
from physical_sandbox.scene import PlacedMachine, Scene, save_scene
from physical_sandbox.site import load_site

urdf = "robots/so101.urdf"
arm = PlacedMachine(load_machine(urdf), urdf, x=-0.025, y=0.251, yaw=-0.242)
save_scene(Scene(load_site("site.json"), {"so101": arm}), "scene.json")
```

Recordings come from numpy (`physical_sandbox.motion.motion_from_numpy`, then `save_record`) or from LeRobot
directly. [`examples/so101-pickplace`](examples/so101-pickplace) runs all of this on a real LeRobot dataset:
the site from one frame, the arm placed on it, two episodes compared.

### In Claude Desktop

The `ps_compare` tool opens the same comparison as a screen in the chat: the site's picture with both machines
as see-through shells, the tool-tip paths, a slider and a play button, and the table of numbers (click a time to
jump there). Its arguments are those of `psbx compare` (`videoCamera` for `--lerobot-camera`, `gripThreshold` for
`--grip-threshold`); from the chat it opens only local files, `rtsp://` cameras and camera numbers.

## License

Apache-2.0. Copyright Atsushi Kojo. The in-chat screens bundle three.js and the MCP Apps SDK; see `NOTICE`.
