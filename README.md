# physical-sandbox

Overlay robots on photos and videos of a real site, at real scale. Fix a site from a single picture (the
floor's four corners), put a machine on it, play a robot motion over the picture, and compare two motions with
numbers and a replay video. The command is `psbx`; `psbx stage` does all of it on one page.

Everything runs on your machine: photos, videos and cut-outs never leave it.

## Try it

Five minutes, one photo of your own desk:

1. Download a motion to play: `huggingface-cli download lerobot/svla_so101_pickplace --repo-type dataset`
2. Put an A4 sheet on a desk and take a photo of it, `desk.jpg`.
3. `psbx stage desk.jpg`. Drag the four corners onto the sheet's corners, pick A4, press 決定 (Decide). The work
   folder is `desk/`, next to the photo.
4. In the robot step pick `physical-sandbox/so101` (it is pulled from the public catalog; the page offers it as
   that one line only while no machine is cached, otherwise type `so101` in the catalog search box and take it
   from the hits), drag it beside the sheet, press 決定 (Decide).
5. In the motion step pick `svla_so101_pickplace` from the list, episode 0: choosing from the list loads the
   motion (読み込む (Load) is for a path you typed). Press 再生 (Play): the arm moves on your desk. 動画に書き出す (Export video) writes the replay.
6. In the compare step pick the same dataset, episode 1, as the second motion and press 比べる (Compare): the
   table of gaps and two shells. 書き出す (Export) writes the report.

The steps are described under [Stage](#stage).

## Install

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e .                  # the core: psbx stage, psbx site fit, psbx compare
.venv/bin/pip install -e '.[sam,mcp]'       # plus SAM 2.1 (cut out objects) and the MCP server (screens in the chat)
.venv/bin/pip install -e '.[sam,mcp,dev]'   # plus what the tests need
```

Python 3.10 or newer, macOS or Linux. The core includes everything the stage needs, mesh thinning
(`fast-simplification`) among it. The extras: `[sam]` cuts out fixed objects with SAM 2.1 (ultralytics), `[mcp]`
is the MCP SDK that serves the in-chat screens (`psbx mcp`), `[dev]` runs the tests. The replay video is drawn
with MuJoCo offscreen: on macOS that works as it is; on a Linux machine without a display set `MUJOCO_GL=egl`
(or `osmesa`).

## Stage

```sh
psbx stage desk.jpg
```

opens one page in your browser with four steps, in this order. A step opens when the one before it is done:
the robot step after the floor is decided, the motion and compare steps after a machine is placed.

1. **床 (Floor).** Drag the four corners onto a rectangle on the ground, going round, 1 to 4. Pick the object's
   size from the menu, or type the lengths of edge 1→2 and edge 2→3 in centimetres. Tick 長さは実測した値
   when you measured them. 決定 writes `site.json`.
2. **ロボット (Robot).** Pick a machine: the ones in `~/.cache/physical-sandbox/machines`, the ones already
   pulled from the catalog, and the folders you gave with `--machine`. The public catalog can be searched and
   pulled from the page, no account needed; you can also type the path of a URDF. Drag the machine's shell on the
   floor and set its yaw. 決定 writes `scene.json`.
3. **動き (Motion).** Pick a motion: a record file (`*.record.json`) in the work folder, or a LeRobot dataset
   found under `$HF_HOME/hub` and `$HF_HOME/lerobot/hub` (`~/.cache/huggingface` unless `HF_HOME` is set) or
   given with `--lerobot`, with its episode number; or type a record's path or `lerobot:<folder>#<episode>`.
   Choosing from the list loads it (読み込む (Load) reads a typed path); 再生 (Play) draws it over the photo. To move
   the robot while the motion plays, use the robot step: the shell there follows the motion, and the motion view
   follows the robot. 動画に書き出す (Export video) writes `replay-<label>.mp4` in the work folder, where the
   label is the record file's name without `.record.json` (`<file>-<motion>` when the file holds several motions;
   `<dataset> ep<N>` for a LeRobot episode). A record with one device
   drives the placed machine, and the page says so.
4. **比べる (Compare).** The motion of the step before is A. Pick a second motion B the same way (and, if you
   like, the しきい値 (threshold) of the grasp, in radians). 比べる (Compare) shows both shells, the tool-tip trails, the line between the two tips and the
   table of figures; 書き出す (Export) writes `compare/<A>-vs-<B>/gaps.json`, `gaps.csv`, `gaps.png` and
   `replay.mp4`, the same files as [`psbx compare`](#compare-two-motions).

The options: `--dir DIR` is the work folder (default: next to the picture, named after it), `--camera side`
names the camera in the site (default: the one the folder's floor was decided with, else `camera`), `--machine DIR` and `--lerobot DIR` offer a machine folder or a
dataset folder (both repeatable), `--port` fixes the port, `--no-browser` only prints the address, for a machine
you reach over SSH (forward the port).

One folder is one site: a folder fixed on another picture is refused. `stage.json` in the folder keeps what the
page had (the picture, the machine, motions A and B, the grasp threshold), so `psbx stage desk/` reopens where
you left off.

The page is served on 127.0.0.1 only, at an address with a random token. The photo and the videos never leave
your machine; the only network use is the catalog search and pull you ask for. The line at the bottom of the page
shows whether `psbx login` has been run (it only looks at the token file). Searching and pulling the public
catalog needs no account, and a kept token also lets you pull your own private items; pushing is
[`psbx push`](#push-to-the-catalog) in a terminal. Ctrl-C in the terminal when done.

## Fix a site

A site is a JSON file with the camera that took the picture (its lens and where it stands) and the floor, in
metres: x forward, y left, z up, the floor at z = 0. The first site comes from four corners of a rectangle on
the ground in one picture and the lengths of two of its edges. Further cameras can then be tied to that site
through fixed objects both pictures show (`physical_sandbox.site.add_camera` in Python). The stage's first step
writes it; the rest of this section is for when you want only the floor, or no browser.

The rectangle needs real lengths, and a picture cannot give them. The easy way is to put something of a known
size on the floor before taking the picture: a sheet of A4 paper on a table (21 × 29.7 cm), a pallet on a
factory floor (110 × 110 cm). Tape on the table and a ruler work too. A lane width or a desk width guessed by eye
works as well, but then every report says the dimensions are estimated.

`psbx site fit photo.jpg --out site.json` is the floor step alone: a page with only the floor, and
`site.json` written where you say (press 決定 again after a change to overwrite it; Ctrl-C when done). `--camera`, `--edges 0.6 0.4` (the lengths to start from, in metres),
`--no-browser` and `--port 8797` work as on the stage.

### From a script

```sh
psbx site fit photo.jpg --corners 42,330 120,360 290,272 219.6,257.3 --edges 0.061 0.129 --out site.json
```

fits the site without the screen, from corners already known in pixels `u,v` (going round; a corner outside
the picture, with a negative `u`, is fine) and `--edges`, the lengths of edge 1-2 and edge 2-3 in metres. Add
`--measured` when you measured them. Nobody has confirmed a site made this way, and the file says so. This is
what [`examples/so101-pickplace`](examples/so101-pickplace) uses, with corners read off a frame of the dataset.

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

## Take a machine from the catalog

```sh
psbx catalog search arm          # name  version  category  display
psbx pull physical-sandbox/so101 # prints the cache folder
```

```python
from physical_sandbox import catalog
from physical_sandbox.machine import load_machine

machine = load_machine(catalog.urdf_path(catalog.pull("physical-sandbox/so101")))
```

Files are checked against the manifest's sha256 and cached under `~/.cache/physical-sandbox/catalog/`
(`$PS_CACHE_DIR`); a second `pull` of the same version does not touch the network. `$PS_CATALOG_URL` points
at another catalog server.

## Get your own machine ready for the catalog

```sh
psbx machine init my-arm/     # product.json and profile.json templates beside the URDF
psbx machine check my-arm/    # one line per check: the URDF reads, the meshes are there, the names and
                              # licence are filled in, wheels are continuous joints
psbx machine init statue.glb  # a mesh-only object: object.json (with its unit, m or mm)
```

The browser draws a display bundle made from the URDF (`physical_sandbox.bundle.make_bundle`): `bundle.json`
and binary meshes, large meshes thinned to 20,000 triangles. The fields of `product.json`, `profile.json` and
`object.json`, and how the SO-101's grasp was measured, are in [docs/machines.md](docs/machines.md). In Claude
Desktop the same check is the `ps_machine_check` tool.

## Push to the catalog

```sh
psbx push <handle>/my-arm:1 my-arm/              # a machine folder: checked, bundled, uploaded
psbx push <handle>/statue statue.stl             # an object (beside its object.json); version 1, then 2, ...
psbx push <handle>/run-1:1 record.json --public  # a record (schemaVersion 3), published
```

The first push signs in: it prints a 6-digit code to enter at `<PS.com>/vr`, and keeps the token in
`~/.config/physical-sandbox/ps-com-token.json` (mode 0600) until it expires (`psbx login` does only this
step). You push under your own handle, decided once on PS.com at `/catalog/me`. Everything is private (only
you see it) unless you add `--public`. A version that is already there is refused (`taken`); without a
version, push takes 1, or one more than the highest number there.

A record whose devices name a local machine (`"product": "so101"`) takes that machine along: when
`<handle>/so101` is not in the catalog, it goes up **privately** as `<handle>/so101:1` from
`~/.cache/physical-sandbox/machines/so101/` or `--machine-dir DIR` (`DIR/so101/`). When it is there and the
files are the same it is used as it is; when they differ, it goes up as the next version. The uploaded record
names `<handle>/so101:<version>`; your file is left as it was. In Claude Desktop, `ps_catalog_push` does the
same (private by default) once `psbx login` has been run in a terminal.

## Compare two motions

```sh
psbx compare --site site.json --scene scene.json --camera camera \
  --a demo.record.json --b policy.record.json --video site.mp4 --out out/
```

writes, into `out/`:

- `gaps.json` and `gaps.csv`: the site's quality first (its residual in pixels, whether its dimensions are
  measured or estimated, whether a person confirmed it), then the gaps: the tool tips' distance at the same
  time (maximum, when, mean), the difference in path shape (time ignored), each joint's gap, the durations, the
  smoothness (jerk) and the grasps (where the gripper closes through the middle of the range it went through;
  the threshold used is in the files, and `--grip-threshold` sets another; a gripper that hardly moved, under
  5 % of its joint's range, gets no grasps and a note instead, so its wobble is never counted). What the numbers
  cannot tell yet is listed with the reason.
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
`--grip-threshold`); from the chat it opens only local files, `rtsp://` cameras and camera numbers. Leave `b`
out to look at one motion alone: the same screen without the gaps between two (it shows the duration and the
grasps).

`ps_scene_place(siteDir, machine, x, y, yaw, scenePath=None, urdf=None)` writes the scene file the screen
needs: it puts `machine` (a catalog ref such as `physical-sandbox/so101`, pulled if it is not cached, or, with
`urdf`, the id to give that URDF file) at `x`, `y` metres and `yaw` radians on the site in `siteDir`, into
`siteDir/scene.json` unless `scenePath` says another file.

## License

Apache-2.0. Copyright Atsushi Kojo. The in-chat screens bundle three.js and the MCP Apps SDK; see `NOTICE`.
