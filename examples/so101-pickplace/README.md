# SO-101 pick-and-place: two episodes compared on the real recording

This example runs `psbx` end to end on real data: the LeRobot dataset
[`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace), where an SO-101
arm picks a red brick off a white table and drops it into a transparent tray. It fixes the site from the
dataset's own `side` camera, places the arm on it, and compares episode A (blue, episode 0 by default) with
episode B (orange, episode 1 by default): gap numbers, a graph, and a replay video with both machines drawn as
see-through shells over episode A's recording.

Nothing from the dataset is in this folder (its licence was not checked): no frames, no pictures. Only the
steps and scripts. Every file the script writes goes to `$OUT`.

## What you need

- The dataset, downloaded by LeRobot or `huggingface-cli` (about 50 episodes, `side` and `up` cameras,
  640x480, 30 fps). Set `PS_LEROBOT_DATASET` to its snapshot folder (the one with `meta/info.json`).
- An SO-101 URDF with its meshes. Set `PS_SO101_URDF` to it. The recommended source is the public
  `so101_new_calib.urdf` in TheRobotStudio's
  [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) repository (under `Simulation/SO101/`, with its
  `assets/` meshes). The script's default is a local copy of that same model,
  `~/work/physical-sandbox-proto/robots/so101/so101.urdf` (its meshes converted to OBJ), and the numbers below
  were chosen with it; another model may need another base placement.
- `physical-sandbox` installed (the core is enough: `pip install -e .`).

## Run it

```sh
. .venv/bin/activate
OUT=out/so101 sh examples/so101-pickplace/run.sh
```

`A` and `B` pick other episodes (`A=3 B=7 sh examples/so101-pickplace/run.sh`); `PSBX` and `PYTHON` point at
other commands. The site and the base placement always come from episode 0 (its frame 0 and the fit below),
whatever `A` is: that assumes the camera and the arm's base did not move between episodes. Episode `A`'s
recording is only the background of the replay. The script does four things:

1. **A still.** Frame 0 of episode 0's `side` recording goes to `$OUT/side-frame0.png`. The arm is at rest.
2. **The site.** `psbx site fit` on that still with the orange tape under the tray as the floor:

   ```sh
   psbx site fit $OUT/side-frame0.png --camera side \
     --corners 42,330 120,360 290,272 219.6,257.3 --edges 0.061 0.129 --out $OUT/site.json
   ```

   The tape is an L, not a whole rectangle in this view: corner 1 is its left end, corner 2 the bend
   nearest the camera, corner 3 its right end next to the arm. Corner 4 closes the rectangle under the tray,
   where nothing is visible (see "What was corrected"). No tape measure was used, so the site says
   `dimensions: estimated` and every report carries 寸法は推定（未確認）.
3. **The scene.** The arm's base at `x = -0.025 m, y = 0.251 m, yaw = -0.242 rad` on the site, under the
   machine id `so101` (the scene's only machine, which the `lerobot:` motions drive). There is no command for
   this yet, so the script calls `physical_sandbox.scene.save_scene` with
   `confirmed_by_person=False`.
4. **The comparison.**

   ```sh
   psbx compare --site $OUT/site.json --scene $OUT/scene.json --camera side \
     --a "lerobot:$PS_LEROBOT_DATASET#0" --b "lerobot:$PS_LEROBOT_DATASET#1" --out $OUT
   ```

   It writes `gaps.json`, `gaps.csv`, `gaps.png` and `replay.mp4` (cut where episode 0's recording ends).
   On a laptop the whole script takes about ten seconds.

## Checking it

Look at the first frame, the grasp and the last frame of `replay.mp4` (the grasp is where the gripper
channel of episode 0 closes on the brick, at about 5.6 s, frame 169). The blue shell should lie on the black
arm: the base on the base, the upper arm on the link with the LeRobot sticker, the forearm and the gripper on
theirs, and at the grasp the gripper tip on the brick. In the first and last frames both episodes are at rest,
so the orange shell covers the blue one; to see blue alone, draw episode 0 by itself with
`physical_sandbox.render.render_overlay`.

**Grasps.** `compare` counts a grasp where the gripper closes through the middle of the range it actually went
through in the two motions: 0.086 rad here (`grip_threshold` in `gaps.json` and `gaps.csv`; `--grip-threshold`
sets another). In this dataset the gripper only opens to about 20-25 of 100 before closing on the brick, so the
middle of its URDF range (0.785 rad) would find none. It finds two closings in each episode: the grasp (episode
0 at 5.60 s, episode 1 at 4.17 s, the tool tips 11 mm apart) and the gripper closing again after the drop (7.70 s
and 6.58 s, 23 mm apart). With these two episodes the tool gap is at most 226 mm (at 5.77 s, while episode 1
is already lifting), 89 mm on average, and the path shape differs by 27 mm.

## What was corrected, and why

**The joint mapping (`JOINT_MAP` in `src/physical_sandbox/lerobot.py`) did not need changing.** The dataset
gives five arm joints in -100..100 and the gripper in 0..100; mapping them linearly onto the URDF's joint
limits, all with sign +1, already puts the arm where the picture has it. Every check of a sign came out
right: at the grasp the tool tip comes out at table height (z within 1 cm of 0) in front of the base, at rest
it folds down in front of the base as in the picture, and when the pan joint goes to about +80 the arm
turns to the side of the tray. A wrong sign on any joint would have put the arm in the air, under the table or
on the wrong side, which no placement can hide. Only the comment that called the map unverified changed.

**The floor: corner 4 and the two lengths.** These are what needed work.

- The first try closed the rectangle with corner 4 from vanishing points, guessing the horizon at about
  y = 100 (a field of view of 65 degrees, as for a usual webcam), with 0.13 m x 0.30 m as a first guess of
  the tray. The arm came out about half its size in the picture: the site was too big.
- Scaling the site so that the base and the reach matched the picture (0.085 m x 0.195 m) put the arm on the
  brick at the grasp, but at rest the shell sat 20-40 px below the real arm.
- Four corners of a small rectangle seen this far off are close to a parallelogram, so the focal length
  that `psbx site fit` derives from them swings from 500 to 3000 px when corner 4 moves by one pixel.
  Corner 4 cannot be read off the picture to that precision (it is under the tray), so it was chosen from
  the arm instead. A small least-squares fit (`fit_site.py` in this folder) took the three visible tape
  corners together with the gripper tip read off nine frames of episode 0 (rest, reaching, grasp, lifting,
  over the tray, returning), held the focal length near 500 px, and solved for the camera, the two lengths
  and the base placement. It keeps the three tape corners within 1 px; the tips stay 1-24 px off (they were
  read off the fixed jaw's teal tip while the URDF's tool frame lies between the jaws, and some of those
  frames are in fast motion). Corner 4 is that camera's view of the rectangle's fourth corner, and the
  lengths and base are its answer: 0.061 m x 0.129 m (a believable size next to the 32 mm brick) and the
  base at (-0.025, 0.251), turned -14 degrees. `psbx site fit` on those corners gives a focal length of
  529 px and a corner residual of 0.33 px.

This uses the arm's own drawing to fix the scale, which is what "estimated from the arm" means here. It
also means that errors in the joint mapping other than a sign could be partly absorbed into the camera, the
scale and the placement rather than shown: an offset of a few degrees in a joint's zero, or a range (gain)
somewhat off. What was checked is the direction of each joint (above); the ends of each range were not
checked separately. `python examples/so101-pickplace/fit_site.py` prints the numbers again.

## What still does not fit

- While the arm moves fast (reaching at about 4 s, coming back at about 8 s) the shell is 10-20 px off the
  real gripper, ahead on the way out and behind on the way back. The recorded joint states and the video may
  be a frame or two apart; nothing here corrects time.
- The real moving jaw sits a little to the right of the drawn one around the grasp: the gripper opening or
  the wrist roll is a few degrees off. Not corrected.
- The scale is estimated. Lengths in `gaps.json` (tool gap, path shape) are in metres of this estimate; the
  fitted lengths moved by about 5 % when the assumed focal length went from 450 to 650 px, and the true error
  is not known. Times and joint angles do not depend on it.
- The site has no fixed objects (the tray, the box at the back): `psbx site fit` takes only the floor. They
  are chosen in the MCP screen.
- Nobody has confirmed the site or the placement (`confirmedByPerson: false` in both files).
