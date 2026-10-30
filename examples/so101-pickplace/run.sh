#!/bin/sh
# Compare two SO-101 pick-and-place episodes of the LeRobot dataset lerobot/svla_so101_pickplace on the
# dataset's own side camera: fix the site from frame 0, place the arm, then gap numbers and a replay video.
# The dataset and its pictures stay where they are; every file this writes goes to $OUT.
#
# Run it with the virtualenv active (psbx and python on PATH), or set PSBX and PYTHON.
set -eu

DATASET="${PS_LEROBOT_DATASET:-$HOME/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/f641879e22172be7e8161d5e6c1503c2d2feb657}"
URDF="${PS_SO101_URDF:-$HOME/work/physical-sandbox-proto/robots/so101/so101.urdf}"
OUT="${OUT:-out/so101}"
PSBX="${PSBX:-psbx}"
PYTHON="${PYTHON:-python}"
A="${A:-0}"  # episode drawn in blue (its side recording is the background of the replay)
B="${B:-1}"  # episode drawn in orange

mkdir -p "$OUT"

# The site and the base placement below come from episode 0 whatever A and B are: this assumes the camera
# and the arm's base stayed put across episodes. fit_site.py shows how the numbers were found.

# 1. A still of the side camera: frame 0 of episode 0 (the arm at rest).
"$PYTHON" - "$DATASET" "$OUT/side-frame0.png" <<'PY'
import sys, cv2
from physical_sandbox.lerobot import lerobot_video
path, start, _ = lerobot_video(sys.argv[1], 0, "side")
cap = cv2.VideoCapture(str(path))
cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
ok, frame = cap.read()
if not ok:
    sys.exit(f"cannot read a frame of {path}")
cv2.imwrite(sys.argv[2], frame)
PY

# 2. The site: the orange tape under the tray. Corners going round: 1 the tape's left end (the tray's back
#    corner on the far side from the arm), 2 the tape's corner nearest the camera, 3 the tape's right end
#    (next to the arm), 4 hidden under the tray (see README). Lengths are estimates (no tape measure was used).
"$PSBX" site fit "$OUT/side-frame0.png" --camera side \
  --corners 42,330 120,360 290,272 219.6,257.3 \
  --edges 0.061 0.129 \
  --out "$OUT/site.json"

# 3. The scene: the arm's base on the site (x, y in metres, yaw in radians), chosen by eye and by the fit
#    described in the README. The machine id "so101" is what the LeRobot motions drive.
"$PYTHON" - "$OUT/site.json" "$URDF" "$OUT/scene.json" <<'PY'
import sys
from physical_sandbox.machine import load_machine
from physical_sandbox.scene import PlacedMachine, Scene, save_scene
from physical_sandbox.site import load_site
site_path, urdf, scene_path = sys.argv[1:]
arm = PlacedMachine(load_machine(urdf), urdf, x=-0.025, y=0.251, yaw=-0.242, confirmed_by_person=False)
save_scene(Scene(load_site(site_path), {"so101": arm}, []), scene_path)
print(f"wrote {scene_path}")
PY

# 4. Compare the two episodes: gaps.json, gaps.csv, gaps.png and replay.mp4 (on episode A's side recording).
"$PSBX" compare --site "$OUT/site.json" --scene "$OUT/scene.json" --camera side \
  --a "lerobot:$DATASET#$A" --b "lerobot:$DATASET#$B" --out "$OUT"
