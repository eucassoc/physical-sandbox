"""How run.sh's corner 4, edge lengths and base placement were found (run once; run.sh has the results).

The three visible tape corners and the gripper tip read by eye off nine frames of episode 0's side video are
fitted together: the camera (focal length held near FOCAL_PX, principal point at the centre), the two tape
lengths and the arm's base x, y, yaw on the site. Prints corner 4 (the camera's view of the rectangle's hidden
corner) and the numbers to put in run.sh. Plain Levenberg-Marquardt in numpy; needs only the core install.

    PS_LEROBOT_DATASET=... PS_SO101_URDF=... python examples/so101-pickplace/fit_site.py
"""
import os
from pathlib import Path

import cv2
import numpy as np

from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_lerobot

DATASET = Path(os.environ.get(
    "PS_LEROBOT_DATASET",
    "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/"
    "f641879e22172be7e8161d5e6c1503c2d2feb657")).expanduser()
URDF = Path(os.environ.get("PS_SO101_URDF", "~/work/physical-sandbox-proto/robots/so101/so101.urdf")).expanduser()
W, H = 640, 480
FOCAL_PX = 500.0      # a usual webcam (about 65 degrees across); the corners alone cannot pin it (see README)
CORNER_WEIGHT = 5.0   # the tape corners are sharper than a gripper tip in a moving frame
# corners 1-3 in the side view: the tape's left end, its bend nearest the camera, its right end
CORNERS_PX = np.array([[42, 330], [120, 360], [290, 272]], float)
# episode 0 frame -> the fixed jaw's tip (teal) in that frame of the side video, read by eye
TIPS_PX = {0: (438, 240), 120: (440, 240), 150: (490, 300), 178: (512, 295), 195: (255, 210),
           210: (155, 185), 225: (152, 182), 240: (262, 170), 260: (470, 200)}


def main():
    machine = load_machine(URDF)
    motion = motion_from_lerobot(DATASET, 0, machine=machine)
    frames = sorted(TIPS_PX)
    tips_local = {i: machine.fk(motion.q[i]) for i in frames}
    observed = np.vstack([CORNERS_PX] + [np.array(TIPS_PX[i], float)[None] for i in frames])

    def project(p):
        f, rv, tv, a, b, x, y, yaw = p[0], p[1:4], p[4:7], p[7], p[8], p[9], p[10], p[11]
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pts = [np.array([[0, 0, 0], [a, 0, 0], [a, b, 0]], float)]
        pts += [(Rz @ tips_local[i] + [x, y, 0])[None] for i in frames]
        px, _ = cv2.projectPoints(np.vstack(pts), rv, tv, K, None)
        return px.reshape(-1, 2), K

    def residuals(p):
        r = project(p)[0] - observed
        r[:3] *= CORNER_WEIGHT
        return np.concatenate([r.ravel(), [p[0] - FOCAL_PX]])

    # start: camera at the arm's front-right looking back at it, a 0.08 x 0.2 m tape, the base behind it
    R0 = cv2.Rodrigues(np.array([1.8, 0.5, -0.4]))[0]
    t0 = -R0 @ np.array([0.3, -0.2, 0.15])
    p = np.concatenate([[FOCAL_PX], cv2.Rodrigues(R0)[0].ravel(), t0, [0.08, 0.2, 0.0, 0.25, 0.0]])
    lam = 1e-2
    for _ in range(300):
        r = residuals(p)
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            d = np.zeros_like(p)
            d[k] = 1e-6 * max(1.0, abs(p[k]))
            J[:, k] = (residuals(p + d) - r) / d[k]
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -J.T @ r)
        if (residuals(p + step) ** 2).sum() < (r ** 2).sum():
            p, lam = p + step, lam * 0.3
        else:
            lam *= 10
    px, K = project(p)
    err = np.linalg.norm(px - observed, axis=1)
    c4, _ = cv2.projectPoints(np.array([[0, p[8], 0.0]]), p[1:4], p[4:7], K, None)
    print(f"focal {p[0]:.1f} px; corner errors {np.round(err[:3], 1)} px; "
          f"tip errors {np.round(err[3:], 1)} px (frames {frames})")
    print(f"--corners 42,330 120,360 290,272 {c4[0, 0, 0]:.1f},{c4[0, 0, 1]:.1f} --edges {p[7]:.3f} {p[8]:.3f}")
    print(f"base x={p[9]:.3f} y={p[10]:.3f} yaw={p[11]:.3f}")


if __name__ == "__main__":
    main()
