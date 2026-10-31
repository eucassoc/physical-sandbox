import json
import math

import numpy as np
import pytest

from physical_sandbox.compare import Comparison, compare
from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_numpy
from physical_sandbox.scene import PlacedMachine, Scene
from physical_sandbox.site import Site

INERTIAL = '<inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'
URDF = f"""<?xml version="1.0"?>
<robot name="rail">
  <link name="base">{INERTIAL}</link>
  <link name="rail">{INERTIAL}</link>
  <link name="carriage">{INERTIAL}</link>
  <link name="jaw">{INERTIAL}</link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="rail"/><origin xyz="0 0 0.1"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="side" type="prismatic">
    <parent link="rail"/><child link="carriage"/><origin xyz="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/>
  </joint>
  <joint name="gripper" type="revolute">
    <parent link="carriage"/><child link="jaw"/><origin xyz="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="0" upper="1" effort="1" velocity="1"/>
  </joint>
</robot>
"""
NAMES = ["slide", "side", "gripper"]
FPS = 30.0


@pytest.fixture
def scene(tmp_path, monkeypatch):
    monkeypatch.setenv("PS_CACHE_DIR", str(tmp_path / "cache"))
    p = tmp_path / "rail.urdf"
    p.write_text(URDF)
    placed = PlacedMachine(machine=load_machine(p), urdf=str(p), x=0.0, y=0.0, yaw=0.0)
    return Scene(site=Site(id="s-1", cameras={}), machines={"rail": placed})


def _motion(slide, grip=None, side=None, *, label="", fps=FPS, machine="rail", names=NAMES):
    slide = np.asarray(slide, float)
    grip = np.zeros_like(slide) if grip is None else np.asarray(grip, float)
    side = np.zeros_like(slide) if side is None else np.asarray(side, float)
    return motion_from_numpy(np.column_stack([slide, side, grip]), fps=fps, joint_names=names,
                             machine=machine, label=label)


def _smooth(duration, slow=1.0, length=0.3):
    t = np.arange(0, duration * slow + 1e-9, 1 / FPS)
    return length * (1 - np.cos(np.pi * t / (duration * slow))) / 2


def test_sideways_offset_is_the_tool_deviation(scene):
    s = _smooth(2.0)
    a, b = _motion(s, label="demo"), _motion(s, side=np.full_like(s, 0.02), label="policy")
    c = compare(scene, a, b)
    assert isinstance(c, Comparison)
    assert c.tool_deviation_m["max"] == pytest.approx(0.02, abs=1e-6)
    assert c.tool_deviation_m["mean"] == pytest.approx(0.02, abs=1e-6)
    assert 0.0 <= c.tool_deviation_m["at_s"] <= 2.0
    assert c.joints["side"]["max_rad"] == pytest.approx(0.02, abs=1e-6)
    assert c.joints["slide"]["max_rad"] == pytest.approx(0.0, abs=1e-9)
    assert c.joints["gripper"]["max_rad"] == pytest.approx(0.0, abs=1e-9)
    assert c.shape_deviation_m == pytest.approx(0.02, abs=1e-6)  # a sideways shift cannot be warped away
    assert c.duration_s == {"a": pytest.approx(2.0), "b": pytest.approx(2.0), "diff": pytest.approx(0.0)}


def test_slower_copy_differs_in_time_but_not_in_shape(scene):
    a = _motion(_smooth(2.0))
    b = _motion(_smooth(2.0, slow=1.5))
    c = compare(scene, a, b)
    assert c.tool_deviation_m["max"] > 0.01
    assert c.shape_deviation_m < 0.002
    assert c.duration_s["a"] == pytest.approx(2.0)
    assert c.duration_s["b"] == pytest.approx(3.0)
    assert c.duration_s["diff"] == pytest.approx(1.0)  # b - a
    # the shorter motion holds its last pose, so the gap runs to the end of the longer one
    assert c.series["t"][-1] >= 3.0 - 1e-9
    assert len(c.series["t"]) == len(c.series["tool_gap_m"])
    assert c.series["tool_gap_m"][-1] == pytest.approx(0.0, abs=1e-9)


def test_constant_velocity_has_no_jerk_and_a_wiggle_has_some(scene):
    line = _motion(np.linspace(0.0, 0.3, 61))
    wiggle = _motion(0.15 + 0.05 * np.sin(np.arange(61) * 0.9))
    c = compare(scene, line, wiggle)
    assert c.jerk["a"] == pytest.approx(0.0, abs=1e-6)
    assert c.jerk["b"] > 1.0


def test_one_grasp_each_pairs_up_with_a_known_time_gap(scene):
    n = 91
    t = np.arange(n) / FPS
    slide = np.linspace(0.0, 0.3, n)

    def closing(at):  # gripper 1 -> 0 over 0.2 s starting at ``at``
        return np.clip(1.0 - (t - at) / 0.2, 0.0, 1.0)

    a = _motion(slide, closing(1.0))
    b = _motion(slide, closing(1.5))
    c = compare(scene, a, b)
    assert len(c.grasps) == 1
    g = c.grasps[0]
    # the default threshold is the middle of the range the gripper went through (0..1 -> 0.5): halfway through
    # the 0.2 s close
    assert g["a_s"] == pytest.approx(1.1, abs=1e-6)
    assert g["b_s"] == pytest.approx(1.6, abs=1e-6)
    assert g["tool_gap_m"] == pytest.approx(0.1 * 0.5, abs=1e-6)  # slide moves 0.1 m/s, 0.5 s apart
    assert "grasps" not in c.unknown
    # a threshold of 0.25 is crossed later in the close
    c2 = compare(scene, a, b, grip_threshold=0.25)
    assert c2.grasps[0]["a_s"] == pytest.approx(1.15, abs=1e-6)


def test_a_gripper_that_opens_only_a_little_is_still_found_by_the_default_threshold(scene):
    """As in the SO-101 dataset: the limits are 0..1 but the gripper only opens to 0.2 before closing."""
    n = 91
    t = np.arange(n) / FPS
    slide = np.linspace(0.0, 0.3, n)

    def small(at):  # 0.2 -> 0.02 over 0.2 s starting at ``at``
        return 0.02 + 0.18 * np.clip(1.0 - (t - at) / 0.2, 0.0, 1.0)

    c = compare(scene, _motion(slide, small(1.0)), _motion(slide, small(1.5)))
    assert c.grip_threshold == pytest.approx(0.11)  # the middle of 0.02..0.2, far below the limits' 0.5
    assert c.grasp_counts == {"a": 1, "b": 1}
    assert c.grasps[0]["a_s"] == pytest.approx(1.1, abs=1e-6)
    assert "grasps" not in c.unknown
    assert c.to_dict()["grip_threshold"] == pytest.approx(0.11)


def test_no_grasp_in_either_motion_is_said_with_the_threshold_not_shown_as_a_bare_zero(scene):
    n = 31
    slide = np.linspace(0.0, 0.3, n)
    opens = np.linspace(0.0, 1.0, n)  # opening only: a range to take the middle of, but no closing through it
    c = compare(scene, _motion(slide, opens), _motion(slide, opens))
    assert c.grasp_counts == {"a": 0, "b": 0}
    assert "grasps" in c.unknown
    assert c.unknown_reasons["grasps"] == "つかむ瞬間が見つかりませんでした（しきい値 0.5）"
    # a gripper that never moves has no middle to speak of: it hardly moved, and no threshold was used
    still = np.full(n, 0.5)
    c0 = compare(scene, _motion(slide, still), _motion(slide, still))
    assert c0.grip_threshold is None and "ほとんど動いていません" in c0.unknown_reasons["grasps"]
    # an explicit threshold the closings never cross says so too, with that threshold
    closes = np.where(np.arange(n) < 15, 0.5, 0.2)
    c2 = compare(scene, _motion(slide, closes), _motion(slide, closes), grip_threshold=0.1)
    assert c2.grasp_counts == {"a": 0, "b": 0} and "しきい値 0.1）" in c2.unknown_reasons["grasps"]
    with pytest.raises(ValueError, match="grip_threshold"):
        compare(scene, _motion(slide, closes), _motion(slide, closes), grip_threshold=float("nan"))


def test_a_gripper_that_hardly_moves_has_no_grasps_counted_and_says_so(scene):
    """Sensor noise on a gripper that stays put: the middle of its range is crossed dozens of times, but none
    is a grasp. Under 5 % of the joint's limit span (the rail's gripper is -1..1, so under 0.1) is 'hardly
    moved', and grasps are unknown; the noise is never counted."""
    n = 121
    slide = np.linspace(0.0, 0.3, n)
    noise = 0.1 + 0.004 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # 0.096 / 0.104, a 0.008 wide wobble
    c = compare(scene, _motion(slide, noise), _motion(slide, noise))
    assert c.grasp_counts == {"a": 0, "b": 0} and c.grasps == []
    assert "grasps" in c.unknown and c.grip_threshold is None
    assert "ほとんど動いていません" in c.unknown_reasons["grasps"] and "0.008" in c.unknown_reasons["grasps"]
    assert c.to_dict()["grip_threshold"] is None
    # a threshold the person chose is used as given, wobble and all
    c2 = compare(scene, _motion(slide, noise), _motion(slide, noise), grip_threshold=0.1)
    assert c2.grasp_counts["a"] > 10 and c2.grip_threshold == pytest.approx(0.1)


def test_a_wobble_around_the_threshold_during_one_close_is_one_grasp(scene):
    """A close that dithers about the middle counts once: a closing is a fall from well above the threshold
    to well below it (a band a tenth of the gripper's range each side)."""
    n = 91
    t = np.arange(n) / FPS
    slide = np.linspace(0.0, 0.3, n)
    v = np.where(t < 1.0, 1.0, 0.0)
    v[30:36] = [0.52, 0.48, 0.53, 0.47, 0.51, 0.49]  # six crossings of 0.5 on the way down
    c = compare(scene, _motion(slide, v), _motion(slide, v))
    assert c.grasp_counts == {"a": 1, "b": 1}
    assert c.grasps[0]["a_s"] == pytest.approx(30 / FPS, abs=0.5 / FPS)  # the first fall through 0.5
    # a wobble that never leaves the band is not a close at all
    w = np.where(t < 1.0, 1.0, 0.45)
    w[40:] = np.resize([0.45, 0.55], n - 40)
    c2 = compare(scene, _motion(slide, w), _motion(slide, w))
    assert c2.grasp_counts == {"a": 1, "b": 1}  # the 1.0 -> 0.45 fall is one grasp; the 0.45/0.55 wobble none


def test_opening_is_not_a_grasp_and_extra_grasps_are_counted(scene):
    n = 91
    t = np.arange(n) / FPS
    opens = np.clip((t - 1.0) / 0.2, 0.0, 1.0)  # 0 -> 1: opening only
    closes_once = np.where(t < 1.0, 1.0, 0.0)
    twice = np.where((t < 0.8) | ((t >= 1.4) & (t < 2.2)), 1.0, 0.0)  # two closes
    slide = np.linspace(0.0, 0.3, n)
    c = compare(scene, _motion(slide, opens), _motion(slide, twice))
    assert c.grasps == []
    assert c.grasp_counts == {"a": 0, "b": 2}
    c2 = compare(scene, _motion(slide, closes_once), _motion(slide, twice))
    assert len(c2.grasps) == 1 and c2.grasp_counts == {"a": 1, "b": 2}


def test_no_gripper_joint_says_grasps_are_unknown(scene):
    import copy
    a = motion_from_numpy(np.zeros((5, 1)), fps=FPS, joint_names=["slide"], machine="rail")
    # a machine without a gripper: a copy of the placed rail with one joint (and a stubbed fk)
    placed = copy.copy(scene.machines["rail"])
    m = copy.copy(placed.machine)
    m.joint_names = ["slide"]
    m.limits = placed.machine.limits[:1]
    m.fk = lambda q: np.array([float(q[0]), 0.0, 0.1])
    scene2 = Scene(site=scene.site, machines={"rail": PlacedMachine(machine=m, urdf=placed.urdf, x=0, y=0, yaw=0)})
    c = compare(scene2, a, a)
    assert c.grasps == []
    assert "grasps" in c.unknown
    assert c.unknown_reasons["grasps"]
    assert c.grip_threshold is None and c.to_dict()["grip_threshold"] is None


def test_to_dict_is_plain_json_with_unknowns_and_labels(scene):
    s = _smooth(1.0)
    c = compare(scene, _motion(s, label="demo"), _motion(s + 0.01, label="policy"), step_s=1 / 20)
    d = c.to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["labels"] == {"a": "demo", "b": "policy"}
    assert d["step_s"] == 1 / 20
    assert d["unknown"] == ["clearance", "reach", "world", "grasps"]  # the gripper never moves here
    assert d["grip_threshold"] is None
    assert d["unknown_reasons"]["clearance"] == "奥行きと対象物は段 3"
    for key in ("tool_deviation_m", "shape_deviation_m", "joints", "duration_s", "jerk", "grasps", "series"):
        assert key in d
    assert set(d["tool_deviation_m"]) == {"max", "mean", "at_s"}
    assert set(d["joints"]["slide"]) == {"mean_rad", "max_rad", "at_s"}
    assert type(d["tool_deviation_m"]["max"]) is float
    assert all(type(x) is float for x in d["series"]["t"])


def test_refuses_different_machines_or_joints(scene):
    a = _motion([0.0, 0.1])
    with pytest.raises(ValueError, match="machine"):
        compare(scene, a, _motion([0.0, 0.1], machine="other"))
    with pytest.raises(ValueError, match="joint"):
        compare(scene, a, _motion([0.0, 0.1], names=["slide", "side", "grip"]))
    with pytest.raises(ValueError, match="step_s"):
        compare(scene, a, a, step_s=0)


# --- DTW (the path shape without time) ---------------------------------------------------------

def _full_dtw_mean(pa, pb):
    """Plain O(N*M) python DTW with the same tie-break (diagonal, then i-1, then j-1)."""
    n, m = len(pa), len(pb)
    acc = np.full((n + 1, m + 1), np.inf)
    length = np.zeros((n + 1, m + 1), int)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            opts = [(acc[i - 1, j - 1], length[i - 1, j - 1]), (acc[i - 1, j], length[i - 1, j]),
                    (acc[i, j - 1], length[i, j - 1])]
            best = min(range(3), key=lambda k: opts[k][0])
            acc[i, j] = np.linalg.norm(pa[i - 1] - pb[j - 1]) + opts[best][0]
            length[i, j] = opts[best][1] + 1
    return acc[n, m] / length[n, m]


def test_dtw_hand_computed_value():
    from physical_sandbox.compare import _dtw_mean

    pa = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], float)
    pb = np.array([[0, 0, 0], [2, 0, 1]], float)
    # costs: a0 [0, sqrt5], a1 [1, sqrt2], a2 [2, 1]; best path (0,0),(1,0),(2,1) = 0 + 1 + 1 over 3 pairs
    # (the 4-pair path (0,0),(1,0),(1,1),(2,1) costs 0+1+sqrt2+1, the diagonal-ish (0,0),(1,1),(2,1) costs 0+sqrt2+1)
    assert _dtw_mean(pa, pb) == pytest.approx(2.0 / 3.0, abs=1e-12)


@pytest.mark.parametrize("n,m", [(12, 12), (15, 9), (7, 20)])
def test_banded_dtw_equals_the_full_one_when_the_band_covers_the_path(n, m):
    from physical_sandbox.compare import _dtw_mean

    rng = np.random.default_rng(n * 100 + m)
    pa, pb = rng.normal(size=(n, 3)), rng.normal(size=(m, 3))
    full = _full_dtw_mean(pa, pb)
    assert _dtw_mean(pa, pb, band=max(n, m)) == pytest.approx(full, abs=1e-12)


def test_a_narrow_band_can_only_cost_more():
    from physical_sandbox.compare import _dtw_mean

    rng = np.random.default_rng(3)
    pa, pb = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
    assert _dtw_mean(pa, pb, band=2) >= _dtw_mean(pa, pb, band=40) - 1e-12


def test_dtw_memory_stays_bounded_for_long_motions():
    import tracemalloc

    from physical_sandbox.compare import _dtw_mean

    t = np.linspace(0, 1, 4000)
    pa = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    tracemalloc.start()
    try:
        _dtw_mean(pa, pa + [0.0, 0.01, 0.0])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 120e6  # the old full cost tensor alone would be 4000 * 4000 * 3 * 8 = 384 MB


def test_dtw_refuses_a_band_matrix_over_the_limit(monkeypatch):
    import physical_sandbox.compare as cmp

    monkeypatch.setattr(cmp, "_MAX_DTW_CELLS", 1000)
    pa = np.zeros((500, 3))
    with pytest.raises(ValueError, match=r"500.*500|DTW"):
        cmp._dtw_mean(pa, pa)


# --- jerk ----------------------------------------------------------------------------------------

def test_jerk_of_a_cubic_ramp_is_six_times_its_coefficient(scene):
    c3 = 0.5  # x(t) = c3 t^3, so the third derivative is 6 c3 = 3 m/s^3 everywhere
    t = np.arange(0, 1.0 + 1e-9, 1 / 60)
    ramp = _motion(c3 * t ** 3, fps=60.0)
    for step in (1 / 60, 1 / 30, 1 / 20):  # every step is a whole number of samples, so the grid is exact
        c = compare(scene, ramp, ramp, step_s=step)
        assert c.jerk["a"] == pytest.approx(6 * c3, rel=1e-6)
        assert c.jerk["b"] == pytest.approx(6 * c3, rel=1e-6)


def test_jerk_is_nan_with_fewer_than_four_samples_and_null_in_the_dict(scene):
    short = _motion([0.0, 0.1, 0.2])  # 3 samples at 30 fps
    c = compare(scene, short, short)
    assert math.isnan(c.jerk["a"]) and math.isnan(c.jerk["b"])
    d = c.to_dict()
    assert d["jerk"] == {"a": None, "b": None}
    assert json.loads(json.dumps(d, allow_nan=False)) == d
