import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from physical_sandbox.lerobot import JOINT_MAP, lerobot_video
from physical_sandbox.machine import load_machine
from physical_sandbox.motion import motion_from_lerobot

DATASET = Path(os.environ.get(
    "PS_LEROBOT_DATASET",
    "~/.cache/huggingface/lerobot/hub/datasets--lerobot--svla_so101_pickplace/snapshots/"
    "f641879e22172be7e8161d5e6c1503c2d2feb657")).expanduser()
SO101 = Path(os.environ.get(
    "PS_SO101_URDF", "~/work/physical-sandbox-proto/robots/so101/so101.urdf")).expanduser()
needs_data = pytest.mark.skipif(
    not (DATASET / "meta" / "info.json").exists() or not SO101.exists(),
    reason="LeRobot dataset or SO-101 URDF not found (set PS_LEROBOT_DATASET, PS_SO101_URDF)")


@pytest.fixture(scope="module")
def machine():
    return load_machine(SO101)


def _episode_row(ep):
    t = pq.read_table(next((DATASET / "meta" / "episodes").glob("chunk-*/file-*.parquet")))
    return t.to_pylist()[ep]


@needs_data
def test_joint_map_has_one_entry_per_joint(machine):
    assert set(JOINT_MAP) == set(machine.joint_names)
    for lo, hi, sign in JOINT_MAP.values():
        assert lo < hi and sign in (1, -1)


@needs_data
def test_episode_0_has_the_length_the_meta_says(machine):
    m = motion_from_lerobot(DATASET, 0, machine=machine)
    row = _episode_row(0)
    assert m.q.shape == (row["length"], 6) == (303, 6)
    assert m.t[0] == 0.0
    assert np.all(np.diff(m.t) > 0)
    assert m.t[-1] == pytest.approx(302 / 30, abs=1e-3)
    assert m.joint_names == machine.joint_names
    assert m.source == "lerobot" and m.machine == "so101"
    assert "0" in m.label


@needs_data
def test_a_different_episode_has_its_own_length(machine):
    m = motion_from_lerobot(DATASET, 1, machine=machine, label="second")
    assert m.q.shape[0] == _episode_row(1)["length"] == 266
    assert m.label == "second"


@needs_data
def test_values_follow_the_map_and_stay_near_the_limits(machine):
    m = motion_from_lerobot(DATASET, 0, machine=machine)
    lo, hi = machine.limits[:, 0], machine.limits[:, 1]
    assert np.all(m.q >= lo - 0.1) and np.all(m.q <= hi + 0.1)
    # frame 0 of episode 0: shoulder_pan 1.956 of -100..100 -> just above the middle
    j = machine.joint_names.index("shoulder_pan")
    assert m.q[0, j] == pytest.approx(lo[j] + (1.9560878 + 100) / 200 * (hi[j] - lo[j]), abs=1e-4)


@needs_data
def test_the_mapping_ends_and_middle(machine, monkeypatch):
    import physical_sandbox.lerobot as lr

    lo, hi = machine.limits[:, 0], machine.limits[:, 1]
    names = machine.joint_names
    g = names.index("gripper")
    arm = [i for i, n in enumerate(names) if n != "gripper"]
    for x, want in ((-100.0, lo), (100.0, hi), (0.0, (lo + hi) / 2)):
        q = lr.map_joints(np.full((1, 6), x), names, machine.limits)
        assert np.allclose(q[0, arm], want[arm])
    # gripper runs 0..100
    assert lr.map_joints(np.full((1, 6), 0.0), names, machine.limits)[0, g] == pytest.approx(lo[g])
    assert lr.map_joints(np.full((1, 6), 100.0), names, machine.limits)[0, g] == pytest.approx(hi[g])
    assert lr.map_joints(np.full((1, 6), 50.0), names, machine.limits)[0, g] == pytest.approx((lo[g] + hi[g]) / 2)
    # a flipped sign swaps the ends
    flipped = dict(JOINT_MAP, shoulder_pan=(-100.0, 100.0, -1))
    monkeypatch.setattr(lr, "JOINT_MAP", flipped)
    q = lr.map_joints(np.full((1, 6), -100.0), names, machine.limits)
    assert q[0, 0] == pytest.approx(hi[0])


@needs_data
def test_action_key_works_too(machine):
    m = motion_from_lerobot(DATASET, 0, machine=machine, key="action")
    assert m.q.shape == (303, 6)
    with pytest.raises(KeyError, match="observation.state"):
        motion_from_lerobot(DATASET, 0, machine=machine, key="nope")


@needs_data
def test_episode_out_of_range_is_refused(machine):
    with pytest.raises(IndexError, match="50"):
        motion_from_lerobot(DATASET, 50, machine=machine)


@needs_data
def test_a_machine_with_other_joint_names_is_refused(machine):
    from dataclasses import replace

    other = replace(machine, joint_names=["j1", "j2", "j3", "j4", "j5", "j6"])
    with pytest.raises(ValueError) as e:
        motion_from_lerobot(DATASET, 0, machine=other)
    assert "shoulder_pan" in str(e.value) and "j1" in str(e.value)


@needs_data
def test_video_gives_the_file_and_the_span_of_the_episode():
    path, a, b = lerobot_video(DATASET, 1, "side")
    assert path.is_file() and path.suffix == ".mp4" and "observation.images.side" in str(path)
    assert (a, b) == pytest.approx((10.1, 18.96666666666667))
    path, a, b = lerobot_video(DATASET, 0, "up")
    assert "observation.images.up" in str(path) and (a, b) == pytest.approx((0.0, 10.1))


@needs_data
def test_video_with_a_missing_camera_names_the_ones_that_exist():
    with pytest.raises(KeyError) as e:
        lerobot_video(DATASET, 0, "top")
    msg = str(e.value)
    assert "top" in msg and "side" in msg and "up" in msg
    with pytest.raises(KeyError):
        lerobot_video(DATASET, 0, "observation.images.top")


@needs_data
def test_video_accepts_the_full_feature_key():
    path, _, _ = lerobot_video(DATASET, 0, "observation.images.side")
    assert path.is_file()


@needs_data
def test_an_episode_survives_a_recording(machine, tmp_path):
    from physical_sandbox.motion import load_record, save_record

    m = motion_from_lerobot(DATASET, 0, machine=machine)
    save_record([m], "site-1", tmp_path / "r.json")
    (back,) = load_record(tmp_path / "r.json")
    assert np.array_equal(back.q, m.q) and np.array_equal(back.t, m.t)
    assert back.source == "lerobot" and back.label == m.label


# ---- a crafted dataset: the paths its metadata names stay inside it ----

from types import SimpleNamespace  # noqa: E402

from physical_sandbox.errors import InputError  # noqa: E402
from mini_lerobot import make_dataset  # noqa: E402

ONE_JOINT = SimpleNamespace(joint_names=["shoulder_pan"], limits=np.array([[-1.0, 1.0]]))


def mini(tmp_path, **kw):
    return make_dataset(tmp_path / "ds", ["shoulder_pan"], [np.linspace(-100, 100, 5).reshape(-1, 1)], **kw)


def test_a_small_dataset_reads_and_its_video_is_inside_it(tmp_path):
    ds = mini(tmp_path)
    m = motion_from_lerobot(ds, 0, machine=ONE_JOINT, machine_name="arm")
    assert m.q[:, 0] == pytest.approx(np.linspace(-1, 1, 5))
    path, start, end = lerobot_video(ds, 0, "side")
    assert path.is_file() and path.is_relative_to(ds) and (start, end) == (0.0, pytest.approx(5 / 30))


@pytest.mark.parametrize("template, why", [
    ("/etc/{chunk_index}-{file_index}.parquet", "inside the dataset folder"),
    ("../outside/{chunk_index}-{file_index}.parquet", "inside the dataset folder"),
    ("data/../../{chunk_index}-{file_index}.parquet", "inside the dataset folder"),
    ("data/{chunk_index.__class__}.parquet", "is not one of"),
    ("data/{file_index[0]}.parquet", "is not one of"),
    ("data/{other}.parquet", "is not one of"),
    ("data/{}.parquet", "is not one of"),
    ("data/{chunk_index!r}.parquet", "plain format"),
])
def test_a_data_path_template_that_could_leave_the_dataset_is_refused(tmp_path, template, why):
    ds = mini(tmp_path, data_path=template)
    with pytest.raises(InputError, match=why) as e:
        motion_from_lerobot(ds, 0, machine=ONE_JOINT, machine_name="arm")
    assert "data_path" in str(e.value)


def test_a_video_path_template_with_dotdot_is_refused(tmp_path):
    ds = mini(tmp_path, video_path="videos/{video_key}/../../../x-{chunk_index}-{file_index}.mp4")
    with pytest.raises(InputError, match="video_path"):
        lerobot_video(ds, 0, "side")


def test_a_camera_name_that_climbs_out_is_refused(tmp_path):
    ds = mini(tmp_path, cameras=("side", "../../../../tmp"))
    with pytest.raises(InputError, match="outside the dataset folder"):
        lerobot_video(ds, 0, "../../../../tmp")


@pytest.mark.parametrize("value", ["../x", "1", 1.5, -1, True])
def test_an_index_that_is_not_a_whole_number_is_refused(tmp_path, value):
    ds = mini(tmp_path, row_overrides={"data/chunk_index": value})
    with pytest.raises(InputError, match="data/chunk_index"):
        motion_from_lerobot(ds, 0, machine=ONE_JOINT, machine_name="arm")


def test_the_check_does_not_follow_links_out_of_the_snapshot(tmp_path):
    """The Hugging Face cache links each file of a snapshot to a blob outside it: that must still read."""
    ds = mini(tmp_path)
    data_file = ds / "data" / "chunk-000" / "file-000.parquet"
    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    data_file.rename(blob)
    data_file.symlink_to(blob)
    assert motion_from_lerobot(ds, 0, machine=ONE_JOINT, machine_name="arm").q.shape == (5, 1)
