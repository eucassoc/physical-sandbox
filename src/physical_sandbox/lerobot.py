"""Read a LeRobot v3.0 dataset: an episode's joint angles as a Motion, and its video.

Only ``meta/info.json``, ``meta/episodes`` and the data files are read; nothing is copied.
"""
from __future__ import annotations

import json
import numbers
import os
import string
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .errors import InputError, InputIndexError, InputKeyError
from .motion import Motion

# Dataset units -> URDF joint range. Per joint: (in_lo, in_hi, sign). The dataset normalises the five
# arm joints to -100..100 and the gripper to 0..100; the value is mapped linearly onto the URDF's
# [lower, upper] (sign -1 flips the direction).
# Checked by eye on lerobot/svla_so101_pickplace (examples/so101-pickplace): with physical-sandbox-proto's
# SO-101 URDF, the shell drawn from these angles lies on the real arm at rest, at the grasp and over the
# tray, and each joint turns the right way (sign +1 for all). The ends of each range are not constrained by
# that check: the site's scale was fitted to the same drawing, so an offset of a few degrees or a range
# somewhat off would be partly absorbed. Another calibration of the same arm may differ.
JOINT_MAP: dict = {
    "shoulder_pan": (-100.0, 100.0, 1),
    "shoulder_lift": (-100.0, 100.0, 1),
    "elbow_flex": (-100.0, 100.0, 1),
    "wrist_flex": (-100.0, 100.0, 1),
    "wrist_roll": (-100.0, 100.0, 1),
    "gripper": (0.0, 100.0, 1),
}


def map_joints(values, joint_names, limits) -> np.ndarray:
    """(N, J) dataset values -> (N, J) radians, through ``JOINT_MAP`` onto ``limits`` (J, 2)."""
    values = np.asarray(values, float)
    out = np.empty_like(values)
    for j, name in enumerate(joint_names):
        if name not in JOINT_MAP:
            raise ValueError(f"no mapping for joint {name!r} in JOINT_MAP (has {sorted(JOINT_MAP)})")
        in_lo, in_hi, sign = JOINT_MAP[name]
        u = (values[:, j] - in_lo) / (in_hi - in_lo)
        if sign < 0:
            u = 1.0 - u
        lower, upper = limits[j]
        out[:, j] = lower + u * (upper - lower)
    return out


def _info(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def _index(row: dict, column: str) -> int:
    """A chunk or file index from the episodes table: a whole number >= 0, never text (it goes into a path)."""
    value = row[column]
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 0:
        raise InputError(f"the dataset's {column!r} must be a whole number >= 0, got {value!r}")
    return int(value)


def _dataset_file(dataset_dir: Path, info: dict, template_key: str, fields: dict) -> Path:
    """The file ``info[template_key]`` names inside ``dataset_dir``, filled with ``fields``.

    The template comes from the dataset's own ``info.json``, so it is checked before use: it must be a relative
    path without ``..``, and its only fields are the expected names (no positional fields, no attribute or
    index access, no conversions). The filled path must stay inside the dataset folder. Symbolic links are not
    followed for that check: the Hugging Face cache links every file of a snapshot to a blob outside it.
    """
    template = info.get(template_key)
    if not isinstance(template, str) or not template:
        raise InputError(f"the dataset's info.json has no {template_key}")

    def bad(why: str) -> InputError:
        return InputError(f"the dataset's {template_key} {template!r} cannot be used: {why}")

    if _escapes(template):
        raise bad("it must be a path inside the dataset folder (relative, without '..')")
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as e:
        raise bad(str(e)) from None
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if name not in fields:
            raise bad(f"the field {{{name}}} is not one of {', '.join(sorted(fields))}")
        if conversion is not None or (spec and "{" in spec):
            raise bad(f"the field {{{name}}} may only carry a plain format such as :03d")
    try:
        rel = template.format(**fields)
    except (ValueError, TypeError, KeyError, IndexError) as e:
        raise bad(str(e)) from None
    if _escapes(rel):
        raise bad(f"it names {rel!r}, outside the dataset folder")
    base = os.path.normpath(os.path.abspath(dataset_dir))
    path = os.path.normpath(os.path.join(base, rel))
    if not Path(path).is_relative_to(base):
        raise bad(f"it names {rel!r}, outside the dataset folder")
    return Path(path)


def _escapes(rel: str) -> bool:
    """Whether a relative path from a dataset's metadata is absolute or climbs out with '..'."""
    posix, windows = PurePosixPath(rel), PureWindowsPath(rel)
    return (posix.is_absolute() or windows.is_absolute() or bool(windows.drive) or rel.startswith(("/", "\\"))
            or ".." in posix.parts or ".." in windows.parts)


def _episode_row(dataset_dir: Path, episode: int) -> dict:
    files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no meta/episodes files under {dataset_dir}")
    total = 0
    for f in files:
        t = pq.read_table(f)
        idx = t.column("episode_index").to_pylist()
        total += len(idx)
        if episode in idx:
            return t.slice(idx.index(episode), 1).to_pylist()[0]
    raise InputIndexError(f"episode {episode} is not in the dataset (it has {total} episodes)")


def read_episode(dataset_dir, episode: int, *, machine, key: str, label, machine_name: str) -> Motion:
    dataset_dir = Path(dataset_dir).expanduser()
    info = _info(dataset_dir)
    feat = info["features"].get(key)
    if feat is None:
        raise InputKeyError(f"feature {key!r} is not in the dataset; it has {sorted(info['features'])}")
    ds_names = [n[:-4] if n.endswith(".pos") else n for n in (feat.get("names") or [])]
    if ds_names != list(machine.joint_names):
        raise InputError(f"the dataset's joints {ds_names} are not the machine's joints {list(machine.joint_names)}")
    row = _episode_row(dataset_dir, episode)
    data_file = _dataset_file(dataset_dir, info, "data_path", {
        "chunk_index": _index(row, "data/chunk_index"), "file_index": _index(row, "data/file_index")})
    table = pq.read_table(data_file, columns=[key, "timestamp", "frame_index", "episode_index"])
    rows = table.filter(pc.equal(table["episode_index"], episode)).to_pydict()
    order = np.argsort(rows["frame_index"], kind="stable")
    vals = np.array(rows[key], float)[order]
    ts = np.array(rows["timestamp"], float)[order]
    if vals.shape[0] != row["length"]:
        raise ValueError(f"episode {episode} has {vals.shape[0]} rows in {data_file.name}, its meta says {row['length']}")
    return Motion(
        machine=machine_name, joint_names=list(machine.joint_names), t=ts - ts[0],
        q=map_joints(vals, machine.joint_names, machine.limits),
        source="lerobot", label=label if label is not None else f"{dataset_dir.name} episode {episode}")


def lerobot_video(dataset_dir, episode: int, camera: str) -> tuple:
    """(mp4 path, from_timestamp, to_timestamp) of ``camera`` for ``episode``.

    ``camera`` is the short name (``side``) or the feature key (``observation.images.side``).
    """
    dataset_dir = Path(dataset_dir).expanduser()
    info = _info(dataset_dir)
    prefix = "observation.images."
    cameras = sorted(k[len(prefix):] for k, f in info["features"].items()
                     if k.startswith(prefix) and f.get("dtype") == "video")
    short = camera[len(prefix):] if camera.startswith(prefix) else camera
    if short not in cameras:
        raise InputKeyError(f"the dataset has no camera {camera!r}; its cameras are {', '.join(cameras)}")
    vkey = prefix + short
    row = _episode_row(dataset_dir, episode)
    path = _dataset_file(dataset_dir, info, "video_path", {
        "video_key": vkey, "chunk_index": _index(row, f"videos/{vkey}/chunk_index"),
        "file_index": _index(row, f"videos/{vkey}/file_index")})
    return path, float(row[f"videos/{vkey}/from_timestamp"]), float(row[f"videos/{vkey}/to_timestamp"])
