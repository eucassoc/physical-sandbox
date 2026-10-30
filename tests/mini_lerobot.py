"""A tiny LeRobot v3.0 dataset written into a test's folder: the same layout as a real one, nothing from one."""
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def make_dataset(root, joint_names, episodes, *, cameras=("side",), size=(320, 240), fps=30,
                 data_path=DATA_PATH, video_path=VIDEO_PATH, row_overrides=None):
    """Write a dataset with ``episodes`` (a list of (N, J) arrays in dataset units) one after the other in one
    data file and one video per camera, where frame k of the video is grey ``(4 k) % 256``. ``row_overrides``
    replaces columns of the episodes table (for crafted, hostile metadata). Returns the dataset folder."""
    root = Path(root)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    features = {"observation.state": {"dtype": "float32", "shape": [len(joint_names)],
                                      "names": [f"{n}.pos" for n in joint_names]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]}}
    for cam in cameras:
        features[f"observation.images.{cam}"] = {"dtype": "video", "shape": [size[1], size[0], 3]}
    info = {"codebase_version": "v3.0", "fps": fps, "data_path": data_path, "video_path": video_path,
            "features": features}
    (root / "meta" / "info.json").write_text(json.dumps(info))

    states, ts, frames, eps, rows = [], [], [], [], []
    start = 0
    for ep, values in enumerate(episodes):
        values = np.asarray(values, float)
        n = len(values)
        states += [list(map(float, r)) for r in values]
        ts += [k / fps for k in range(n)]
        frames += list(range(n))
        eps += [ep] * n
        row = {"episode_index": ep, "data/chunk_index": 0, "data/file_index": 0, "length": n}
        for cam in cameras:
            key = f"videos/observation.images.{cam}"
            row.update({f"{key}/chunk_index": 0, f"{key}/file_index": 0,
                        f"{key}/from_timestamp": start / fps, f"{key}/to_timestamp": (start + n) / fps})
        rows.append(row)
        start += n
    for row in rows:
        row.update(row_overrides or {})
    pq.write_table(pa.Table.from_pylist(rows), root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    data_file = root / DATA_PATH.format(chunk_index=0, file_index=0)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"observation.state": states, "timestamp": pa.array(ts, pa.float32()),
                             "frame_index": frames, "episode_index": eps}), data_file)
    for cam in cameras:
        video = root / VIDEO_PATH.format(video_key=f"observation.images.{cam}", chunk_index=0, file_index=0)
        video.parent.mkdir(parents=True, exist_ok=True)
        w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
        for k in range(start):
            w.write(np.full((size[1], size[0], 3), (4 * k) % 256, np.uint8))
        w.release()
    return root
