"""CityFlowV2 data contracts. Inference readers never load identity annotations."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

import cv2
import numpy as np
from PIL import Image

from rhd.common import file_hash


def read_pairs(path: Path, cast):
    result = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        key, value = line.split()
        camera = int(key.removeprefix("c"))
        if camera in result:
            raise ValueError(f"Repeated camera in {path}")
        result[camera] = cast(value)
    return result


def scene_metadata(dataset: Path, split: str, scenario: str) -> dict:
    dataset = dataset.resolve()
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown official split")
    folder = dataset / split / scenario
    timestamp_file = dataset / "cam_timestamp" / f"{scenario}.txt"
    count_file = dataset / "cam_framenum" / f"{scenario}.txt"
    offsets = read_pairs(timestamp_file, float)
    counts = read_pairs(count_file, int)
    cameras = {}
    for video in sorted(folder.glob("c*/vdo.avi")):
        camera = int(video.parent.name[1:])
        cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise ValueError(f"Unreadable video: {video}")
            meta = {"camera": camera, "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "timestamp_offset_s": offsets[camera], "video": str(video),
                    "video_sha256": file_hash(video),
                    "roi": str(video.parent / "roi.jpg"),
                    "roi_sha256": file_hash(video.parent / "roi.jpg")}
        finally:
            cap.release()
        expected_fps = 8. if scenario == "S03" and camera == 15 else 10.
        if abs(meta["fps"] - expected_fps) > .001 or meta["frames"] != counts[camera]:
            raise ValueError(f"Native video metadata disagrees with dataset documentation: {video}")
        if meta["width"] < 1 or meta["height"] < 1 or not math.isfinite(offsets[camera]) or offsets[camera] < 0:
            raise ValueError("Invalid camera metadata")
        cameras[str(camera)] = meta
    if not cameras or set(map(int, cameras)) != set(offsets) or set(offsets) != set(counts):
        raise ValueError(f"Missing camera video/offset/frame count for {scenario}")
    return {"dataset_root": str(dataset), "split": split, "scenario": scenario,
            "cameras": cameras, "timestamps_sha256": file_hash(timestamp_file),
            "framecounts_sha256": file_hash(count_file),
            "clock": "offset_s + zero_based_local_frame/native_fps; integer microsecond events",
            "timing_limit": "Supplied start offsets only. Transmission skips/stuck frames are not repaired; no label-derived synchronization."}


def event_time_us(meta: dict, local_frame: int) -> int:
    if type(local_frame) is not int or not 0 <= local_frame < meta["frames"]:
        raise ValueError("Local frame out of range")
    return round((meta["timestamp_offset_s"] + local_frame / meta["fps"]) * 1_000_000)


def load_roi(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        roi = np.asarray(image, dtype=np.uint8)
    if roi.ndim != 2:
        raise ValueError(f"Expected grayscale official ROI: {path}")
    return roi


def roi_accepts(box, roi: np.ndarray) -> bool:
    """Official four-corner rule after coordinate truncation; out-of-bounds corners ignored.

    This is image-only filtering. It does not inspect which identities are annotated.
    Width/height are truncated separately, as in the official 9-column export.
    """
    x1, y1, x2, y2 = box
    x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
    height, width = roi.shape
    for xx in (x, x+w):
        for yy in (y, y+h):
            if 0 <= xx < width and 0 <= yy < height and roi[yy, xx] < 255:
                return False
    return True


def raw_detections(path: Path, frames: int, roi: np.ndarray, minimum_score=.1):
    """Read only official det_*.txt files, never GT or MTSC trajectories."""
    if path.parent.name != "det" or path.name not in ("det_mask_rcnn.txt", "det_yolo3.txt", "det_ssd512.txt"):
        raise ValueError("Only supplied raw detector files are accepted")
    if not 0 <= minimum_score <= 1:
        raise ValueError("Invalid minimum confidence")
    result = defaultdict(list)
    counts = {"raw": 0, "below_confidence": 0, "outside_roi": 0, "retained": 0}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        a = line.split(",")
        if len(a) != 10 or int(a[1]) != -1:
            raise ValueError(f"Unexpected raw detection at {path}:{number}")
        frame = int(a[0])-1
        x, y, w, h, score = map(float, a[2:7])
        if not 0 <= frame < frames or not np.isfinite([x, y, w, h, score]).all() or w <= 0 or h <= 0 or not 0 <= score <= 1:
            raise ValueError(f"Invalid detection at {path}:{number}")
        counts["raw"] += 1
        if score < minimum_score:
            counts["below_confidence"] += 1
            continue
        box = [x, y, x+w, y+h]
        if not roi_accepts(box, roi):
            counts["outside_roi"] += 1
            continue
        result[frame].append({"box": box, "score": score, "class_id": 2})
        counts["retained"] += 1
    return dict(result), counts


def annotation_rows(scene: dict, start_s: float, stop_s: float) -> list[dict]:
    """Calibration/evaluation-only label reader; reject ambiguous boxes."""
    if not 0 <= start_s < stop_s:
        raise ValueError("Invalid label interval")
    result, seen = [], set()
    for camera_text, meta in scene["cameras"].items():
        path = Path(meta["video"]).parent / "gt" / "gt.txt"
        for line in path.read_text().splitlines():
            a = line.split(",")
            frame = int(a[0])-1
            time_us = event_time_us(meta, frame)
            if not round(start_s*1e6) <= time_us < round(stop_s*1e6):
                continue
            identity = int(a[1]); camera = int(camera_text)
            key = (camera, identity, frame)
            if key in seen:
                raise ValueError(f"Repeated GT key {key}")
            seen.add(key)
            x, y, w, h = map(float, a[2:6])
            if not np.isfinite([x,y,w,h]).all() or w <= 0 or h <= 0:
                raise ValueError(f"Invalid GT box {key}")
            result.append({"camera": camera, "id": identity, "frame": frame,
                           "time_s": time_us/1e6, "box": [x,y,x+w,y+h]})
    return result


def contiguous_spans(rows: list[dict]) -> dict[int, list[list[dict]]]:
    by_track = defaultdict(list)
    for row in rows:
        by_track[(row["id"], row["camera"])].append(row)
    result = defaultdict(list)
    for (identity, _), track in by_track.items():
        current = []
        for row in sorted(track, key=lambda r:r["frame"]):
            if current and row["frame"] != current[-1]["frame"]+1:
                result[identity].append(current); current=[]
            current.append(row)
        if current:
            result[identity].append(current)
    for identity in result:
        result[identity].sort(key=lambda s:(s[0]["time_s"], s[0]["camera"]))
    return dict(result)
