"""Pinned upstream metrics and raw detection sources for desktop benchmarks.

Kept outside rhd so existing experiment signatures remain valid. No ground-truth
data is passed to a tracker or used to remove predictions.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import zipfile

import numpy as np

from rhd.cache import DetectionCache
from rhd.common import digest, file_hash, iou_matrix
from rhd.evaluation import group_frames, identity_metrics, read_rows

ROOT = Path(__file__).resolve().parents[1]
TRACK_SHA = "12c8791b303e0a0b50f753af204249e622d0281a"
BOX_SHA = "6dbc31b68c54c9305dc1f31841aff621f4914bb6"
AUTHOR_SHA = "8a8d416b65486a43a2d2641876f585095912f5ac"


def verify_vendor():
    path = ROOT / "third_party/benchmark_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Run scripts/setup_benchmarks.py first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        if file_hash(ROOT / "third_party" / relative) != expected:
            raise ValueError(f"Modified upstream benchmark file: {relative}")
    return manifest


@contextmanager
def legacy_numpy():
    """Process-local compatibility for unmodified legacy upstream dependencies.

    Restore numpy afterwards; no vendor source or matching rule is changed.
    """
    aliases = {"int": int, "float": float,
               "asfarray": lambda a, dtype=float: np.asarray(a, dtype=dtype)}
    added = {name: value for name, value in aliases.items() if name not in np.__dict__}
    for name, value in added.items():
        setattr(np, name, value)
    try:
        yield
    finally:
        for name in added:
            delattr(np, name)


def upstream_metrics():
    sys.path.insert(0, str(ROOT / "third_party" / f"TrackEval-{TRACK_SHA}"))
    import trackeval
    return [trackeval.metrics.Identity({"THRESHOLD": .5, "PRINT_CONFIG": False}),
            trackeval.metrics.CLEAR({"THRESHOLD": .5, "PRINT_CONFIG": False}),
            trackeval.metrics.HOTA()]


def metric_data(gt, pred, cameras, start, stop, global_ids=False):
    """Each camera/frame is a unique observation time, including empty frames."""
    def key(row):
        return row["id"] if global_ids else (row["camera"], row["id"])
    gmap = {v: i for i, v in enumerate(sorted({key(r) for r in gt}))}
    pmap = {v: i for i, v in enumerate(sorted({key(r) for r in pred}))}
    groups_g, groups_p = group_frames(gt), group_frames(pred)
    gids, pids, scores = [], [], []
    for camera in cameras:
        for frame in range(start, stop):
            g, p = groups_g.get((camera, frame), []), groups_p.get((camera, frame), [])
            gids.append(np.asarray([gmap[key(r)] for r in g], dtype=np.int64))
            pids.append(np.asarray([pmap[key(r)] for r in p], dtype=np.int64))
            scores.append(iou_matrix([r["box"] for r in g], [r["box"] for r in p]))
    return {"gt_ids": gids, "tracker_ids": pids, "similarity_scores": scores,
            "num_gt_ids": len(gmap), "num_tracker_ids": len(pmap),
            "num_gt_dets": len(gt), "num_tracker_dets": len(pred),
            "num_timesteps": len(gids)}


def compact_metrics(results):
    result = {}
    for block in results:
        for key in ("IDF1", "IDP", "IDR", "IDTP", "IDFP", "IDFN", "MOTA", "IDSW",
                    "CLR_TP", "CLR_FP", "CLR_FN", "HOTA", "DetA", "AssA", "LocA"):
            if key in block:
                value = block[key]
                number = float(np.mean(value)) if isinstance(value, np.ndarray) else float(value)
                result[key] = number if np.isfinite(number) else None
    return result


def sct_metrics(gt, pred, cameras, start, stop):
    metrics = upstream_metrics()
    per_camera, raw = {}, {}
    with legacy_numpy():
        for camera in cameras:
            g = [r for r in gt if r["camera"] == camera]
            p = [r for r in pred if r["camera"] == camera]
            data = metric_data(g, p, [camera], start, stop)
            blocks = [metric.eval_sequence(data) for metric in metrics]
            per_camera[str(camera)] = compact_metrics(blocks)
            raw[str(camera)] = blocks
        combined = [metric.combine_sequences({cam: blocks[i] for cam, blocks in raw.items()})
                    for i, metric in enumerate(metrics)]
    aggregate = compact_metrics(combined)
    custom = identity_metrics(gt, pred, .5, local=True)
    parity = all(abs(aggregate[key] - custom[key]) < 1e-9
                 for key in ("IDTP", "IDFP", "IDFN", "IDF1", "IDP", "IDR"))
    if not parity:
        raise ValueError("Identity metric disagreement with pinned TrackEval; investigate before reporting")
    mean = {key: float(np.mean([v[key] for v in per_camera.values()]))
            for key in ("IDF1", "IDP", "IDR", "HOTA", "DetA", "AssA", "MOTA", "IDSW")}
    return {"per_camera": per_camera, "combined_counts": aggregate,
            "unweighted_camera_mean": mean, "custom_identity_parity": parity}


def official_sct_audit(gt, pred, cameras, start, stop):
    """Execute author loading function and matching threshold unchanged.

    Also evaluate standard IoU .5 using the same motmetrics package. This audit
    does not silently replace the author's zero-distance threshold.
    """
    import pandas as pd
    import motmetrics as mm
    path = ROOT / "third_party/roundabouthd_evaluation_8a8d416/eval_sct.py"
    spec = importlib.util.spec_from_file_location("roundabouthd_official_sct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def dataframe(rows, camera):
        with tempfile.TemporaryDirectory(prefix="rhd_official_sct_") as temporary:
            path = Path(temporary) / "input.txt"
            path.write_text("".join(f'{r["frame"] + 1} {r["id"]} ' + " ".join(map(str, r["box"])) + " 2\n"
                                    for r in rows if r["camera"] == camera), encoding="utf-8")
            return module.load_data(path)
    output = {}
    fields = ["idf1", "idp", "idr", "idtp", "idfp", "idfn", "mota", "num_switches"]
    with legacy_numpy():
        handler = mm.metrics.create()
        for camera in cameras:
            g, p = dataframe(gt, camera), dataframe(pred, camera)
            values = {}
            for name, distance in [("author_unmodified_iou_distance_0", 0.), ("standard_iou_0.5", .5)]:
                acc = mm.utils.compare_to_groundtruth(g, p, dist="iou", distth=distance)
                summary = handler.compute(acc, metrics=fields, name="sequence").iloc[0]
                values[name] = {key: float(v) if np.isfinite(v) else None for key, v in summary.items()}
            output[str(camera)] = values
    return {"source_sha256": file_hash(path), "per_camera": output,
            "note": "Author SCT code sets IoU distance threshold 0 (exact boxes), and positional CLI arguments are prediction then groundtruth. Standard TrackEval results are reported separately. Publication metric parity is unresolved."}


class CacheSource:
    def __init__(self, path, cfg):
        self.cache = DetectionCache(path, readonly=True)
        self.cache.validate_range(cfg["cameras"], cfg["start_frame"], cfg["stop_frame"], cfg["fps"])
        self.metadata = {c: self.cache.get_meta(f"camera:{c}") for c in cfg["cameras"]}
        self.provenance = {"kind": "shared_local_detection_cache", "settings": self.cache.get_meta("detector"),
                           "signature": self.cache.signature(cfg["cameras"], cfg["start_frame"], cfg["stop_frame"])}

    def frame(self, camera, frame):
        return self.cache.frame(camera, frame)[0]

    def close(self):
        self.cache.close()


class ArchiveSource:
    """Only labels_xy contains the author's raw detector outputs.

    labels_test is SCT-derived (scores exactly 1); labels_GT is corrected.
    Both are forbidden as tracker inputs.
    """
    def __init__(self, dataset, cfg):
        self.archives, self.members, self.metadata = {}, {}, {}
        hashes = {}
        for camera in cfg["cameras"]:
            path = dataset / f"imagesc{camera:03}/detection/labels_xy.zip"
            archive = zipfile.ZipFile(path)
            members = {}
            for name in archive.namelist():
                if not name.endswith(".txt"):
                    continue
                stem = Path(name).stem
                if not stem.startswith("img") or not stem[3:].isdigit():
                    raise ValueError(f"Unrecognized detection filename: {name}")
                frame = int(stem[3:])
                if frame in members:
                    raise ValueError("Duplicate detection frame in archive")
                members[frame] = name
            if any(f not in members for f in range(cfg["start_frame"], cfg["stop_frame"])):
                raise ValueError(f"Missing raw detection frames for camera {camera}")
            self.archives[camera], self.members[camera] = archive, members
            self.metadata[camera] = {"width": 3840, "height": 2160, "fps": 15}
            hashes[str(camera)] = file_hash(path)
        self.provenance = {"kind": "author_raw_labels_xy", "archive_sha256": hashes,
                           "frame_base": 0, "model_reported": "YOLOv12x", "detector_timing": "unavailable",
                           "exact_weights_and_inference_parameters": "not included in archive; not verified",
                           "descriptor": "unavailable; SCT only", "signature": digest(hashes)}

    def frame(self, camera, frame):
        data = self.archives[camera].read(self.members[camera][frame]).decode("utf-8-sig")
        result = []
        for line in data.splitlines():
            if not line.strip():
                continue
            values = np.asarray([float(v) for v in line.split()])
            if len(values) != 6 or not np.isfinite(values).all():
                raise ValueError("Raw detections need class,x1,y1,x2,y2,confidence")
            cls, x1, y1, x2, y2, score = values
            if cls != int(cls) or not 0 <= score <= 1 or x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid raw detection")
            if int(cls) in [2, 3, 5, 7]:
                result.append({"box": [x1, y1, x2, y2], "score": score, "class_id": int(cls), "feature": [0.] * 48})
        return result

    def close(self):
        for archive in self.archives.values():
            archive.close()


class MotionGate:
    """Optional causal moving-track gate, applied identically after every tracker.

    Requires observed displacement, never future points or ground truth. Once
    confirmed, a stopped track remains eligible. False detections can still pass.
    This is an internal alternative, not the unpublished author parking filter.
    """
    def __init__(self, settings, width, height):
        self.settings, self.width, self.height = settings, width, height
        self.state = {}

    def accept(self, identity, box, frame):
        if self.settings["kind"] == "none":
            return True
        center = np.asarray([(box[0] + box[2]) / (2 * self.width), (box[1] + box[3]) / (2 * self.height)])
        record = self.state.setdefault(identity, {"points": [], "confirmed": False})
        if record["confirmed"]:
            return True
        points = record["points"]
        points.append((frame, center))
        points[:] = [(f, p) for f, p in points if frame - f < self.settings["window_frames"]]
        if len(points) >= self.settings["minimum_observations"]:
            coords = np.asarray([p for _, p in points])
            record["confirmed"] = bool(np.linalg.norm(np.ptp(coords, axis=0)) >= self.settings["displacement_fraction"])
        return record["confirmed"]


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            x1, y1, x2, y2 = row["box"]
            f.write(f'{row["camera"]} {row["id"]} {row["frame"]} {x1:.6f} {y1:.6f} {x2-x1:.6f} {y2-y1:.6f} -1 -1\n')
