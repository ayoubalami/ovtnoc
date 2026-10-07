"""Pinned Fast Online MTMCT association-only CPU adaptation on a frozen local stream.

This is not a reproduction of the published perception pipeline or CityFlow
table. No ground-truth path is accepted. The native association statements are
extracted from verified, unmodified MIT-licensed upstream source at runtime.
"""
from __future__ import annotations

import argparse
import ast
import copy
from collections import defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform

from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, iou_matrix, load_config, write_json
from rhd.evaluation import read_rows

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/fast_online_mtmct_f920c396"
COMMIT = "f920c396aaa820e231bd3fb7ec12d563ce75c396"
MATCH_THRESHOLD = 0.65  # Native upstream default, frozen; no test tuning.
FEATURE_MATCH_IOU = 0.5


def verify_upstream(root=UPSTREAM):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("commit") != COMMIT:
        raise ValueError("Unexpected published baseline commit")
    for name, item in manifest["files"].items():
        if file_hash(root / name) != item["sha256"]:
            raise ValueError(f"Modified upstream source: {name}")
    return manifest


class _NumpyCompatibility:
    """Supply removed aliases to legacy functions without changing numpy globally."""
    def __getattr__(self, name):
        return {"int": int, "bool": bool}.get(name) or getattr(np, name)


class _NoOverlap:
    def __getitem__(self, index):
        return 0


def _module(nodes, filename):
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(filename), "exec")


def load_upstream(retention_frames):
    """Return verified upstream functions and its exact frame-association block."""
    verify_upstream()
    directory = UPSTREAM / "2. online_MTMC"
    main = ast.parse((directory / "run_mtmc.py").read_text())
    ns = {"np": _NumpyCompatibility(), "copy": copy, "cdist": cdist, "pdist": pdist,
          "squareform": squareform, "linkage": linkage, "fcluster": fcluster,
          "opt": SimpleNamespace(get_feat_mode="best", mtmc_match_thr=MATCH_THRESHOLD,
                                 max_time_differ=retention_frames)}
    cluster = next(node for node in main.body if isinstance(node, ast.ClassDef) and node.name == "Cluster")
    exec(_module([cluster], directory / "run_mtmc.py"), ns)
    utility_path = directory / "utils/utils.py"
    utility = ast.parse(utility_path.read_text())
    function = next(node for node in utility.body if isinstance(node, ast.FunctionDef) and node.name == "pairwise_tracks_dist")
    exec(_module([function], utility_path), ns)
    native_distance = ns["pairwise_tracks_dist"]
    # Missing zone-restricted descriptors are unavailable, not a zero-distance match.
    ns["pairwise_tracks_dist"] = lambda *a, **k: np.nan_to_num(native_distance(*a, **k), nan=1.0, posinf=1.0, neginf=1.0)
    for filename in ("utils/scipy_linear_assignment.py", "utils/sklearn_dunn.py"):
        path = directory / filename
        nodes = [node for node in ast.parse(path.read_text()).body
                 if not (isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names))]
        exec(_module(nodes, path), ns)
    track_path = directory / "tracking/track.py"
    track_cls = next(node for node in ast.parse(track_path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "Track")
    feature = next(node for node in track_cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_feature")
    exec(_module([feature], track_path), ns)
    run = next(node for node in main.body if isinstance(node, ast.FunctionDef) and node.name == "run_mtmc")
    loop = next(node for node in run.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "fdx")
    def assigned(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    begin = next(i for i, node in enumerate(loop.body) if assigned(node, "online_global_ids"))
    end = next(i for i, node in enumerate(loop.body[begin:], begin)
               if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript)
               and isinstance(node.target.value, ast.Name) and node.target.value.id == "total_times")
    block = loop.body[begin:end]
    cluster_begin = next(i for i, node in enumerate(block) if assigned(node, "online_feats"))
    cluster_end = next(i for i, node in enumerate(block) if assigned(node, "num_cluster"))
    return ns, _module(block, directory / "run_mtmc.py"), _module(block[:cluster_begin] + block[cluster_end:], directory / "run_mtmc.py")


class _Track:
    def __init__(self, camera, local_id, dimension, native_feature):
        self.camera, self.cam = camera, f"c{camera + 5:03d}"
        self.local_id, self.global_id = local_id, None
        self.end_frame, self.dimension = -1, dimension
        self.obs_history, self.smooth_feat = [], None
        self.native_feature = native_feature

    def get_feature(self, mode="best"):
        return self.native_feature(self, mode) if self.obs_history else np.zeros(self.dimension)

    def observe(self, frame, box, score, feature):
        feature = np.asarray(feature, dtype=float)
        if feature.shape != (self.dimension,) or not np.isfinite(feature).all():
            raise ValueError("Malformed cached vehicle descriptor")
        norm = np.linalg.norm(feature)
        if norm == 0:
            return False
        feature = feature / norm
        self.smooth_feat = feature.copy() if self.smooth_feat is None else .9 * self.smooth_feat + .1 * feature
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        self.obs_history.append((frame, box, score, feature.copy()))
        return True


def preflight(cfg, cache_path, local_run, retention):
    if any(c not in (1, 2, 3, 4) for c in cfg["cameras"]):
        raise ValueError("Adapter supports at most four cameras using native c006..c009 aliases")
    if retention not in ("native", "development_route"):
        raise ValueError("Unknown retention variant")
    local_run = Path(local_run)
    manifest = json.loads((local_run / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError("A completed signed local run is required")
    for name, expected in manifest["artifact_hashes"].items():
        if file_hash(local_run / name) != expected:
            raise ValueError(f"Changed local run artifact: {name}")
    summary = json.loads((local_run / "summary.json").read_text())
    for name in ("cameras", "start_frame", "stop_frame", "fps"):
        if summary[name] != cfg[name]:
            raise ValueError(f"Local run/config scope mismatch: {name}")
    paths = [local_run / name for name in ("manifest.json", "local_predictions.txt", "decisions.jsonl", "summary.json")]
    local_rows = read_rows(paths[1])
    if any(r["camera"] not in cfg["cameras"] or not cfg["start_frame"] <= r["frame"] < cfg["stop_frame"] for r in local_rows):
        raise ValueError("Local rows outside the declared replay range")
    choices = {}
    for line in paths[2].read_text().splitlines():
        record = json.loads(line)
        key = record["camera"], record["frame"]
        if key in choices or type(record.get("detect")) is not bool:
            raise ValueError("Duplicate/malformed local detector decision")
        choices[key] = record["detect"]
    expected_keys = {(c, f) for c in cfg["cameras"] for f in range(cfg["start_frame"], cfg["stop_frame"])}
    if choices.keys() != expected_keys:
        raise ValueError("Incomplete local detector decision trace")
    frames = 30
    if retention == "development_route":
        if not cfg.get("routes") or not cfg.get("route_calibration"):
            raise ValueError("Development-derived retention requires frozen calibrated routes")
        frames = math.ceil(max(r["travel_time_s"][1] for r in cfg["routes"]) * cfg["fps"]) + cfg["tracker"].get("track_buffer", 30)
    ns, full_block, singleton_block = load_upstream(frames)
    with DetectionCache(cache_path, readonly=True) as cache:
        cache.validate_range(cfg["cameras"], cfg["start_frame"], cfg["stop_frame"], cfg["fps"])
        detector = cache.get_meta("detector")
        if detector.get("descriptor") != "vehicle-reid-0001_512_v1":
            raise ValueError("This adapter requires the shared official 512-dimensional vehicle cache")
        cache_signature = cache.signature(cfg["cameras"], cfg["start_frame"], cfg["stop_frame"])
    options = {"config": cfg, "upstream_commit": COMMIT, "vendor_manifest_sha256": file_hash(UPSTREAM / "manifest.json"),
               "driver_sha256": file_hash(Path(__file__)), "retention_variant": retention, "retention_frames": frames,
               "match_threshold": MATCH_THRESHOLD, "feature_match_iou": FEATURE_MATCH_IOU, "feature_mode": "best",
               "cache_signature": cache_signature, "local_input_hashes": {p.name: file_hash(p) for p in paths},
               "local_policy": summary["policy"], "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn")},
               "adaptations": ["Association-only; shared frozen ByteTrack local stream and official vehicle OSNet cache replace native YOLOv7/BoTSORT/ResNet-IBN.",
                               "RoundaboutHD non-overlap masks are zero; native historical no-revisit gate remains unchanged.",
                               "Features are aligned to emitted local boxes at IoU>=0.5 on actual detector decisions only; missing/zero features cannot match.",
                               "Empty/singleton current sets bypass undefined native hierarchical clustering; legacy NumPy aliases are supplied locally.",
                               "Local boxes/output coverage are retained; native S02 timestamp corrections, annotation-specific filtering and 1.5x box expansion are not applied.",
                               "Frozen scheduling decisions are replayed; global baseline identities do not feed back to scheduling."]}
    return options, local_rows, choices, summary, ns, full_block, singleton_block


def run_baseline(cfg, cache_path, local_run, output, retention="native", preflight_only=False):
    options, rows, choices, source_summary, ns, full_block, singleton_block = preflight(cfg, cache_path, local_run, retention)
    signature = digest(options)
    if preflight_only:
        return {"status": "preflight_passed", "signature": signature, "local_rows": len(rows), "options": options}
    output = Path(output)
    if (output / "manifest.json").is_file():
        existing = json.loads((output / "manifest.json").read_text())
        if existing.get("signature") != signature or existing.get("status") != "complete":
            raise ValueError("Output belongs to another/incomplete baseline; use a new versioned folder")
        for name, expected in existing["artifact_hashes"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"Changed completed baseline artifact: {name}")
        return json.loads((output / "summary.json").read_text())
    if output.exists() and any(output.iterdir()):
        raise ValueError("Baseline output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "manifest.json", {"status": "running", "signature": signature, "options": options})
    by_frame, tracks = defaultdict(list), {}
    for row in rows:
        by_frame[(row["camera"], row["frame"])].append(row)
    ns.update(clusters_dict={}, next_global_id=0)
    aliases = [f"c{c + 5:03d}" for c in cfg["cameras"]]
    ns["overlap_regions"] = {a: {b: _NoOverlap() for b in aliases} for a in aliases}
    matched_features = detection_exposures = missing_features = 0
    messages = []
    begin = time.perf_counter()
    with DetectionCache(cache_path, readonly=True) as cache, (output / "predictions.txt").open("w") as predictions, (output / "availability.jsonl").open("w") as availability:
        for frame in range(cfg["start_frame"], cfg["stop_frame"]):
            active, row_tracks = [], []
            for camera in sorted(cfg["cameras"]):
                current = by_frame[(camera, frame)]
                current_tracks = []
                for row in current:
                    key = camera, row["id"]
                    track = tracks.setdefault(key, _Track(camera, row["id"], 512, ns["get_feature"]))
                    track.end_frame, track.x1y1x2y2 = frame, np.asarray(row["box"])
                    current_tracks.append(track)
                    row_tracks.append((row, track))
                if choices[(camera, frame)]:
                    detection_exposures += 1
                    detections, _, _ = cache.frame(camera, frame)
                    if current and detections:
                        ious = iou_matrix([r["box"] for r in current], [d["box"] for d in detections])
                        a, b = linear_sum_assignment(np.where(ious >= FEATURE_MATCH_IOU, 1 - ious, 1e6))
                        for x, y in zip(a, b):
                            if ious[x, y] >= FEATURE_MATCH_IOU:
                                matched_features += current_tracks[x].observe(frame, detections[y]["box"], detections[y]["score"], detections[y]["feature"])
                active.extend(current_tracks)
            unassigned = [t for t in active if t.global_id is None]
            previous_next = ns["next_global_id"]
            ns.update(online_tracks=active, fdx=frame)
            if len(active) < 2:
                ns["cluster"] = np.arange(len(active))
                exec(singleton_block, ns)
            else:
                exec(full_block, ns)
            for track in unassigned:
                reason = "upstream_historical_match" if track.global_id < previous_next else "upstream_new_identity"
                messages.append({"event": "identity_committed", "time_s": frame / cfg["fps"], "camera": track.camera,
                                 "local_id": track.local_id, "global_id": track.global_id + 1, "reason": reason})
            for row, track in row_tracks:
                if track.global_id is None:
                    raise RuntimeError("Native association left a local track without a global ID")
                missing_features += int(not track.obs_history)
                x1, y1, x2, y2 = row["box"]
                predictions.write(f"{row['camera']} {track.global_id + 1} {frame} {x1:.6f} {y1:.6f} {x2-x1:.6f} {y2-y1:.6f} -1 -1\n")
                availability.write(canonical({"camera": row["camera"], "local_id": row["id"], "global_id": track.global_id + 1,
                                               "frame": frame, "published_at_s": frame / cfg["fps"], "published_at_frame": frame,
                                               "reason": "published_online"}) + "\n")
    for name in ("local_predictions.txt", "decisions.jsonl"):
        shutil.copyfile(Path(local_run) / name, output / name)
    with (output / "messages.jsonl").open("w") as stream:
        for event in messages:
            stream.write(canonical(event) + "\n")
    summary = {"method": "Fast_Online_MTMCT association-only CPU adaptation", "policy": source_summary["policy"],
               "cameras": cfg["cameras"], "start_frame": cfg["start_frame"], "stop_frame": cfg["stop_frame"], "fps": cfg["fps"],
               "prediction_rows": len(rows), "detector_calls": detection_exposures,
               "calls_by_camera": source_summary["calls_by_camera"], "budget_fraction": source_summary["budget_fraction"],
               "opportunities": source_summary["opportunities"], "retention_variant": retention,
               "retention_frames": options["retention_frames"], "match_threshold": MATCH_THRESHOLD,
               "actual_descriptor_observations_aligned": matched_features, "prediction_rows_without_descriptor_history": missing_features,
               "identity_commits": len(messages), "association_replay_elapsed_s": time.perf_counter() - begin, "signature": signature,
               "runtime_note": "Frozen local/decision stream replay; excludes detector, descriptor inference, original SCT and decoding. This is not full published-pipeline speed.",
               "claim_scope": "Adapted published association comparison on shared perception; no published CityFlow score reproduction or unseen-scene claim."}
    if detection_exposures != source_summary["detector_calls"]:
        raise ValueError("Exposure counts differ from the frozen local decision trace")
    read_rows(output / "predictions.txt")  # Reject any native same-camera duplicate global ID.
    write_json(output / "summary.json", summary)
    names = ("predictions.txt", "availability.jsonl", "local_predictions.txt", "decisions.jsonl", "messages.jsonl", "summary.json")
    write_json(output / "manifest.json", {"status": "complete", "signature": signature, "options": options,
                                          "artifact_hashes": {name: file_hash(output / name) for name in names}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--local-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retention", choices=("native", "development_route"), default="native")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = run_baseline(load_config(args.config), args.cache, args.local_run, args.output, args.retention, args.preflight)
    print(canonical(result), flush=True)


if __name__ == "__main__":
    main()
