"""Pinned published association adapted to selected S05 native local streams.

This is an association-only adaptation, not the published CUDA pipeline or a
reproduction of its CityFlow score. Ground truth is never accepted or opened.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from cityflow.replay import local_frame_range, native_time_us
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, iou_matrix, write_json
from rhd.evaluation import read_rows
from run_published_association_baseline_v0_4 import (
    COMMIT, FEATURE_MATCH_IOU, MATCH_THRESHOLD, UPSTREAM, _NoOverlap,
    load_upstream, verify_upstream)

ARTIFACTS = ("predictions.txt", "availability.jsonl", "local_predictions.txt",
             "decisions.jsonl", "messages.jsonl", "summary.json")
CAMERA_ALIASES = {29: 6, 33: 7, 34: 8, 35: 9}


def config_from_artifacts(selection, plan, scene):
    """Construct the disclosed S05 scope without accepting an annotation input.

    Model/tracker/association options are frozen on S02 development. The S05
    plan supplies target-prefix route calibration and fixed replay intervals.
    Native published threshold stays .65 independently of selected association.
    """
    if selection.get('status') != 'frozen_after_development':
        raise ValueError('Require frozen S02 development selection before S05 replay')
    development = selection.get('development', {})
    if development.get('scenario') != 'S02' or development.get('interval_s') != [0.,60.]:
        raise ValueError('Selection must use only S02 development [0,60)')
    selected = selection.get('config')
    required = {'tracker','association','communication','output_filter','seed','fps','perception'}
    if not isinstance(selected, dict) or not required <= set(selected):
        raise ValueError('Frozen selection config is incomplete')
    if plan.get('scenario') != 'S05' or scene.get('scenario') != 'S05':
        raise ValueError('The v0.6 association adapter supports S05 only')
    if plan.get('split') != 'validation' or scene.get('split') != 'validation':
        raise ValueError('S05 plan/scene must use the official validation split')
    if sorted(plan.get('cameras', [])) != sorted(CAMERA_ALIASES):
        raise ValueError('Frozen four-camera S05 plan differs from adapter inventory')
    if not set(map(str,CAMERA_ALIASES)) <= set(scene.get('cameras', {})):
        raise ValueError('Selected scene is missing a planned S05 camera')
    calibration = plan.get('calibration_interval_s')
    if (not isinstance(calibration, list) or len(calibration) != 2 or calibration[0] != 0 or
            not 0 < calibration[1] <= plan.get('replay_start_s', -1)):
        raise ValueError('Prefix calibration must finish before holdout replay')
    cfg = {'version':'cityflow_published_association_v0_6', **selected,
           'dataset': scene['dataset_root'], 'split':'validation', 'scenario':'S05',
           'cameras': sorted(CAMERA_ALIASES), 'routes': plan['routes'],
           'replay_start_s':plan['replay_start_s'], 'score_start_s':plan['score_start_s'],
           'stop_s':plan['stop_s'], 'scene_signature':digest(scene),
           'route_calibration':{'start_s':calibration[0], 'stop_s':calibration[1],
                                'origin':'S05 prefix-only frozen holdout plan'},
           'scope':'Adapted published association with shared frozen perception; not the published full pipeline.'}
    if not 0 <= cfg['replay_start_s'] <= cfg['score_start_s'] < cfg['stop_s']:
        raise ValueError('Invalid frozen holdout replay/score interval')
    for route in cfg['routes']:
        if route.get('source') not in CAMERA_ALIASES or route.get('destination') not in CAMERA_ALIASES:
            raise ValueError('Plan route lies outside selected S05 cameras')
        lo, hi = route['travel_time_s']
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo < hi:
            raise ValueError('Invalid prefix-calibrated travel window')
    return cfg


class Track:
    def __init__(self, camera, local_id, native_feature):
        if camera not in CAMERA_ALIASES:
            raise ValueError('Unknown original S05 camera')
        self.camera, self.cam = camera, f"c{CAMERA_ALIASES[camera]:03d}"
        self.local_id, self.global_id = local_id, None
        self.end_frame = -1
        self.obs_history, self.smooth_feat = [], None
        self.native_feature = native_feature

    def get_feature(self, mode="best"):
        return self.native_feature(self, mode) if self.obs_history else np.zeros(512)

    def observe(self, event_us, box, score, feature):
        value = np.asarray(feature, dtype=float)
        if value.shape != (512,) or not np.isfinite(value).all():
            raise ValueError("Malformed shared vehicle descriptor")
        norm = np.linalg.norm(value)
        if not norm:
            return False
        value = value / norm
        self.smooth_feat = value.copy() if self.smooth_feat is None else .9 * self.smooth_feat + .1 * value
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)
        self.obs_history.append((event_us, np.asarray(box), float(score), value.copy()))
        return True


def preflight(cfg, cache_path, local_run, retention):
    cameras = sorted(cfg["cameras"])
    if cfg.get("scenario") != "S05" or cameras != sorted(CAMERA_ALIASES):
        raise ValueError("This adapter supports original S05 cameras29,33,34,35")
    if retention not in ("native", "development_route"):
        raise ValueError("Unknown retention variant")
    local_run = Path(local_run)
    manifest = json.loads((local_run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Require a completed signed shared local run")
    if digest(manifest.get('options')) != manifest.get('signature'):
        raise ValueError('Shared local manifest options/signature changed')
    local_config = manifest['options'].get('config', {})
    if local_config.get('tracker') != cfg['tracker'] or local_config.get('output_filter') != cfg.get('output_filter'):
        raise ValueError('Shared local tracker/output filter differs from frozen selection')
    if local_config.get('tracker_box_mode','kalman') != cfg.get('tracker_box_mode','kalman'):
        raise ValueError('Shared local observation-box mode differs from frozen selection')
    for required in ('local_predictions.txt', 'decisions.jsonl', 'summary.json'):
        if required not in manifest.get('artifact_hashes', {}):
            raise ValueError('Shared local manifest misses a required artifact')
    for name, expected in manifest["artifact_hashes"].items():
        if file_hash(local_run / name) != expected:
            raise ValueError(f"Shared local artifact changed: {name}")
    summary = json.loads((local_run / "summary.json").read_text(encoding="utf-8"))
    for name in ("cameras", "scenario", "split", "replay_start_s", "score_start_s", "stop_s"):
        if summary[name] != cfg[name]:
            raise ValueError(f"Shared local/config scope mismatch: {name}")
    if summary.get("policy") != "full" or summary.get("detector_call_fraction") != 1:
        raise ValueError("CityFlow v0.5 baseline requires the shared full native detection stream")
    rows = read_rows(local_run / "local_predictions.txt")
    if any(row["camera"] not in cameras or row["frame"] < 1 for row in rows):
        raise ValueError("Local output requires original cameras and one-based native frames")
    choices, by_time = {}, defaultdict(list)
    for line in (local_run / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        camera, frame = record["camera"], record["frame"]
        if type(camera) is not int or type(frame) is not int or record.get("detect") is not True:
            raise ValueError("Malformed shared native detection decision")
        key = camera, frame
        if key in choices:
            raise ValueError("Duplicate shared native decision")
        if not math.isfinite(record["time_s"]):
            raise ValueError("Non-finite native observation time")
        choices[key] = record
        by_time[round(record["time_s"] * 1_000_000)].append(record)
    metadata, ranges = {}, {}
    with DetectionCache(cache_path, readonly=True) as cache:
        scene = cache.get_meta('scene')
        if (not scene or scene.get('scenario') != 'S05' or scene.get('split') != cfg['split'] or
                digest(scene) != cfg.get('scene_signature') or
                Path(scene['dataset_root']).resolve() != Path(cfg['dataset']).resolve()):
            raise ValueError('Cache scene differs from the frozen selected S05 scene')
        detector = cache.get_meta("detector")
        if not detector or detector.get("descriptor") != "vehicle-reid-0001_512_v1":
            raise ValueError("Require the shared 512-dimensional official vehicle ReID cache")
        for camera in cameras:
            meta = cache.get_meta(f"camera:{camera}")
            if meta != summary["native_camera_metadata"][str(camera)]:
                raise ValueError("Cache/local native metadata mismatch")
            if meta != scene['cameras'][str(camera)]:
                raise ValueError('Cache camera metadata differs from its signed scene')
            bounds = local_frame_range(meta, cfg["replay_start_s"], cfg["stop_s"])
            metadata[camera], ranges[camera] = meta, bounds
            for local_frame in range(*bounds):
                decision = choices.get((camera, local_frame + 1))
                if decision is None or round(decision["time_s"] * 1_000_000) != native_time_us(meta, local_frame):
                    raise ValueError("Shared native clock/coverage mismatch")
            found = cache.db.execute("SELECT COUNT(*) FROM frames WHERE camera=? AND frame>=? AND frame<?",
                                     (camera, *bounds)).fetchone()[0]
            if found != bounds[1] - bounds[0]:
                raise ValueError("Incomplete native detection cache")
        expected_keys = {(camera, local_frame + 1) for camera, bounds in ranges.items()
                         for local_frame in range(*bounds)}
        if set(choices) != expected_keys or any((r["camera"], r["frame"]) not in choices for r in rows):
            raise ValueError("Local rows/decisions exceed declared native replay scope")
        cache_signature = digest({str(camera): cache.signature([camera], *bounds)
                                  for camera, bounds in ranges.items()})
    if cfg["fps"] != 10 or any(meta["fps"] != 10 for meta in metadata.values()):
        raise ValueError("Native 30-reference-frame retention requires declared 10 FPS S05")
    # A video can end while its last emitted track is still active. Retire that
    # camera at the exact coverage boundary, without inventing a native frame
    # or waiting for another camera's next observation.
    replay_start_us = round(cfg["replay_start_s"] * 1_000_000)
    replay_stop_us = round(cfg["stop_s"] * 1_000_000)
    for meta in metadata.values():
        coverage_end_us = native_time_us(meta, meta["frames"])
        if replay_start_us <= coverage_end_us < replay_stop_us:
            by_time.setdefault(coverage_end_us, [])
    retention_s = 30 / cfg["fps"]
    if retention == "development_route":
        if not cfg.get("routes") or not cfg.get("route_calibration"):
            raise ValueError("Derived retention requires frozen development-only routes")
        retention_s = max(route["travel_time_s"][1] for route in cfg["routes"]) + max(
            int(meta["fps"] / 30. * cfg["tracker"].get("track_buffer",30)) / meta["fps"]
            for meta in metadata.values())
    retention_us = math.ceil(retention_s * 1_000_000)
    ns, full_block, singleton_block = load_upstream(retention_us)
    paths = [local_run / n for n in ("manifest.json", "local_predictions.txt", "decisions.jsonl", "summary.json")]
    cache_sha256 = file_hash(cache_path)
    if manifest['options'].get('cache_sha256') != cache_sha256:
        raise ValueError('Baseline cache differs from the shared local replay cache')
    options = {"config": cfg, "upstream_commit": COMMIT,
        "vendor_manifest_sha256": file_hash(UPSTREAM / "manifest.json"),
        "driver_sha256": file_hash(Path(__file__)),
        "upstream_loader_sha256": file_hash(Path(__file__).with_name("run_published_association_baseline_v0_4.py")),
        "native_clock_helper_sha256": file_hash(Path(__file__).parents[1] / "src/cityflow/replay.py"),
        "retention_variant": retention, "retention_s": retention_us / 1_000_000,
        "retention_microseconds": retention_us, "match_threshold": MATCH_THRESHOLD,
        "feature_match_iou": FEATURE_MATCH_IOU, "feature_mode": "best",
        "cache_signature": cache_signature, "cache_sha256": cache_sha256,
        "original_to_native_alias": {str(k):v for k,v in CAMERA_ALIASES.items()},
        "shared_local_source_signature": manifest['options'].get('source_signature'),
        "local_input_hashes": {p.name: file_hash(p) for p in paths},
        "packages": {n: importlib.metadata.version(n) for n in ("numpy", "scipy", "scikit-learn")},
        "adaptations": [
            "Association-only: identical frozen ByteTrack local stream and 512-dimensional vehicle OSNet cache replace native YOLOv7/BoTSORT/ResNet-IBN.",
            "Original S05 cameras29,33,34,35 map internally to native c006,c007,c008,c009. Every exported camera/frame remains original. Alias assignment is inventory order, not a physical overlap/topology equivalence.",
            "Dataset-specific native frame repairs and horizontal image flips are not applied to S05; supplied timestamp offsets and native one-based observation frames remain unchanged.",
            "Ordered-pair overlap masks are explicitly zero for the non-overlapping handoff study; simultaneous cross-camera clustering is disabled. This is not an overlap-aware reproduction.",
            "Native frame subtraction and retention operate in common-clock integer microseconds; 30 reference frames at 10 FPS equal 3 seconds.",
            "Each observation event updates only observed cameras; last active sets of other cameras persist to their next native observation or exact video-coverage-end event. All currently active tracks get current end_frame, preserving native active-cluster exclusion.",
            "Actual descriptor observations are aligned at IoU>=0.5 once per camera/native frame; latched active tracks receive no new appearance observations.",
            "Native best-feature selection, .65 threshold, clustering and historical Hungarian statements and no-camera-revisit gate are preserved.",
            "The optional development_route variant changes only historical retention using frozen target-prefix route upper bounds plus local buffer. It adds no route gate and is distinct from native 3-second retention.",
            "Missing descriptor distances are unavailable; empty/singleton clustering guards and local legacy NumPy aliases retain the earlier documented CPU adaptation.",
            "Identical float local boxes and row coverage are retained; native annotation-specific size/border/NMS filtering and 1.5x output expansion are omitted."]}
    return options, rows, choices, by_time, metadata, summary, ns, full_block, singleton_block


def run_baseline(cfg, cache_path, local_run, output, retention="native", preflight_only=False,
                 *, input_artifact_paths=None):
    values = preflight(cfg, cache_path, local_run, retention)
    options, rows, choices, by_time, metadata, source_summary, ns, full_block, singleton_block = values
    if input_artifact_paths is not None:
        if len({Path(path).name for path in input_artifact_paths}) != len(input_artifact_paths):
            raise ValueError('Configuration artifact filenames must be distinct')
        options['configuration_artifact_sha256'] = {Path(path).name:file_hash(Path(path))
                                                  for path in input_artifact_paths}
    signature = digest(options)
    if preflight_only:
        return {"status": "preflight_passed", "signature": signature, "local_rows": len(rows), "options": options}
    output, local_run = Path(output), Path(local_run)
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (existing.get("status") != "complete" or existing.get("signature") != signature or
                digest(existing.get('options')) != existing.get('signature')):
            raise ValueError("Baseline output belongs to another/incomplete experiment")
        if set(existing.get("artifact_hashes", {})) != set(ARTIFACTS):
            raise ValueError("Baseline manifest artifact inventory differs")
        for name, expected in existing["artifact_hashes"].items():
            if file_hash(output / name) != expected:
                raise ValueError(f"Changed completed baseline artifact: {name}")
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty versioned baseline output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {"status": "running", "signature": signature, "options": options})
    by_frame, tracks = defaultdict(list), {}
    for row in rows:
        by_frame[(row["camera"], row["frame"])].append(row)
    cameras = sorted(cfg["cameras"])
    aliases = [f"c{CAMERA_ALIASES[c]:03d}" for c in cameras]
    ns.update(clusters_dict={}, next_global_id=0,
              overlap_regions={a: {b: _NoOverlap() for b in aliases} for a in aliases})
    active_by_camera = {c: [] for c in cameras}
    matched_features = missing_features = calls = emitted_rows = 0
    commits = []
    begin = time.perf_counter()
    with DetectionCache(cache_path, readonly=True) as cache, \
            (output / "predictions.txt").open("w", encoding="utf-8") as predictions, \
            (output / "availability.jsonl").open("w", encoding="utf-8") as availability:
        for event_us, decisions in sorted(by_time.items()):
            now = event_us / 1_000_000
            observed_rows = []
            for decision in sorted(decisions, key=lambda r: r["camera"]):
                camera, frame = decision["camera"], decision["frame"]
                current_rows = by_frame[(camera, frame)]
                current_tracks = []
                for row in current_rows:
                    key = camera, row["id"]
                    track = tracks.setdefault(key, Track(camera, row["id"], ns["get_feature"]))
                    track.x1y1x2y2 = np.asarray(row["box"])
                    current_tracks.append(track)
                    observed_rows.append((row, track))
                detections, _, _ = cache.frame(camera, frame - 1)
                calls += 1
                if current_rows and detections:
                    ious = iou_matrix([row["box"] for row in current_rows], [d["box"] for d in detections])
                    a, b = linear_sum_assignment(np.where(ious >= FEATURE_MATCH_IOU, 1 - ious, 1e6))
                    for x, y in zip(a, b):
                        if ious[x, y] >= FEATURE_MATCH_IOU:
                            detection = detections[y]
                            matched_features += current_tracks[x].observe(event_us, detection["box"],
                                                                          detection["score"], detection["feature"])
                active_by_camera[camera] = current_tracks
            # End real camera coverage without inventing empty native frames.
            for camera in cameras:
                coverage_end = native_time_us(metadata[camera], metadata[camera]["frames"])
                if coverage_end <= event_us:
                    active_by_camera[camera] = []
            active = [track for camera in cameras for track in active_by_camera[camera]]
            for track in active:
                track.end_frame = event_us
            unassigned = [track for track in active if track.global_id is None]
            previous_next = ns["next_global_id"]
            ns.update(online_tracks=active, fdx=event_us)
            if len(active) < 2:
                ns["cluster"] = np.arange(len(active))
                exec(singleton_block, ns)
            else:
                exec(full_block, ns)
            for track in unassigned:
                reason = "upstream_historical_match" if track.global_id < previous_next else "upstream_new_identity"
                commits.append({"event": "identity_committed", "time_s": now, "camera": track.camera,
                                "local_id": track.local_id, "global_id": track.global_id + 1, "reason": reason})
            for row, track in observed_rows:
                if track.global_id is None:
                    raise RuntimeError("Published association left an unassigned local track")
                missing_features += int(not track.obs_history)
                x1, y1, x2, y2 = row["box"]
                predictions.write(f"{row['camera']} {track.global_id + 1} {row['frame']} {x1:.6f} {y1:.6f} {x2-x1:.6f} {y2-y1:.6f} -1 -1\n")
                availability.write(canonical({"camera": row["camera"], "local_id": row["id"],
                    "global_id": track.global_id + 1, "frame": row["frame"],
                    "local_frame_zero_based": row["frame"] - 1, "time_s": now,
                    "published_at_s": now, "reason": "published_online"}) + "\n")
                emitted_rows += 1
    if calls != source_summary["detector_calls"] or emitted_rows != len(rows):
        raise ValueError("Baseline changed the shared detection calls or output coverage")
    read_rows(output / "predictions.txt")  # Reject native same-camera global-ID collisions.
    for name in ("local_predictions.txt", "decisions.jsonl"):
        shutil.copyfile(local_run / name, output / name)
    with (output / "messages.jsonl").open("w", encoding="utf-8") as stream:
        for event in commits:
            stream.write(canonical(event) + "\n")
    summary = {"method": "Fast_Online_MTMCT association-only CPU adaptation", "policy": "full",
        "scenario": cfg["scenario"], "split": cfg["split"], "cameras": cameras,
        "reference_fps": cfg["fps"], "replay_start_s": cfg["replay_start_s"],
        "score_start_s": cfg["score_start_s"], "stop_s": cfg["stop_s"],
        "prediction_rows": emitted_rows, "detector_calls": calls,
        "opportunities": source_summary["opportunities"], "calls_by_camera": source_summary["calls_by_camera"],
        "actual_descriptor_observations_aligned": matched_features,
        "prediction_rows_without_descriptor_history": missing_features,
        "identity_commits": len(commits), "retention_variant": retention,
        "retention_s": options["retention_s"], "match_threshold": MATCH_THRESHOLD,
        "original_to_native_alias": options['original_to_native_alias'],
        "native_camera_metadata": source_summary['native_camera_metadata'],
        "detector_call_fraction": 1.,
        "association_replay_elapsed_s": time.perf_counter() - begin,
        "runtime_note": "Shared cached local/detection replay; decoding, detector, descriptor inference and local tracking excluded. No full published-pipeline runtime claim.",
        "claim_scope": "Published association adaptation with zero overlap masks for a non-overlap handoff study. Not full published-method reproduction or an official leaderboard comparison.",
        "signature": signature}
    write_json(output / "summary.json", summary)
    write_json(manifest_path, {"status": "complete", "signature": signature, "options": options,
                              "artifact_hashes": {name: file_hash(output / name) for name in ARTIFACTS}})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "plan", "scene", "cache", "local-run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--retention", choices=("native", "development_route"), default="native")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    scene = json.loads(args.scene.read_text(encoding='utf-8'))
    cfg = config_from_artifacts(selection, plan, scene)
    print(canonical(run_baseline(cfg, args.cache, args.local_run, args.output,
                                args.retention, args.preflight,
                                input_artifact_paths=(args.selection,args.plan,args.scene))), flush=True)


if __name__ == "__main__":
    main()
