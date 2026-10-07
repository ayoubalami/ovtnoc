"""Causal native-frame replay for cameras with different starts and frame rates.

The RoundaboutHD coordinator and local tracker are reused unchanged. This module
owns the clock: camera observations, message deliveries, and confirmation
deadlines are separate events. It never resamples a camera or reads identities.
"""
from __future__ import annotations

import heapq
import importlib.metadata
import json
import math
import platform
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from rhd.bounded import BoundedCoordinator, validate_options
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, write_json
from rhd.vision import CameraTracker


MICROSECONDS = 1_000_000
ARTIFACTS = ("predictions.txt", "local_predictions.txt", "availability.jsonl",
             "decisions.jsonl", "messages.jsonl", "timeline.jsonl", "summary.json")


def native_time_us(metadata: dict, local_frame: int) -> int:
    """Nearest microsecond of offset + zero-based native frame / native FPS.

    Fraction arithmetic prevents two algebraically equal camera timestamps from
    landing on opposite sides of a rounding boundary because of float error.
    """
    timestamp = (Fraction(str(metadata["timestamp_offset_s"]))
                 + Fraction(local_frame) / Fraction(str(metadata["fps"])))
    return round(timestamp * MICROSECONDS)


def _seconds_us(value: float) -> int:
    return round(Fraction(str(value)) * MICROSECONDS)


def _scheduled_us(value: float) -> int:
    # An internal timer is processed at or after its scheduled time, never early.
    return math.ceil(value * MICROSECONDS - 1e-7)


def local_frame_range(metadata: dict, start_s: float, stop_s: float) -> tuple[int, int]:
    """Native zero-based indices whose rounded timestamps lie in [start, stop)."""
    def lower_bound(target):
        left, right = 0, metadata["frames"]
        while left < right:
            middle = (left + right) // 2
            if native_time_us(metadata, middle) < target:
                left = middle + 1
            else:
                right = middle
        return left

    return lower_bound(_seconds_us(start_s)), lower_bound(_seconds_us(stop_s))


def source_signature() -> str:
    source = Path(__file__).resolve().parents[1]
    return digest({str(path.relative_to(source)): file_hash(path)
                   for package in ("rhd", "cityflow")
                   for path in sorted((source / package).glob("*.py"))})


def _validate_config(cfg: dict):
    cameras = cfg.get("cameras", [])
    if not cameras or any(type(c) is not int or c < 1 for c in cameras) or len(set(cameras)) != len(cameras):
        raise ValueError("cameras must be distinct positive integer IDs")
    for name in ("dataset", "split", "scenario", "tracker", "association", "communication", "routes", "seed"):
        if name not in cfg:
            raise ValueError(f"Missing CityFlow replay option: {name}")
    if not math.isfinite(cfg.get("fps", 0)) or cfg["fps"] <= 0:
        raise ValueError("fps must be a positive reference clock rate")
    for name in ("replay_start_s", "score_start_s", "stop_s"):
        if not math.isfinite(cfg.get(name, float("nan"))):
            raise ValueError(f"{name} must be finite")
    if not 0 <= cfg["replay_start_s"] <= cfg["score_start_s"] < cfg["stop_s"]:
        raise ValueError("Need 0 <= replay_start_s <= score_start_s < stop_s")
    if _seconds_us(cfg["stop_s"]) <= _seconds_us(cfg["replay_start_s"]):
        raise ValueError("Replay interval is empty at microsecond clock precision")
    if cfg["association"].get("kind") != "bounded":
        raise ValueError("CityFlow v0.5 uses the common bounded kernel, including K=0")
    validate_options(cfg["association"])
    delay = cfg["communication"].get("delay_s", 0.)
    loss = cfg["communication"].get("loss_probability", 0.)
    if not math.isfinite(delay) or delay < 0 or not math.isfinite(loss) or not 0 <= loss <= 1:
        raise ValueError("Invalid communication delay or loss probability")
    for route in cfg["routes"]:
        if route["source"] not in cameras or route["destination"] not in cameras:
            raise ValueError("Configured route uses a camera outside this replay")


def _validate_cache(cache: DetectionCache, cfg: dict):
    scene = cache.get_meta("scene")
    if not scene or scene.get("split") != cfg["split"] or scene.get("scenario") != cfg["scenario"]:
        raise ValueError("Cache scene split/scenario does not match the experiment")
    dataset = Path(cfg["dataset"]).resolve()
    if Path(scene.get("dataset_root", "")).resolve() != dataset:
        raise ValueError("Cache dataset root does not match the experiment")
    scene_cameras = scene.get("cameras", {})
    if not set(cfg["cameras"]).issubset(map(int, scene_cameras)):
        raise ValueError("Cache scene does not contain all configured cameras")
    if "scene_signature" in cfg and digest(scene) != cfg["scene_signature"]:
        raise ValueError("Cache scene signature does not match the frozen protocol")
    if scene.get("reference_fps", cfg["fps"]) != cfg["fps"]:
        raise ValueError("Cache reference clock does not match the experiment")
    detector = cache.get_meta("detector")
    if not detector or not isinstance(detector.get("kind"), str):
        raise ValueError("Cache detector provenance is missing")
    metadata, ranges = {}, {}
    for camera in sorted(cfg["cameras"]):
        meta = cache.get_meta(f"camera:{camera}")
        if not meta:
            raise ValueError(f"Missing camera metadata for {camera}")
        if isinstance(scene_cameras, dict):
            scene_camera = scene_cameras.get(str(camera), scene_cameras.get(camera))
            if not isinstance(scene_camera, dict) or digest(meta) != digest(scene_camera):
                raise ValueError(f"Camera {camera} cache metadata disagrees with its signed scene metadata")
        for name in ("width", "height", "frames"):
            if type(meta.get(name)) is not int or meta[name] <= 0:
                raise ValueError(f"Invalid camera {camera} metadata: {name}")
        for name in ("fps", "timestamp_offset_s"):
            if not isinstance(meta.get(name), (float, int)) or not math.isfinite(meta[name]):
                raise ValueError(f"Invalid camera {camera} metadata: {name}")
        if meta["fps"] <= 0 or native_time_us(meta, 1) <= native_time_us(meta, 0):
            raise ValueError(f"Invalid or too fine camera {camera} native frame rate")
        if "camera_metadata" in cfg:
            expected = cfg["camera_metadata"].get(str(camera), cfg["camera_metadata"].get(camera))
            if expected is None or any(expected.get(k) != meta[k]
                                       for k in ("width", "height", "frames", "fps", "timestamp_offset_s")):
                raise ValueError(f"Camera {camera} metadata does not match the frozen protocol")
        start, stop = local_frame_range(meta, cfg["replay_start_s"], cfg["stop_s"])
        found = cache.db.execute("SELECT COUNT(*) FROM frames WHERE camera=? AND frame>=? AND frame<?",
                                 (camera, start, stop)).fetchone()[0]
        if found != stop - start:
            raise ValueError(f"Camera {camera}: missing native cache frames in [{start},{stop}); found {found}")
        metadata[camera], ranges[camera] = meta, (start, stop)
    if not sum(stop - start for start, stop in ranges.values()):
        raise ValueError("No real camera observations in the selected interval")
    return metadata, ranges


def _cache_signature(cache: DetectionCache, ranges: dict) -> str:
    return digest({str(camera): cache.signature([camera], start, stop)
                   for camera, (start, stop) in sorted(ranges.items())})


def run(cfg: dict, cache_path: Path, output: Path) -> dict:
    """Run full-detection native observations with exact delivery/deadline events.

    Cache local indices are zero based. Both prediction files use the original
    CityFlow camera IDs and original one-based local video frame numbers. Score
    boundaries are metadata only: inference never opens a ground-truth file.
    """
    _validate_config(cfg)
    cache_path, output = Path(cache_path), Path(output)
    cameras = sorted(cfg["cameras"])
    with DetectionCache(cache_path, readonly=True) as cache:
        metadata, ranges = _validate_cache(cache, cfg)
        options = {"config": cfg, "policy": "full", "source_signature": source_signature(),
                   "cache_sha256": file_hash(cache_path), "cache_signature": _cache_signature(cache, ranges),
                   "native_frame_ranges": {str(c): list(r) for c, r in ranges.items()},
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("ultralytics", "lap", "numpy", "scipy")},
                   "clock": "nearest microsecond of offset + zero-based native frame / native FPS"}
        signature = digest(options)
        manifest = output / "manifest.json"
        if manifest.is_file():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous.get("signature") != signature:
                raise ValueError("Existing CityFlow run belongs to a different signed experiment")
            if digest(previous.get("options")) != previous["signature"]:
                raise ValueError("Completed CityFlow manifest options changed")
            if previous.get("status") != "complete":
                raise ValueError("Incomplete CityFlow run; use a new attempt directory")
            if set(previous.get("artifact_hashes", {})) != set(ARTIFACTS):
                raise ValueError("Completed CityFlow manifest has an invalid artifact set")
            for name, expected in previous["artifact_hashes"].items():
                if not (output / name).is_file() or file_hash(output / name) != expected:
                    raise ValueError(f"Completed CityFlow artifact changed or missing: {name}")
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))
        if output.exists() and any(output.iterdir()):
            raise ValueError("CityFlow output directory must be empty")
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest, {"status": "running", "signature": signature, "options": options})
        coordinator = BoundedCoordinator(cfg)
        trackers = {camera: CameraTracker(metadata[camera]["fps"], metadata[camera]["width"],
                                           metadata[camera]["height"], cfg["tracker"],
                                           cfg.get("output_filter")) for camera in cameras}
        video_events = []
        coverage_events = []
        stop_us = _seconds_us(cfg["stop_s"])
        for camera, (start, stop) in ranges.items():
            if start < stop:
                heapq.heappush(video_events, (native_time_us(metadata[camera], start), camera, start))
                coverage_end = native_time_us(metadata[camera], metadata[camera]["frames"])
                if coverage_end < stop_us:
                    heapq.heappush(coverage_events, (coverage_end, camera))
        buffered = {}
        calls = Counter({camera: 0 for camera in cameras})
        events = Counter()
        detector_ms = appearance_ms = 0.
        processing_ms = []
        output_rows = pending_at_end = terminal_future_commits = 0
        begin_run = time.perf_counter()
        with (output / "predictions.txt").open("w", encoding="utf-8") as predictions, \
                (output / "local_predictions.txt").open("w", encoding="utf-8") as local_predictions, \
                (output / "decisions.jsonl").open("w", encoding="utf-8") as decisions, \
                (output / "availability.jsonl").open("w", encoding="utf-8") as availability, \
                (output / "timeline.jsonl").open("w", encoding="utf-8") as timeline:
            def row_text(camera, identity, local_frame, box):
                x1, y1, x2, y2 = box
                return f"{camera} {identity} {local_frame+1} {x1:.6f} {y1:.6f} {x2-x1:.6f} {y2-y1:.6f} -1 -1\n"

            def emit(key, local_frame, box, observation_time, publication_time):
                nonlocal output_rows
                gid = coordinator.mapping[key]
                predictions.write(row_text(key[0], gid, local_frame, box))
                availability.write(canonical({"camera": key[0], "local_id": key[1], "global_id": gid,
                                              "frame": local_frame + 1, "local_frame_zero_based": local_frame,
                                              "time_s": observation_time,
                                              "published_at_s": max(observation_time, publication_time),
                                              "reason": coordinator.commits[key]["reason"]}) + "\n")
                output_rows += 1

            def emit_ready(now, terminal=False):
                for key in sorted(list(buffered)):
                    if key in coordinator.mapping:
                        committed = coordinator.commits[key]["time_s"]
                        published = max(now, committed) if terminal else now
                        for frame, box, observed in buffered.pop(key):
                            emit(key, frame, box, observed, published)

            def next_internal():
                due = [_scheduled_us(state["deadline"]) for key, state in coordinator.states.items()
                       if key not in coordinator.mapping]
                if coordinator.pending:
                    due.append(_scheduled_us(coordinator.pending[0][0]))
                return min(due, default=stop_us)

            while True:
                next_us = min(video_events[0][0] if video_events else stop_us,
                              coverage_events[0][0] if coverage_events else stop_us, next_internal())
                if next_us >= stop_us:
                    break
                begin_event = time.perf_counter()
                now = next_us / MICROSECONDS
                before_pending = len(coordinator.pending)
                coordinator.deliver(now)
                delivered = before_pending - len(coordinator.pending)
                batch, ended = {}, []
                while coverage_events and coverage_events[0][0] == next_us:
                    _, camera = heapq.heappop(coverage_events)
                    coordinator.set_active(camera, [])
                    ended.append(camera)
                    events["camera_coverage_end"] += 1
                while video_events and video_events[0][0] == next_us:
                    _, camera, frame = heapq.heappop(video_events)
                    detections, recorded_detector, recorded_appearance = cache.frame(camera, frame)
                    detector_ms += recorded_detector
                    appearance_ms += recorded_appearance
                    calls[camera] += 1
                    tracks = trackers[camera].step(detections, frame)
                    batch[camera] = (frame, tracks)
                    coordinator.set_active(camera, tracks)
                    if frame + 1 < ranges[camera][1]:
                        heapq.heappush(video_events, (native_time_us(metadata[camera], frame + 1), camera, frame + 1))
                    decisions.write(canonical({"camera": camera, "frame": frame + 1,
                                               "local_frame_zero_based": frame, "time_s": now,
                                               "detect": True, "policy": "full"}) + "\n")
                    for track in tracks:
                        local_predictions.write(row_text(camera, track["local_id"], frame, track["box"]))
                # All tied observations update the active sets before the joint
                # resolution; another camera keeps its last active set between
                # its own native observations.
                for camera, (_, tracks) in sorted(batch.items()):
                    coordinator.update(camera, tracks, now)
                coordinator.resolve(now)
                emit_ready(now)
                for camera, (frame, tracks) in sorted(batch.items()):
                    for track in tracks:
                        key = (camera, track["local_id"])
                        if key not in coordinator.mapping:
                            buffered.setdefault(key, []).append((frame, track["box"], now))
                        else:
                            emit(key, frame, track["box"], now, now)
                # Every observation is published after the batch assignment.
                # Empty observe calls only publish a previously observed exit
                # whose source identity just committed at a timer event.
                for camera in cameras:
                    coordinator.observe(camera, batch.get(camera, (None, []))[1], now)
                event_kind = "video_batch" if batch else "timer"
                events[event_kind] += 1
                events["messages_delivered"] += delivered
                timeline.write(canonical({"event": event_kind, "time_s": now,
                                          "camera_frames": {str(c): f + 1 for c, (f, _) in sorted(batch.items())},
                                          "coverage_ended_cameras": ended,
                                          "messages_delivered": delivered,
                                          "unresolved_tracks": sum(k not in coordinator.mapping for k in coordinator.states)}) + "\n")
                processing_ms.append((time.perf_counter() - begin_event) * 1000)
            pending_at_end = sum(key not in coordinator.mapping for key in coordinator.states)
            coordinator.flush(cfg["stop_s"])
            terminal_future_commits = sum(record["reason"] == "end_of_segment" and record["time_s"] > cfg["stop_s"]
                                          for record in coordinator.commits.values())
            emit_ready(cfg["stop_s"], terminal=True)
            for camera in cameras:
                coordinator.set_active(camera, [])
            timeline.write(canonical({"event": "interval_end", "time_s": cfg["stop_s"],
                                      "pending_tracks_flushed": pending_at_end,
                                      "fresh_fallbacks_at_future_deadline_without_future_evidence": terminal_future_commits}) + "\n")
            if buffered:
                raise RuntimeError("Unpublished global rows remain after terminal flush")
        elapsed = time.perf_counter() - begin_run
        with (output / "messages.jsonl").open("w", encoding="utf-8") as messages:
            for record in coordinator.log:
                messages.write(canonical(record) + "\n")
        pending_commits = [record for record in coordinator.commits.values() if "first_time_s" in record]
        diagnostic_counts = Counter(record.get("diagnostics", {}).get("confirmation_outcome", "unreported")
                                    for record in coordinator.commits.values())
        opportunities = sum(stop - start for start, stop in ranges.values())
        detector_provenance = cache.get_meta("detector")
        summary = {"policy": "full", "runtime_mode": "asynchronous_native_frame_cached_replay",
                   "scenario": cfg["scenario"], "split": cfg["split"], "cameras": cameras,
                   "reference_fps": cfg["fps"], "replay_start_s": cfg["replay_start_s"],
                   "score_start_s": cfg["score_start_s"], "stop_s": cfg["stop_s"],
                   "native_frame_ranges_zero_based": {str(c): list(r) for c, r in ranges.items()},
                   "native_camera_metadata": {str(c): m for c, m in metadata.items()},
                   "opportunities": opportunities, "detector_calls": sum(calls.values()),
                   "detector_call_fraction": sum(calls.values()) / opportunities,
                   "calls_by_camera": {str(c): calls[c] for c in cameras}, "prediction_only_frames": 0,
                   "prediction_rows": output_rows, "metadata_messages": coordinator.messages,
                   "metadata_bytes": coordinator.bytes, "messages_dropped": coordinator.dropped,
                   "event_counts": dict(events), "confirmation_outcome_counts": dict(diagnostic_counts),
                   "bounded_confirmation": {"confirmation_frames": cfg["association"].get("confirmation_frames", 15),
                                            "pending_at_end": pending_at_end, "pending_tracks": len(pending_commits),
                                            "confirmed_handoffs": sum(r["reason"] == "confirmed_handoff" for r in pending_commits),
                                            "maximum_pending_lifetime_s": max((r["time_s"] - r["first_time_s"] for r in pending_commits), default=0.),
                                            "deadline_violations": sum(r["time_s"] > r["deadline_s"] + 1e-9 for r in pending_commits)},
                   "terminal_fresh_fallbacks_at_future_deadline": terminal_future_commits,
                   "recorded_selected_detector_time_s": (detector_ms / 1000 if detector_provenance.get("detector_timing_available", True) else None),
                   "detector_timing_available": detector_provenance.get("detector_timing_available", True),
                   "recorded_selected_appearance_time_s": appearance_ms / 1000,
                   "output_filter_statistics": {str(c): trackers[c].output_filter.statistics for c in cameras},
                   "data_kind": detector_provenance["kind"], "replay_elapsed_s": elapsed,
                   "replay_event_p95_ms": float(np.percentile(processing_ms, 95)),
                   "aggregate_camera_frames_per_wall_second": opportunities / elapsed,
                   "runtime_note": "Cached replay excludes decoding and detector inference. Native observation times are quantized to the nearest microsecond. Internal delivery/deadline timers consume no new observations. Full detection is used; no scheduling or edge runtime claim is made.",
                   "terminal_note": "Entrants unresolved when the interval ends receive fresh IDs at their original deadline or the interval end, whichever is later. No post-interval observations or deliveries are consumed.",
                   "frame_convention": "Original one-based camera-local video frame numbers in prediction files; cache indices are zero based.",
                   "platform": {"python": platform.python_version(), "system": platform.platform()},
                   "signature": signature}
        write_json(output / "summary.json", summary)
        write_json(manifest, {"status": "complete", "signature": signature, "options": options,
                              "artifact_hashes": {name: file_hash(output / name) for name in ARTIFACTS}})
        return summary
