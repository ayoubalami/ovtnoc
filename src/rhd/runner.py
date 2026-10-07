from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from . import __version__
from .cache import DetectionCache
from .common import canonical, digest, file_hash, write_json
from .coordination import Coordinator, Scheduler
from .evaluation import evaluate
from .vision import CameraTracker


def source_signature():
    return digest({p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def run(cfg: dict, cache_path: Path | None, output: Path, policy="full", budget=1.0, interval=1, live=None):
    """Run the same causal engine with a detector cache or real on-demand video inference."""
    bounded = cfg["association"].get("kind") == "bounded"
    if policy in ("coordinated", "identity") and not cfg["routes"]:
        raise ValueError("Coordinated scheduling needs audited routes and arrival windows; configure routes first")
    if policy == "identity" and not bounded:
        raise ValueError("Identity scheduling requires bounded association")
    cameras = sorted(cfg["cameras"])
    scheduler = Scheduler(cameras, policy, budget, interval, cfg["scheduler"])
    start, stop, fps = cfg["start_frame"], cfg["stop_frame"], cfg["fps"]
    options = {"config": cfg, "policy": policy, "budget": budget, "interval": interval,
               "source_signature": source_signature(), "package_version": __version__,
               "packages": {n: importlib.metadata.version(n) for n in ("ultralytics", "lap", "numpy", "scipy")}}
    if live is not None:
        from .vision import LiveSource
        source = LiveSource(**live)
    else:
        source = DetectionCache(cache_path, readonly=True)
    with source as cache:
        cache.validate_range(cameras, start, stop, fps)
        options["cache_signature"] = cache.signature(cameras, start, stop)
        signature = digest(options)
        manifest = output / "manifest.json"
        if manifest.is_file():
            previous = json.loads(manifest.read_text(encoding="utf-8"))
            if previous.get("signature") != signature:
                raise ValueError(f"Existing output {output} belongs to a different experiment; use a new output folder")
            if previous.get("status") == "complete":
                for name, expected_hash in previous["artifact_hashes"].items():
                    if not (output / name).is_file() or file_hash(output / name) != expected_hash:
                        raise ValueError(f"Completed artifact changed or missing: {output / name}")
                print(f"completed run reused: {output}", flush=True)
                return json.loads((output / "summary.json").read_text(encoding="utf-8"))
            raise ValueError(f"Incomplete run at {output}; use a new folder. Sweep resumes completed runs and makes a new attempt for interrupted ones")
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"Output folder is not empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        write_json(manifest, {"status": "running", "signature": signature, "options": options})
        if bounded:
            from .bounded import BoundedCoordinator
            coordinator = BoundedCoordinator(cfg)
        else:
            coordinator = Coordinator(cfg)
        trackers = {c: CameraTracker(fps, cache.get_meta(f"camera:{c}")["width"],
                                     cache.get_meta(f"camera:{c}")["height"], cfg["tracker"],
                                     cfg.get("output_filter")) for c in cameras}
        previous_tracks = {c: [] for c in cameras}
        calls_by_camera = {c: 0 for c in cameras}
        detector_time = appearance_time = 0.0
        tick_times = []
        output_rows = 0
        overdue_unserved = 0
        maximum_gap = {c: 0 for c in cameras}
        predicted_frames = 0
        buffered = {}
        pending_at_end = 0
        begin_run = time.perf_counter()
        with (output / "predictions.txt").open("w", encoding="utf-8") as predictions, (output / "decisions.jsonl").open("w", encoding="utf-8") as decisions_file, (output / "local_predictions.txt").open("w", encoding="utf-8") as local_predictions, (output / "availability.jsonl").open("w", encoding="utf-8") as availability:
            def row_text(camera, identity, frame, box):
                x1, y1, x2, y2 = box
                return f"{camera} {identity} {frame} {x1:.6f} {y1:.6f} {x2-x1:.6f} {y2-y1:.6f} -1 -1\n"

            def emit(key, frame, box, published):
                nonlocal output_rows
                gid = coordinator.mapping[key]
                predictions.write(row_text(key[0], gid, frame, box))
                reason = coordinator.commits[key]["reason"] if bounded else "immediate"
                availability.write(canonical({"camera": key[0], "local_id": key[1], "global_id": gid,
                                               "frame": frame, "published_at_s": published,
                                               "published_at_frame": math.ceil(published * fps - 1e-9),
                                               "reason": reason}) + "\n")
                output_rows += 1

            def emit_ready(now):
                for key in sorted(list(buffered)):
                    if key in coordinator.mapping:
                        published = coordinator.commits[key]["time_s"] if bounded else now
                        for original_frame, box in buffered.pop(key):
                            emit(key, original_frame, box, max(published, original_frame / fps))

            for tick, frame in enumerate(range(start, stop)):
                begin = time.perf_counter()
                now = frame / fps
                coordinator.deliver(now)
                decisions = scheduler.choose(tick, previous_tracks, coordinator, now)
                current_tracks = {}
                for camera in cameras:
                    if live is not None:
                        cache.advance(camera, frame)
                    choice = decisions[camera]
                    if choice["detect"]:
                        detections, detector_ms, feature_ms = cache.frame(camera, frame)
                        detector_time += detector_ms
                        appearance_time += feature_ms
                        calls_by_camera[camera] += 1
                    else:
                        detections = None
                        predicted_frames += 1
                        overdue_unserved += int(choice["overdue"])
                    maximum_gap[camera] = max(maximum_gap[camera], choice["age"] if tick > 0 else 0)
                    tracks = trackers[camera].step(detections, frame)
                    current_tracks[camera] = tracks
                    coordinator.set_active(camera, tracks)
                    if bounded:
                        coordinator.update(camera, tracks, now)
                    for track in tracks:
                        local_predictions.write(row_text(camera, track["local_id"], frame, track["box"]))
                    decisions_file.write(canonical({"camera": camera, "frame": frame, "time_s": now, **choice}) + "\n")
                if bounded:
                    coordinator.resolve(now)
                emit_ready(now)
                for camera in cameras:
                    for track in current_tracks[camera]:
                        key = (camera, track["local_id"])
                        global_id = coordinator.assign(camera, track, now)
                        if global_id is None:
                            buffered.setdefault(key, []).append((frame, track["box"]))
                        else:
                            emit(key, frame, track["box"], now)
                # Events are published only after every camera's decision for this tick.
                for camera in cameras:
                    coordinator.observe(camera, current_tracks[camera], now)
                previous_tracks = current_tracks
                tick_times.append((time.perf_counter() - begin) * 1000)
            if bounded:
                pending_at_end = sum(k not in coordinator.mapping for k in coordinator.states)
                coordinator.flush((stop - 1) / fps)
                emit_ready((stop - 1) / fps)
            if buffered:
                raise RuntimeError("Unpublished global rows remain")
        replay_elapsed = time.perf_counter() - begin_run
        with (output / "messages.jsonl").open("w", encoding="utf-8") as f:
            for record in coordinator.log:
                f.write(canonical(record) + "\n")
        opportunities = (stop - start) * len(cameras)
        summary = {"policy": policy, "budget_fraction": budget, "fixed_interval": interval,
                   "cameras": cameras, "start_frame": start, "stop_frame": stop, "fps": fps,
                   "opportunities": opportunities, "detector_calls": scheduler.calls,
                   "detector_call_fraction": scheduler.calls / opportunities,
                   "calls_by_camera": {str(k): v for k, v in calls_by_camera.items()}, "prediction_only_frames": predicted_frames,
                   "maximum_detection_age_frames": {str(k): v for k, v in maximum_gap.items()}, "overdue_camera_frames_not_served": overdue_unserved,
                   "prediction_rows": output_rows, "metadata_messages": coordinator.messages,
                   "metadata_bytes": coordinator.bytes, "messages_dropped": coordinator.dropped,
                   "replay_elapsed_s": replay_elapsed, "replay_tick_p95_ms": float(np.percentile(tick_times, 95)),
                   "recorded_selected_detector_time_s": (detector_time / 1000 if cache.get_meta("detector").get("detector_timing_available", True) else None),
                   "detector_timing_available": cache.get_meta("detector").get("detector_timing_available", True),
                   "recorded_selected_appearance_time_s": appearance_time / 1000,
                   "output_filter_statistics": {str(c): trackers[c].output_filter.statistics for c in cameras},
                   "data_kind": cache.get_meta("detector")["kind"],
                   "runtime_mode": "live_video_inference" if live is not None else "cached_replay",
                   "aggregate_camera_frames_per_wall_second": opportunities / replay_elapsed,
                   "runtime_note": ("Live sequential desktop replay includes decoding, scheduled inference, tracking, coordination, and output writes. Model loading and pre-range seeking are excluded. Tick p95 is processing time, not streaming latency or edge-device performance."
                                    if live is not None else "Cached replay timing excludes video decode and detector inference. Summed recorded invocation time is a compute proxy, not live latency or energy."),
                   "budget_note": "All cameras receive a startup call. Subsequent adaptive calls share a token cap; a safety deadline can be missed if the budget is insufficient.",
                   "platform": {"python": platform.python_version(), "system": platform.platform()}, "signature": signature}
        write_json(output / "summary.json", summary)
        if bounded:
            pending_commits = [r for r in coordinator.commits.values() if "first_time_s" in r]
            summary["bounded_confirmation"] = {
                "confirmation_frames": cfg["association"].get("confirmation_frames", 15),
                "pending_at_end": pending_at_end, "pending_tracks": len(pending_commits),
                "confirmed_handoffs": sum(r["reason"] == "confirmed_handoff" for r in pending_commits),
                "maximum_pending_lifetime_s": max((r["time_s"] - r["first_time_s"] for r in pending_commits), default=0.),
                "deadline_violations": sum(r["time_s"] > r["deadline_s"] + 1e-9 for r in pending_commits)}
            write_json(output / "summary.json", summary)
        names = ("predictions.txt", "local_predictions.txt", "availability.jsonl", "decisions.jsonl", "messages.jsonl", "summary.json")
        write_json(manifest, {"status": "complete", "signature": signature, "options": options,
                              "artifact_hashes": {n: file_hash(output / n) for n in names}})
        return summary


def score_run(cfg: dict, output: Path, ground_truth: Path, label_scope="multicam"):
    metrics = evaluate(ground_truth, output / "predictions.txt", cfg["cameras"], cfg["start_frame"],
                       cfg["stop_frame"], cfg["fps"], label_scope=label_scope)
    metrics["ground_truth_sha256"] = file_hash(ground_truth)
    write_json(output / "metrics.json", metrics)
    return metrics


def sweep(cfg: dict, cache: Path, ground_truth: Path, output: Path, budgets=(0.25, 0.5, 0.75), intervals=(2, 3, 5)):
    # Preflight all inputs before producing any primary result.
    if not cfg["routes"]:
        raise ValueError("Sweep includes coordinated scheduling; configure routes first")
    if not ground_truth.is_file():
        raise FileNotFoundError(ground_truth)
    definitions = [("full", 1.0, 1)] + [("fixed", 1.0, i) for i in intervals]
    definitions += [(p, b, 1) for b in budgets for p in ("independent", "coordinated")]
    records = []
    for policy, budget, interval in definitions:
        tag = f"{policy}_b{budget:g}_n{interval}"
        directory = output / tag
        attempt = 1
        while (directory / "manifest.json").is_file():
            state = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if state.get("status") == "complete":
                break
            attempt += 1
            directory = output / f"{tag}_attempt{attempt}"
        summary = run(cfg, cache, directory, policy, budget, interval)
        metrics = score_run(cfg, directory, ground_truth)
        row = {"policy": policy, "budget_fraction": budget, "fixed_interval": interval,
               "detector_calls": summary["detector_calls"], "detector_call_fraction": summary["detector_call_fraction"],
               "global_IDF1": metrics["global_identity"]["IDF1"], "local_IDF1": metrics["local_identity"]["IDF1"],
               "handoff_recall": metrics["handoffs"]["handoff_recall"],
               "handoff_precision": metrics["handoffs"]["handoff_precision"],
               "recorded_detector_time_s": summary["recorded_selected_detector_time_s"], "data_kind": summary["data_kind"]}
        records.append(row)
        print(canonical(row), flush=True)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    plot_tradeoff(records, output / "identity_compute.png")
    return records


def plot_tradeoff(records: list[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for policy in Scheduler.policies:
        points = sorted([r for r in records if r["policy"] == policy], key=lambda r: r["detector_call_fraction"])
        if points:
            ax.plot([r["detector_call_fraction"] * 100 for r in points],
                    [r["global_IDF1"] * 100 for r in points], "o-", label=policy)
    ax.set(xlabel="Detector calls (% of camera-frame opportunities)", ylabel="Global IDF1 (%)")
    synthetic = any(r["data_kind"] == "synthetic_detector_responses" for r in records)
    ax.set_title("Synthetic pipeline check — not dataset evidence" if synthetic else "Identity versus detection workload")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)

