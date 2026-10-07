"""Synthetic shared-stream checks for the published association adaptation."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_cityflow_published_v0_5 import preflight, run_baseline
from cityflow.replay import local_frame_range, native_time_us
from rhd.cache import DetectionCache
from rhd.common import canonical, file_hash, write_json
from rhd.evaluation import read_rows


def stream(tmp_path, active_overlap=False, short_camera=False):
    cfg = {"cameras": [6, 7, 8, 9], "fps": 10, "scenario": "S02", "split": "validation",
           "replay_start_s": 0., "score_start_s": 4., "stop_s": 8.,
           "tracker": {"track_buffer": 30}, "route_calibration": {"synthetic_test": True},
           "routes": [{"travel_time_s": [0., 6.]}]}
    metadata = {camera: {"fps": 10., "frames": 80, "width": 100, "height": 100,
                        "timestamp_offset_s": offset}
                for camera, offset in zip(cfg["cameras"], [0., .061, .421, .660])}
    if short_camera:
        metadata[6]["frames"] = 5
    visible = {(6, f) for f in range(1, 6)}
    if short_camera:
        visible |= {(8, f) for f in range(3, 7)}
    elif active_overlap:
        visible |= {(8, 1)}
    else:
        visible |= {(7, f) for f in range(22, 26)}
        visible |= {(8, f) for f in range(72, 75)}
    feature = np.zeros(512); feature[0] = 1
    cache_path = tmp_path / "cache.sqlite"
    choices, rows = [], []
    with DetectionCache(cache_path) as cache:
        cache.ensure_meta("detector", {"descriptor": "vehicle-reid-0001_512_v1"})
        for camera, meta in metadata.items():
            cache.ensure_meta(f"camera:{camera}", meta)
            for local_frame in range(*local_frame_range(meta, 0, 8)):
                frame = local_frame + 1
                ds = []
                if (camera, frame) in visible:
                    ds = [{"box": [10, 10, 20, 20], "score": .9, "class_id": 2,
                           "feature": feature.tolist()}]
                    rows.append(f"{camera} 1 {frame} 10.000000 10.000000 10.000000 10.000000 -1 -1\n")
                cache.add_frame(camera, local_frame, ds, 0., 0.)
                choices.append({"camera": camera, "frame": frame, "detect": True,
                                "time_s": native_time_us(meta, local_frame) / 1_000_000})
        cache.commit()
    local = tmp_path / "local"
    local.mkdir()
    choices.sort(key=lambda d: (d["time_s"], d["camera"]))
    (local / "decisions.jsonl").write_text("".join(canonical(d) + "\n" for d in choices))
    (local / "local_predictions.txt").write_text("".join(rows))
    summary = {**{name: cfg[name] for name in ("cameras", "scenario", "split", "replay_start_s", "score_start_s", "stop_s")},
               "policy": "full", "detector_call_fraction": 1.,
               "native_camera_metadata": {str(c): m for c, m in metadata.items()},
               "detector_calls": len(choices), "opportunities": len(choices),
               "calls_by_camera": {str(c): sum(d["camera"] == c for d in choices) for c in metadata}}
    write_json(local / "summary.json", summary)
    names = ("decisions.jsonl", "local_predictions.txt", "summary.json")
    write_json(local / "manifest.json", {"status": "complete",
               "artifact_hashes": {n: file_hash(local / n) for n in names}})
    return cfg, cache_path, local


def test_native_vs_derived_retention_and_shared_boxes(tmp_path):
    cfg, cache, local = stream(tmp_path)
    native, derived = tmp_path / "native", tmp_path / "derived"
    native_summary = run_baseline(cfg, cache, local, native, "native")
    derived_summary = run_baseline(cfg, cache, local, derived, "development_route")
    nrows, drows = read_rows(native / "predictions.txt"), read_rows(derived / "predictions.txt")
    assert {r["id"] for r in nrows if r["camera"] in (6, 7)} == {1}
    assert {r["id"] for r in nrows if r["camera"] == 8} == {2}
    assert {r["id"] for r in drows} == {1}
    assert native_summary["retention_s"] == 3
    assert derived_summary["retention_s"] == 9
    assert native_summary["prediction_rows_without_descriptor_history"] == 0
    local_rows = read_rows(local / "local_predictions.txt")
    spatial = lambda records: sorted((r["camera"], r["frame"], tuple(r["box"])) for r in records)
    assert spatial(local_rows) == spatial(nrows) == spatial(drows)
    assert file_hash(native / "local_predictions.txt") == file_hash(local / "local_predictions.txt")
    assert file_hash(native / "decisions.jsonl") == file_hash(local / "decisions.jsonl")
    assert run_baseline(cfg, cache, local, native, "native") == native_summary


def test_latched_active_camera_cannot_be_historical_source(tmp_path):
    cfg, cache, local = stream(tmp_path, active_overlap=True)
    output = tmp_path / "published"
    run_baseline(cfg, cache, local, output)
    rows = read_rows(output / "predictions.txt")
    assert {r["id"] for r in rows if r["camera"] == 6} == {1}
    assert {r["id"] for r in rows if r["camera"] == 8} == {2}
    availability = [json.loads(line) for line in (output / "availability.jsonl").read_text().splitlines()]
    assert all(row["time_s"] == row["published_at_s"] for row in availability)


def test_video_end_retires_a_still_active_track_without_an_empty_frame(tmp_path):
    cfg, cache, local = stream(tmp_path, short_camera=True)
    values = preflight(cfg, cache, local, "native")
    by_time = values[3]
    assert by_time[500_000] == []  # Exact cam006 coverage boundary, no fake frame.
    output = tmp_path / "published"
    summary = run_baseline(cfg, cache, local, output)
    rows = read_rows(output / "predictions.txt")
    assert {row["id"] for row in rows if row["camera"] in (6, 8)} == {1}
    assert max(row["frame"] for row in rows if row["camera"] == 6) == 5
    assert summary["detector_calls"] == json.loads((local / "summary.json").read_text())["detector_calls"]
    assert file_hash(output / "decisions.jsonl") == file_hash(local / "decisions.jsonl")


def test_changed_shared_source_and_completed_result_are_rejected(tmp_path):
    cfg, cache, local = stream(tmp_path)
    output = tmp_path / "published"
    run_baseline(cfg, cache, local, output)
    (output / "predictions.txt").write_text("tampered\n")
    with pytest.raises(ValueError, match="Changed completed"):
        run_baseline(cfg, cache, local, output)
    (local / "decisions.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError, match="Shared local artifact changed"):
        preflight(cfg, cache, local, "native")
