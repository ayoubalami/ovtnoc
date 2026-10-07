"""Full synthetic study integration; scores here are never research evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_cityflow_v0_5 as study
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, write_json


def _observations(camera, frame):
    tracks = []
    if camera == 6:
        tracks.append((10, [60, 10, 90, 40]))
        if frame < 5:
            tracks.append((1, [10, 10, 40, 40]))
    elif camera == 7 and frame >= 5:
        tracks.append((1, [10, 10, 40, 40]))
    elif camera in (8, 9):
        tracks.append((camera, [20, 20, 50, 50]))
    return tracks


@pytest.fixture
def synthetic_study(tmp_path):
    licensed = Path(os.environ.get("CITYFLOW_DATASET_ROOT",
                    "data/CityFlowV2")) / "eval" / "eval.py"
    if not licensed.is_file():
        pytest.skip("Set CITYFLOW_DATASET_ROOT to the licensed AIC22 dataset")
    dataset = tmp_path / "synthetic_dataset"
    (dataset / "eval").mkdir(parents=True)
    shutil.copyfile(licensed, dataset / "eval" / "eval.py")
    cameras = {}
    official_rows = []
    for camera, offset in zip((6, 7, 8, 9), (0., .061, .421, .660)):
        directory = dataset / "validation" / "S02" / f"c{camera:03d}"
        (directory / "gt").mkdir(parents=True)
        video = directory / "vdo.avi"
        video.write_bytes(b"synthetic cache test; no video decoder is used")
        roi = directory / "roi.jpg"
        Image.fromarray(np.full((80, 100), 255, dtype=np.uint8)).save(roi, format="PNG")
        cameras[str(camera)] = {"camera": camera, "fps": 10., "frames": 20,
                                "width": 100, "height": 80, "timestamp_offset_s": offset,
                                "video": str(video), "video_sha256": file_hash(video),
                                "roi": str(roi), "roi_sha256": file_hash(roi)}
        local_rows = []
        for frame in range(20):
            for identity, (x1, y1, x2, y2) in _observations(camera, frame):
                local_rows.append(f"{frame+1},{identity},{x1},{y1},{x2-x1},{y2-y1},1,-1,-1,-1\n")
                official_rows.append(f"{camera} {identity} {frame+1} {x1} {y1} {x2-x1} {y2-y1} -1 -1\n")
        (directory / "gt" / "gt.txt").write_text("".join(local_rows), encoding="utf-8")
    # Real validation GT also contains S05 cameras. This extra camera verifies
    # that scenario selection happens outside inference and before evaluation.
    official_rows.append("10 100 1 1 1 10 10 -1 -1\n")
    (dataset / "eval" / "ground_truth_validation.txt").write_text("".join(official_rows), encoding="utf-8")
    scene = {"dataset_root": str(dataset), "split": "validation", "scenario": "S02",
             "cameras": cameras, "timestamps_sha256": "synthetic offsets",
             "framecounts_sha256": "synthetic counts", "clock": "native",
             "timing_limit": "Synthetic integration test only"}
    cfg = {"version": "synthetic_cityflow_v0_5", "dataset": str(dataset),
           "split": "validation", "scenario": "S02", "cameras": [6, 7, 8, 9],
           "fps": 10., "seed": 7, "replay_start_s": .2, "score_start_s": .5, "stop_s": 2.,
           "tracker": {}, "communication": {"delay_s": 0., "loss_probability": 0.},
           "association": {"kind": "bounded", "confirmation_frames": 8,
                           "minimum_observations": 1, "source_retention_frames": 8,
                           "gallery_size": 5, "max_cosine_distance": .8,
                           "ambiguity_margin": 0., "time_weight": 0., "diagnostics": True},
           "routes": [{"source": 6, "destination": 7, "direction": [0, 0],
                       "exit_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                       "entry_polygon": [[0, 0], [.5, 0], [.5, 1], [0, 1]],
                       "travel_time_s": [0., 1.]}],
           "route_calibration": {"start_s": 0., "stop_s": .2},
           "controls": {"confirmation_frames": [0, 2, 4, 8],
                        "minimum_observations": 1, "source_retention_frames": 8},
           "scope": "Synthetic integration fixture only; no research evidence.",
           "perception": "Synthetic detections with synthetic 512-dimensional descriptors.",
           "scene_signature": digest(scene)}
    protocol = tmp_path / "protocol"
    write_json(protocol / "protocol.json", cfg)
    write_json(protocol / "scene.json", scene)
    write_json(protocol / "freeze.json", {"protocol_sha256": file_hash(protocol / "protocol.json"),
                                          "scene_sha256": file_hash(protocol / "scene.json")})
    cache_path = tmp_path / "synthetic_cache.sqlite"
    with DetectionCache(cache_path) as cache:
        cache.ensure_meta("scene", scene)
        # Mimic the frozen production metadata contract. All payloads remain
        # explicitly synthetic and this temporary fixture is never evidence.
        cache.ensure_meta("detector", {"kind": "provided_raw_cityflow_detector_cache",
                                       "model": "CityFlowV2 supplied mask_rcnn", "minimum_confidence": .1,
                                       "feature_scope": "every retained detection on its own native video frame; no region/cadence restriction",
                                       "descriptor": "vehicle-reid-0001_512_v1",
                                       "synthetic_test_only": True,
                                       "detector_timing_available": False})
        for camera in cfg["cameras"]:
            cache.ensure_meta(f"camera:{camera}", cameras[str(camera)])
            for frame in range(20):
                detections = []
                for identity, box in _observations(camera, frame):
                    descriptor = np.zeros(512); descriptor[identity] = 1.
                    detections.append({"box": box, "score": .9, "class_id": 2,
                                       "feature": descriptor.tolist()})
                cache.add_frame(camera, frame, detections)
        cache.commit()
    return protocol / "protocol.json", cache_path, tmp_path / "study"


def _execute(monkeypatch, protocol, cache, output, published=True):
    arguments = ["run_cityflow_v0_5.py", "--protocol", str(protocol),
                 "--cache", str(cache), "--output", str(output)]
    if published:
        arguments.append("--published-baseline")
    monkeypatch.setattr(sys, "argv", arguments)
    study.main()


def test_full_synthetic_study_baselines_resume_and_metric_tamper(synthetic_study, monkeypatch):
    protocol, cache, output = synthetic_study
    _execute(monkeypatch, protocol, cache, output)
    state = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert len(state["completed"]) == 7  # four delays, local IDs, two published variants
    assert state["signed_inputs"]["published_baseline"] is True
    assert set(state["artifact_hashes"]) >= {"results.csv", "results.md", "shared_stream_proof.json",
                                             "ground_truth_S02_cohort.txt", "evaluation_inputs.json"}
    assert not any(line.startswith("10 ") for line in
                   (output / "ground_truth_S02_cohort.txt").read_text().splitlines())
    proof = json.loads((output / "shared_stream_proof.json").read_text())
    assert proof["all_deadline_controls_identical"]
    local_hashes, trace_hashes, calls = [], [], []
    for k in (0, 2, 4, 8):
        directory = output / f"K{k}_n1"
        local_hashes.append(file_hash(directory / "local_predictions.txt"))
        trace_hashes.append(file_hash(directory / "decisions.jsonl"))
        summary = json.loads((directory / "summary.json").read_text())
        calls.append(summary["detector_calls"])
        metrics = json.loads((directory / "metrics.json").read_text())
        assert set(metrics) >= {"official_benchmark", "roi_only_identity", "unfiltered_identity", "handoffs"}
        assert metrics["handoffs"]["reference_handoffs"] == 1
        pending = json.loads((directory / "pending_diagnostics.json").read_text())
        assert pending["scored_pending_tracks"] >= 1
    assert len(set(local_hashes)) == len(set(trace_hashes)) == len(set(calls)) == 1
    local_metrics = json.loads((output / "local_ID_control" / "metrics.json").read_text())
    assert local_metrics["official_benchmark"]["idf1"] == 0.
    assert local_metrics["roi_only_identity"]["idf1"] > 0.
    assert local_metrics["handoffs"]["confirmed_handoffs"] == 0
    for variant in ("native", "development_route"):
        directory = output / f"published_association_{variant}"
        assert file_hash(directory / "local_predictions.txt") == local_hashes[0]
        assert file_hash(directory / "decisions.jsonl") == trace_hashes[0]
        assert json.loads((directory / "summary.json").read_text())["retention_variant"] == variant
        assert (directory / "metrics.json").is_file()
    previous_progress = (output / "progress.json").read_bytes()
    _execute(monkeypatch, protocol, cache, output)
    assert (output / "progress.json").read_bytes() == previous_progress
    assert all(file_hash(output / name) == expected for name, expected in state["artifact_hashes"].items())
    # The study resume validates supplemental score artifacts, beyond replay's
    # own prediction/availability manifest, before returning a completed run.
    (output / "K2_n1" / "metrics.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Study artifact changed"):
        _execute(monkeypatch, protocol, cache, output)
    assert (output / "progress.json").read_bytes() == previous_progress


def test_driver_rejects_a_detector_outside_the_frozen_protocol(synthetic_study, monkeypatch):
    protocol, cache_path, output = synthetic_study
    with DetectionCache(cache_path) as cache:
        detector = cache.get_meta("detector")
        detector["model"] = "CityFlowV2 supplied yolo3"
        cache.db.execute("UPDATE metadata SET value=? WHERE key='detector'", (canonical(detector),))
        cache.commit()
    with pytest.raises(ValueError, match="frozen perception protocol"):
        _execute(monkeypatch, protocol, cache_path, output)
    assert not output.exists()
