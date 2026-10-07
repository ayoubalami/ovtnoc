import copy
import json

import numpy as np
import pytest

from rhd.cache import DetectionCache
from rhd.common import load_config, point_in_polygon, write_json
from rhd.coordination import Coordinator, Scheduler
from rhd.demo import generate
from rhd.evaluation import evaluate, identity_metrics, read_rows
from rhd.runner import run, sweep
from rhd.routes import fit_routes
from rhd.vision import CameraTracker


def config():
    return {"fps": 15, "cameras": [1, 2], "seed": 7, "routes": [], "association": {}, "communication": {}}


def row(cam, identity, frame, box=None):
    return {"camera": cam, "id": identity, "frame": frame, "box": box or [0, 0, 10, 10]}


def track(local_id=1, feature=None):
    return {"local_id": local_id, "point": [0.9, 0.5], "velocity": [0.01, 0],
            "feature": feature or [1, 0], "score": 0.9, "age_since_detection": 0, "uncertainty": 0.0}


def routed_config():
    cfg = config()
    cfg["routes"] = [{"source": 1, "destination": 2, "exit_polygon": [[0.8, 0], [1, 0], [1, 1], [0.8, 1]],
                      "entry_polygon": [[0, 0], [0.4, 0], [0.4, 1], [0, 1]],
                      "direction": [1, 0], "travel_time_s": [0, 2]}]
    return cfg


def test_global_idf1_penalizes_camera_identity_break():
    gt = [row(1, 1, 0), row(2, 1, 10)]
    split = [row(1, 10, 0), row(2, 20, 10)]
    assert identity_metrics(gt, split)["IDF1"] == 0.5
    assert identity_metrics(gt, split, local=True)["IDF1"] == 1.0


def test_idf1_counts_false_positives_and_missing_frames():
    gt = [row(1, 1, 0), row(1, 1, 1)]
    pred = [row(1, 10, 0), row(1, 20, 0, [50, 50, 60, 60])]
    result = identity_metrics(gt, pred)
    assert (result["IDTP"], result["IDFN"], result["IDFP"]) == (1, 1, 1)
    assert result["IDF1"] == 0.5


def test_multicamera_observations_are_not_matched_across_frame_or_camera():
    assert identity_metrics([row(1, 1, 0)], [row(2, 1, 0)])["IDTP"] == 0
    assert identity_metrics([row(1, 1, 0)], [row(1, 1, 1)])["IDTP"] == 0


def test_input_rejects_duplicate_or_malformed_annotations(tmp_path):
    file = tmp_path / "rows.txt"
    file.write_text("1 1 0 0 0 10 10 -1 -1\n1 1 0 0 0 10 10 -1 -1\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_rows(file)
    file.write_text("1 1 0 0 0 -10 10 -1 -1\n")
    with pytest.raises(ValueError, match="invalid"):
        read_rows(file)


def test_skipped_frame_predicts_without_turning_track_into_missed_detection():
    tracker = CameraTracker(15, 320, 180, {})
    d1 = {"box": [30, 40, 70, 64], "score": 0.9, "class_id": 2, "feature": [1, 0]}
    first = tracker.step([d1], 0)
    d2 = {**d1, "box": [35, 40, 75, 64]}
    second = tracker.step([d2], 1)
    predicted = tracker.step(None, 2)
    assert first[0]["local_id"] == second[0]["local_id"] == predicted[0]["local_id"]
    assert predicted[0]["box"][0] > second[0]["box"][0]
    assert predicted[0]["age_since_detection"] == 1
    assert not tracker.step([], 3)  # An actual detector miss has different semantics.


def test_message_is_not_available_within_source_tick():
    coord = Coordinator(routed_config())
    t = track()
    coord.assign(1, t, 0)
    coord.observe(1, [t], 0)
    coord.deliver(0)
    assert coord.arrival_priority(2, 0) == 0
    coord.deliver(1 / 15)
    assert coord.arrival_priority(2, 1 / 15) > 0
    destination = {**track(local_id=2), "point": [0.1, 0.5]}
    assert coord.assign(2, destination, 1 / 15) == coord.mapping[(1, 1)]


def test_lost_or_expired_messages_do_not_trigger_handoffs():
    cfg = routed_config()
    cfg["communication"]["loss_probability"] = 1
    coord = Coordinator(cfg)
    t = track()
    coord.assign(1, t, 0)
    coord.observe(1, [t], 0)
    coord.deliver(1)
    assert coord.arrival_priority(2, 1) == 0
    assert coord.dropped == 1
    coord = Coordinator(routed_config())
    coord.assign(1, t, 0)
    coord.observe(1, [t], 0)
    coord.deliver(3)
    assert coord.arrival_priority(2, 3) == 0


def test_ambiguous_identity_candidates_are_rejected():
    coord = Coordinator(routed_config())
    for local in (1, 2):
        t = track(local_id=local)
        coord.assign(1, t, 0)
        coord.observe(1, [t], 0)
    coord.deliver(1 / 15)
    gid = coord.assign(2, {**track(3), "point": [0.1, 0.5]}, 1 / 15)
    assert gid not in [coord.mapping[(1, 1)], coord.mapping[(1, 2)]]


def test_adaptive_policies_have_equal_call_counts_at_equal_budget():
    cfg = routed_config()
    coord = Coordinator(cfg)
    independent = Scheduler([1, 2], "independent", 0.25, 1, {})
    coordinated = Scheduler([1, 2], "coordinated", 0.25, 1, {})
    observations = {1: [], 2: []}
    for tick in range(101):
        independent.choose(tick, observations, coord, tick / 15)
        coordinated.choose(tick, observations, coord, tick / 15)
    assert independent.calls == coordinated.calls == 52  # 2 startup + 100*2*.25


def test_low_budget_records_lack_of_hard_detection_deadline():
    sched = Scheduler([1, 2], "independent", 0.01, 1, {"max_detection_gap": 5})
    coord = Coordinator(config())
    unserved = 0
    for tick in range(20):
        choices = sched.choose(tick, {1: [], 2: []}, coord, tick / 15)
        unserved += sum(d["overdue"] and not d["detect"] for d in choices.values())
    assert unserved > 0


def test_missing_cache_frames_fail_before_experiment(tmp_path):
    with DetectionCache(tmp_path / "data.sqlite") as cache:
        cache.ensure_meta("camera:1", {"fps": 15})
        cache.add_frame(1, 0, [])
        cache.commit()
        with pytest.raises(ValueError, match="Resume extraction"):
            cache.validate_range([1], 0, 2, 15)
        assert cache.frame(1, 0)[0] == []
        with pytest.raises(ValueError, match="absent"):
            cache.frame(1, 1)


def test_resume_rejects_changed_cache_provenance(tmp_path):
    with DetectionCache(tmp_path / "data.sqlite") as cache:
        cache.ensure_meta("detector", {"model_sha256": "a"})
        with pytest.raises(ValueError, match="provenance mismatch"):
            cache.ensure_meta("detector", {"model_sha256": "b"})


def test_closed_loop_demo_has_predictions_on_skipped_frames_and_resume(tmp_path):
    cfg, cache, gt = generate(tmp_path / "inputs")
    output = tmp_path / "fixed"
    summary = run(cfg, cache, output, "fixed", interval=3)
    assert summary["detector_calls"] == 100
    assert summary["prediction_only_frames"] == 200
    rows = read_rows(output / "predictions.txt")
    assert any(row["frame"] % 3 for row in rows)
    metrics = evaluate(gt, output / "predictions.txt", cfg["cameras"], 0, 150, 15)
    assert metrics["global_identity"]["IDTP"] > 0
    before = (output / "predictions.txt").stat().st_mtime_ns
    assert run(cfg, cache, output, "fixed", interval=3) == summary
    assert (output / "predictions.txt").stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="different experiment"):
        run(cfg, cache, output, "fixed", interval=2)


def test_output_tampering_is_not_silently_reused(tmp_path):
    cfg, cache, _ = generate(tmp_path / "inputs")
    cfg["stop_frame"] = 4
    output = tmp_path / "full"
    run(cfg, cache, output)
    (output / "predictions.txt").write_text("")
    with pytest.raises(ValueError, match="artifact changed"):
        run(cfg, cache, output)


def test_future_cached_detections_do_not_change_past_decisions(tmp_path):
    cfg, cache_path, _ = generate(tmp_path / "inputs")
    run(cfg, cache_path, tmp_path / "a", "coordinated", budget=0.5)
    with DetectionCache(cache_path) as cache:
        cache.db.execute("UPDATE frames SET payload='[]' WHERE frame>=100")
        cache.commit()
    run(cfg, cache_path, tmp_path / "b", "coordinated", budget=0.5)
    def prefix(path):
        return [line for line in path.read_text().splitlines() if json.loads(line)["frame"] < 100]
    assert prefix(tmp_path / "a" / "decisions.jsonl") == prefix(tmp_path / "b" / "decisions.jsonl")


def test_route_config_rejects_pixel_coordinates_and_invalid_endpoints(tmp_path):
    cfg = routed_config()
    cfg["start_frame"], cfg["stop_frame"] = 0, 100
    cfg["routes"][0]["exit_polygon"][0] = [256, 100]
    p = tmp_path / "cfg.json"
    write_json(p, cfg)
    with pytest.raises(ValueError, match="normalized"):
        load_config(p)


def test_polygon_boundaries_are_included():
    assert point_in_polygon([0, 0.5], [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert not point_in_polygon([1.1, 0.5], [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_route_calibration_records_training_range_and_blocks_test_overlap(tmp_path):
    cfg, cache, gt = generate(tmp_path / "inputs")
    fitted = fit_routes(cfg, cache, gt, 0, 150, min_samples=1)
    assert fitted["route_calibration"]["routes"][0]["samples"] == 2
    assert fitted["routes"][0]["travel_time_s"][0] < fitted["routes"][0]["travel_time_s"][1]
    assert cfg["routes"][0]["travel_time_s"] == [0.7, 1.8]  # input is preserved
    fitted["phase"] = "test"
    path = tmp_path / "test_config.json"
    write_json(path, fitted)
    with pytest.raises(ValueError, match="overlaps"):
        load_config(path)
    fitted["start_frame"], fitted["stop_frame"] = 150, 200
    write_json(path, fitted)
    assert load_config(path)["phase"] == "test"


def test_insufficient_route_examples_are_not_fabricated(tmp_path):
    cfg, cache, gt = generate(tmp_path / "inputs")
    with pytest.raises(ValueError, match="only 2 development examples"):
        fit_routes(cfg, cache, gt, 0, 150)


def test_sweep_reuses_completed_retry_after_an_interruption(tmp_path):
    cfg, cache, gt = generate(tmp_path / "inputs")
    cfg["stop_frame"] = 4
    output = tmp_path / "sweep"
    first = output / "full_b1_n1"
    write_json(first / "manifest.json", {"status": "running"})
    sweep(cfg, cache, gt, output, budgets=(0.5,), intervals=(2,))
    retry = output / "full_b1_n1_attempt2" / "predictions.txt"
    before = retry.stat().st_mtime_ns
    sweep(cfg, cache, gt, output, budgets=(0.5,), intervals=(2,))
    assert retry.stat().st_mtime_ns == before
    assert not (output / "full_b1_n1_attempt3").exists()
