"""Native clock, causal deadlines, fixed perception, and signed replay checks."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cityflow import replay
from rhd.cache import DetectionCache
from rhd.common import canonical, digest


def config(dataset, **association):
    return {"dataset": str(dataset), "split": "validation", "scenario": "S02",
            "cameras": [6, 7], "fps": 10., "replay_start_s": 0.,
            "score_start_s": 0., "stop_s": 1., "seed": 7,
            "tracker": {}, "communication": {}, "routes": [],
            "association": {"kind": "bounded", "confirmation_frames": 2,
                            "source_retention_frames": 8, "minimum_observations": 1,
                            "gallery_size": 5, "max_cosine_distance": .8,
                            "ambiguity_margin": .03, "diagnostics": True, **association}}


def route(source=6, destination=7):
    return {"source": source, "destination": destination,
            "exit_polygon": [[.8, 0], [1, 0], [1, 1], [.8, 1]],
            "entry_polygon": [[0, 0], [.4, 0], [.4, 1], [0, 1]],
            "direction": [0, 0], "travel_time_s": [0, 1]}


def detection(local=1, feature=(1., 0.), point=(.1, .5)):
    return {"local_id": local, "box": [10., 10., 40., 40.], "score": .9,
            "class_id": 2, "feature": list(feature), "point": list(point)}


class SnapshotTracker:
    """Exact observation snapshots make the coordinator causality test explicit."""
    initializations = []

    def __init__(self, fps, width, height, options, output_filter):
        self.initializations.append((fps, width, height))
        self.output_filter = SimpleNamespace(statistics={})

    def step(self, detections, local_frame):
        assert detections is not None
        return [{"local_id": d["local_id"], "box": d["box"], "score": d["score"],
                 "feature": d["feature"], "observation_feature": d["feature"],
                 "point": d["point"], "velocity": [.01, 0],
                 "age_since_detection": 0, "uncertainty": 0.} for d in detections]


def make_cache(path, cfg, cameras):
    with DetectionCache(path) as cache:
        cache.ensure_meta("scene", {"dataset_root": cfg["dataset"], "split": cfg["split"],
                                    "scenario": cfg["scenario"], "cameras": sorted(cameras),
                                    "reference_fps": cfg["fps"]})
        cache.ensure_meta("detector", {"kind": "synthetic_clock_test", "detector_timing_available": False})
        for camera, (fps, offset, detections) in cameras.items():
            cache.ensure_meta(f"camera:{camera}", {"fps": fps, "timestamp_offset_s": offset,
                                                   "width": 100, "height": 80, "frames": len(detections)})
            for index, observations in enumerate(detections):
                cache.add_frame(camera, index, observations)
        cache.commit()


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(autouse=True)
def snapshots(monkeypatch):
    SnapshotTracker.initializations = []
    monkeypatch.setattr(replay, "CameraTracker", SnapshotTracker)


def test_native_frames_keep_offsets_and_different_rates_without_resampling(tmp_path):
    cfg = config(tmp_path / "data")
    cfg["replay_start_s"] = .13
    cfg["score_start_s"] = .15
    cfg["stop_s"] = .51
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[detection()]] * 6),
                           7: (8., .05, [[detection()]] * 5)})
    result = replay.run(cfg, cache, tmp_path / "run")
    decisions = records(tmp_path / "run" / "decisions.jsonl")
    assert [(r["camera"], r["frame"], r["time_s"]) for r in decisions] == [
        (7, 2, .175), (6, 3, .2), (6, 4, .3), (7, 3, .3), (6, 5, .4), (7, 4, .425), (6, 6, .5)]
    assert result["opportunities"] == result["detector_calls"] == 7
    assert result["calls_by_camera"] == {"6": 4, "7": 3}
    assert SnapshotTracker.initializations == [(10., 100, 80), (8., 100, 80)]
    availability = records(tmp_path / "run" / "availability.jsonl")
    assert all(r["frame"] == r["local_frame_zero_based"] + 1 for r in availability)
    assert all(r["published_at_s"] >= r["time_s"] for r in availability)


def test_deadline_runs_before_a_later_native_observation(tmp_path):
    cfg = config(tmp_path / "data", minimum_observations=2)
    cfg["routes"] = [route()]
    cfg["stop_s"] = .6
    cache = tmp_path / "cache.sqlite"
    # The destination's second observation is at .25; K=2 at reference 10
    # means a .2 deadline. That future descriptor cannot justify confirmation.
    make_cache(cache, cfg, {6: (2., 0., [[]] * 2),
                           7: (4., 0., [[detection()], [detection()], [detection()]])})
    result = replay.run(cfg, cache, tmp_path / "run")
    commits = [r for r in records(tmp_path / "run" / "messages.jsonl") if r["event"] == "identity_committed"]
    assert len(commits) == 1
    assert commits[0]["time_s"] == .2
    assert commits[0]["reason"] == "confirmation_deadline"
    assert commits[0]["observations"] == 1
    assert result["bounded_confirmation"]["deadline_violations"] == 0
    availability = records(tmp_path / "run" / "availability.jsonl")
    assert availability[0]["published_at_s"] == .2
    assert any(r["event"] == "timer" and r["time_s"] == .2
               for r in records(tmp_path / "run" / "timeline.jsonl"))


def test_message_delivery_is_an_event_between_camera_observations(tmp_path):
    cfg = config(tmp_path / "data")
    cfg["routes"] = [route()]
    cfg["stop_s"] = .4
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (5., 0., [[detection(point=(.9, .5))], []]),
                           7: (5., .15, [[detection()], []])})
    replay.run(cfg, cache, tmp_path / "run")
    delivered = [r for r in records(tmp_path / "run" / "messages.jsonl") if r["event"] == "delivered"]
    assert delivered[0]["scheduled_delivery_s"] == delivered[0]["time_s"] == .1
    # Source activity persists across its sparse observations. The source is
    # active at destination .15, then is removed by its own .2 observation.
    commits = [r for r in records(tmp_path / "run" / "messages.jsonl") if r["event"] == "identity_committed"]
    assert commits[-1]["reason"] == "confirmed_handoff"
    assert commits[-1]["time_s"] == .2
    assert commits[-1]["global_id"] == commits[0]["global_id"]


def test_tied_camera_observations_are_resolved_jointly_and_order_independent(tmp_path):
    cfg = config(tmp_path / "data")
    cfg["routes"] = [route()]
    cfg["stop_s"] = .3
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[detection(point=(.9, .5))], [], []]),
                           7: (10., .1, [[detection()], []])})
    replay.run(cfg, cache, tmp_path / "first")
    cfg["cameras"].reverse()
    replay.run(cfg, cache, tmp_path / "second")
    for name in ("local_predictions.txt", "predictions.txt", "availability.jsonl", "decisions.jsonl", "messages.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    commits = [r for r in records(tmp_path / "first" / "messages.jsonl") if r["event"] == "identity_committed"]
    assert commits[-1]["reason"] == "confirmed_handoff"
    assert commits[-1]["time_s"] == .1


def test_delay_controls_keep_exactly_the_same_native_local_stream(tmp_path):
    cfg = config(tmp_path / "data")
    cfg["routes"] = [route()]
    cfg["stop_s"] = .51
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[detection(point=(.9, .5))], [], [], [], [], []]),
                           7: (8., .05, [[detection()], [detection()], [detection()], [detection()]])})
    streams = []
    for delay in (0, 2, 4, 8):
        variant = deepcopy(cfg)
        variant["association"]["confirmation_frames"] = delay
        result = replay.run(variant, cache, tmp_path / f"k{delay}")
        streams.append(((tmp_path / f"k{delay}" / "local_predictions.txt").read_bytes(),
                        (tmp_path / f"k{delay}" / "decisions.jsonl").read_bytes(), result["detector_calls"]))
    assert all(stream == streams[0] for stream in streams)


def test_terminal_flush_waits_until_deadline_without_consuming_future_frames(tmp_path):
    cfg = config(tmp_path / "data", confirmation_frames=8)
    cfg["routes"] = [route()]
    cfg["stop_s"] = .1
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[]] * 5), 7: (10., 0., [[detection()]] * 5)})
    result = replay.run(cfg, cache, tmp_path / "run")
    assert result["detector_calls"] == 2
    availability = records(tmp_path / "run" / "availability.jsonl")
    assert len(availability) == 1
    assert availability[0]["frame"] == 1
    assert availability[0]["published_at_s"] == .8
    assert availability[0]["reason"] == "end_of_segment"


def test_camera_coverage_end_releases_its_last_active_identity(tmp_path):
    cfg = config(tmp_path / "data", confirmation_frames=4)
    cfg["routes"] = [route()]
    cfg["stop_s"] = .5
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[detection(point=(.9, .5))]]),
                           7: (10., .15, [[detection()]] * 3)})
    result = replay.run(cfg, cache, tmp_path / "run")
    commits = [r for r in records(tmp_path / "run" / "messages.jsonl") if r["event"] == "identity_committed"]
    assert commits[-1]["reason"] == "confirmed_handoff"
    assert commits[-1]["time_s"] == .15
    assert result["event_counts"]["camera_coverage_end"] == 2


def test_scene_and_native_cache_gaps_are_rejected_before_output(tmp_path):
    cfg = config(tmp_path / "data")
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[]] * 5), 7: (10., 0., [[]] * 5)})
    wrong = deepcopy(cfg)
    wrong["scenario"] = "S01"
    with pytest.raises(ValueError, match="scene"):
        replay.run(wrong, cache, tmp_path / "wrong")
    with DetectionCache(cache) as stored:
        stored.db.execute("DELETE FROM frames WHERE camera=7 AND frame=2")
        stored.commit()
    with pytest.raises(ValueError, match="missing native cache"):
        replay.run(cfg, cache, tmp_path / "missing")
    assert not (tmp_path / "wrong").exists()
    assert not (tmp_path / "missing").exists()


def test_resume_checks_artifacts_configuration_cache_and_source(tmp_path, monkeypatch):
    cfg = config(tmp_path / "data")
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[]] * 5), 7: (10., 0., [[]] * 5)})
    first = replay.run(cfg, cache, tmp_path / "run")
    assert replay.run(cfg, cache, tmp_path / "run") == first
    changed = deepcopy(cfg)
    changed["association"]["confirmation_frames"] = 4
    with pytest.raises(ValueError, match="different signed experiment"):
        replay.run(changed, cache, tmp_path / "run")
    monkeypatch.setattr(replay, "source_signature", lambda: "changed")
    with pytest.raises(ValueError, match="different signed experiment"):
        replay.run(cfg, cache, tmp_path / "run")
    monkeypatch.undo()
    (tmp_path / "run" / "timeline.jsonl").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        replay.run(cfg, cache, tmp_path / "run")


def test_resume_rejects_changed_cache_bytes(tmp_path):
    cfg = config(tmp_path / "data")
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[]] * 5), 7: (10., 0., [[]] * 5)})
    replay.run(cfg, cache, tmp_path / "run")
    with DetectionCache(cache) as stored:
        stored.ensure_meta("added_provenance", {"changed": True})
    with pytest.raises(ValueError, match="different signed experiment"):
        replay.run(cfg, cache, tmp_path / "run")


def test_resume_rejects_tampered_manifest_options(tmp_path):
    cfg = config(tmp_path / "data")
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[]] * 5), 7: (10., 0., [[]] * 5)})
    replay.run(cfg, cache, tmp_path / "run")
    path = tmp_path / "run" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["options"]["clock"] = "changed description"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest options changed"):
        replay.run(cfg, cache, tmp_path / "run")


def test_real_scene_dictionary_contract_and_frozen_scene_signature(tmp_path):
    cfg = config(tmp_path / "data")
    cache = tmp_path / "cache.sqlite"
    make_cache(cache, cfg, {6: (10., 0., [[]] * 5), 7: (10., 0., [[]] * 5)})
    with DetectionCache(cache) as stored:
        scene = stored.get_meta("scene")
        # scene_metadata's real layout is a dictionary with string camera keys,
        # carrying full per-camera provenance rather than just a camera list.
        scene["cameras"] = {str(c): stored.get_meta(f"camera:{c}") for c in (6, 7)}
        scene["timestamps_sha256"] = "frozen-offsets"
        scene["framecounts_sha256"] = "frozen-counts"
        stored.db.execute("UPDATE metadata SET value=? WHERE key='scene'", (canonical(scene),))
        stored.commit()
    cfg["scene_signature"] = digest(scene)
    assert replay.run(cfg, cache, tmp_path / "run")["detector_calls"] == 10
    invalid = deepcopy(cfg)
    invalid["scene_signature"] = "different"
    with pytest.raises(ValueError, match="scene signature"):
        replay.run(invalid, cache, tmp_path / "wrong_scene")
    with DetectionCache(cache) as stored:
        metadata = stored.get_meta("camera:7")
        metadata["timestamp_offset_s"] = .2
        stored.db.execute("UPDATE metadata SET value=? WHERE key='camera:7'", (canonical(metadata),))
        stored.commit()
    with pytest.raises(ValueError, match="signed scene metadata"):
        replay.run(cfg, cache, tmp_path / "wrong_camera")


def test_fraction_clock_batches_mathematically_equal_timestamps():
    assert replay.native_time_us({"fps": 10., "timestamp_offset_s": .1}, 2) == 300_000
    assert replay.native_time_us({"fps": 8., "timestamp_offset_s": .05}, 2) == 300_000
