"""Causal evidence, identity immutability, and budget invariants for bounded handoffs."""

import heapq
import json

import numpy as np
import pytest

from rhd.bounded import BoundedCoordinator, validate_options
from rhd.cache import DetectionCache
from rhd.coordination import Scheduler
from rhd.evaluation import read_rows
from rhd.runner import run
from rhd.vision import CameraTracker


def config(**association):
    return {
        "fps": 15,
        "cameras": [1, 2],
        "seed": 7,
        "communication": {},
        "association": {
            "kind": "bounded", "confirmation_frames": 15,
            "minimum_observations": 3, "gallery_size": 5,
            "max_cosine_distance": .8, "ambiguity_margin": .05,
            "time_weight": .1, **association,
        },
        "routes": [{
            "source": 1, "destination": 2,
            "exit_polygon": [[.8, 0], [1, 0], [1, 1], [.8, 1]],
            "entry_polygon": [[0, 0], [.4, 0], [.4, 1], [0, 1]],
            "direction": [1, 0], "travel_time_s": [0, 2],
        }],
    }


def track(local=1, feature=(1., 0.), point=(.1, .5), age=0, raw=None):
    return {
        "local_id": local, "box": [10, 10, 50, 40], "point": list(point),
        "velocity": [.01, 0], "feature": list(feature),
        "observation_feature": list(feature if raw is None else raw),
        "score": .9, "age_since_detection": age, "uncertainty": 0.,
    }


def message(identity=101, serial=1, feature=(1., 0.), end=2.):
    return {
        "id": serial, "source": 1, "destination": 2, "exit_time_s": 0.,
        "begin_s": 0., "end_s": end, "global_id": identity,
        "feature": list(feature), "confidence": .9,
        "entry_polygon": [[0, 0], [.4, 0], [.4, 1], [0, 1]],
    }


def update(coord, now, tracks, camera=2):
    coord.set_active(camera, tracks)
    coord.update(camera, tracks, now)
    coord.resolve(now)


def test_late_delivered_message_can_resolve_an_already_pending_entrant():
    coord = BoundedCoordinator(config())
    heapq.heappush(coord.pending, (.2, 1, message()))
    for now in (0., .1):
        coord.deliver(now)
        update(coord, now, [track()])
        assert coord.assign(2, track(), now) is None
    coord.deliver(.2)
    update(coord, .2, [track()])
    assert coord.mapping[(2, 1)] == 101


def test_on_time_entry_can_confirm_after_travel_window_expires():
    coord = BoundedCoordinator(config())
    coord.available.append(message(end=.2))
    update(coord, .1, [track()])
    assert (2, 1) not in coord.mapping
    # Later observations can be outside the entry polygon: the saved actual
    # entry observation determines arrival compatibility, not prediction drift.
    for now in (.2, .3):
        coord.deliver(now)
        update(coord, now, [track(point=(.5, .5))])
    assert coord.mapping[(2, 1)] == 101


def test_only_actual_nonzero_raw_observations_count_toward_confirmation():
    coord = BoundedCoordinator(config())
    coord.available.append(message())
    update(coord, 0., [track()])
    for now, age in ((.1, 1), (.2, 2)):
        update(coord, now, [track(age=age)])
        assert (2, 1) not in coord.mapping
    # An old EMA remains nonzero, but this frame has no extracted descriptor.
    update(coord, .3, [track(raw=(0., 0.))])
    assert (2, 1) not in coord.mapping
    update(coord, .4, [track()])
    assert (2, 1) not in coord.mapping
    update(coord, .5, [track()])
    assert coord.mapping[(2, 1)] == 101


def test_ema_gallery_uses_smoothed_feature_only_on_actual_nonzero_raw_observations():
    coord = BoundedCoordinator(config(feature_mode="ema_observed"))
    coord.available.append(message())
    for now in (0., .1):
        update(coord, now, [track(feature=(1., 0.), raw=(0., 1.))])
    update(coord, .2, [track(feature=(1., 0.), raw=(0., 0.))])
    update(coord, .3, [track(feature=(1., 0.), raw=(0., 1.), age=1)])
    assert len(coord.states[(2, 1)]["gallery"]) == 2
    assert (2, 1) not in coord.mapping
    update(coord, .4, [track(feature=(1., 0.), raw=(0., 1.))])
    assert coord.mapping[(2, 1)] == 101


def test_exit_prototype_does_not_rank_by_a_lucky_historical_source_descriptor():
    coord = BoundedCoordinator(config(source_pooling="exit_prototype", minimum_observations=2,
                                     feature_mode="ema_observed", ambiguity_margin=0., time_weight=0.))
    misleading = message(feature=(.7, .7))
    misleading["gallery"] = [[1., 0.]]
    correct = message(identity=202, serial=2, feature=(.95, .312))
    coord.available.extend([misleading, correct])
    update(coord, 0., [track()])
    update(coord, .1, [track()])
    assert coord.mapping[(2, 1)] == 202


@pytest.mark.parametrize("reverse", [False, True])
def test_batch_matching_is_one_to_one_and_independent_of_track_list_order(reverse):
    coord = BoundedCoordinator(config(minimum_observations=1))
    coord.available.extend([message(), message(202, 2, (0., 1.))])
    observations = [track(1), track(2, (0., 1.))]
    update(coord, 0., list(reversed(observations)) if reverse else observations)
    assert coord.mapping == {(2, 1): 101, (2, 2): 202}
    assert len(set(coord.mapping.values())) == len(coord.mapping)


def test_competing_entrants_cannot_both_reuse_the_same_global_id():
    results = []
    for reverse in (False, True):
        coord = BoundedCoordinator(config(minimum_observations=1, ambiguity_margin=0.))
        coord.available.append(message())
        observations = [track(1, (1., .01)), track(2)]
        update(coord, 0., list(reversed(observations)) if reverse else observations)
        assert list(coord.mapping.values()).count(101) == 1
        results.append(dict(coord.mapping))
    assert results[0] == results[1]


def test_mature_entrant_cannot_consume_identity_while_better_competitor_lacks_evidence():
    coord = BoundedCoordinator(config())
    coord.available.append(message())
    mature = track(1, (.98, .2))
    update(coord, 0., [mature])
    update(coord, .1, [mature])
    better = track(2)
    update(coord, .2, [mature, better])
    # Track 1 has reached the evidence minimum; track 2 has only one actual
    # observation. Its better candidate still blocks premature consumption.
    assert not coord.mapping
    update(coord, .3, [mature, better])
    assert not coord.mapping
    update(coord, .4, [mature, better])
    assert coord.mapping.get((2, 1)) != 101
    assert coord.mapping[(2, 2)] == 101


def test_one_source_identity_cannot_commit_to_two_destination_cameras_in_one_tick():
    cfg = config(minimum_observations=1, ambiguity_margin=0.)
    cfg["cameras"].append(3)
    cfg["routes"].append({**cfg["routes"][0], "destination": 3})
    assignments = []
    for camera_order in ((2, 3), (3, 2)):
        coord = BoundedCoordinator(cfg)
        coord.available.extend([message(), {**message(serial=2), "destination": 3}])
        # All camera observations enter the batch before any identity resolves.
        for camera in camera_order:
            coord.update(camera, [track()], 0.)
        coord.resolve(0.)
        assert list(coord.mapping.values()).count(101) == 1
        assignments.append(dict(coord.mapping))
        coord.resolve(1.)
        assert len(coord.mapping) == 2
        assert len(set(coord.mapping.values())) == 2
    assert assignments[0] == assignments[1]


def test_confirmed_mapping_cannot_be_rewritten_by_later_better_evidence():
    coord = BoundedCoordinator(config(minimum_observations=1))
    coord.available.append(message())
    update(coord, 0., [track()])
    coord.available.append(message(202, 2, (0., 1.)))
    for now in (.1, .2, 1.1):
        update(coord, now, [track(feature=(0., 1.))])
        assert coord.assign(2, track(), now) == 101


def test_ambiguous_identity_gets_a_new_id_at_deadline_and_never_late_merges():
    coord = BoundedCoordinator(config(minimum_observations=1))
    coord.available.extend([message(), message(202, 2)])
    update(coord, 0., [track()])
    assert coord.assign(2, track(), 0.) is None
    coord.resolve(.99)
    assert (2, 1) not in coord.mapping
    coord.resolve(1.)
    identity = coord.mapping[(2, 1)]
    assert identity not in (101, 202)
    coord.available = [message()]
    update(coord, 1.1, [track()])
    assert coord.mapping[(2, 1)] == identity


def test_entry_after_travel_window_cannot_reuse_retained_message():
    coord = BoundedCoordinator(config(minimum_observations=1))
    coord.available.append(message(end=.2))
    update(coord, .3, [track()])
    assert coord.mapping.get((2, 1)) != 101
    coord.resolve(1.3)
    assert coord.mapping[(2, 1)] != 101


def test_early_entrant_cannot_become_eligible_by_remaining_in_zone_until_window_opens():
    cfg = config(confirmation_frames=30)
    cfg["routes"][0]["travel_time_s"] = [1., 2.]
    coord = BoundedCoordinator(cfg)
    coord.available.append({**message(), "begin_s": 1.})
    for now in (0., .1, .2, 1., 1.1, 1.2):
        update(coord, now, [track()])
        assert coord.mapping.get((2, 1)) != 101
    coord.resolve(2.)
    assert coord.mapping[(2, 1)] != 101


def test_message_delivered_after_confirmation_deadline_cannot_resolve_old_evidence():
    coord = BoundedCoordinator(config())
    heapq.heappush(coord.pending, (1.1, 1, message()))
    for now in (0., .1, .2):
        update(coord, now, [track()])
    coord.deliver(1.1)
    coord.resolve(1.1)
    assert coord.mapping[(2, 1)] != 101


def test_end_of_segment_flush_finalizes_without_future_identity_evidence():
    coord = BoundedCoordinator(config())
    update(coord, 0., [track()])
    assert (2, 1) not in coord.mapping
    coord.flush(.1)
    assert coord.mapping[(2, 1)] > 0
    committed = dict(coord.mapping)
    coord.available.append(message())
    update(coord, 1.1, [track()])
    assert coord.mapping == committed


def test_prediction_only_exit_does_not_publish_a_handoff_message():
    coord = BoundedCoordinator(config())
    source = track(point=(.7, .5))
    update(coord, 0., [source], camera=1)
    assert (1, 1) in coord.mapping  # No incoming route: no confirmation delay.
    predicted = track(point=(.9, .5), age=1)
    update(coord, .1, [predicted], camera=1)
    coord.observe(1, [predicted], .1)
    assert coord.messages == 0
    observed = track(point=(.9, .5))
    update(coord, .2, [observed], camera=1)
    coord.observe(1, [observed], .2)
    assert coord.messages == 1
    assert coord.pending[0][2]["exit_time_s"] == .2


def test_pending_exit_is_published_with_original_observed_exit_time_after_commit():
    cfg = config()
    cfg["cameras"].append(3)
    cfg["routes"].append({
        "source": 2, "destination": 3,
        "exit_polygon": [[0, 0], [.4, 0], [.4, 1], [0, 1]],
        "entry_polygon": [[0, 0], [.4, 0], [.4, 1], [0, 1]],
        "direction": [1, 0], "travel_time_s": [2, 3],
    })
    coord = BoundedCoordinator(cfg)
    coord.available.append(message())
    update(coord, 0., [track()])
    coord.observe(2, [track()], 0.)
    assert coord.messages == 0
    for now in (.1, .2):
        current = track(point=(.5, .5))
        update(coord, now, [current])
        coord.observe(2, [current], now)
    assert coord.mapping[(2, 1)] == 101
    assert coord.messages == 1
    outgoing = coord.pending[0][2]
    assert outgoing["global_id"] == 101
    assert outgoing["exit_time_s"] == 0.
    assert outgoing["begin_s"] == 2.
    assert coord.pending[0][0] >= .2 + 1 / cfg["fps"]


def test_tracker_exposes_current_raw_descriptor_without_reusing_ema_on_skips():
    tracker = CameraTracker(15, 320, 180, {})
    detection = {"box": [30, 40, 70, 64], "score": .9,
                 "class_id": 2, "feature": [1., 0.]}
    observed = tracker.step([detection], 0)[0]
    assert np.allclose(observed["observation_feature"], [1., 0.])
    predicted = tracker.step(None, 1)[0]
    assert np.linalg.norm(predicted["observation_feature"]) == 0
    zero_observation = tracker.step([{**detection, "feature": [0., 0.]}], 2)[0]
    assert np.linalg.norm(zero_observation["feature"]) > 0
    assert np.linalg.norm(zero_observation["observation_feature"]) == 0


def test_zero_identity_weight_has_exact_coordinated_decision_parity():
    coord = BoundedCoordinator(config())
    coord.available.append(message())
    update(coord, 0., [track()])
    options = {"max_detection_gap": 6, "handoff_weight": .25, "identity_weight": 0.}
    current = {1: [], 2: [track(age=1)]}
    coordinated = Scheduler([1, 2], "coordinated", .5, 1, options)
    identity = Scheduler([1, 2], "identity", .5, 1, options)
    for tick in range(60):
        now = tick / 15
        assert coordinated.choose(tick, current, coord, now) == identity.choose(tick, current, coord, now)
    assert coordinated.calls == identity.calls


@pytest.mark.parametrize("budget", [.25, .5, .75])
def test_identity_priority_preserves_the_exact_shared_detector_call_budget(budget):
    class Hints:
        @staticmethod
        def arrival_priority(camera, now):
            return float(camera == 2)

        @staticmethod
        def identity_priority(camera, now):
            return .95 if camera == 1 else .05

    options = {"max_detection_gap": 6, "handoff_weight": .25, "identity_weight": 2.}
    identity = Scheduler([1, 2], "identity", budget, 1, options)
    independent = Scheduler([1, 2], "independent", budget, 1, options)
    tracks = {1: [], 2: []}
    for tick in range(101):
        identity.choose(tick, tracks, Hints(), tick / 15)
        independent.choose(tick, tracks, Hints(), tick / 15)
    assert identity.calls == independent.calls == 2 + int(100 * 2 * budget)


def test_identity_priority_is_finite_bounded_and_zero_after_confirmation():
    coord = BoundedCoordinator(config())
    coord.available.extend([message(), message(202, 2)])
    update(coord, 0., [track()])
    for now in (0., .1, .5, .99):
        value = coord.identity_priority(2, now)
        assert np.isfinite(value) and 0. <= value <= 1.
    coord.resolve(1.)
    assert coord.identity_priority(2, 1.) == 0.
    assert coord.identity_priority(1, 1.) == 0.


def test_competing_full_evidence_entrants_still_have_nonzero_identity_priority():
    coord = BoundedCoordinator(config(minimum_observations=1))
    coord.available.append(message())
    update(coord, 0., [track(1, (1., .01)), track(2)])
    assert not coord.mapping  # Column ambiguity blocks both near-tied entrants.
    assert coord.identity_priority(2, .5) > 0.


def small_cache(path):
    with DetectionCache(path) as cache:
        cache.ensure_meta("detector", {"kind": "synthetic_detector_responses"})
        for camera in (1, 2):
            cache.ensure_meta(f"camera:{camera}", {
                "fps": 15, "frames": 10, "width": 320, "height": 180,
            })
            for frame in range(10):
                visible = frame < 3 if camera == 1 else frame >= 3
                box = [270, 40, 310, 70] if camera == 1 else [10, 40, 50, 70]
                detections = [{"box": box, "score": .9, "class_id": 2,
                               "feature": [1., 0.]}] if visible else []
                cache.add_frame(camera, frame, detections, detector_ms=0., appearance_ms=0.)
            cache.commit()


def test_buffered_global_rows_have_one_bounded_availability_event_and_causal_prefix(tmp_path):
    cfg = config()
    cfg.update({"start_frame": 0, "stop_frame": 10,
                "tracker": {"match_thresh": .9}, "scheduler": {"max_detection_gap": 6}})
    cfg["routes"][0]["direction"] = [0, 0]
    cache_path = tmp_path / "detections.sqlite"
    small_cache(cache_path)
    first = tmp_path / "first"
    run(cfg, cache_path, first)

    def events(directory):
        return [json.loads(line) for line in (directory / "availability.jsonl").read_text().splitlines()]

    def validate(directory):
        records = events(directory)
        predictions = read_rows(directory / "predictions.txt")
        assert len(records) == len(predictions)
        keys = [(r["camera"], r["global_id"], r["frame"]) for r in records]
        assert len(keys) == len(set(keys))
        assert set(keys) == {(r["camera"], r["id"], r["frame"]) for r in predictions}
        for record in records:
            delay = record["published_at_s"] - record["frame"] / cfg["fps"]
            assert -1e-10 <= delay <= cfg["association"]["confirmation_frames"] / cfg["fps"] + 1e-10
        return records

    first_records = validate(first)
    assert any(r["camera"] == 2 and r["published_at_frame"] > r["frame"] for r in first_records)
    # Alter only future detections. Buffered earlier observations may eventually
    # resolve differently, but observations already published by the cutoff are
    # immutable and cannot depend on those later boxes/descriptors.
    with DetectionCache(cache_path) as cache:
        rows = cache.db.execute("SELECT camera,frame,payload FROM frames WHERE frame>=5").fetchall()
        for camera, frame, payload in rows:
            detections = json.loads(payload)
            for detection in detections:
                detection["box"] = [200, 80, 240, 110]
                detection["feature"] = [-1., 0.]
            cache.db.execute("UPDATE frames SET payload=? WHERE camera=? AND frame=?",
                             (json.dumps(detections), camera, frame))
        cache.commit()
    second = tmp_path / "future_changed"
    run(cfg, cache_path, second)
    second_records = validate(second)
    cutoff = 4 / cfg["fps"]
    assert [r for r in first_records if r["published_at_s"] <= cutoff + 1e-10] == [
        r for r in second_records if r["published_at_s"] <= cutoff + 1e-10]


@pytest.mark.parametrize("wait", [0, 2, 4, 8])
def test_same_kernel_checks_current_evidence_before_zero_or_positive_deadline(wait):
    coord = BoundedCoordinator(config(confirmation_frames=wait, minimum_observations=1,
                                     ambiguity_margin=0., time_weight=0., diagnostics=True,
                                     source_retention_frames=8))
    coord.available.append(message())
    update(coord, 0., [track()])
    assert coord.mapping[(2, 1)] == 101
    commit = coord.commits[(2, 1)]
    assert commit["time_s"] == 0.
    assert commit["diagnostics"]["confirmation_outcome"] == (
        "deadline_accepted" if wait == 0 else "early_accepted")
    assert commit["diagnostics"]["decision_status"] == "accepted"
    assert commit["diagnostics"]["legal_candidate_count"] == 1
    assert commit["diagnostics"]["observations"] == 1


def test_zero_window_does_not_manufacture_two_observations_from_one_tick():
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=2,
                                     ambiguity_margin=0., time_weight=0., diagnostics=True))
    coord.available.append(message())
    coord.update(2, [track()], 0.)
    coord.update(2, [track()], 0.)
    coord.resolve(0.)
    assert coord.mapping[(2, 1)] != 101
    details = coord.commits[(2, 1)]["diagnostics"]
    assert details["confirmation_outcome"] == "deadline_fresh"
    assert details["decision_status"] == "insufficient_observations"
    assert details["observations"] == details["legal_candidate_count"] == 1
    assert details["evidence_shortfall"] == 1


def test_zero_window_cannot_use_a_next_tick_message_or_rewrite_a_fresh_identity():
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=1,
                                     ambiguity_margin=0., time_weight=0., diagnostics=True))
    heapq.heappush(coord.pending, (1 / coord.fps, 1, message()))
    coord.deliver(0.)
    update(coord, 0., [track()])
    identity = coord.mapping[(2, 1)]
    assert identity != 101
    details = coord.commits[(2, 1)]["diagnostics"]
    assert details["confirmation_outcome"] == "deadline_fresh"
    assert details["decision_status"] == "no_legal_candidate"
    coord.deliver(1 / coord.fps)
    update(coord, 1 / coord.fps, [track(feature=(0., 1.))])
    assert coord.available[0]["global_id"] == 101
    assert coord.mapping[(2, 1)] == identity
    assert len(coord.commits) == 1


def test_zero_window_runner_publishes_current_global_rows_without_confirmation_delay(tmp_path):
    cfg = config(confirmation_frames=0, minimum_observations=1,
                 source_retention_frames=8, diagnostics=True,
                 ambiguity_margin=0., time_weight=0.)
    cfg.update({"start_frame": 0, "stop_frame": 10,
                "tracker": {"match_thresh": .9}, "scheduler": {"max_detection_gap": 6}})
    cfg["routes"][0]["direction"] = [0, 0]
    cache_path = tmp_path / "zero_window.sqlite"
    small_cache(cache_path)
    output = tmp_path / "zero_window"
    summary = run(cfg, cache_path, output)
    events = [json.loads(line) for line in (output / "availability.jsonl").read_text().splitlines()]
    assert events
    assert any(e["reason"] == "confirmed_handoff" for e in events)
    assert all(e["published_at_frame"] == e["frame"] for e in events)
    assert all(abs(e["published_at_s"] - e["frame"] / cfg["fps"]) < 1e-10 for e in events)
    assert summary["bounded_confirmation"]["maximum_pending_lifetime_s"] == 0.
    assert summary["bounded_confirmation"]["deadline_violations"] == 0


@pytest.mark.parametrize("wait", [0, 2, 4, 8])
def test_explicit_source_retention_is_identical_for_every_confirmation_window(wait):
    coord = BoundedCoordinator(config(confirmation_frames=wait, source_retention_frames=8))
    coord.available.append(message(end=.2))
    cutoff = .2 + 8 / coord.fps
    coord.deliver(cutoff)
    assert len(coord.available) == 1
    coord.deliver(cutoff + 1e-6)
    assert coord.available == []


def test_omitted_source_retention_keeps_original_window_dependent_expiry():
    coord = BoundedCoordinator(config(confirmation_frames=2))
    coord.available.append(message(end=.2))
    coord.deliver(.2 + 2 / coord.fps)
    assert len(coord.available) == 1
    coord.deliver(.2 + 2 / coord.fps + 1e-6)
    assert coord.available == []


def test_fixed_retention_cannot_admit_an_actual_arrival_after_the_travel_window():
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=1,
                                     source_retention_frames=8, diagnostics=True))
    coord.available.append(message(end=.2))
    coord.deliver(.3)
    assert len(coord.available) == 1
    update(coord, .3, [track()])
    assert coord.mapping[(2, 1)] != 101
    assert coord.commits[(2, 1)]["diagnostics"]["legal_candidate_count"] == 0


def test_zero_window_identity_priority_is_defined_before_resolve_and_zero_after_commit():
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=1,
                                     ambiguity_margin=.05))
    coord.available.extend([message(), message(identity=202, serial=2)])
    coord.update(2, [track()], 0.)
    state = coord.states[(2, 1)]
    # An external caller may request priority between update and resolve. The
    # normal runner resolves within the tick; neither API path divides by zero.
    state["candidates"] = coord._candidates((2, 1), state)
    assert coord.identity_priority(2, 0.) == 1.
    coord.resolve(0.)
    assert coord.identity_priority(2, 0.) == 0.


def test_diagnostics_distinguish_deadline_acceptance_and_terminal_fallback():
    accepted = BoundedCoordinator(config(confirmation_frames=2, minimum_observations=2,
                                        ambiguity_margin=0., time_weight=0., diagnostics=True))
    accepted.available.append(message())
    update(accepted, 0., [track()])
    update(accepted, 2 / accepted.fps, [track()])
    details = accepted.commits[(2, 1)]["diagnostics"]
    assert details["confirmation_outcome"] == "deadline_accepted"
    assert details["decision_status"] == "accepted"
    assert details["evidence_shortfall"] == 0

    terminal = BoundedCoordinator(config(confirmation_frames=8, minimum_observations=2,
                                        diagnostics=True))
    terminal.available.append(message())
    update(terminal, 0., [track()])
    heapq.heappush(terminal.pending, (.2, 2, message(identity=202, serial=2)))
    terminal.flush(.1)
    commit = terminal.commits[(2, 1)]
    assert commit["time_s"] == 8 / terminal.fps
    assert commit["diagnostics"]["confirmation_outcome"] == "interval_end_fresh"
    assert commit["diagnostics"]["decision_status"] == "insufficient_observations"
    assert commit["diagnostics"]["last_evaluated_at_s"] == 0.
    assert len(terminal.pending) == 1  # Flush did not consume future messages.


def test_terminal_diagnostics_leave_unevaluated_candidates_unknown():
    coord = BoundedCoordinator(config(diagnostics=True))
    coord.update(2, [track()], 0.)
    coord.flush(.1)
    details = coord.commits[(2, 1)]["diagnostics"]
    assert details["legal_candidate_count"] is None
    assert details["last_evaluated_at_s"] is None
    assert details["decision_status"] == "insufficient_observations"


def test_diagnostics_record_row_margin_rejection_without_claiming_gt_correctness():
    coord = BoundedCoordinator(config(confirmation_frames=2, minimum_observations=1,
                                     ambiguity_margin=.05, time_weight=0., diagnostics=True))
    coord.available.extend([message(), message(identity=202, serial=2)])
    update(coord, 0., [track()])
    assert not coord.mapping
    coord.resolve(2 / coord.fps)
    details = coord.commits[(2, 1)]["diagnostics"]
    assert details["confirmation_outcome"] == "deadline_fresh"
    assert details["decision_status"] == "competition_rejected"
    assert details["blocking_checks"] == ["row_margin"]
    assert details["row_gap"] == 0.
    assert details["column_gap"] is None
    assert details["legal_candidate_count"] == 2


def test_diagnostics_include_an_immature_competitor_blocking_a_mature_entrant():
    coord = BoundedCoordinator(config(confirmation_frames=2, minimum_observations=2,
                                     ambiguity_margin=.05, time_weight=0., diagnostics=True))
    coord.available.append(message())
    mature = track(1, (.98, .2))
    update(coord, 0., [mature])
    update(coord, 1 / coord.fps, [mature, track(2)])
    assert not coord.mapping
    coord.resolve(2 / coord.fps)
    details = coord.commits[(2, 1)]["diagnostics"]
    assert details["decision_status"] == "competition_rejected"
    assert details["blocking_checks"] == ["column_margin"]
    assert details["column_gap"] < 0
    assert details["competing_entrants"] == 1
    assert (2, 2) not in coord.mapping


def test_diagnostics_distinguish_a_batch_collision_from_a_screening_rejection():
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=1,
                                     ambiguity_margin=0., time_weight=0., diagnostics=True))
    coord.available.append(message())
    update(coord, 0., [track(1), track(2)])
    accepted = [c for c in coord.commits.values() if c["reason"] == "confirmed_handoff"]
    fallback = [c for c in coord.commits.values() if c["reason"] == "confirmation_deadline"]
    assert len(accepted) == len(fallback) == 1
    details = fallback[0]["diagnostics"]
    assert details["decision_status"] == "unassigned_after_batch"
    assert details["competing_entrants"] == 1
    assert details["blocking_checks"] == []


def test_opt_in_diagnostics_do_not_change_identity_or_message_behavior():
    controls = [BoundedCoordinator(config(minimum_observations=1, time_weight=0.,
                                         ambiguity_margin=0., **options))
                for options in ({}, {"diagnostics": True, "source_retention_frames": 15})]
    for coord in controls:
        coord.available.append(message())
        update(coord, 0., [track(local=7)])
        source = track(local=1, point=(.9, .5))
        update(coord, .1, [source], camera=1)
        coord.observe(1, [source], .1)
        coord.deliver(.2)
    assert controls[0].mapping == controls[1].mapping
    assert controls[0].pending == controls[1].pending
    assert controls[0].available == controls[1].available
    stripped = [{k: v for k, v in record.items() if k != "diagnostics"}
                for record in controls[1].log]
    assert controls[0].log == stripped
    outside = controls[1].commits[(1, 1)]["diagnostics"]
    assert outside["confirmation_outcome"] == "outside_entry"
    assert outside["legal_candidate_count"] is None
    assert all("diagnostics" not in record for record in controls[0].log)


@pytest.mark.parametrize("option,value", [
    ("confirmation_frames", -1), ("confirmation_frames", 1.5), ("confirmation_frames", True),
    ("source_retention_frames", -1), ("source_retention_frames", 1.5),
    ("source_retention_frames", None), ("source_retention_frames", True),
    ("diagnostics", "true"), ("diagnostics", 1),
])
def test_new_window_and_diagnostic_options_reject_invalid_types_or_values(option, value):
    with pytest.raises(ValueError):
        validate_options(config(**{option: value})["association"])


@pytest.mark.parametrize("gate", [
    "used_message", "other_destination", "delivered_after_deadline", "active_identity",
    "no_entry_observation", "travel_window", "source_descriptor_incompatible",
    "source_descriptor_missing", "appearance_distance",
])
def test_candidate_diagnostics_count_only_the_first_rejected_gate(gate):
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=1,
                                     diagnostics=True, ambiguity_margin=0., time_weight=0.))
    item = message()
    if gate == "used_message":
        coord.used.add(1)
        item["destination"] = 1  # The later gate must not also be counted.
    elif gate == "other_destination":
        item["destination"] = 1
    elif gate == "delivered_after_deadline":
        item["delivered_at_s"] = 1 / coord.fps
    elif gate == "active_identity":
        coord.mapping[(1, 9)] = 101
        coord.set_active(1, [track(local=9)])
    elif gate == "no_entry_observation":
        item["entry_polygon"] = [[.5, 0], [1, 0], [1, 1], [.5, 1]]
    elif gate == "travel_window":
        item["begin_s"] = 1.
    elif gate == "source_descriptor_incompatible":
        item["feature"] = [1., 0., 0.]
    elif gate == "source_descriptor_missing":
        item["feature"] = [0., 0.]
    elif gate == "appearance_distance":
        item["feature"] = [0., 1.]
    coord.available.append(item)
    update(coord, 0., [track()])
    details = coord.commits[(2, 1)]["diagnostics"]["candidate_gates"]
    assert details["scanned_messages"] == 1
    assert details["legal_message_edges"] == details["eligible_distinct_ids"] == 0
    assert details["first_rejection_counts"][gate] == 1
    assert sum(details["first_rejection_counts"].values()) == 1


def test_candidate_diagnostics_separate_legal_messages_from_distinct_identities():
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=1,
                                     diagnostics=True, ambiguity_margin=0., time_weight=0.))
    coord.available.extend([message(), message(serial=2, feature=(.99, .1))])
    update(coord, 0., [track()])
    details = coord.commits[(2, 1)]["diagnostics"]["candidate_gates"]
    assert details["scanned_messages"] == details["legal_message_edges"] == 2
    assert details["eligible_distinct_ids"] == 1
    assert sum(details["first_rejection_counts"].values()) == 0


def test_candidate_diagnostics_do_not_invent_gate_failures_without_destination_evidence():
    coord = BoundedCoordinator(config(confirmation_frames=0, minimum_observations=1,
                                     diagnostics=True))
    coord.available.append(message())
    update(coord, 0., [track(raw=(0., 0.))])
    details = coord.commits[(2, 1)]["diagnostics"]["candidate_gates"]
    assert details["destination_gallery_observations"] == details["scanned_messages"] == 0
    assert sum(details["first_rejection_counts"].values()) == 0
