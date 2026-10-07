"""Logged decisions and GT endpoint truth remain separate in offline reports."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/deadline_diagnostics_v0_4.py"
SPEC = importlib.util.spec_from_file_location("deadline_diagnostics_v0_4", SCRIPT)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


def config(**options):
    return {"fps": 10., "start_frame": 0, "stop_frame": 100, "cameras": [1, 2],
            "association": {"kind": "bounded", "minimum_observations": 2},
            "routes": [{"source": 1, "destination": 2}], **options}


def commit(camera, local, gid, reason, now, first=None, deadline=None, **extra):
    row = {"event": "identity_committed", "camera": camera, "local_id": local,
           "global_id": gid, "reason": reason, "time_s": now, **extra}
    if first is not None:
        row.update(first_time_s=first, deadline_s=deadline)
    return row


def make_run(tmp_path, commits, observations):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "messages.jsonl").write_text("".join(json.dumps(e) + "\n" for e in commits))
    by_key = {(e["camera"], e["local_id"]): e for e in commits}
    events, predictions = [], []
    for camera, local, gid, frame in observations:
        event = by_key.get((camera, local), {})
        events.append({"camera": camera, "local_id": local, "global_id": gid,
                       "frame": frame, "published_at_s": max(frame / 10., event.get("time_s", 0)),
                       "reason": event.get("reason", "immediate")})
        predictions.append(f"{camera} {gid} {frame} 10 10 10 10 -1 -1\n")
    (folder / "availability.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (folder / "predictions.txt").write_text("".join(predictions))
    return folder


def gt_file(tmp_path, rows):
    path = tmp_path / "gt.txt"
    path.write_text("".join(f"{cam} {identity} {frame} 10 10 10 10 -1 -1\n"
                            for cam, identity, frame in rows))
    return path


def test_legacy_outcomes_separate_early_deadline_and_fresh_without_inventing_causes(tmp_path):
    commits = [commit(2, 1, 101, "confirmed_handoff", .1, 0., .2, observations=2),
               commit(2, 2, 102, "confirmed_handoff", .3, .1, .3, observations=2),
               commit(2, 3, 103, "confirmation_deadline", .4, .2, .4, observations=1),
               commit(2, 4, 104, "end_of_segment", .5, .3, .5, observations=1),
               commit(1, 5, 105, "outside_entry_zone", 0.)]
    run = make_run(tmp_path, commits, [(2, 1, 101, 0), (2, 2, 102, 1),
                                     (2, 3, 103, 2), (2, 4, 104, 3), (1, 5, 105, 0)])
    report = diagnostics.build_report(run, config())
    counts = report["populations"]["all_scored_local_tracks"]["outcomes"]
    assert [counts[k] for k in ["early_accepted", "deadline_accepted", "deadline_fresh",
                               "interval_end_fresh", "outside_entry"]] == [1] * 5
    pending = report["populations"]["all_scored_pending_tracks"]
    assert pending["decision_status"] == {"not_logged": 4}
    assert pending["logged_conditions_nonexclusive"]["positive_evidence_shortfall"] == 0
    assert report["reference_routes"] is None


def test_explicit_diagnostics_report_overlapping_conditions_and_competition(tmp_path):
    record = commit(2, 1, 1, "confirmation_deadline", .2, 0., .2, observations=1,
                    diagnostics={"confirmation_outcome": "deadline_fresh",
                                 "decision_status": "insufficient_observations",
                                 "evidence_shortfall": 1, "legal_candidate_count": 0,
                                 "assignment_eligible": False, "blocking_checks": ["column_margin"],
                                 "candidate_gates": {"scanned_messages": 3, "legal_message_edges": 0,
                                                      "eligible_distinct_ids": 0,
                                                      "first_rejection_counts": {"travel_window": 2,
                                                                                 "used_message": 1}},
                                 "last_evaluated_at_s": .2})
    run = make_run(tmp_path, [record], [(2, 1, 1, 0)])
    report = diagnostics.build_report(run, config())
    pending = report["populations"]["all_scored_pending_tracks"]
    assert pending["decision_status"] == {"insufficient_observations": 1}
    assert pending["logged_conditions_nonexclusive"] == {
        "positive_evidence_shortfall": 1, "zero_legal_candidates": 1, "assignment_ineligible": 1}
    assert pending["blocking_checks"] == {"column_margin": 1}
    assert pending["last_candidate_evaluation"]["first_rejection_counts"] == {
        "travel_window": 2, "used_message": 1}


def test_accepted_identity_is_not_assumed_ground_truth_correct(tmp_path):
    commits = [commit(1, 11, 1, "outside_entry_zone", 0.),
               commit(2, 22, 2, "confirmed_handoff", 1.1, 1., 1.2, observations=2)]
    run = make_run(tmp_path, commits, [(1, 11, 1, 0), (1, 11, 1, 1),
                                     (2, 22, 2, 10), (2, 22, 2, 11)])
    gt = gt_file(tmp_path, [(1, 77, 0), (1, 77, 1), (2, 77, 10), (2, 77, 11)])
    report = diagnostics.build_report(run, config(), gt)
    route = report["reference_routes"][0]
    assert route["reference_handoffs"] == 1 and route["broken"] == 1
    assert route["accepted_broken"] == 1 and route["accepted_confirmed"] == 0
    assert report["reference_handoffs"][0]["destination_prediction_box_xyxy"] == [10., 10., 20., 20.]
    assert report["populations"]["reference_destination_matched_pending_tracks"]["tracks"] == 1


def test_intermediate_camera_cannot_invent_a_pair_handoff(tmp_path):
    run = make_run(tmp_path, [commit(1, 1, 1, "outside_entry_zone", 0.),
                             commit(2, 2, 1, "confirmed_handoff", .5, .4, .6)],
                   [(1, 1, 1, 0), (1, 1, 1, 1), (2, 2, 1, 4), (2, 2, 1, 5)])
    gt = gt_file(tmp_path, [(1, 77, 0), (1, 77, 1), (3, 77, 2),
                            (3, 77, 3), (2, 77, 4), (2, 77, 5)])
    report = diagnostics.build_report(run, config(), gt)
    assert report["reference_handoffs"] == []
    assert report["reference_routes"][0]["reference_handoffs"] == 0
    assert report["reference_routes"][0]["endpoint_recall"] is None


def test_other_camera_duplicate_boxes_only_contribute_topological_presence(tmp_path):
    run = make_run(tmp_path, [commit(1, 1, 1, "outside_entry_zone", 0.),
                             commit(2, 2, 1, "confirmed_handoff", .5, .4, .6)],
                   [(1, 1, 1, 0), (2, 2, 1, 4)])
    gt = gt_file(tmp_path, [(1, 77, 0), (3, 77, 2), (2, 77, 4)])
    with gt.open("a") as stream:
        stream.write("3 77 2 100 100 20 20 -1 -1\n")
    report = diagnostics.build_report(run, config(), gt)
    assert report["reference_handoffs"] == []
    assert report["validation"]["non_evaluated_camera_duplicate_topology_rows"] == 1
    with gt.open("a") as stream:
        stream.write("2 77 4 100 100 20 20 -1 -1\n")
    with pytest.raises(ValueError, match="duplicate"):
        diagnostics.build_report(run, config(), gt)


def test_scored_arrival_can_use_a_source_endpoint_from_warmup(tmp_path):
    run = make_run(tmp_path, [commit(1, 1, 1, "outside_entry_zone", .1),
                             commit(2, 2, 1, "confirmed_handoff", 1.2, 1.1, 1.3)],
                   [(1, 1, 1, 1), (2, 2, 1, 11)])
    gt = gt_file(tmp_path, [(1, 77, 1), (2, 77, 11)])
    report = diagnostics.build_report(run, config(), gt, score_start_frame=10, score_stop_frame=20)
    assert report["populations"]["all_scored_local_tracks"]["tracks"] == 1
    assert report["reference_routes"][0]["confirmed"] == 1
    assert report["reference_routes"][0]["source_endpoint_in_warmup"] == 1
    assert report["reference_handoffs"][0]["source_endpoint_in_warmup"] is True


def test_rewritten_global_id_is_rejected(tmp_path):
    run = make_run(tmp_path, [commit(2, 2, 1, "outside_entry_zone", 0.)], [(2, 2, 1, 0)])
    with (run / "availability.jsonl").open("a") as stream:
        stream.write(json.dumps({"camera": 2, "local_id": 2, "global_id": 2,
                                 "frame": 1, "published_at_s": .1}) + "\n")
    with pytest.raises(ValueError, match="rewritten"):
        diagnostics.build_report(run, config())


def test_reports_reuse_identical_evidence_but_preserve_changed_outputs(tmp_path):
    run = make_run(tmp_path, [commit(2, 2, 1, "outside_entry_zone", 0.)], [(2, 2, 1, 0)])
    report = diagnostics.build_report(run, config())
    output = tmp_path / "diagnostics"
    hashes = diagnostics.write_report(report, output)
    assert diagnostics.write_report(report, output) == hashes
    changed = copy.deepcopy(report)
    changed["scope"]["score_start_frame"] = 1
    with pytest.raises(ValueError, match="Refusing to replace"):
        diagnostics.write_report(changed, output)
    assert diagnostics.write_report(report, output) == hashes
