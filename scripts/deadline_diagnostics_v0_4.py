"""Offline confirmation diagnostics from immutable predictions and event logs.

Acceptance is an algorithm decision, not a ground-truth correctness label. Old
logs support outcome/timing counts; unlogged candidate rejection causes remain
unknown. Ground truth is optional and is used only after inference.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import math
from pathlib import Path

import numpy as np

from rhd.common import canonical, digest, file_hash
from rhd.evaluation import read_rows, spatial_matches, transitions

TOLERANCE_S = 1e-9
OUTCOMES = (
    "early_accepted", "deadline_accepted", "deadline_fresh", "interval_end_fresh",
    "outside_entry", "accepted_after_deadline", "other_pending", "other_nonpending",
    "commit_not_logged",
)


def _jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: expected an object")
                yield value


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field}: expected a finite number")
    return float(value)


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field}: expected an integer >= {minimum}")
    return value


def _distribution(values):
    return {"count": len(values),
            "p50_s": float(np.percentile(values, 50)) if values else None,
            "p95_s": float(np.percentile(values, 95)) if values else None,
            "max_s": float(max(values)) if values else None}


def _outcome(commit):
    if commit is None:
        return "commit_not_logged"
    pending = "first_time_s" in commit
    reason = commit.get("reason")
    if reason == "confirmed_handoff":
        if not pending:
            raise ValueError("A confirmed_handoff commit lacks its pending time/deadline")
        delta = commit["time_s"] - commit["deadline_s"]
        return ("early_accepted" if delta < -TOLERANCE_S else
                "deadline_accepted" if delta <= TOLERANCE_S else "accepted_after_deadline")
    if reason == "confirmation_deadline" and pending:
        return "deadline_fresh"
    if reason == "end_of_segment" and pending:
        return "interval_end_fresh"
    if reason == "outside_entry_zone" and not pending:
        return "outside_entry"
    return "other_pending" if pending else "other_nonpending"


def _population(records):
    records = list(records)
    counts = Counter(r["confirmation_outcome"] for r in records)
    pending = [r for r in records if r["pending"] is True]
    gates = [r["candidate_gates"] for r in pending if r["candidate_gates"] is not None]
    gate_rejections = Counter()
    for gate in gates:
        gate_rejections.update(gate.get("first_rejection_counts", {}))
    return {
        "tracks": len(records), "pending_tracks": len(pending),
        "outcomes": {name: counts[name] for name in OUTCOMES},
        "pending_wait": _distribution([r["wait_s"] for r in pending]),
        "decision_status": dict(sorted(Counter(r["decision_status"] for r in pending).items())),
        "blocking_checks": dict(sorted(Counter(
            check for r in pending for check in r["blocking_checks"]).items())),
        "pending_without_diagnostic_fields": sum(not r["diagnostics_logged"] for r in pending),
        "logged_conditions_nonexclusive": {
            "positive_evidence_shortfall": sum(r["evidence_shortfall"] is not None and
                                              r["evidence_shortfall"] > 0 for r in pending),
            "zero_legal_candidates": sum(r["legal_candidate_count"] == 0 for r in pending),
            "assignment_ineligible": sum(r["assignment_eligible"] is False for r in pending),
        },
        "last_candidate_evaluation": {
            "tracks_with_logged_gates": len(gates),
            "scanned_messages": sum(g.get("scanned_messages", 0) for g in gates),
            "legal_message_edges": sum(g.get("legal_message_edges", 0) for g in gates),
            "eligible_distinct_ids": sum(g.get("eligible_distinct_ids", 0) for g in gates),
            "first_rejection_counts": dict(sorted(gate_rejections.items())),
        },
    }


def _endpoint(mapping, identity, segment, source):
    if source:
        frames = range(segment[1], max(segment[0] - 1, segment[1] - 15), -1)
    else:
        frames = range(segment[0], min(segment[1] + 1, segment[0] + 15))
    for frame in frames:
        local_id = mapping.get((segment[2], frame, identity))
        if local_id is not None:
            return segment[2], local_id, frame
    return None


def _ground_truth(path, cameras, start, stop):
    # Boxes in scored cameras remain subject to the evaluator's strict duplicate
    # rejection. Topology needs presence, not boxes, in non-evaluated cameras.
    scored_rows = read_rows(path, cameras=sorted(cameras), start=start, stop=stop)
    topology, seen, duplicate_other_presence = [], set(), 0
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            # read_rows above validates the entire input syntax and numeric boxes.
            values = line.replace(",", " ").split()
            camera, identity, frame = (int(float(v)) for v in values[:3])
            if not start <= frame < stop:
                continue
            key = camera, identity, frame
            if key in seen:
                if camera not in cameras:
                    duplicate_other_presence += 1
                continue
            seen.add(key)
            topology.append({"camera": camera, "id": identity, "frame": frame})
    return scored_rows, topology, duplicate_other_presence


def build_report(run_directory, config, ground_truth=None, *, score_start_frame=None,
                 score_stop_frame=None):
    """Read a completed replay and return JSON-serializable diagnostic evidence.

    Commit cohort: first filtered local row inside the scored interval. Reference
    cohort: real all-camera consecutive GT transitions whose destination starts in
    the scored interval, source endpoint is within replay, and route is configured.
    Thus a source endpoint may occur during warmup. Original GT transitions are
    constructed before selecting cameras/routes; omitted intermediate cameras
    cannot create new transitions.
    """
    run_directory = Path(run_directory)
    cfg = dict(config)
    fps = _number(cfg["fps"], "fps")
    if fps <= 0:
        raise ValueError("fps must be positive")
    replay_start = _integer(cfg["start_frame"], "start_frame")
    replay_stop = _integer(cfg["stop_frame"], "stop_frame")
    score_start = _integer(score_start_frame if score_start_frame is not None else
                           cfg.get("score_start_frame", replay_start), "score_start_frame")
    score_stop = _integer(score_stop_frame if score_stop_frame is not None else
                          cfg.get("score_stop_frame", replay_stop), "score_stop_frame")
    if not replay_start <= score_start < score_stop <= replay_stop:
        raise ValueError("The nonempty scoring interval must lie inside the replay interval")
    cameras = set(cfg["cameras"])
    paths = {name: run_directory / name for name in
             ("messages.jsonl", "availability.jsonl", "predictions.txt")}
    commits = {}
    for event in _jsonl(paths["messages.jsonl"]):
        if event.get("event") != "identity_committed":
            continue
        key = (_integer(event.get("camera"), "commit camera", 1),
               _integer(event.get("local_id"), "commit local_id", 1))
        _integer(event.get("global_id"), "commit global_id", 1)
        event["time_s"] = _number(event.get("time_s"), "commit time_s")
        if ("first_time_s" in event) != ("deadline_s" in event):
            raise ValueError("A pending commit must log both first_time_s and deadline_s")
        if "first_time_s" in event:
            for name in ("first_time_s", "deadline_s"):
                event[name] = _number(event[name], name)
            if event["time_s"] < event["first_time_s"] - TOLERANCE_S:
                raise ValueError("Commit precedes pending identity start")
            if event["deadline_s"] < event["first_time_s"] - TOLERANCE_S:
                raise ValueError("Deadline precedes pending identity start")
        if key in commits:
            raise ValueError("Duplicate identity commit for one local track")
        commits[key] = event

    availability, first_frames, global_ids = {}, {}, {}
    for event in _jsonl(paths["availability.jsonl"]):
        camera = _integer(event.get("camera"), "availability camera", 1)
        local_id = _integer(event.get("local_id"), "availability local_id", 1)
        global_id = _integer(event.get("global_id"), "availability global_id", 1)
        frame = _integer(event.get("frame"), "availability frame")
        if camera not in cameras or not replay_start <= frame < replay_stop:
            raise ValueError("Availability row outside the replay scope")
        published = _number(event.get("published_at_s"), "published_at_s")
        if published < frame / fps - TOLERANCE_S:
            raise ValueError("An output is published before its observation")
        key, row_key = (camera, local_id), (camera, global_id, frame)
        if row_key in availability:
            raise ValueError("Duplicate global availability row")
        if key in global_ids and global_ids[key] != global_id:
            raise ValueError("Published global identity was rewritten for a local track")
        if key in commits and commits[key]["global_id"] != global_id:
            raise ValueError("Availability and identity commitment disagree")
        if key in commits and published < commits[key]["time_s"] - TOLERANCE_S:
            raise ValueError("A row was published before its identity commitment")
        first_frames[key] = min(first_frames.get(key, frame), frame)
        global_ids[key] = global_id
        availability[row_key] = event

    predictions = read_rows(paths["predictions.txt"])
    prediction_keys = {(r["camera"], r["id"], r["frame"]) for r in predictions}
    if prediction_keys != availability.keys():
        raise ValueError("Predictions and availability rows are not a bijection")
    local_predictions = [{**r, "id": availability[(r["camera"], r["id"], r["frame"])]["local_id"]}
                         for r in predictions]
    local_by_key = {(r["camera"], r["id"], r["frame"]): r for r in local_predictions}

    records = {}
    for key in sorted(global_ids):
        event = commits.get(key)
        diagnostics = event.get("diagnostics", {}) if event else {}
        if not isinstance(diagnostics, dict):
            raise ValueError("Commit diagnostics must be an object")
        outcome = _outcome(event)
        if diagnostics.get("confirmation_outcome") not in (None, outcome):
            raise ValueError("Logged diagnostic outcome disagrees with commitment reason/time")
        checks = diagnostics.get("blocking_checks", [])
        if not isinstance(checks, list) or not all(isinstance(s, str) for s in checks):
            raise ValueError("blocking_checks must be a list of explicit logged strings")
        gates = diagnostics.get("candidate_gates")
        if gates is not None:
            if not isinstance(gates, dict):
                raise ValueError("candidate_gates must be an explicit logged object")
            rejections = gates.get("first_rejection_counts", {})
            if not isinstance(rejections, dict):
                raise ValueError("first_rejection_counts must be an object")
            for name, count in rejections.items():
                _integer(count, f"candidate first rejection {name}")
        pending = "first_time_s" in event if event else None
        records[key] = {
            "camera": key[0], "local_id": key[1], "global_id": global_ids[key],
            "first_filtered_frame": first_frames[key],
            "in_scored_commit_cohort": score_start <= first_frames[key] < score_stop,
            "pending": pending, "confirmation_outcome": outcome,
            "reason": event.get("reason") if event else None,
            "accepted": event.get("reason") == "confirmed_handoff" if event else None,
            "first_time_s": event.get("first_time_s") if event else None,
            "deadline_s": event.get("deadline_s") if event else None,
            "commit_time_s": event.get("time_s") if event else None,
            "wait_s": max(0., event["time_s"] - event["first_time_s"]) if pending else None,
            "observations": event.get("observations") if event else None,
            "diagnostics_logged": bool(diagnostics),
            "decision_status": diagnostics.get("decision_status", "not_logged"),
            "blocking_checks": checks,
            "legal_candidate_count": diagnostics.get("legal_candidate_count"),
            "minimum_observations": diagnostics.get("minimum_observations"),
            "evidence_shortfall": diagnostics.get("evidence_shortfall"),
            "assignment_eligible": diagnostics.get("assignment_eligible"),
            "last_evaluated_at_s": diagnostics.get("last_evaluated_at_s"),
            "row_gap": diagnostics.get("row_gap"), "column_gap": diagnostics.get("column_gap"),
            "selected_global_id": diagnostics.get("selected_global_id"),
            "selected_cost": diagnostics.get("selected_cost"),
            "candidate_global_ids": diagnostics.get("candidate_global_ids"),
            "best_candidate_global_id": diagnostics.get("best_candidate_global_id"),
            "best_candidate_cost": diagnostics.get("best_candidate_cost"),
            "competing_entrants": diagnostics.get("competing_entrants"),
            "best_row_gap": diagnostics.get("best_row_gap"),
            "best_column_gap": diagnostics.get("best_column_gap"),
            "candidate_gates": gates,
            "reference_handoff_indices": [], "gt_correct_reference_endpoints": 0,
            "gt_broken_reference_endpoints": 0, "gt_missed_reference_endpoints": 0,
            "diagnostics": diagnostics,
        }
    selected = [r for r in records.values() if r["in_scored_commit_cohort"]]
    references = []
    route_counts = defaultdict(Counter)
    reference_keys = set()
    duplicate_other_presence = None
    if ground_truth is not None:
        ground_truth = Path(ground_truth)
        # Reading all cameras is essential: pair filtering before transitions()
        # can invent 1->2 when the true consecutive sequence is 1->3->2.
        gt, gt_topology, duplicate_other_presence = _ground_truth(
            ground_truth, cameras, replay_start, replay_stop)
        matched, _ = spatial_matches(gt, local_predictions, .5)
        routes = {(r["source"], r["destination"]) for r in cfg.get("routes", [])}
        if not routes:
            routes = {(a, b) for a in cameras for b in cameras if a != b}
        for route in routes:
            route_counts[route]
        for identity, source, destination in transitions(gt_topology):
            if ((source[2], destination[2]) not in routes or source[2] not in cameras or
                    destination[2] not in cameras or not score_start <= destination[0] < score_stop):
                continue
            src = _endpoint(matched, identity, source, True)
            dst = _endpoint(matched, identity, destination, False)
            src_key, dst_key = src[:2] if src else None, dst[:2] if dst else None
            src_gid, dst_gid = global_ids.get(src_key), global_ids.get(dst_key)
            continuity = "missed" if src_gid is None or dst_gid is None else (
                "confirmed" if src_gid == dst_gid else "broken")
            dst_record = records.get(dst_key)
            accepted = dst_record["accepted"] if dst_record else None
            index = len(references)
            reference = {
                "reference_index": index, "gt_identity": identity,
                "source_camera": source[2], "destination_camera": destination[2],
                "source_segment_start_frame": source[0], "source_segment_end_frame": source[1],
                "destination_segment_start_frame": destination[0],
                "destination_segment_end_frame": destination[1],
                "source_endpoint_in_warmup": source[1] < score_start,
                "source_matched_frame": src[2] if src else None,
                "destination_matched_frame": dst[2] if dst else None,
                "source_local_id": src[1] if src else None,
                "destination_local_id": dst[1] if dst else None,
                "source_global_id": src_gid, "destination_global_id": dst_gid,
                "continuity": continuity, "destination_accepted": accepted,
                "destination_pending": dst_record["pending"] if dst_record else None,
                "destination_confirmation_outcome": dst_record["confirmation_outcome"] if dst_record else None,
                "destination_decision_status": dst_record["decision_status"] if dst_record else None,
                "destination_commit_in_scored_cohort": dst_record["in_scored_commit_cohort"] if dst_record else None,
                "destination_commit_time_s": dst_record["commit_time_s"] if dst_record else None,
                "destination_deadline_s": dst_record["deadline_s"] if dst_record else None,
                "source_prediction_box_xyxy": local_by_key[src]["box"] if src else None,
                "destination_prediction_box_xyxy": local_by_key[dst]["box"] if dst else None,
            }
            references.append(reference)
            route = route_counts[(source[2], destination[2])]
            route["reference_handoffs"] += 1
            route[continuity] += 1
            route["source_endpoint_in_warmup"] += int(source[1] < score_start)
            route["destination_endpoint_matched"] += int(dst is not None)
            if accepted:
                route["accepted_destination"] += 1
                route[f"accepted_{continuity}"] += 1
            if dst_record:
                reference_keys.add(dst_key)
                dst_record["reference_handoff_indices"].append(index)
                dst_record[{"confirmed": "gt_correct_reference_endpoints",
                            "broken": "gt_broken_reference_endpoints",
                            "missed": "gt_missed_reference_endpoints"}[continuity]] += 1
    reference_selected = [records[k] for k in sorted(reference_keys)
                          if records[k]["in_scored_commit_cohort"]]
    routes_summary = [{"source_camera": source, "destination_camera": destination,
                      **{k: counter[k] for k in (
                          "reference_handoffs", "confirmed", "broken", "missed",
                          "destination_endpoint_matched", "source_endpoint_in_warmup",
                          "accepted_destination", "accepted_confirmed", "accepted_broken", "accepted_missed")},
                      "endpoint_recall": counter["confirmed"] / counter["reference_handoffs"]
                      if counter["reference_handoffs"] else None}
                     for (source, destination), counter in sorted(route_counts.items())]
    scope = {"cameras": sorted(cameras), "replay_start_frame": replay_start,
             "replay_stop_frame": replay_stop, "score_start_frame": score_start,
             "score_stop_frame": score_stop, "fps": fps}
    return {
        "schema": "deadline_diagnostics_v0_4", "scope": scope,
        "provenance": {"run_directory": str(run_directory.resolve()),
                       "config_sha256": digest(cfg),
                       "input_sha256": {name: file_hash(path) for name, path in paths.items()},
                       "ground_truth_path": str(ground_truth.resolve()) if ground_truth else None,
                       "ground_truth_sha256": file_hash(ground_truth) if ground_truth else None,
                       "script_sha256": file_hash(Path(__file__))},
        "measurement_contract": {
            "commit_cohort": "First filtered local output is in the scored interval; warmup commits excluded.",
            "reference_cohort": "All-camera consecutive GT transitions, then configured-route selection; destination first visibility scored, source endpoint within replay (may be warmup).",
            "reference_subset": "Unique matched destination local tracks also in the scored commit cohort; multiple references per local track are retained separately.",
            "matching": "IoU >= 0.5; final 15 source frames and first 15 destination frames, same endpoint rule as existing evaluator.",
            "acceptance": "confirmed_handoff is a logged decision, not a GT correctness assertion.",
            "unknown_causes": "No rejection cause is inferred from unlogged fields; no_legal_candidate alone does not identify which route/time/appearance gate rejected candidates.",
            "logged_decision_fields": "Latest logged candidate evaluation at last_evaluated_at_s; decision_status is not an exhaustive root-cause taxonomy and logged conditions may overlap.",
            "candidate_gate_counts": "Sum of each committed entrant's last evaluation only, not cumulative lifetime or unique-message counts. Each scanned message contributes its first rejected gate; gate order does not establish a GT causal root cause.",
            "other_camera_annotations": "Duplicate non-evaluated-camera keys contribute one topological presence only; no box is selected among conflicting annotations. Scored-camera GT duplicates remain strict errors.",
            "clock": "Video scene time after first filtered local output, excluding inference/decode wall time.",
        },
        "populations": {
            "all_scored_local_tracks": _population(selected),
            "all_scored_pending_tracks": _population(r for r in selected if r["pending"] is True),
            "reference_destination_matched_tracks": _population(reference_selected) if ground_truth else None,
            "reference_destination_matched_pending_tracks": _population(
                r for r in reference_selected if r["pending"] is True) if ground_truth else None,
        },
        "validation": {
            "published_track_count": len(global_ids), "logged_commit_count": len(commits),
            "published_tracks_without_logged_commit": sum(k not in commits for k in global_ids),
            "logged_commits_without_published_rows": sum(k not in global_ids for k in commits),
            "pending_commit_deadline_violations": sum(
                r["commit_time_s"] > r["deadline_s"] + TOLERANCE_S
                for r in selected if r["pending"] is True),
            "reference_destination_tracks_outside_commit_cohort": len(reference_keys) - len(reference_selected),
            "non_evaluated_camera_duplicate_topology_rows": duplicate_other_presence,
        },
        "reference_routes": routes_summary if ground_truth else None,
        "reference_handoffs": references if ground_truth else None,
        "track_diagnostics": list(records.values()),
    }


def _csv(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: canonical(row[name]) if isinstance(row.get(name), (dict, list)) else
                         row.get(name) for name in fields})
    return stream.getvalue()


def _markdown(report):
    scope = report["scope"]
    lines = ["# Confirmation decision diagnostics", "",
             f"Replay [{scope['replay_start_frame']},{scope['replay_stop_frame']}), "
             f"scored [{scope['score_start_frame']},{scope['score_stop_frame']}), "
             f"cameras {scope['cameras']}.", "",
             "Accepted identities are algorithm decisions, not ground-truth correctness labels. "
             "Old unlogged rejection causes remain unknown. Delays use video scene time.", "",
             "## Commitment populations", "",
             "| Population | Tracks | Early accepted | Deadline accepted | Deadline fresh | Interval-end fresh | Outside entry | Unlogged/other |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, population in report["populations"].items():
        if population is None:
            continue
        counts = population["outcomes"]
        other = sum(counts[k] for k in ("accepted_after_deadline", "other_pending", "other_nonpending", "commit_not_logged"))
        lines.append(f"| {label.replace('_', ' ')} | {population['tracks']} | "
                     f"{counts['early_accepted']} | {counts['deadline_accepted']} | "
                     f"{counts['deadline_fresh']} | {counts['interval_end_fresh']} | "
                     f"{counts['outside_entry']} | {other} |")
    lines += ["", "Commit counts use first filtered output in the scored interval. Reference-subset "
              "counts use unique matched destination local tracks; reference-transition counts "
              "below use individual real GT transitions. Their denominators differ.", "",
              "## Logged decisions for all scored pending tracks", "",
              "| Logged status | Tracks |", "|---|---:|"]
    pending = report["populations"]["all_scored_pending_tracks"]
    for status, count in pending["decision_status"].items():
        lines.append(f"| {status} | {count} |")
    dist = pending["pending_wait"]
    if dist["count"]:
        lines += ["", f"Pending wait: n={dist['count']}; p50={dist['p50_s']:.3f} s, "
                  f"p95={dist['p95_s']:.3f} s, max={dist['max_s']:.3f} s."]
    else:
        lines += ["", "Pending wait distribution is empty."]
    lines += ["", "A logged `no_legal_candidate` identifies a final eligibility result, not its "
              "unlogged time/route/appearance cause. Margin blocking checks may overlap and "
              "must not be added as mutually exclusive outcomes. Logged fields describe the "
              "latest candidate evaluation at `last_evaluated_at_s`; missing evidence and zero "
              "legal candidates can coexist."]
    gates = pending["last_candidate_evaluation"]
    if gates["tracks_with_logged_gates"]:
        lines += ["", "### First rejected gates in the last logged candidate evaluation", "",
                  "| First rejected gate | Evaluated message instances |", "|---|---:|"]
        for name, count in gates["first_rejection_counts"].items():
            lines.append(f"| {name} | {count} |")
        lines += ["", f"Logged gate evaluations cover {gates['tracks_with_logged_gates']} pending tracks; "
                  f"summed scanned message instances: {gates['scanned_messages']}; "
                  f"legal message edges: {gates['legal_message_edges']}.", "",
                  "These are sums of each entrant's last evaluation, not lifetime or unique-message "
                  "counts. First rejection follows code gate order and does not establish a "
                  "ground-truth causal root cause."]
    if report["reference_routes"] is not None:
        lines += ["", "## Offline reference-handoff endpoints by route", "",
                  "| Route | Reference | Confirmed | Broken | Missed | Endpoint recall (%) | Accepted correct / broken / missed |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for route in report["reference_routes"]:
            recall = f"{100 * route['endpoint_recall']:.2f}" if route["endpoint_recall"] is not None else "n/a"
            lines.append(f"| {route['source_camera']}→{route['destination_camera']} | "
                         f"{route['reference_handoffs']} | {route['confirmed']} | "
                         f"{route['broken']} | {route['missed']} | "
                         f"{recall} | "
                         f"{route['accepted_confirmed']}/{route['accepted_broken']}/{route['accepted_missed']} |")
        lines += ["", "GT transitions are constructed with all annotated cameras before route "
                  "selection. A source endpoint can be in replay warmup; the destination first "
                  "visibility must be in the scored interval. Acceptance and endpoint correctness "
                  "are reported independently. Predicted-transition precision is not inferred "
                  "from these reference-route counts."]
        duplicates = report["validation"]["non_evaluated_camera_duplicate_topology_rows"]
        if duplicates:
            lines += ["", f"Non-evaluated-camera duplicate key rows: {duplicates}. They contribute "
                      "one topological presence per camera/ID/frame; no conflicting box is chosen "
                      "for matching. Scored-camera duplicates would stop the report."]
    else:
        lines += ["", "Ground truth was not requested; correctness and reference-route results are unavailable."]
    lines += ["", "## Evidence", "",
              "See `commit_diagnostics.csv` for entrant decisions, "
              "`reference_handoff_diagnostics.csv` for endpoint frames/boxes and "
              "`diagnostics.json` for complete fields, validation checks and input hashes.", ""]
    return "\n".join(lines)


def write_report(report, output_directory):
    """Write JSON/CSV/Markdown; reuse identical files and reject changed reports."""
    output = Path(output_directory)
    tracks = report["track_diagnostics"]
    track_fields = [name for name in tracks[0] if name != "diagnostics"] if tracks else [
        "camera", "local_id", "global_id", "confirmation_outcome"]
    references = report["reference_handoffs"] or []
    ref_fields = list(references[0]) if references else [
        "reference_index", "gt_identity", "source_camera", "destination_camera", "continuity"]
    summary_rows = []
    for population, value in report["populations"].items():
        if value is not None:
            for outcome in OUTCOMES:
                summary_rows.append({"population": population, "tracks": value["tracks"],
                                     "outcome": outcome, "count": value["outcomes"][outcome]})
    contents = {"diagnostics.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
                "commit_diagnostics.csv": _csv(tracks, track_fields),
                "reference_handoff_diagnostics.csv": _csv(references, ref_fields),
                "summary.csv": _csv(summary_rows, ["population", "tracks", "outcome", "count"]),
                "diagnostics.md": _markdown(report)}
    for name, content in contents.items():
        path = output / name
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to replace a changed diagnostic artifact: {path}")
    output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        path = output / name
        if not path.exists():
            path.write_text(content, encoding="utf-8", newline="")
    return {name: file_hash(output / name) for name in contents}


def generate_report(run_directory, config, output_directory, ground_truth=None, **score_options):
    report = build_report(run_directory, config, ground_truth, **score_options)
    hashes = write_report(report, output_directory)
    return report, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth", "--mcvt-gt", dest="ground_truth", type=Path)
    parser.add_argument("--score-start-frame", type=int)
    parser.add_argument("--score-stop-frame", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    report, _ = generate_report(args.run, cfg, args.output, args.ground_truth,
                               score_start_frame=args.score_start_frame,
                               score_stop_frame=args.score_stop_frame)
    population = report["populations"]["all_scored_pending_tracks"]
    print(json.dumps({"output": str(args.output), "pending_tracks": population["tracks"],
                      "outcomes": population["outcomes"],
                      "reference_routes": report["reference_routes"]}, indent=2))


if __name__ == "__main__":
    main()
