"""Evaluation-only handoff evidence on the native, offset-aware CityFlow clock.

The full scene's annotation topology is built before selecting routes or time
windows. This prevents a missing/intervening camera from creating a reference
handoff. These endpoint metrics are custom, not the official CityFlow benchmark.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from rhd.evaluation import spatial_matches
from .data import contiguous_spans, event_time_us


def _rows(source, scene):
    if isinstance(source, (str, Path)):
        from .evaluation import read_submission
        table = read_submission(source, cameras=map(int, scene["cameras"]),
                                allow_float_boxes=True)
        result = []
        for r in table.itertuples(index=False, name=None):
            camera, identity, frame, x, y, w, h, _, _ = r
            result.append({"camera": int(camera), "id": int(identity),
                           "frame": int(frame)-1,
                           "box": [float(x), float(y), float(x+w), float(y+h)]})
    else:
        result = [dict(r) for r in source]
    seen = set()
    for row in result:
        camera, identity, frame = row["camera"], row["id"], row["frame"]
        if any(type(x) is not int for x in (camera, identity, frame)) or identity < 0:
            raise ValueError("Handoff rows need integral camera/identity/native frame")
        meta = scene["cameras"].get(str(camera))
        if meta is None:
            raise ValueError("Handoff row outside declared scene cameras")
        expected = event_time_us(meta, frame)/1e6
        if "time_s" in row and (not math.isfinite(row["time_s"]) or
                                abs(row["time_s"]-expected) > 1.1e-6):
            raise ValueError("Handoff row timestamp disagrees with the native camera clock")
        row["time_s"] = expected
        box = row["box"]
        if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("Invalid handoff box")
        key = (camera, identity, frame)
        if key in seen:
            raise ValueError("Repeated handoff camera/identity/frame key")
        seen.add(key)
    return result


def _route_pairs(routes):
    result = set()
    for route in routes:
        pair = (route["source"], route["destination"]) if isinstance(route, dict) else tuple(route)
        if len(pair) != 2 or any(type(c) is not int or c < 1 for c in pair) or pair[0] == pair[1]:
            raise ValueError("Invalid handoff route")
        result.add(pair)
    return result


def _topology(rows):
    spans = contiguous_spans(rows)
    transitions, same_camera = [], 0
    membership = {}
    for identity, visits in spans.items():
        for index, visit in enumerate(visits):
            for row in visit:
                membership[(row["camera"], row["frame"], identity)] = (identity, index)
        for index, (source, destination) in enumerate(zip(visits, visits[1:])):
            if source[0]["camera"] == destination[0]["camera"]:
                same_camera += 1
                continue
            transitions.append({"identity": identity, "source": source,
                                "destination": destination,
                                "source_visit": (identity, index),
                                "destination_visit": (identity, index+1),
                                "route": (source[0]["camera"], destination[0]["camera"]),
                                "gap_s": destination[0]["time_s"]-source[-1]["time_s"]})
    return transitions, membership, same_camera


def _scope(transition, replay_start_s, score_start_s, score_stop_s):
    source, destination = transition["source"], transition["destination"]
    if not score_start_s <= destination[0]["time_s"] < score_stop_s:
        return "destination_outside_score"
    if transition["gap_s"] <= 0:
        return "overlap_or_nonpositive_gap"
    if not replay_start_s <= source[-1]["time_s"] < score_stop_s:
        return "source_outside_replay"
    return "eligible"


def _endpoint(mapping, identity, span, endpoint_frames, *, first, start_s, stop_s):
    endpoint = span[0]["frame"] if first else span[-1]["frame"]
    candidates = span if first else reversed(span)
    for row in candidates:
        distance = row["frame"]-endpoint if first else endpoint-row["frame"]
        if distance >= endpoint_frames:
            break
        if not start_s <= row["time_s"] < stop_s:
            continue
        match = mapping.get((row["camera"], row["frame"], identity))
        if match is not None:
            return match, row
    return None, None


def _distribution(values, missing=0):
    return {"samples": len(values), "missing_availability_rows": missing,
            "median_s": float(np.median(values)) if values else None,
            "p95_s": float(np.quantile(values, .95)) if values else None,
            "maximum_s": float(max(values)) if values else None}


def _availability(source, scene, prediction_rows):
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    else:
        rows = source
    predictions = {(r["camera"], r["frame"], r["id"]) for r in prediction_rows}
    result = {}
    for row in rows:
        camera, global_id = row["camera"], row["global_id"]
        if type(camera) is not int or type(global_id) is not int or global_id < 0:
            raise ValueError("Availability camera and global ID must be integral")
        if "local_frame_zero_based" in row:
            frame = row["local_frame_zero_based"]
            if "frame" in row and row["frame"] != frame+1:
                raise ValueError("Availability native frame columns disagree")
        else:
            frame = row["frame"]-1
        key = (camera, frame, global_id)
        if key in result or key not in predictions:
            raise ValueError("Availability must uniquely describe emitted prediction rows")
        observed = event_time_us(scene["cameras"][str(camera)], frame)/1e6
        if "time_s" in row and (not math.isfinite(row["time_s"]) or
                                abs(row["time_s"]-observed) > 1.1e-6):
            raise ValueError("Availability observation timestamp disagrees with the native clock")
        published = row["published_at_s"]
        if not math.isfinite(published) or published < observed-1.1e-6:
            raise ValueError("Availability publication precedes observation")
        result[key] = max(0., published-observed)
    return result


def evaluate_handoffs(scene, ground_truth_rows, predictions, *, routes,
                      replay_start_s=60., score_start_s=120., score_stop_s=212.,
                      threshold=.5, endpoint_frames=15, availability=None):
    """Score non-overlap, route-scoped endpoints after full-scene topology.

    Dictionary input rows use zero-based native frames; nine-column file input
    uses the official one-based native frames. GT must contain the full scene,
    including unconfigured/intervening cameras and observations before replay.
    The caller is responsible for supplying full GT, not a prefiltered cohort.
    Every endpoint searches at most 15 native frames by default (1.5 s at 10 FPS).
    Publication wait measures when a matched destination box becomes available,
    separately from the time needed to obtain its first spatial match.
    """
    if not 0 <= replay_start_s <= score_start_s < score_stop_s or not 0 < threshold <= 1:
        raise ValueError("Invalid handoff scoring interval or IoU threshold")
    if type(endpoint_frames) is not int or endpoint_frames < 1:
        raise ValueError("Endpoint window must contain a positive integer frame count")
    configured = _route_pairs(routes)
    if any(c not in set(map(int, scene["cameras"])) for pair in configured for c in pair):
        raise ValueError("Configured handoff route outside scene")
    gt, pred = _rows(ground_truth_rows, scene), _rows(predictions, scene)
    references, gt_membership, same_camera = _topology(gt)
    proposed, _, predicted_same_camera = _topology(pred)
    by_gt, by_pred = spatial_matches(gt, pred, threshold)
    waits = _availability(availability, scene, pred)
    route_counts = defaultdict(Counter)
    eligible, exclusions = [], Counter()
    for transition in references:
        route = transition["route"]
        route_counts[route]["full_topology_cross_camera_pairs"] += 1
        route_counts[route]["full_topology_overlap_or_nonpositive_gap" if transition["gap_s"] <= 0 else
                            "full_topology_positive_gap"] += 1
        scope = _scope(transition, replay_start_s, score_start_s, score_stop_s)
        if scope == "destination_outside_score":
            route_counts[route]["destination_outside_score"] += 1
            exclusions[scope] += 1
            continue
        route_counts[route]["destination_entries_in_score"] += 1
        if scope != "eligible":
            route_counts[route][scope] += 1
            exclusions[scope] += 1
            continue
        route_counts[route]["eligible_nonoverlap_transitions"] += 1
        if route not in configured:
            route_counts[route]["unsupported_route"] += 1
            exclusions["unsupported_route"] += 1
            continue
        eligible.append(transition)
        route_counts[route]["reference_handoffs"] += 1
    eligible_visits = {(t["source_visit"], t["destination_visit"]) for t in eligible}
    outcomes, latencies, matched_waits, confirmed_waits = [], [], [], []
    missing_waits = confirmed_missing_waits = 0
    for transition in eligible:
        identity, source, destination = transition["identity"], transition["source"], transition["destination"]
        previous, source_row = _endpoint(by_gt, identity, source, endpoint_frames,
                                          first=False, start_s=replay_start_s, stop_s=score_stop_s)
        current, destination_row = _endpoint(by_gt, identity, destination, endpoint_frames,
                                             first=True, start_s=score_start_s, stop_s=score_stop_s)
        outcome = "missed_handoffs" if previous is None or current is None else (
            "confirmed_handoffs" if previous == current else "identity_breaks")
        route_counts[transition["route"]][outcome] += 1
        if outcome == "confirmed_handoffs":
            latencies.append(destination_row["time_s"]-destination[0]["time_s"])
        wait = None
        if waits is not None and current is not None:
            wait = waits.get((destination_row["camera"], destination_row["frame"], current))
            if wait is None:
                missing_waits += 1
                if outcome == "confirmed_handoffs":
                    confirmed_missing_waits += 1
            else:
                matched_waits.append(wait)
                if outcome == "confirmed_handoffs":
                    confirmed_waits.append(wait)
        outcomes.append({"gt_id": identity, "source_camera": transition["route"][0],
                         "destination_camera": transition["route"][1],
                         "source_last_native_frame_1based": source[-1]["frame"]+1,
                         "destination_first_native_frame_1based": destination[0]["frame"]+1,
                         "source_end_time_s": source[-1]["time_s"],
                         "destination_start_time_s": destination[0]["time_s"],
                         "travel_gap_s": transition["gap_s"], "outcome": outcome,
                         "source_predicted_id": previous, "destination_predicted_id": current,
                         "source_matched_native_frame_1based": source_row["frame"]+1 if source_row else None,
                         "destination_matched_native_frame_1based": destination_row["frame"]+1 if destination_row else None,
                         "destination_publication_wait_s": wait})
    prediction_counts, prediction_exclusions = Counter(), Counter()
    for transition in proposed:
        scope = _scope(transition, replay_start_s, score_start_s, score_stop_s)
        if scope != "eligible" or transition["route"] not in configured:
            prediction_exclusions[scope if scope != "eligible" else "unsupported_route"] += 1
            continue
        prediction_counts["predicted_handoffs"] += 1
        identity = transition["identity"]
        previous, source_row = _endpoint(by_pred, identity, transition["source"], endpoint_frames,
                                          first=False, start_s=replay_start_s, stop_s=score_stop_s)
        current, destination_row = _endpoint(by_pred, identity, transition["destination"], endpoint_frames,
                                             first=True, start_s=score_start_s, stop_s=score_stop_s)
        if previous is None or current is None:
            prediction_counts["unmatched_associations"] += 1
        else:
            visits = (gt_membership[(source_row["camera"], source_row["frame"], previous)],
                      gt_membership[(destination_row["camera"], destination_row["frame"], current)])
            if previous == current and visits in eligible_visits:
                prediction_counts["correct_associations"] += 1
            else:
                prediction_counts["false_associations"] += 1
    totals = Counter(row["outcome"] for row in outcomes)
    table = []
    for route in sorted(set(route_counts) | configured):
        counts = route_counts[route]
        row = {"source": route[0], "destination": route[1], "configured": route in configured}
        for key in ("full_topology_cross_camera_pairs", "full_topology_positive_gap",
                    "full_topology_overlap_or_nonpositive_gap", "destination_outside_score", "destination_entries_in_score",
                    "overlap_or_nonpositive_gap", "source_outside_replay", "eligible_nonoverlap_transitions",
                    "unsupported_route", "reference_handoffs", "confirmed_handoffs", "identity_breaks", "missed_handoffs"):
            row[key] = counts[key]
        row["handoff_recall"] = counts["confirmed_handoffs"]/counts["reference_handoffs"] if counts["reference_handoffs"] else None
        table.append(row)
    predicted_count = prediction_counts["predicted_handoffs"]
    return {"scope": {"scenario": scene.get("scenario"), "configured_routes": [list(r) for r in sorted(configured)],
                       "replay_interval_s": [replay_start_s, score_stop_s],
                       "destination_entry_score_interval_s": [score_start_s, score_stop_s],
                       "iou_threshold": threshold, "endpoint_native_frames": endpoint_frames,
                       "endpoint_duration_s_by_camera": {c:endpoint_frames/m["fps"] for c,m in scene["cameras"].items()}},
            "reference_handoffs": len(eligible),
            "confirmed_handoffs": totals["confirmed_handoffs"],
            "missed_handoffs": totals["missed_handoffs"], "identity_breaks": totals["identity_breaks"],
            "handoff_recall": totals["confirmed_handoffs"]/len(eligible) if eligible else None,
            **{key:prediction_counts[key] for key in ("predicted_handoffs", "correct_associations", "false_associations", "unmatched_associations")},
            "handoff_precision": prediction_counts["correct_associations"]/predicted_count if predicted_count else None,
            "first_spatial_match_delay": _distribution(latencies),
            "matched_destination_publication_wait": _distribution(matched_waits, missing_waits) if waits is not None else None,
            "confirmed_destination_publication_wait": _distribution(confirmed_waits, confirmed_missing_waits) if waits is not None else None,
            "topology": {"full_gt_cross_camera_pairs": len(references), "same_camera_adjacencies": same_camera,
                         "full_gt_overlap_or_nonpositive_gap": sum(t["gap_s"] <= 0 for t in references),
                         "excluded_reference_pairs": dict(exclusions),
                         "full_prediction_cross_camera_pairs": len(proposed),
                         "predicted_same_camera_adjacencies": predicted_same_camera,
                         "excluded_prediction_pairs": dict(prediction_exclusions)},
            "routes": table, "reference_outcomes": outcomes,
            "metric_note": "Custom non-overlap handoff metric, not official CityFlow scoring. Full-scene contiguous GT topology precedes route/time filtering. Precision requires matched endpoints belonging to an eligible adjacent GT visit pair; a skipped intervening camera cannot invent a correct association. Original floating-point prediction boxes are used. Publication wait and spatial-match delay are separate."}
