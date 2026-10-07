"""Offline scoring of immutable global outputs and their publication deadlines.

Ground truth is read only here, after inference. Eventual tracking scores are
complemented by scores that omit outputs unavailable at each declared deadline.
An unavailable output remains a missing prediction; it is never moved to a later
image frame or backdated to its observation time.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from rhd.common import file_hash, write_json
from rhd.evaluation import evaluate, read_rows, spatial_matches, transitions

DELAYS_S = (0.0, 0.5, 1.0)
TOLERANCE_S = 1e-9


def _number(value, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{location}: expected a finite number")
    return float(value)


def _integer(value, location: str, minimum=1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{location}: expected an integer >= {minimum}")
    return value


def _jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield number, value


def _distribution(values):
    return {"count": len(values),
            "p50_s": float(np.percentile(values, 50)) if values else None,
            "p95_s": float(np.percentile(values, 95)) if values else None,
            "max_s": float(max(values)) if values else None}


def _key(row):
    return row["camera"], row["id"], row["frame"]


def _publication_handoffs(gt, predictions, events, fps):
    """Use the existing 15-frame endpoint rule, but actual publication times.

    Endpoint identity correctness intentionally follows evaluate() rather than
    selecting a more favorable later match. This is distinct from its historical
    first-spatial-match delay statistic.
    """
    matched, _ = spatial_matches(gt, predictions, 0.5)
    reference = transitions(gt)
    delays = []
    for identity, source, destination in reference:
        source_match = destination_match = None
        for frame in range(source[1], max(source[0] - 1, source[1] - 15), -1):
            gid = matched.get((source[2], frame, identity))
            if gid is not None:
                source_match = source[2], gid, frame
                break
        for frame in range(destination[0], min(destination[1] + 1, destination[0] + 15)):
            gid = matched.get((destination[2], frame, identity))
            if gid is not None:
                destination_match = destination[2], gid, frame
                break
        if source_match is not None and destination_match is not None and source_match[1] == destination_match[1]:
            published = max(events[source_match]["published_at_s"], events[destination_match]["published_at_s"])
            delays.append(max(0.0, published - destination[0] / fps))
    return {"reference_handoffs": len(reference), "eventually_confirmed_handoffs": len(delays),
            "publication_confirmation_delay": _distribution(delays),
            "confirmed_by_arrival_deadline": [
                {"allowed_delay_s": delay, "confirmed_handoffs": sum(v <= delay + TOLERANCE_S for v in delays),
                 "handoff_recall": sum(v <= delay + TOLERANCE_S for v in delays) / len(reference) if reference else None}
                for delay in DELAYS_S],
            "note": "Publication delay is measured from first GT destination visibility and requires both matched endpoint IDs to be available. Endpoint matching uses the same custom 15-frame rule as evaluate()."}


def evaluate_availability(cfg: dict, folder: Path, ground_truth: Path) -> dict:
    """Validate publication records, write deadline-filtered rows, and score them.

    This function is an offline evaluation helper. It does not change the original
    predictions, availability records, tracker state, or selected configuration.
    """
    folder, ground_truth = Path(folder), Path(ground_truth)
    fps = _number(cfg["fps"], "config fps")
    if fps <= 0:
        raise ValueError("config fps must be positive")
    cameras, start, stop = cfg["cameras"], cfg["start_frame"], cfg["stop_frame"]
    prediction_path, availability_path = folder / "predictions.txt", folder / "availability.jsonl"
    predictions = read_rows(prediction_path)
    if any(r["camera"] not in cameras or not start <= r["frame"] < stop for r in predictions):
        raise ValueError("Global predictions contain rows outside the configured cameras/frame interval")
    prediction_keys = {_key(r) for r in predictions}
    events, local_keys, local_mapping = {}, set(), {}
    publication_delays = []
    for number, event in _jsonl(availability_path):
        location = f"{availability_path}:{number}"
        camera = _integer(event.get("camera"), f"{location} camera")
        local_id = _integer(event.get("local_id"), f"{location} local_id")
        gid = _integer(event.get("global_id"), f"{location} global_id")
        frame = _integer(event.get("frame"), f"{location} frame", 0)
        published = _number(event.get("published_at_s"), f"{location} published_at_s")
        if camera not in cameras or not start <= frame < stop:
            raise ValueError(f"{location}: availability record outside configured cameras/frame interval")
        if published < frame / fps - TOLERANCE_S:
            raise ValueError(f"{location}: output published before its observation frame")
        if not isinstance(event.get("reason"), str) or not event["reason"]:
            raise ValueError(f"{location}: missing publication reason")
        if "published_at_frame" in event:
            available_frame = _integer(event["published_at_frame"], f"{location} published_at_frame", 0)
            if available_frame != math.ceil(published * fps - TOLERANCE_S):
                raise ValueError(f"{location}: inconsistent published_at_frame and published_at_s")
        key, local_key, track_key = (camera, gid, frame), (camera, local_id, frame), (camera, local_id)
        if key in events or local_key in local_keys:
            raise ValueError(f"{location}: duplicate global or local publication row")
        if track_key in local_mapping and local_mapping[track_key] != gid:
            raise ValueError(f"{location}: published global identity rewritten for one local track")
        events[key], local_mapping[track_key] = event, gid
        local_keys.add(local_key)
        publication_delays.append(max(0.0, published - frame / fps))
    if events.keys() != prediction_keys:
        raise ValueError("Availability/global predictions are not a bijection: "
                         f"{len(prediction_keys - events.keys())} missing events, "
                         f"{len(events.keys() - prediction_keys)} extra events")

    local_path = folder / "local_predictions.txt"
    local_rows = read_rows(local_path) if local_path.is_file() else None
    if local_rows is not None:
        local_rows_by_key = {_key(r): r for r in local_rows}
        if local_rows_by_key.keys() != local_keys:
            raise ValueError("Immediate local predictions and availability records differ in row coverage")
        for row in predictions:
            event = events[_key(row)]
            local = local_rows_by_key[(row["camera"], event["local_id"], row["frame"])]
            if not np.allclose(row["box"], local["box"], atol=1e-6, rtol=0):
                raise ValueError("Buffered global output changed a box from the immediate local stream")

    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("prediction_rows") is not None and summary["prediction_rows"] != len(predictions):
        raise ValueError("Summary prediction_rows disagrees with predictions/availability records")
    messages_path = folder / "messages.jsonl"
    commits, pending_delays, reasons, commit_violations = {}, [], Counter(), 0
    for number, record in _jsonl(messages_path):
        if record.get("event") != "identity_committed":
            continue
        location = f"{messages_path}:{number}"
        key = (_integer(record.get("camera"), f"{location} camera"),
               _integer(record.get("local_id"), f"{location} local_id"))
        gid = _integer(record.get("global_id"), f"{location} global_id")
        now = _number(record.get("time_s"), f"{location} time_s")
        if key in commits:
            raise ValueError(f"{location}: duplicate identity commit for one local track")
        if key in local_mapping and local_mapping[key] != gid:
            raise ValueError(f"{location}: identity commit disagrees with published global identity")
        commits[key] = record
        reasons[record.get("reason", "unspecified")] += 1
        if "first_time_s" in record:
            first = _number(record["first_time_s"], f"{location} first_time_s")
            if now < first - TOLERANCE_S:
                raise ValueError(f"{location}: identity committed before pending track began")
            pending_delays.append(max(0.0, now - first))
        if "deadline_s" in record:
            deadline = _number(record["deadline_s"], f"{location} deadline_s")
            commit_violations += int(now > deadline + TOLERANCE_S)
    for event in events.values():
        commit = commits.get((event["camera"], event["local_id"]))
        if commit and event["published_at_s"] < commit["time_s"] - TOLERANCE_S:
            raise ValueError("A global row was published before its identity was committed")

    bounded_summary = summary.get("bounded_confirmation", {})
    association = cfg.get("association", {})
    confirmation_frames = bounded_summary.get("confirmation_frames")
    if confirmation_frames is None:
        confirmation_frames = association.get("confirmation_frames") if commits else 0
    configured_delay_s = confirmation_frames / fps if confirmation_frames is not None else None
    if bounded_summary.get("deadline_violations") is not None and bounded_summary["deadline_violations"] != commit_violations:
        raise ValueError("Summary deadline violations disagree with identity commit records")
    if bounded_summary.get("maximum_pending_lifetime_s") is not None:
        measured_max = max(pending_delays, default=0.0)
        if not math.isclose(bounded_summary["maximum_pending_lifetime_s"], measured_max, rel_tol=0, abs_tol=TOLERANCE_S):
            raise ValueError("Summary pending lifetime disagrees with identity commit records")

    # Preserve original row text and order rather than rounding boxes a second time.
    original_lines = {}
    with prediction_path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.replace(",", " ").split()
            original_lines[tuple(int(float(v)) for v in fields[:3])] = line.rstrip("\r\n")
    curve, generated_paths = [], []
    for allowed in DELAYS_S:
        path = folder / f"predictions_available_{round(allowed * 1000):04d}ms.txt"
        accepted = [r for r in predictions if events[_key(r)]["published_at_s"] <= r["frame"] / fps + allowed + TOLERANCE_S]
        with path.open("w", encoding="utf-8") as stream:
            for row in accepted:
                stream.write(original_lines[_key(row)] + "\n")
        generated_paths.append(path)
        curve.append({"allowed_delay_s": allowed, "prediction_file": path.name,
                      "available_prediction_rows": len(accepted), "late_prediction_rows": len(predictions) - len(accepted),
                      "publication_coverage": len(accepted) / len(predictions) if predictions else None,
                      "metrics": evaluate(ground_truth, path, cameras, start, stop, fps, label_scope="multicam")})
    gt = read_rows(ground_truth, cameras, start, stop)
    result = {"schema_version": 1, "scope": {"cameras": cameras, "start_frame": start, "stop_frame": stop, "fps": fps},
              "eventual_metrics": evaluate(ground_truth, prediction_path, cameras, start, stop, fps, label_scope="multicam"),
              "availability_curve": curve, "publication_delay": _distribution(publication_delays),
              "published_handoffs": _publication_handoffs(gt, predictions, events, fps),
              "integrity": {"prediction_rows": len(predictions), "availability_rows": len(events),
                            "local_prediction_rows": len(local_rows) if local_rows is not None else None,
                            "duplicate_rows": 0, "missing_events": 0, "extra_events": 0,
                            "published_identity_rewrites": 0, "identity_commits": len(commits)},
              "deadline_statistics": {"configured_maximum_wait_s": configured_delay_s,
                                      "commit_deadline_violations": commit_violations,
                                      "publication_rows_over_configured_wait": sum(v > configured_delay_s + TOLERANCE_S for v in publication_delays) if configured_delay_s is not None else None,
                                      "pending_confirmation_delay": _distribution(pending_delays),
                                      "commit_reasons": dict(sorted(reasons.items())),
                                      "reported_summary": bounded_summary},
              "input_hashes": {p.name: file_hash(p) for p in (prediction_path, availability_path, summary_path, messages_path)},
              "ground_truth_sha256": file_hash(ground_truth), "helper_sha256": file_hash(Path(__file__)),
              "generated_prediction_hashes": {p.name: file_hash(p) for p in generated_paths},
              "availability_clock": "video_replay_time",
              "note": "Offline GT evaluation only. Publication times use the video/replay clock: confirmation waiting excludes wall-clock detector/decoding time and is not live streaming latency. Deadline scores omit unavailable original-frame rows, which become missing predictions. Eventual scores permit causal confirmation delay; they are not zero-latency online accuracy. The existing first-spatial-match handoff delay remains distinct from publication-based confirmation delay."}
    if local_path.is_file():
        result["input_hashes"][local_path.name] = file_hash(local_path)
    write_json(folder / "latency_metrics.json", result)
    return result


# Compatibility for callers using the task's initial helper name.
score_availability = evaluate_availability
