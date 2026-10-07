from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import iou_matrix


def read_rows(path: Path, cameras=None, start=0, stop=None) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.replace(",", " ").split()
            if len(fields) != 9:
                raise ValueError(f"{path}:{line_number}: expected 9 MCVT fields")
            vals = [float(v) for v in fields]
            if not np.isfinite(vals).all() or any(v != int(v) for v in vals[:3]):
                raise ValueError(f"{path}:{line_number}: non-finite value or non-integral ID/frame")
            camera, identity, frame = map(int, vals[:3])
            if camera < 1 or identity < 1 or frame < 0 or vals[5] <= 0 or vals[6] <= 0:
                raise ValueError(f"{path}:{line_number}: invalid camera/identity/frame/box")
            if (cameras is not None and camera not in cameras) or frame < start or (stop is not None and frame >= stop):
                continue
            key = (camera, identity, frame)
            if key in seen:
                raise ValueError(f"{path}:{line_number}: duplicate camera/identity/frame")
            seen.add(key)
            x, y, w, h = vals[3:7]
            rows.append({"camera": camera, "id": identity, "frame": frame, "box": [x, y, x + w, y + h]})
    return rows


def group_frames(rows):
    result = defaultdict(list)
    for row in rows:
        result[(row["camera"], row["frame"])].append(row)
    return result


def identity_metrics(gt: list[dict], predictions: list[dict], threshold=0.5, local=False) -> dict:
    """Global one-to-one trajectory assignment with IoU-gated potential matches.

    Camera+frame is the observation key. IDs stay global across cameras; computing
    per-camera IDF1 and averaging would hide handoff identity errors.
    """
    def key(row):
        return (row["camera"], row["id"]) if local else row["id"]

    gt_count, pred_count = Counter(map(key, gt)), Counter(map(key, predictions))
    gt_ids, pred_ids = sorted(gt_count), sorted(pred_count)
    gi, pi = {v: i for i, v in enumerate(gt_ids)}, {v: i for i, v in enumerate(pred_ids)}
    overlap = np.zeros((len(gi), len(pi)), dtype=np.int64)
    grouped_g, grouped_p = group_frames(gt), group_frames(predictions)
    for frame_key in grouped_g.keys() & grouped_p.keys():
        g, p = grouped_g[frame_key], grouped_p[frame_key]
        valid = iou_matrix([r["box"] for r in g], [r["box"] for r in p]) >= threshold
        for a, b in zip(*np.nonzero(valid)):
            overlap[gi[key(g[a])], pi[key(p[b])]] += 1
    n, m = overlap.shape
    true_positive = 0
    if n and m:
        gcount = np.asarray([gt_count[v] for v in gt_ids])
        pcount = np.asarray([pred_count[v] for v in pred_ids])
        cost = np.full((n + m, n + m), float(2 * (len(gt) + len(predictions)) + 1))
        cost[:n, :m] = gcount[:, None] + pcount[None, :] - 2 * overlap
        cost[np.arange(n), m + np.arange(n)] = gcount
        cost[n + np.arange(m), np.arange(m)] = pcount
        cost[n:, m:] = 0
        a, b = linear_sum_assignment(cost)
        true_positive = int(sum(overlap[x, y] for x, y in zip(a, b) if x < n and y < m))
    false_negative, false_positive = len(gt) - true_positive, len(predictions) - true_positive
    denom = len(gt) + len(predictions)
    return {"IDF1": 2 * true_positive / denom if denom else 0.0,
            "IDP": true_positive / len(predictions) if predictions else 0.0,
            "IDR": true_positive / len(gt) if gt else 0.0,
            "IDTP": true_positive, "IDFN": false_negative, "IDFP": false_positive,
            "gt_observations": len(gt), "predicted_observations": len(predictions)}


def spatial_matches(gt, predictions, threshold):
    by_gt, by_pred = {}, {}
    groups_g, groups_p = group_frames(gt), group_frames(predictions)
    for frame_key in groups_g.keys() & groups_p.keys():
        g, p = groups_g[frame_key], groups_p[frame_key]
        ious = iou_matrix([r["box"] for r in g], [r["box"] for r in p])
        # Gate before assignment; otherwise an invalid edge can steal a valid match.
        a, b = linear_sum_assignment(np.where(ious >= threshold, 1 - ious, 1e6))
        for x, y in zip(a, b):
            if ious[x, y] >= threshold:
                by_gt[(g[x]["camera"], g[x]["frame"], g[x]["id"])] = p[y]["id"]
                by_pred[(p[y]["camera"], p[y]["frame"], p[y]["id"])] = g[x]["id"]
    return by_gt, by_pred


def segments(rows, max_gap=1):
    by_track = defaultdict(list)
    for row in rows:
        by_track[(row["id"], row["camera"])].append(row["frame"])
    by_id = defaultdict(list)
    for (identity, camera), frames in by_track.items():
        frames.sort()
        begin = end = frames[0]
        for frame in frames[1:]:
            if frame > end + max_gap:
                by_id[identity].append((begin, end, camera))
                begin = frame
            end = frame
        by_id[identity].append((begin, end, camera))
    for value in by_id.values():
        value.sort()
    return by_id


def transitions(rows):
    result = []
    for identity, spans in segments(rows).items():
        for source, destination in zip(spans, spans[1:]):
            if source[2] != destination[2] and destination[0] > source[1]:
                result.append((identity, source, destination))
    return result


def handoff_metrics(gt, predictions, threshold, fps, confirmation_frames=15):
    by_gt, by_pred = spatial_matches(gt, predictions, threshold)

    def last_match(mapping, identity, source):
        for frame in range(source[1], max(source[0] - 1, source[1] - confirmation_frames), -1):
            value = mapping.get((source[2], frame, identity))
            if value is not None:
                return value
        return None

    def first_match(mapping, identity, destination):
        for frame in range(destination[0], min(destination[1] + 1, destination[0] + confirmation_frames)):
            value = mapping.get((destination[2], frame, identity))
            if value is not None:
                return value, frame
        return None, None

    confirmed = missed = broken = 0
    latencies = []
    reference = transitions(gt)
    for identity, source, destination in reference:
        previous = last_match(by_gt, identity, source)
        current, frame = first_match(by_gt, identity, destination)
        if previous is None or current is None:
            missed += 1
        elif previous == current:
            confirmed += 1
            latencies.append((frame - destination[0]) / fps)
        else:
            broken += 1
    proposed = transitions(predictions)
    correct = false = unmatched = 0
    for identity, source, destination in proposed:
        previous = last_match(by_pred, identity, source)
        current, _ = first_match(by_pred, identity, destination)
        if previous is None or current is None:
            unmatched += 1
        elif previous == current:
            correct += 1
        else:
            false += 1
    return {"reference_handoffs": len(reference), "confirmed_handoffs": confirmed,
            "missed_handoffs": missed, "identity_breaks": broken,
            "handoff_recall": confirmed / len(reference) if reference else None,
            "predicted_handoffs": len(proposed), "correct_associations": correct,
            "false_associations": false, "unmatched_associations": unmatched,
            "handoff_precision": correct / len(proposed) if proposed else None,
            "median_confirmation_delay_s": float(np.median(latencies)) if latencies else None}


def evaluate(gt_path: Path, prediction_path: Path, cameras, start, stop, fps, threshold=0.5, label_scope="multicam") -> dict:
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    gt = read_rows(gt_path, cameras, start, stop)
    pred = read_rows(prediction_path, cameras, start, stop)
    if not gt:
        raise ValueError("No ground-truth observations in the selected cameras/frame range")
    if label_scope not in ("multicam", "sct"):
        raise ValueError("label_scope must be multicam or sct")
    result = {"scope": {"cameras": cameras, "start_frame": start, "stop_frame": stop, "iou_threshold": threshold, "label_scope": label_scope},
            "local_identity": identity_metrics(gt, pred, threshold, local=True),
            "metric_note": "ID metrics use global trajectory assignment. Handoff metrics are custom, with a 15-frame endpoint confirmation window. Validate parity against the dataset evaluator before a paper table."}
    if label_scope == "multicam":
        result["global_identity"] = identity_metrics(gt, pred, threshold)
        result["handoffs"] = handoff_metrics(gt, pred, threshold, fps)
    else:
        result["metric_note"] = "SCT-only labels: local identity accuracy is measured within each camera. Global identity/handoff metrics are omitted because these labels do not establish cross-camera correspondence."
    return result
