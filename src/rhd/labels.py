from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .common import digest, file_hash, write_json
from .dataset import find_ground_truth


def _audit_rows(rows):
    seen, conflicts, duplicate_extra = {}, [], 0
    counts, identities, ranges = Counter(), defaultdict(set), {}
    for number, values in enumerate(rows, 1):
        camera, identity, frame = map(int, values[:3])
        counts[camera] += 1
        identities[camera].add(identity)
        lo, hi = ranges.get(camera, (frame, frame))
        ranges[camera] = (min(lo, frame), max(hi, frame))
        key = (camera, identity, frame)
        if key in seen:
            duplicate_extra += 1
            first_line, first_values = seen[key]
            conflicts.append({"camera": camera, "id": identity, "frame": frame, "first_line": first_line,
                              "extra_line": number, "identical": first_values == values,
                              "boxes": [first_values[3:7], values[3:7]]})
        else:
            seen[key] = number, values
    return {"rows": len(rows), "identities": len({int(r[1]) for r in rows}),
            "camera_rows": dict(counts), "camera_identities": {c: len(v) for c, v in identities.items()},
            "frame_bounds": ranges, "duplicate_extra_rows": duplicate_extra,
            "duplicate_details": conflicts}


def prepare_labels(dataset: Path, output: Path, source_frame_base: int):
    """Preserve source data, shift declared frame base, and report every duplicate.

    Conflicting annotations are retained, so evaluation remains fail-fast if a
    selected interval contains them. No arbitrary deduplication is performed.
    """
    if source_frame_base not in (0, 1):
        raise ValueError("source_frame_base must be 0 or 1")
    multi = find_ground_truth(dataset)
    sct_files = sorted(dataset.glob("imagesc*/SCT/SCT_GT.txt"))
    if not sct_files:
        raise FileNotFoundError("No SCT/SCT_GT.txt files found")
    source_hashes = {str(p.resolve()): file_hash(p) for p in [multi, *sct_files]}
    signature = digest({"source_hashes": source_hashes, "source_frame_base": source_frame_base,
                        "converter": "rhd_labels_v1"})
    report_path = output / "annotation_audit.json"
    if report_path.is_file():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous["signature"] != signature:
            raise ValueError("Prepared labels belong to different inputs; choose a new output directory")
        for name, expected in previous["artifact_hashes"].items():
            if not (output / name).is_file() or file_hash(output / name) != expected:
                raise ValueError(f"Prepared annotation changed: {output / name}")
        return previous
    if output.exists() and any(output.iterdir()):
        raise ValueError("Prepared-label output directory is not empty")
    multi_rows = []
    for line in multi.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 9:
            raise ValueError("MCVT annotation needs 9 fields")
        values[2] -= source_frame_base
        multi_rows.append(values)
    sct_rows = []
    for path in sct_files:
        camera = int(path.parents[1].name.removeprefix("imagesc"))
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            if len(values) != 7:
                raise ValueError(f"SCT annotation needs 7 fields: {path}")
            frame, local_id, x1, y1, x2, y2, _ = values
            if local_id != int(local_id) or not 0 <= local_id < 1_000_000:
                raise ValueError("SCT ID is outside the supported camera namespace")
            sct_rows.append([camera, camera * 1_000_000 + local_id, frame - source_frame_base,
                             x1, y1, x2 - x1, y2 - y1, -1, -1])
    for values in multi_rows + sct_rows:
        if not all(math.isfinite(v) for v in values) or any(v != int(v) for v in values[:3]) or values[0] < 1 or values[1] < 1 or values[2] < 0 or values[5] <= 0 or values[6] <= 0:
            raise ValueError(f"Invalid normalized annotation row: {values}")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("multicam_zero_based.txt", multi_rows), ("sct_zero_based.txt", sct_rows)):
        temporary = output / (name + ".tmp")
        with temporary.open("w", encoding="utf-8") as f:
            for v in rows:
                f.write(f"{int(v[0])} {int(v[1])} {int(v[2])} " + " ".join(f"{x:.6f}" for x in v[3:]) + "\n")
        temporary.replace(output / name)
    report = {"signature": signature, "source_hashes": source_hashes,
              "source_frame_base": source_frame_base, "prepared_frame_base": 0,
              "source_data_preserved": True, "multicam": _audit_rows(multi_rows), "sct": _audit_rows(sct_rows),
              "label_scope_note": "MCVT and SCT cover different identity/observation sets. SCT IDs are namespaced per camera and cannot establish cross-camera identity correctness.",
              "conflict_note": "Duplicate rows are retained and listed. Scoring an interval containing a duplicate fails until it is explicitly resolved or excluded.",
              "artifact_hashes": {name: file_hash(output / name) for name in ("multicam_zero_based.txt", "sct_zero_based.txt")}}
    write_json(report_path, report)
    return report
