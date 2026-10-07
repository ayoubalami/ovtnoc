"""Resumable, common-kernel confirmation-deadline experiment (desktop v0.4).

Ground truth enters only offline evaluation. Full detection and exact local-stream
hash equality isolate the confirmation window from detector scheduling.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from rhd.cache import DetectionCache
from rhd.common import digest, file_hash, load_config, write_json
from rhd.evaluation import evaluate, read_rows
from rhd.runner import run, source_signature
from benchmark_support import sct_metrics, verify_vendor
from bounded_metrics import evaluate_availability

ROOT = Path(__file__).resolve().parents[1]


def definitions(cfg, deadlines=(0, 2, 4, 8), retention=8, minimum_two=True):
    if not deadlines or len(set(deadlines)) != len(deadlines):
        raise ValueError("Deadlines must be a nonempty set of distinct integers")
    if any(type(k) is not int or k < 0 for k in deadlines):
        raise ValueError("Deadlines must be nonnegative integer frame counts")
    if type(retention) is not int or retention < max(deadlines):
        raise ValueError("Fixed source retention must cover every tested deadline")
    if cfg["association"].get("kind") != "bounded":
        raise ValueError("A bounded source configuration is required")
    result = []
    for frames in deadlines:
        candidate = copy.deepcopy(cfg)
        candidate["association"].update(confirmation_frames=frames,
            minimum_observations=1, source_retention_frames=retention, diagnostics=True)
        result.append((f"delay_k{frames}_n1", "delay_only", candidate))
    if minimum_two:
        candidate = copy.deepcopy(cfg)
        candidate["association"].update(confirmation_frames=8,
            minimum_observations=2, source_retention_frames=retention, diagnostics=True)
        if retention < 8:
            raise ValueError("The separate K=8/minimum-two control needs retention >=8")
        result.append(("evidence_k8_n2", "observation_requirement", candidate))
    return result


def detector_trace(path):
    """Exclude priority scores; validate the complete actual detector sequence."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    trace = []
    for record in records:
        key = (record["camera"], record["frame"])
        if key in seen or type(record["detect"]) is not bool:
            raise ValueError("Invalid or duplicate detector decision")
        seen.add(key)
        trace.append((*key, record["detect"]))
    return digest(sorted(trace)), len(trace), sum(row[2] for row in trace)


def verify_hashes(root, hashes):
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Changed or missing completed artifact: {path}")


def tree_hashes(folder, root):
    return {str(p.relative_to(root)).replace("\\", "/"): file_hash(p)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def execute(args):
    cfg = load_config(args.config)
    score_start = cfg.get("score_start_frame", cfg["start_frame"])
    if type(score_start) is not int or not cfg["start_frame"] <= score_start < cfg["stop_frame"]:
        raise ValueError("score_start_frame must lie in the replay interval")
    candidates = definitions(cfg, tuple(args.deadlines), args.retention_frames,
                             not args.no_minimum_two_control)
    vendor = verify_vendor()
    with DetectionCache(args.cache, readonly=True) as cache:
        meta = cache.get_meta("detector")
        if meta.get("descriptor") != "vehicle-reid-0001_512_v1":
            raise ValueError("Use the verified observed vehicle descriptor cache")
        cache.validate_range(cfg["cameras"], cfg["start_frame"], cfg["stop_frame"], cfg["fps"])
        cache_signature = cache.signature(cfg["cameras"], cfg["start_frame"], cfg["stop_frame"])
    for path in (args.mcvt_gt, args.sct_gt):
        if not path.is_file():
            raise FileNotFoundError(path)
        read_rows(path, cfg["cameras"], cfg["start_frame"], cfg["stop_frame"])
    script_names = ("run_deadline_ablation_v0_4.py", "bounded_metrics.py",
                    "benchmark_support.py", "deadline_diagnostics_v0_4.py")
    options = {"schema_version": 1, "experiment": "common_kernel_deadline_v0_4",
        "config": cfg, "definitions": candidates, "policy": "full", "budget": 1.,
        "score_start_frame": score_start, "cache_signature": cache_signature,
        "core_source_sha256": source_signature(),
        "script_hashes": {n: file_hash(ROOT / "scripts" / n) for n in script_names},
        "vendor_manifest_sha256": file_hash(ROOT / "third_party/benchmark_manifest.json"),
        "ground_truth_hashes": {"mcvt": file_hash(args.mcvt_gt), "sct": file_hash(args.sct_gt)},
        "scope": "Previously exposed RoundaboutHD scene; confirmatory mechanism analysis, not an untouched independent test.",
        "rules": {"selection": "No model or hyperparameter selection from these results.",
                  "inference": "No GT enters tracking, candidate construction or identity assignment.",
                  "comparison": "Full detection; identical local prediction and detector decision hashes required.",
                  "n2": "Separate observation-requirement control, excluded from the delay-only claim.",
                  "warmup": "Track/coordinator state evolves causally through replay start to score start; no retroactive ID rewriting."}}
    signature = digest(options)
    progress_path = args.output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["signature"] != signature:
            raise ValueError("Study inputs changed: preserve this folder and use a new version")
        if file_hash(args.output / "protocol.json") != progress["protocol_sha256"]:
            raise ValueError("Prepared protocol changed")
        for record in progress["completed"].values():
            verify_hashes(args.output, record["hashes"])
        verify_hashes(args.output, progress.get("summary_hashes", {}))
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Refusing a nonempty unsigned output directory")
        progress = {"status": "prepared", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                    "signature": signature, "options": options, "completed": {}}
        write_json(args.output / "protocol.json", {"signature": signature, "options": options,
                                                  "prepared_at_utc": progress["prepared_at_utc"]})
        progress["protocol_sha256"] = file_hash(args.output / "protocol.json")
        write_json(progress_path, progress)
    if args.prepare_only:
        print(f"Verified/prepared protocol {signature}; {len(candidates)} controls", flush=True)
        return progress
    from deadline_diagnostics_v0_4 import build_report, write_report
    progress["status"] = "running"
    write_json(progress_path, progress)
    local_metrics = None
    for name, family, candidate in candidates:
        if name in progress["completed"]:
            print(f"Verified completed {name}", flush=True)
            continue
        folder = args.output / name
        attempt = 0
        while (folder / "manifest.json").is_file():
            state = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            if state["status"] == "complete":
                break
            attempt += 1
            folder = args.output / f"{name}_attempt{attempt}"
        summary = run(candidate, args.cache, folder, "full", 1.)
        latency = evaluate_availability(candidate, folder, args.mcvt_gt)
        metrics = evaluate(args.mcvt_gt, folder / "predictions.txt", cfg["cameras"],
                           score_start, cfg["stop_frame"], cfg["fps"])
        write_json(folder / "scored_metrics.json", metrics)
        if local_metrics is None:
            local_metrics = sct_metrics(read_rows(args.sct_gt, cfg["cameras"], score_start, cfg["stop_frame"]),
                read_rows(folder / "local_predictions.txt", cfg["cameras"], score_start, cfg["stop_frame"]),
                cfg["cameras"], score_start, cfg["stop_frame"])
        write_json(folder / "sct_metrics.json", local_metrics)
        diagnostic = build_report(folder, candidate, args.mcvt_gt, score_start_frame=score_start)
        write_report(diagnostic, folder / "diagnostics")
        trace_hash, decisions, actual_calls = detector_trace(folder / "decisions.jsonl")
        expected = len(cfg["cameras"]) * (cfg["stop_frame"] - cfg["start_frame"])
        if decisions != expected or actual_calls != expected or summary["detector_calls"] != expected:
            raise ValueError("Full-detection control does not have the complete common detector sequence")
        row = {"name": name, "family": family,
            "confirmation_frames": candidate["association"]["confirmation_frames"],
            "minimum_observations": candidate["association"]["minimum_observations"],
            "source_retention_frames": args.retention_frames, "detector_calls": actual_calls,
            "global_IDF1": metrics["global_identity"]["IDF1"],
            "local_IDF1": local_metrics["combined_counts"]["IDF1"],
            "local_HOTA": local_metrics["combined_counts"]["HOTA"],
            **{k: metrics["handoffs"][k] for k in ("handoff_recall", "handoff_precision",
                "reference_handoffs", "confirmed_handoffs", "correct_associations", "predicted_handoffs")},
            "topology_reference_handoffs": sum(r["reference_handoffs"] for r in diagnostic["reference_routes"]),
            "topology_confirmed_handoffs": sum(r["confirmed"] for r in diagnostic["reference_routes"]),
            "pending_p50_s": latency["deadline_statistics"]["pending_confirmation_delay"]["p50_s"],
            "pending_p95_s": latency["deadline_statistics"]["pending_confirmation_delay"]["p95_s"],
            "pending_max_s": latency["deadline_statistics"]["pending_confirmation_delay"]["max_s"],
            "deadline_violations": latency["deadline_statistics"]["commit_deadline_violations"],
            "local_predictions_sha256": file_hash(folder / "local_predictions.txt"),
            "detector_trace_sha256": trace_hash, "run_directory": str(folder.resolve())}
        row["topology_handoff_recall"] = (row["topology_confirmed_handoffs"]/row["topology_reference_handoffs"]
                                          if row["topology_reference_handoffs"] else None)
        row["pending_statistics_scope"] = "complete replay; scored entrant cohort in diagnostics"
        for available in latency["availability_curve"]:
            scored = evaluate(args.mcvt_gt, folder / available["prediction_file"], cfg["cameras"],
                              score_start, cfg["stop_frame"], cfg["fps"])
            row[f"IDF1_available_{round(available['allowed_delay_s']*1000)}ms"] = scored["global_identity"]["IDF1"]
        if row["deadline_violations"] or latency["deadline_statistics"]["publication_rows_over_configured_wait"]:
            raise ValueError("Configured identity publication deadline violated")
        for prior in progress["completed"].values():
            for key in ("local_predictions_sha256", "detector_trace_sha256", "detector_calls"):
                if row[key] != prior["row"][key]:
                    raise ValueError(f"Delay-only comparison changed the common local stream: {key}")
        progress["completed"][name] = {"row": row, "hashes": tree_hashes(folder, args.output)}
        write_json(progress_path, progress)
        print(f"{name}: IDF1={100*row['global_IDF1']:.2f}%; handoffs={row['confirmed_handoffs']}/{row['reference_handoffs']}; max pending={row['pending_max_s']}", flush=True)
    rows = [progress["completed"][name]["row"] for name, _, _ in candidates]
    zero = next((r for r in rows if r["name"] == "delay_k0_n1"), None)
    differences = [{"name": r["name"], "delta_IDF1_pp": 100*(r["global_IDF1"]-zero["global_IDF1"]),
                    "delta_confirmed_handoffs": r["confirmed_handoffs"]-zero["confirmed_handoffs"]}
                   for r in rows if r["family"] == "delay_only"] if zero else []
    write_json(args.output / "results.json", {"options": options, "rows": rows,
        "paired_differences_from_k0": differences,
        "integrity": {"identical_local_predictions": True, "identical_detector_sequence": True,
                      "identical_detector_call_count": True},
        "interpretation": "Mechanism experiment on an already observed scene. No deadline is selected using these evaluation scores. Negative and unchanged outcomes are retained."})
    with (args.output / "table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    progress["status"] = "complete"
    progress["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    progress["summary_hashes"] = {p.name: file_hash(p) for p in args.output.iterdir()
                                 if p.is_file() and p.name != "progress.json"}
    write_json(progress_path, progress)
    print(f"Complete: {args.output}", flush=True)
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mcvt-gt", type=Path, default=Path("data/prepared_labels/multicam_zero_based.txt"))
    parser.add_argument("--sct-gt", type=Path, default=Path("data/prepared_labels/sct_zero_based.txt"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadlines", nargs="+", type=int, default=[0, 2, 4, 8])
    parser.add_argument("--retention-frames", type=int, default=8)
    parser.add_argument("--no-minimum-two-control", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
