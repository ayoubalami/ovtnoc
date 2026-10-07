"""Reproduce two frozen published-association adaptations and score offline."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
from rhd.common import digest, file_hash, load_config, write_json
from rhd.evaluation import evaluate, read_rows
from bounded_metrics import evaluate_availability
from run_deadline_ablation_v0_4 import verify_hashes, tree_hashes
from run_published_association_baseline_v0_4 import run_baseline, COMMIT


def execute(args):
    cfg=load_config(args.config)
    score_start=cfg.get("score_start_frame",cfg["start_frame"])
    read_rows(args.ground_truth,cfg["cameras"],cfg["start_frame"],cfg["stop_frame"])
    options={"config":cfg,"gt_sha256":file_hash(args.ground_truth),
        "local_manifest_sha256":file_hash(args.local_run/"manifest.json"),
        "driver_sha256":file_hash(Path(__file__)),"variants":["native","development_route"],
        "score_start_frame":score_start,"upstream_commit":COMMIT}
    protocol=args.output/"evaluation_protocol.json"
    results=args.output/"results.json"
    if results.exists():
        previous=json.loads(results.read_text())
        verify_hashes(args.output,previous["artifact_hashes"])
        # Each original baseline verifies its exact config/cache/local inputs.
        for variant in ("native","development_route"):
            run_baseline(cfg,args.cache,args.local_run,args.output/variant,variant)
        print(f"Verified completed baseline study {args.output}",flush=True)
        return previous
    if protocol.exists():
        if json.loads(protocol.read_text())["signature"] != digest(options):
            raise ValueError("Changed baseline evaluation protocol; use a new folder")
    else:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(protocol,{"signature":digest(options),"options":options,
            "prepared_at_utc":datetime.now(timezone.utc).isoformat(),
            "gt_scope":"Offline evaluation only; GT not passed to association."})
    rows=[]
    score_cfg=copy.deepcopy(cfg)
    score_cfg["association"]={"kind":"immediate","confirmation_frames":0}
    for variant in ("native","development_route"):
        folder=args.output/variant
        summary=run_baseline(cfg,args.cache,args.local_run,folder,variant)
        latency=evaluate_availability(score_cfg,folder,args.ground_truth)
        metrics=evaluate(args.ground_truth,folder/"predictions.txt",cfg["cameras"],score_start,cfg["stop_frame"],cfg["fps"])
        write_json(folder/"scored_metrics.json",metrics)
        row={"variant":variant,"retention_frames":summary["retention_frames"],
            "match_threshold":summary["match_threshold"],
            "global_IDF1":metrics["global_identity"]["IDF1"],
            "local_IDF1_mcvt":metrics["local_identity"]["IDF1"],
            **{k:metrics["handoffs"][k] for k in ("handoff_recall","handoff_precision","reference_handoffs","confirmed_handoffs")},
            "detector_calls":summary["detector_calls"],"prediction_rows":summary["prediction_rows"],
            "descriptor_observations":summary["actual_descriptor_observations_aligned"],
            "rows_without_descriptor_history":summary["prediction_rows_without_descriptor_history"],
            "rows_without_descriptor_fraction":summary["prediction_rows_without_descriptor_history"]/summary["prediction_rows"],
            "publication_delay_max_s":latency["publication_delay"]["max_s"],
            "deadline_violations":latency["deadline_statistics"]["commit_deadline_violations"]}
        if row["publication_delay_max_s"] != 0 or row["deadline_violations"]:
            raise ValueError("Immediate baseline has delayed output")
        for name in ("local_predictions.txt","decisions.jsonl"):
            if file_hash(folder/name) != file_hash(args.local_run/name):
                raise ValueError("Baseline changed the shared local stream")
        rows.append(row)
        print(f"{variant}: global IDF1={100*row['global_IDF1']:.2f}%; handoffs={row['confirmed_handoffs']}/{row['reference_handoffs']}",flush=True)
    report={"status":"complete","options":options,"scope":{"cameras":cfg["cameras"],
        "start_frame":cfg["start_frame"],"stop_frame":cfg["stop_frame"],"score_start_frame":score_start},
        "rows":rows,"artifact_hashes":tree_hashes(args.output,args.output),
        "caveats":["CPU association-only adaptation, not the full published detector/tracker/descriptor pipeline.",
          "Threshold0.65 and no-camera-revisit gate preserved. Native and development-derived retention both reported.",
          "Zone-restricted observed descriptors aligned at IoU>=0.5 on actual detector decisions; missing descriptor history disclosed.",
          "Same local boxes and detector exposures, no baseline identity feedback to scheduling, no GT input.",
          "Reported global metrics use the scored interval; supplementary availability metrics cover full replay."]}
    write_json(results,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("config","cache","local-run","ground-truth","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
