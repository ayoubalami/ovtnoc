from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import load_config, write_json


def main():
    parser = argparse.ArgumentParser(description="Desktop RoundaboutHD detector-scheduling research")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Safely extract/resume dataset assets from the downloaded ZIP")
    prepare.add_argument("--archive", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--include-reid", action="store_true", help="Also extract the image-based ReID subset")
    labels = sub.add_parser("prepare-labels", help="Make frame-aligned derivative labels and an explicit annotation audit")
    labels.add_argument("--dataset", type=Path, required=True)
    labels.add_argument("--output", type=Path, required=True)
    labels.add_argument("--source-frame-base", type=int, choices=(0, 1), required=True)
    audit = sub.add_parser("audit", help="Inspect extracted videos and label presence")
    audit.add_argument("--dataset", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    cache = sub.add_parser("cache", help="Extract/resume actual YOLO detections and bootstrap appearance")
    cache.add_argument("--dataset", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--model", type=Path, required=True, help="Existing pretrained .pt weights")
    cache.add_argument("--stop-frame", type=int, help="Exclusive; initially use 300")
    cache.add_argument("--device", default="cpu")
    cache.add_argument("--image-size", type=int, default=640)
    cache.add_argument("--confidence", type=float, default=0.1)
    cache.add_argument("--cameras", type=int, nargs="+", help="Camera IDs to extract; default all videos")
    run = sub.add_parser("run", help="Causal cached replay of one scheduling policy")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--cache", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--policy", choices=("full", "fixed", "independent", "coordinated"), default="full")
    run.add_argument("--budget", type=float, default=1.0)
    run.add_argument("--interval", type=int, default=1)
    run.add_argument("--ground-truth", type=Path, help="Used only after replay finishes, for scoring")
    run.add_argument("--label-scope", choices=("multicam", "sct"), default="multicam")
    live = sub.add_parser("live", help="Measure actual sequential desktop video inference using the same policy engine")
    live.add_argument("--config", type=Path, required=True)
    live.add_argument("--dataset", type=Path, required=True)
    live.add_argument("--model", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--policy", choices=("full", "fixed", "independent", "coordinated"), default="full")
    live.add_argument("--budget", type=float, default=1.0)
    live.add_argument("--interval", type=int, default=1)
    live.add_argument("--device", default="cpu")
    live.add_argument("--image-size", type=int, default=640)
    live.add_argument("--confidence", type=float, default=0.1)
    live.add_argument("--ground-truth", type=Path)
    live.add_argument("--label-scope", choices=("multicam", "sct"), default="multicam")
    sweep = sub.add_parser("sweep", help="Resumable baseline/coordination budget sweep")
    sweep.add_argument("--config", type=Path, required=True)
    sweep.add_argument("--cache", type=Path, required=True)
    sweep.add_argument("--ground-truth", type=Path, required=True)
    sweep.add_argument("--output", type=Path, required=True)
    sweep.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    sweep.add_argument("--intervals", type=int, nargs="+", default=[2, 3, 5])
    score = sub.add_parser("evaluate", help="Score MCVT predictions on every frame in a declared range")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--ground-truth", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path)
    score.add_argument("--iou", type=float, default=0.5)
    score.add_argument("--label-scope", choices=("multicam", "sct"), default="multicam")
    demo = sub.add_parser("demo", help="Synthetic pipeline check; not paper evidence")
    demo.add_argument("--output", type=Path, default=Path("runs/demo"))
    fit = sub.add_parser("fit-routes", help="Fit arrival priors using only a declared development-label interval")
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--cache", type=Path, required=True)
    fit.add_argument("--ground-truth", type=Path, required=True)
    fit.add_argument("--start-frame", type=int, required=True)
    fit.add_argument("--stop-frame", type=int, required=True)
    fit.add_argument("--min-samples", type=int, default=5)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            from .dataset import prepare_archive
            result = prepare_archive(args.archive, args.output, args.include_reid)
        elif args.command == "prepare-labels":
            from .labels import prepare_labels
            report = prepare_labels(args.dataset, args.output, args.source_frame_base)
            result = {"report": str((args.output / "annotation_audit.json").resolve()),
                      "source_frame_base": report["source_frame_base"], "prepared_frame_base": 0,
                      "multicam_rows": report["multicam"]["rows"], "sct_rows": report["sct"]["rows"],
                      "duplicate_rows": {k: report[k]["duplicate_extra_rows"] for k in ("multicam", "sct")}}
        elif args.command == "audit":
            from .vision import audit
            result = audit(args.dataset, args.output)
        elif args.command == "cache":
            from .vision import extract
            if args.stop_frame is not None and args.stop_frame < 1:
                raise ValueError("stop-frame must be positive")
            if args.image_size < 32 or not 0 < args.confidence <= 1:
                raise ValueError("invalid image size or confidence")
            extract(args.dataset, args.output, args.model, args.stop_frame, args.device, args.image_size, args.confidence, args.cameras)
            result = {"cache": str(args.output.resolve())}
        elif args.command in ("run", "live"):
            from .runner import run, score_run
            cfg = load_config(args.config)
            if args.command == "live":
                options = {"dataset": args.dataset, "model_path": args.model, "device": args.device,
                           "image_size": args.image_size, "confidence": args.confidence}
                result = run(cfg, None, args.output, args.policy, args.budget, args.interval, live=options)
            else:
                result = run(cfg, args.cache, args.output, args.policy, args.budget, args.interval)
            if args.ground_truth:
                score_run(cfg, args.output, args.ground_truth, args.label_scope)
        elif args.command == "sweep":
            from .runner import sweep
            result = sweep(load_config(args.config), args.cache, args.ground_truth, args.output, args.budgets, args.intervals)
        elif args.command == "evaluate":
            from .evaluation import evaluate
            cfg = load_config(args.config)
            result = evaluate(args.ground_truth, args.predictions, cfg["cameras"], cfg["start_frame"], cfg["stop_frame"], cfg["fps"], args.iou, args.label_scope)
            if args.output:
                write_json(args.output, result)
        elif args.command == "fit-routes":
            from .routes import fit_routes
            if args.output.exists():
                raise ValueError("Fitted config already exists; choose a new output to preserve it")
            result = fit_routes(load_config(args.config), args.cache, args.ground_truth, args.start_frame, args.stop_frame, args.min_samples)
            write_json(args.output, result)
        else:
            from .demo import demo
            result = demo(args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
