"""Diagnose fixed-perception CityFlow endpoints on the declared development prefix."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cityflow_dev_v0_6.diagnostics import generate_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--start-s', type=float, default=0.)
    parser.add_argument('--stop-s', type=float, default=60.)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--local-run', type=Path)
    parser.add_argument('--global-run', type=Path)
    parser.add_argument('--detector', choices=('mask_rcnn', 'yolo3', 'ssd512'), default='mask_rcnn')
    parser.add_argument('--minimum-score', type=float, default=.1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = generate_report(args.scene, args.output, start_s=args.start_s, stop_s=args.stop_s,
                             cache_path=args.cache, local_run=args.local_run, global_run=args.global_run,
                             detector=args.detector, minimum_score=args.minimum_score)
    print(json.dumps({'output': str(args.output), 'reference_handoffs': report['reference_handoffs'],
                      'stages': {name: {'annotation_recall': stage['annotation_recall'],
                                       'both_endpoints_covered': stage['both_endpoints_covered']}
                                 for name, stage in report['stages'].items()}}), flush=True)


if __name__ == '__main__':
    main()
