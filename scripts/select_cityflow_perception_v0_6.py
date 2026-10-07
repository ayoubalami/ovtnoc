"""Freeze perception using S02 prefix spatial evidence, before S05 scoring."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from rhd.cache import DetectionCache
from rhd.common import digest, file_hash, write_json
from run_cityflow_development_v0_6 import PROFILES


def rank(row):
    # Association outcomes do not choose perception or the held-out deadline.
    return (-row['tracked_both_endpoints'], -row['local_annotation_recall'],
            row['front_end'], row['tracker'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development', type=Path, nargs='+', required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--cache', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Selection is immutable; use a new version if required')
    records, evidence = [], {}
    for directory in args.development:
        state = json.loads((directory/'progress.json').read_text())
        if state['status'] != 'complete' or state['signed_inputs']['interval_s'] != [0.,60.]:
            raise ValueError('Perception selection requires completed S02 [0,60) matrices')
        if digest(state['signed_inputs']) != state['signature']:
            raise ValueError('Development signature is invalid')
        for name, expected in state['artifact_hashes'].items():
            if file_hash(directory/name) != expected:
                raise ValueError(f'Development artifact changed: {name}')
        values = json.loads((directory/'results.json').read_text())
        if any(row['reference_handoffs'] != 36 for row in values):
            raise ValueError('Development cohorts differ')
        records.extend(values)
        evidence[str(directory/'progress.json')] = file_hash(directory/'progress.json')
    chosen = min(records, key=rank)
    configuration = json.loads(args.protocol.read_text())
    expected_cache = json.loads((Path(chosen['run'])/'manifest.json').read_text())['options']['cache_sha256']
    matches = [path for path in args.cache if file_hash(path) == expected_cache]
    if len(matches) != 1:
        raise ValueError('Provide the unique cache matching the selected completed run')
    with DetectionCache(matches[0], readonly=True) as cache:
        detector = cache.get_meta('detector')
    keys = ('tracker','association','communication','output_filter','seed','fps','perception')
    cfg = {key: copy.deepcopy(configuration[key]) for key in keys}
    cfg['tracker'] = copy.deepcopy(PROFILES[chosen['tracker']])
    cfg['tracker_box_mode'] = chosen.get('box_mode','kalman')
    cfg['perception'] = (f"{detector['model']}, confidence >=0.1; native images, official image-only ROI, "
                         "vehicle-reid-0001; preserved observation indices and measured observation boxes; no dataset training.")
    variable = {'scene_json_sha256','scene_signature','interval','raw_file_hashes','builder_sha256'}
    invariant_detector = {key:value for key,value in detector.items() if key not in variable}
    write_json(args.output, dict(status='frozen_after_development', version='cityflow_perception_selection_v0_6',
         config=cfg, detector_provenance=invariant_detector,
         development_detector_provenance=detector,
         selected_record=chosen, development=dict(scenario='S02', interval_s=[0.,60.],
         criterion='Maximize spatial coverage of both reference handoff endpoints; then maximize local annotation recall; ties lexicographic. Cross-camera association outcomes and S05 scores do not choose perception.',
         candidates=records, artifact_hashes=evidence),
         driver_sha256=file_hash(Path(__file__)), profile_driver_sha256=file_hash(Path(__file__).with_name('run_cityflow_development_v0_6.py'))))
    print(json.dumps(chosen), flush=True)


if __name__ == '__main__':
    main()
