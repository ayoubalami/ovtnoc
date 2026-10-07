"""Compare fixed perception profiles exclusively on S02 [0,60) development data."""
from __future__ import annotations
import argparse
import copy
import csv
import json
from pathlib import Path

from cityflow_dev_v0_6.replay import run, source_signature
from cityflow_dev_v0_6.diagnostics import generate_report
from rhd.cache import DetectionCache
from rhd.common import digest, file_hash, write_json


PROFILES = {
    'legacy': dict(track_high_thresh=.6, track_low_thresh=.1, new_track_thresh=.6,
                   track_buffer=30, match_thresh=.9, fuse_score=True),
    'moderate': dict(track_high_thresh=.4, track_low_thresh=.1, new_track_thresh=.4,
                     track_buffer=30, match_thresh=.8, fuse_score=True),
    'standard': dict(track_high_thresh=.25, track_low_thresh=.1, new_track_thresh=.25,
                     track_buffer=30, match_thresh=.8, fuse_score=True),
}


def development_config(protocol, profile):
    cfg = copy.deepcopy(protocol)
    if cfg['scenario'] != 'S02':
        raise ValueError('This development driver only permits S02')
    cfg.update(version='cityflow_development_v0_6', replay_start_s=0.,
               score_start_s=0., stop_s=60.)
    cfg['tracker'] = copy.deepcopy(PROFILES[profile])
    cfg['phase'] = 'development_only'
    cfg['scope'] = 'S02 labeled prefix development; no independent test claim.'
    return cfg


def attempt_directory(root, tag):
    attempt = 1
    while True:
        directory = root / (tag if attempt == 1 else f'{tag}_attempt{attempt}')
        manifest = directory / 'manifest.json'
        if not directory.exists() or not any(directory.iterdir()):
            return directory
        if manifest.is_file() and json.loads(manifest.read_text())['status'] == 'complete':
            return directory
        attempt += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--front-end', required=True)
    p.add_argument('--box-mode', choices=['kalman','kalman_indexed','observed'], default='kalman_indexed')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    args = p.parse_args()
    cfg = json.loads(args.protocol.read_text())
    scene = json.loads(args.scene.read_text())
    if scene['scenario'] != 'S02' or cfg['scenario'] != 'S02':
        raise ValueError('Development is restricted to S02')
    with DetectionCache(args.cache, readonly=True) as cache:
        if digest(cache.get_meta('scene')) != digest(scene):
            raise ValueError('Scene and cache differ')
        detector = cache.get_meta('detector')
    signed = dict(protocol_sha256=file_hash(args.protocol), scene_sha256=file_hash(args.scene),
                  cache_sha256=file_hash(args.cache), core_signature=source_signature(),
                  driver_sha256=file_hash(Path(__file__)),
                  diagnostic_sha256=file_hash(Path(__file__).parents[1]/'src/cityflow_dev_v0_6/diagnostics.py'),
                  front_end=args.front_end, profiles=args.profiles, box_mode=args.box_mode,
                  interval_s=[0.,60.], detector=detector)
    signature = digest(signed)
    args.output.mkdir(parents=True, exist_ok=True)
    progress = args.output/'progress.json'
    if progress.is_file():
        state = json.loads(progress.read_text())
        if state['signature'] != signature:
            raise ValueError('Development inputs changed; use a new versioned folder')
        for name, expected in state.get('artifact_hashes', {}).items():
            if file_hash(args.output/name) != expected:
                raise ValueError(f'Development artifact changed: {name}')
        if state['status'] == 'complete':
            print(f'Verified completed development matrix: {args.output}', flush=True)
            return
    elif any(args.output.iterdir()):
        raise ValueError('Development output must be new')
    write_json(progress, dict(status='running', signature=signature, signed_inputs=signed))
    records = []
    for profile in args.profiles:
        candidate = development_config(cfg, profile)
        candidate['tracker_box_mode'] = args.box_mode
        directory = attempt_directory(args.output, profile)
        summary = run(candidate, args.cache, directory)
        diagnostic = directory/'endpoint_diagnosis'
        if diagnostic.exists():
            report = json.loads((diagnostic/'report.json').read_text())
            proof = json.loads((diagnostic/'proof.json').read_text())
            if proof['report_sha256'] != file_hash(diagnostic/'report.json'):
                raise ValueError('Diagnostic artifact changed')
        else:
            report = generate_report(args.scene, diagnostic, cache_path=args.cache,
                                     local_run=directory, global_run=directory)
        stages = report['stages']
        local, global_stage = stages['local_tracker'], stages['global_assignment']
        records.append(dict(front_end=args.front_end, tracker=profile,box_mode=args.box_mode,
                            detector_annotation_recall=stages['cached_detections']['annotation_recall'],
                            local_annotation_recall=local['annotation_recall'],
                            reference_handoffs=report['reference_handoffs'],
                            detected_both_endpoints=stages['cached_detections']['both_endpoints_covered'],
                            tracked_both_endpoints=local['both_endpoints_covered'],
                            same_global_identity=global_stage['same_global_identity'],
                            global_identity_break=global_stage['global_identity_break'],
                            detector_calls=summary['detector_calls'], run=str(directory)))
        print(json.dumps(records[-1]), flush=True)
        write_json(args.output/'results.json', records)
        artifact_hashes = {str(path.relative_to(args.output)): file_hash(path)
                           for path in directory.rglob('*') if path.is_file()}
        previous = json.loads(progress.read_text()).get('artifact_hashes', {})
        previous.update(artifact_hashes)
        write_json(progress, dict(status='running', signature=signature, signed_inputs=signed,
                                 completed=records, artifact_hashes=previous))
    with (args.output/'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    lines = ['# S02 prefix perception development (v0.6)', '',
             'Only [0,60) s is used. These are development diagnostics, not an independent result.', '',
             '| Front end | Tracker | Detection annotation recall | Tracking annotation recall | Both endpoints tracked | Same global identity |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for row in records:
        lines.append(f"| {row['front_end']} | {row['tracker']} | {100*row['detector_annotation_recall']:.2f}% | {100*row['local_annotation_recall']:.2f}% | {row['tracked_both_endpoints']}/{row['reference_handoffs']} | {row['same_global_identity']}/{row['reference_handoffs']} |")
    lines += ['', 'Association K=8, route geometry, appearance, stopped-vehicle policy, and all other gates remain fixed.',
              'Both-endpoint coverage measures spatial observability. Same global identity measures assignment correctness.',
              'CityFlow annotations cover multi-camera vehicles only; unmatched predictions are not all false positives.', '']
    (args.output/'results.md').write_text('\n'.join(lines), encoding='utf-8')
    artifacts = {str(path.relative_to(args.output)): file_hash(path)
                 for path in args.output.rglob('*') if path.is_file() and path != progress}
    write_json(progress, dict(status='complete', signature=signature, signed_inputs=signed,
                             completed=records, artifact_hashes=artifacts))


if __name__ == '__main__':
    main()
