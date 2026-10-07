"""Freeze a developed configuration before opening CityFlow S05 suffix labels.

Inference uses a selected four-camera cache. Evaluation builds full 19-camera
GT topology before route/time filtering. The RoundaboutHD and CityFlow v0.5 core
modules are reused unchanged. This script does not extract detections.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from cityflow.data import annotation_rows
from cityflow.evaluation import evaluate_predictions, export_official_predictions, prepare_rois
from cityflow.handoffs import evaluate_handoffs
from cityflow.replay import _validate_config, _validate_cache
from cityflow_dev_v0_6.replay import run, source_signature
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, write_json

DEADLINES = [0, 2, 4, 8]
STATUS = 'frozen_before_annotation_access'
NATIVE_KEYS = ('camera', 'fps', 'frames', 'width', 'height', 'timestamp_offset_s',
               'video', 'video_sha256', 'roi', 'roi_sha256')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise ValueError(f'Immutable artifact changed: {path}')
    else:
        write_json(path, value)


def verify_artifacts(root, artifacts):
    root = Path(root).resolve()
    for name, expected in artifacts.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f'Study artifact changed or missing: {name}')


def validate_selection(selection):
    if selection.get('status') != 'frozen_after_development':
        raise ValueError('Selection must be frozen_after_development')
    development = selection.get('development', {})
    if development.get('scenario') != 'S02' or development.get('interval_s') != [0, 60]:
        raise ValueError('Selection must use the declared S02 [0,60) development interval')
    config = selection.get('config', {})
    if not isinstance(config, dict) or not {'tracker','association','communication','output_filter','seed','fps','perception'} <= set(config):
        raise ValueError('Selection is missing the declared inference configuration')
    if config['fps'] != 10.:
        raise ValueError('S05 deadline controls use the native 10 FPS reference clock')
    association = config['association']
    if association.get('minimum_observations') != 1 or association.get('source_retention_frames') != 8:
        raise ValueError('Delay controls require minimum_observations=1 and fixed source_retention_frames=8')
    expected = selection.get('detector_provenance', {})
    if (not isinstance(expected, dict) or not {'kind','descriptor'} <= set(expected) or
            not ('model' in expected or 'detector_assets' in expected)):
        raise ValueError('Selection must specify invariant kind/descriptor and model or detector asset provenance')
    # Optional explicit path/hash evidence is checked without opening annotations.
    evidence = {**selection.get('development',{}).get('artifact_hashes',{}),
                **selection.get('development_artifact_hashes',{})}
    for path, expected_hash in evidence.items():
        candidate = Path(path)
        if not candidate.is_file() and candidate.parts and candidate.parts[0] == 'desktop_impl':
            candidate = Path(__file__).parents[1]/Path(*candidate.parts[1:])
        if not candidate.is_file() or file_hash(candidate) != expected_hash:
            raise ValueError(f'Development selection evidence changed: {path}')


def derive_config(plan, selection, selected_scene, full_scene):
    validate_selection(selection)
    if plan.get('scenario') != 'S05' or plan.get('split') != 'validation':
        raise ValueError('Holdout must use validation/S05')
    if plan.get('status') != 'holdout_scene_routes_and_intervals_frozen_pending_development_model_selection':
        raise ValueError('Unknown holdout selection status')
    cameras = sorted(plan['cameras'])
    if len(cameras) != 4 or len(set(cameras)) != 4:
        raise ValueError('Frozen holdout requires exactly four distinct cameras')
    if set(map(int, selected_scene.get('cameras', {}))) != set(cameras):
        raise ValueError('Selected scene must contain exactly the frozen four cameras')
    if set(map(int, full_scene.get('cameras', {}))) != set(plan['prefix_topology_cameras']):
        raise ValueError('Full evaluation scene does not preserve prefix topology cameras')
    if digest(full_scene) != plan['scene_metadata_signature']:
        raise ValueError('Full frozen scene metadata changed')
    for scene in (selected_scene, full_scene):
        if (scene.get('scenario'),scene.get('split')) != ('S05','validation'):
            raise ValueError('Scene split/scenario changed')
        if Path(scene['dataset_root']).resolve() != Path(plan['dataset_root']).resolve():
            raise ValueError('Dataset root differs from frozen holdout plan')
    for camera in cameras:
        selected = selected_scene['cameras'][str(camera)]
        original = full_scene['cameras'][str(camera)]
        if any(selected.get(key) != original.get(key) for key in NATIVE_KEYS):
            raise ValueError(f'Selected camera {camera} is not a native derivative of frozen scene')
    calibration = plan['calibration_interval_s']
    if calibration != [0.,120.] or calibration[1] > plan['replay_start_s'] or not plan['replay_start_s'] <= plan['score_start_s'] < plan['stop_s']:
        raise ValueError('Calibration, replay and scoring intervals changed or overlap')
    cfg = copy.deepcopy(selection['config'])
    cfg.update({'version':'cityflow_holdout_v0_6', 'dataset':plan['dataset_root'],
        'split':'validation', 'scenario':'S05', 'cameras':cameras,
        'routes':copy.deepcopy(plan['routes']), 'replay_start_s':plan['replay_start_s'],
        'score_start_s':plan['score_start_s'], 'stop_s':plan['stop_s'],
        'scene_signature':digest(selected_scene), 'camera_metadata':selected_scene['cameras'],
        'route_calibration':{'start_s':calibration[0],'stop_s':calibration[1],
                             'source':'frozen S05 full-scene prefix only'},
        'scope':plan['scope'], 'clock_limit':plan['clock_limit'],
        'controls':{'confirmation_frames':DEADLINES,'minimum_observations':1,
                    'source_retention_frames':8,'only_deadline_changes':True}})
    cfg['association']['confirmation_frames'] = 8
    _validate_config(cfg)
    return cfg


def verify_cache_contract(cache_path, cfg, selected_scene, selection):
    with DetectionCache(cache_path, readonly=True) as cache:
        if digest(cache.get_meta('scene')) != digest(selected_scene):
            raise ValueError('Cache scene differs from selected native scene')
        actual = cache.get_meta('detector') or {}
        for key, expected in selection['detector_provenance'].items():
            if actual.get(key) != expected:
                raise ValueError(f'Cache invariant detector provenance changed: {key}')
        _validate_cache(cache, cfg)  # Native frame coverage and camera metadata.
        return actual


def freeze_before_annotations(output, signed):
    """Write/verify a complete inference freeze before any annotation reader runs."""
    output = Path(output); signature = digest(signed)
    path = output/'study_freeze.json'
    value = {'status':STATUS,'signature':signature,'signed_inputs':signed,
             'note':'Inference configuration, native scene, cache, sources and controls frozen before suffix annotation access.'}
    if path.exists():
        previous = read_json(path)
        if previous != value or digest(previous.get('signed_inputs')) != signature:
            raise ValueError('Frozen study input changed; use a new output folder')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('New study output must be empty')
        output.mkdir(parents=True, exist_ok=True)
        write_json(path, value)
    return signature


def require_annotation_freeze(output, signature):
    path = Path(output)/'study_freeze.json'
    if not path.is_file():
        raise ValueError('Suffix annotation access requires study_freeze.json')
    value = read_json(path)
    if (value.get('status') != STATUS or value.get('signature') != signature or
            digest(value.get('signed_inputs')) != signature):
        raise ValueError('Suffix annotation access refused: freeze signature changed')


def load_evaluation_inputs(output, signature, cfg, full_scene):
    """This is the only annotation-access entry point; the freeze gate runs first."""
    require_annotation_freeze(output, signature)
    root = Path(cfg['dataset']); source = root/'eval'/f"ground_truth_{cfg['split']}.txt"
    full_stop = max(m['timestamp_offset_s']+m['frames']/m['fps']
                    for m in full_scene['cameras'].values())
    # IMPORTANT: Full scene/full coverage first. Do not prefilter selected cameras.
    gt_rows = annotation_rows(full_scene, 0., full_stop)
    originals = {str(camera):file_hash(Path(meta['video']).parent/'gt'/'gt.txt')
                 for camera,meta in full_scene['cameras'].items()}
    cohort_text = '\n'.join(line for line in source.read_text().splitlines()
                            if int(line.split()[0]) in cfg['cameras'])+'\n'
    cohort = Path(output)/'ground_truth_S05_cohort.txt'
    if cohort.exists():
        if cohort.read_text(encoding='utf-8') != cohort_text:
            raise ValueError('Frozen official GT cohort changed')
    else:
        cohort.write_text(cohort_text, encoding='utf-8')
    provenance = {'study_signature':signature,'annotation_access_after_freeze':True,
        'full_topology_cameras':sorted(map(int,full_scene['cameras'])),
        'full_topology_interval_s':[0.,full_stop], 'full_topology_rows':len(gt_rows),
        'original_per_camera_gt_sha256':originals,'official_gt_sha256':file_hash(source),
        'official_selected_cohort_sha256':file_hash(cohort),
        'note':'Cohort filtering is official evaluation only. Custom handoffs use all scene cameras and entire native coverage before route/time filtering. No GT reaches replay.'}
    immutable_json(Path(output)/'annotation_provenance.json', provenance)
    return cohort, gt_rows


def score_handoffs(cfg, full_scene, gt_rows, predictions, availability=None):
    return evaluate_handoffs(full_scene, gt_rows, predictions, routes=cfg['routes'],
        replay_start_s=cfg['replay_start_s'],score_start_s=cfg['score_start_s'],
        score_stop_s=cfg['stop_s'],availability=availability)


def score_run(cfg, full_scene, directory, cohort, gt_rows, roi_directory):
    source = directory/'predictions.txt'; official = directory/'official_predictions.txt'
    if official.exists():
        temporary = directory/'official_predictions.verification.tmp'
        if temporary.exists():
            raise ValueError('Unexpected interrupted official-export verification file')
        try:
            export_official_predictions(source, temporary, cameras=cfg['cameras'])
            if file_hash(temporary) != file_hash(official):
                raise ValueError('Frozen official prediction export changed')
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        export_official_predictions(source, official, cameras=cfg['cameras'])
    # The official metric requires exactly the scored camera cohort's clocks.
    clocks = {c:full_scene['cameras'][str(c)] for c in cfg['cameras']}
    metrics = evaluate_predictions(official_eval_path=Path(cfg['dataset'])/'eval'/'eval.py',
        ground_truth_path=cohort,prediction_path=official,roi_directory=roi_directory,
        split=cfg['split'],cameras=cfg['cameras'],
        score_interval_s=[cfg['score_start_s'],cfg['stop_s']],camera_clocks=clocks)
    availability = directory/'availability.jsonl'
    metrics['handoffs'] = score_handoffs(cfg, full_scene, gt_rows, source,
                                       availability if availability.is_file() else None)
    immutable_json(directory/'metrics.json', metrics)
    return metrics


def pending_statistics(directory, start_s, stop_s):
    commits = [json.loads(line) for line in (directory/'messages.jsonl').read_text().splitlines()]
    commits = [r for r in commits if r.get('event') == 'identity_committed' and
               'first_time_s' in r and start_s <= r['first_time_s'] < stop_s]
    waits = [max(0.,r['time_s']-r['first_time_s']) for r in commits]
    return {'scored_pending_tracks':len(commits),
        'confirmation_outcomes':dict(Counter(r.get('diagnostics',{}).get('confirmation_outcome','unreported') for r in commits)),
        'median_wait_s':float(np.median(waits)) if waits else None,
        'p95_wait_s':float(np.quantile(waits,.95)) if waits else None,
        'maximum_wait_s':max(waits) if waits else None,
        'note':'Tracks first observed during the score interval; commitment is not GT correctness.'}


def attempt_directory(root, tag):
    directory = root/tag; attempt = 1
    while directory.exists() and any(directory.iterdir()):
        manifest = directory/'manifest.json'
        if manifest.is_file() and read_json(manifest).get('status') == 'complete':
            return directory
        attempt += 1; directory = root/f'{tag}_attempt{attempt}'
    return directory


def write_local_control(source, output):
    output.mkdir(parents=True, exist_ok=True); lines = []
    for line in source.read_text().splitlines():
        values = line.split(); values[1] = str(int(values[0])*1_000_000_000+int(values[1]))
        lines.append(' '.join(values))
    text = '\n'.join(lines)+('\n' if lines else ''); destination = output/'predictions.txt'
    if destination.exists():
        if destination.read_text() != text:
            raise ValueError('Camera-local control changed')
    else:
        destination.write_text(text, encoding='utf-8')
    immutable_json(output/'summary.json', {'method':'camera_local_ID_control',
        'shared_local_source_sha256':file_hash(source),
        'note':'Official singleton-ID filtering may remove all local IDs. Unfiltered metric retains them.'})


def result_row(method, k, metrics, pending, summary, directory):
    handoffs = metrics['handoffs']
    return {'method':method,'confirmation_frames':k,'deadline_s':k/10 if k is not None else None,
        'official_IDF1':metrics['official_benchmark']['idf1'],
        'roi_only_IDF1':metrics['roi_only_identity']['idf1'],
        'unfiltered_IDF1':metrics['unfiltered_identity']['idf1'],
        'reference_handoffs':handoffs['reference_handoffs'],
        'confirmed_handoffs':handoffs['confirmed_handoffs'],
        'missed_handoffs':handoffs['missed_handoffs'], 'identity_breaks':handoffs['identity_breaks'],
        'handoff_recall':handoffs['handoff_recall'], 'handoff_precision':handoffs['handoff_precision'],
        'detector_calls':summary['detector_calls'],
        'median_pending_wait_s':pending.get('median_wait_s'),
        'p95_pending_wait_s':pending.get('p95_wait_s'), 'run':str(directory)}


def build_report(records, cfg, output):
    keys = ['method','confirmation_frames','deadline_s','official_IDF1','roi_only_IDF1',
            'unfiltered_IDF1','reference_handoffs','confirmed_handoffs','missed_handoffs',
            'identity_breaks','handoff_recall','handoff_precision','detector_calls',
            'median_pending_wait_s','p95_pending_wait_s']
    with (output/'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    lines = ['# CityFlowV2 S05 developed-perception holdout (v0.6)','',cfg['scope'],'',
        f"Calibration [0,120) s; causal replay [{cfg['replay_start_s']:g},{cfg['stop_s']:g}) s; score [{cfg['score_start_s']:g},{cfg['stop_s']:g}) s.",'',
        f"Frozen cameras: {cfg['cameras']}; routes: "+', '.join(f"{r['source']} -> {r['destination']}" for r in cfg['routes'])+'.','',
        cfg['perception'],'',
        '| Method | Deadline (s) | Official IDF1 (%) | ROI-only IDF1 (%) | Unfiltered IDF1 (%) | Confirmed/reference handoffs |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in records:
        deadline = '--' if row['deadline_s'] is None else f"{row['deadline_s']:.1f}"
        lines.append(f"| {row['method']} | {deadline} | {100*row['official_IDF1']:.2f} | {100*row['roi_only_IDF1']:.2f} | {100*row['unfiltered_IDF1']:.2f} | {row['confirmed_handoffs']}/{row['reference_handoffs']} |")
    lines += ['','## Interpretation and provenance','',
        '- Selection uses only S02 [0,60) development evidence; S02 suffix was previously observed and is archived as observed evidence.',
        '- S05 route/zone calibration uses the disclosed [0,120) prefix. Scene, routes and intervals were chosen before S05 suffix scores. Exact configuration/code/cache were frozen before suffix labels were opened.',
        '- Full 19-camera GT topology precedes route/time selection. Overlaps, unsupported routes, source coverage exclusions and endpoint misses are preserved in metrics.json.',
        '- K=0/2/4/8 uses the same local boxes, detector calls, gates and assignment logic. Only the deadline varies; one required observation and eight-frame source retention are fixed.',
        '- The selected v0.6 adapter preserves original detection indices through both confidence partitions, so each track receives its own image descriptor. Observed-box mode uses measured boxes; Kalman state still supplies motion. Earlier v0.5 association results have a known descriptor-index limitation and remain archived.',
        '- Ultralytics rescales track_buffer from its 30 FPS reference: track_buffer=30 at 10 FPS means 10 lost frames, or 1 second. The holdout selection used a conservative 3.8-second planning allowance and 40-second warmup; this overestimate does not reduce warmup coverage.',
        '- Official evaluation drops singleton-camera predicted IDs. ROI-only and unfiltered IDF1 expose fresh-ID failures. Custom handoffs search 15 native endpoint frames and are not official CityFlow ranking metrics.',
        '- Published association controls, when requested, replace the original perception pipeline and disable overlap clustering. They are adapted association comparisons, not full published-method reproductions.',
        '- Cached replay timing excludes decoding/detector inference. timing_summary.json separately records measured extraction inference costs and replay costs. No real-time or edge claim is inferred.',
        '- S03/S04/S05 share physical location S0345. S05 is distinct from S02/RoundaboutHD but includes labeled target-scene calibration; no zero-shot transfer or full-benchmark claim is made.',
        '- All outcomes, including weak/negative results, are retained. No suffix score selects a preferred detector, tracker, route or deadline.','']
    (output/'results.md').write_text('\n'.join(lines), encoding='utf-8')


def collect_artifacts(output, final=False):
    mutable_reports = {'results.md','results.csv','progress.json'}
    return {str(path.relative_to(output)):file_hash(path) for path in sorted(output.rglob('*'))
            if path.is_file() and path.name != 'progress.json' and
            (final or path.parent != output or path.name not in mutable_reports)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan','selected-scene','selection','cache','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cache-validation', type=Path)
    parser.add_argument('--published-baseline-script', type=Path)
    args = parser.parse_args()
    plan = read_json(args.plan); selection = read_json(args.selection)
    selected_scene = read_json(args.selected_scene)
    holdout_freeze = read_json(args.plan.parent/'holdout_freeze.json')
    verify_artifacts(args.plan.parent, holdout_freeze['artifact_sha256'])
    if holdout_freeze['artifact_sha256'].get(args.plan.name) != file_hash(args.plan):
        raise ValueError('Plan is not the frozen holdout artifact')
    full_scene_path = args.plan.parent/'holdout_scene_metadata.json'
    full_scene = read_json(full_scene_path)
    cfg = derive_config(plan, selection, selected_scene, full_scene)
    detector = verify_cache_contract(args.cache, cfg, selected_scene, selection)
    if args.cache_validation:
        validation = read_json(args.cache_validation)
        if validation.get('status') != 'complete' or validation.get('cache_sha256') != file_hash(args.cache):
            raise ValueError('Completed cache verification missing or changed')
    baseline = args.published_baseline_script
    if baseline is not None and not baseline.is_file():
        raise FileNotFoundError(baseline)
    root = Path(__file__).resolve().parents[1]
    baseline_dependencies = {}
    if baseline is not None:
        for name in ('run_cityflow_published_v0_5.py','run_published_association_baseline_v0_4.py','benchmark_support.py'):
            path = Path(__file__).with_name(name)
            if path.is_file():
                baseline_dependencies[path.relative_to(root).as_posix()] = file_hash(path)
        vendor = root/'third_party'/'fast_online_mtmct_f920c396'
        for path in sorted(vendor.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                baseline_dependencies[path.relative_to(root).as_posix()] = file_hash(path)
    signed = {'config':cfg,'source_signature':source_signature(),
        'driver_sha256':file_hash(Path(__file__)), 'plan_sha256':file_hash(args.plan),
        'holdout_freeze_sha256':file_hash(args.plan.parent/'holdout_freeze.json'),
        'full_scene_sha256':file_hash(full_scene_path),
        'selected_scene_sha256':file_hash(args.selected_scene),
        'selection_sha256':file_hash(args.selection),'cache_sha256':file_hash(args.cache),
        'detector_metadata':detector,
        'cache_validation_sha256':file_hash(args.cache_validation) if args.cache_validation else None,
        'baseline_script_sha256':file_hash(baseline) if baseline else None,
        'baseline_dependencies':baseline_dependencies}
    signature = freeze_before_annotations(args.output, signed)
    immutable_json(args.output/'protocol.json', cfg)
    state_path = args.output/'progress.json'
    if state_path.exists():
        state = read_json(state_path)
        if state.get('signature') != signature:
            raise ValueError('Study progress signature changed')
        verify_artifacts(args.output, state.get('artifact_hashes', {}))
        if state.get('status') == 'complete':
            # Validate previously accessed GT without interpreting it again.
            provenance = read_json(args.output/'annotation_provenance.json')
            for camera, expected in provenance['original_per_camera_gt_sha256'].items():
                if file_hash(Path(full_scene['cameras'][camera]['video']).parent/'gt'/'gt.txt') != expected:
                    raise ValueError('Completed study GT changed')
            official_gt = Path(cfg['dataset'])/'eval'/f"ground_truth_{cfg['split']}.txt"
            if file_hash(official_gt) != provenance['official_gt_sha256']:
                raise ValueError('Completed official GT changed')
            print(f'Verified completed S05 holdout reused: {args.output}', flush=True); return
    write_json(state_path, {'status':'running','signature':signature,'completed':[]})
    # The freeze exists and has been verified before this first GT reader call.
    cohort, gt_rows = load_evaluation_inputs(args.output, signature, cfg, full_scene)
    rois = args.output/'ROIs'
    roi_provenance = prepare_rois(cfg['dataset'],cfg['split'],cfg['scenario'],cfg['cameras'],rois)
    immutable_json(args.output/'roi_provenance.json', roi_provenance)
    records, proofs, timings = [], [], []
    local_hash = trace_hash = None; reference = None; reference_summary = None
    for k in DEADLINES:
        candidate = copy.deepcopy(cfg); candidate['association']['confirmation_frames'] = k
        directory = attempt_directory(args.output, f'K{k}_n1')
        summary = run(candidate, args.cache, directory)
        hashes = (file_hash(directory/'local_predictions.txt'),file_hash(directory/'decisions.jsonl'))
        if local_hash is None:
            local_hash, trace_hash = hashes; reference = directory; reference_summary = summary
        if hashes != (local_hash,trace_hash):
            raise RuntimeError('Delay controls changed local boxes or detector call traces')
        if summary['detector_calls'] != reference_summary['detector_calls'] or summary['calls_by_camera'] != reference_summary['calls_by_camera']:
            raise RuntimeError('Delay controls changed detector call counts')
        metrics = score_run(cfg,full_scene,directory,cohort,gt_rows,rois)
        pending = pending_statistics(directory,cfg['score_start_s'],cfg['stop_s'])
        immutable_json(directory/'pending_diagnostics.json', pending)
        records.append(result_row(f'Common kernel K={k}',k,metrics,pending,summary,directory))
        proofs.append({'confirmation_frames':k,'run':str(directory),
                       'local_predictions_sha256':hashes[0],'decisions_sha256':hashes[1],
                       'detector_calls':summary['detector_calls'],'calls_by_camera':summary['calls_by_camera']})
        timings.append({'method':f'K{k}',**{key:summary.get(key) for key in (
            'recorded_selected_detector_time_s','recorded_selected_appearance_time_s',
            'replay_elapsed_s','replay_event_p95_ms','aggregate_camera_frames_per_wall_second','runtime_note')}})
        build_report(records,cfg,args.output)
        write_json(state_path, {'status':'running','signature':signature,'completed':records,
                               'artifact_hashes':collect_artifacts(args.output)})
        print(canonical(records[-1]), flush=True)
    local = args.output/'local_ID_control'; write_local_control(reference/'local_predictions.txt',local)
    extra = [('Camera-local ID control',local)]
    if baseline:
        for retention in ('native','development_route'):
            directory = args.output/f'published_association_{retention}'
            subprocess.run([sys.executable,str(baseline),'--selection',str(args.selection),
                '--plan',str(args.plan),'--scene',str(args.selected_scene),'--cache',str(args.cache),
                '--local-run',str(reference),'--output',str(directory),'--retention',retention],check=True)
            extra.append((f'Published association ({retention} retention)',directory))
    for method, directory in extra:
        metrics = score_run(cfg,full_scene,directory,cohort,gt_rows,rois)
        records.append(result_row(method,None,metrics,{},reference_summary,directory))
    immutable_json(args.output/'shared_stream_proof.json', {
        'all_deadline_controls_identical':True,'controls':proofs,
        'local_predictions_sha256':local_hash,'decisions_sha256':trace_hash,
        'fixed_minimum_observations':1,'fixed_source_retention_frames':8,
        'only_changed_parameter':'association.confirmation_frames'})
    immutable_json(args.output/'timing_summary.json', {
        'cache_detector_timing_available':detector.get('detector_timing_available',False),
        'methods':timings,'note':'Extraction detector/appearance inference and cached replay reported separately. Decode/I/O excluded from recorded inference costs; end-to-end streaming performance not measured.'})
    build_report(records,cfg,args.output)
    write_json(state_path, {'status':'complete','signature':signature,'completed':records,
                           'artifact_hashes':collect_artifacts(args.output,final=True)})
    print(f'CityFlow S05 holdout complete: {args.output}', flush=True)


if __name__ == '__main__':
    main()
