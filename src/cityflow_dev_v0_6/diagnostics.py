"""Development-only perception and endpoint diagnostics; never used for inference.

Ground truth in CityFlow labels only vehicles seen by multiple cameras. Therefore
this module reports annotation recall and unmatched predictions, not detector
precision or false positives. Raw detections have no cross-camera identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from cityflow.data import annotation_rows, contiguous_spans, event_time_us, load_roi, roi_accepts
from cityflow.replay import local_frame_range
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, iou_matrix, write_json


def _key(row):
    return row['camera'], row['frame'], row['id']


def _validated_rows(rows, scene, start_s, stop_s):
    result, seen = [], set()
    for original in rows:
        row = dict(original)
        if any(type(row.get(k)) is not int for k in ('camera', 'frame', 'id')):
            raise ValueError('Rows need integral camera, native frame and identity')
        if row['id'] < 0:
            raise ValueError('Rows need nonnegative identities')
        meta = scene['cameras'].get(str(row['camera']))
        if meta is None:
            raise ValueError('Row outside declared scene cameras')
        timestamp = event_time_us(meta, row['frame']) / 1e6
        if 'time_s' in row and (not math.isfinite(row['time_s']) or abs(row['time_s']-timestamp) > 1.1e-6):
            raise ValueError('Row timestamp disagrees with native camera clock')
        row['time_s'] = timestamp
        box = row['box']
        if len(box) != 4 or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Invalid diagnostic box')
        if not start_s <= timestamp < stop_s:
            continue
        key = _key(row)
        if key in seen:
            raise ValueError('Duplicate camera/identity/native frame diagnostic row')
        seen.add(key)
        result.append(row)
    return result


def development_references(prefix_gt, routes, start_s=0., stop_s=60.):
    """Reproduce fit_prefix's contiguous topology and boundary/short-span gates.

    All prefix cameras precede route selection. No label beyond stop_s is used.
    The 0.1 s censor margin matches v0.5's S02 calibration, whose cameras are 10Hz.
    """
    if not 0 <= start_s < stop_s:
        raise ValueError('Invalid development reference interval')
    configured = {tuple(r) if not isinstance(r, dict) else (r['source'], r['destination']) for r in routes}
    if any(len(r) != 2 or r[0] == r[1] for r in configured):
        raise ValueError('Invalid reference route')
    spans = contiguous_spans(prefix_gt)
    references, exclusions = [], Counter()
    for identity, visits in sorted(spans.items()):
        for source, destination in zip(visits, visits[1:]):
            if source[0]['camera'] == destination[0]['camera']:
                exclusions['same_camera'] += 1
                continue
            if destination[0]['time_s'] <= source[-1]['time_s']:
                exclusions['overlap'] += 1
                continue
            if min(len(source), len(destination)) < 5:
                exclusions['short_span'] += 1
                continue
            if (source[0]['frame'] == 0 or destination[0]['frame'] == 0 or
                    destination[-1]['time_s'] >= stop_s-.1 or source[-1]['time_s'] >= stop_s-.1):
                exclusions['prefix_censored'] += 1
                continue
            if (source[0]['camera'], destination[0]['camera']) not in configured:
                exclusions['unconfigured_route'] += 1
                continue
            if source[-1]['time_s'] < start_s or destination[0]['time_s'] < start_s:
                exclusions['before_development_start'] += 1
                continue
            references.append({'identity': identity, 'source': source, 'destination': destination,
                               'gap_s': destination[0]['time_s']-source[-1]['time_s']})
    return references, dict(exclusions)


def match_rows(gt, predictions, threshold=.5):
    """One-to-one, IoU-gated frame assignment, with actual matched row evidence."""
    if not 0 < threshold <= 1:
        raise ValueError('Invalid IoU threshold')
    grouped_gt, grouped_pred = defaultdict(list), defaultdict(list)
    for row in gt:
        grouped_gt[(row['camera'], row['frame'])].append(row)
    for row in predictions:
        grouped_pred[(row['camera'], row['frame'])].append(row)
    matches = {}
    for key in sorted(grouped_gt.keys() & grouped_pred.keys()):
        gs, ps = grouped_gt[key], grouped_pred[key]
        ious = iou_matrix([r['box'] for r in gs], [r['box'] for r in ps])
        left, right = linear_sum_assignment(np.where(ious >= threshold, 1-ious, 1e6))
        for a, b in zip(left, right):
            if ious[a, b] >= threshold:
                matches[_key(gs[a])] = {'prediction': ps[b], 'iou': float(ious[a, b])}
    return matches


def _frame_evidence(gt_row, matches):
    result = {'camera': gt_row['camera'], 'native_frame1': gt_row['frame']+1,
              'time_s': gt_row['time_s'], 'gt_box': gt_row['box']}
    match = matches.get(_key(gt_row))
    if match is None:
        result['matched'] = False
    else:
        result.update({'matched': True, 'prediction_id': match['prediction']['id'],
                       'prediction_box': match['prediction']['box'], 'iou': match['iou']})
        if 'score' in match['prediction']:
            result['prediction_score'] = match['prediction']['score']
    return result


def _raw_gate_evidence(gt_row, raw_grouped, threshold, minimum_score):
    candidates = raw_grouped.get((gt_row['camera'], gt_row['frame']), [])
    ious = iou_matrix([gt_row['box']], [r['box'] for r in candidates]).ravel()
    valid = [(r, float(iou)) for r, iou in zip(candidates, ious) if iou >= threshold]
    above = [(r, iou) for r, iou in valid if r['score'] >= minimum_score]
    accepted = [(r, iou) for r, iou in above if r['roi_accepted']]
    best = max(zip(candidates, ious), key=lambda pair: pair[1], default=None)
    reason = ('no_raw_iou_candidate' if not valid else 'all_iou_candidates_below_confidence'
              if not above else 'all_above_confidence_iou_candidates_rejected_by_roi'
              if not accepted else 'eligible_raw_iou_candidate_exists')
    result = {'iou_candidates': len(valid), 'above_confidence_iou_candidates': len(above),
              'roi_accepted_iou_candidates': len(accepted), 'gate_evidence': reason}
    if best is not None:
        row, iou = best
        result['best_raw_candidate'] = {'box': row['box'], 'score': row['score'],
                                        'iou': float(iou), 'roi_accepted': row['roi_accepted']}
    return result


def diagnose(scene, prefix_gt, stage_rows, *, routes=((7,9), (8,6)), start_s=0., stop_s=60.,
             threshold=.5, endpoint_frames=15, raw_all_scores=None, minimum_score=.1,
             identity_stages=('global_assignment',)):
    """Pure diagnostic API. Stage rows and prefix GT use zero-based native indices.

    Endpoint coverage means a spatially matched box in each endpoint window. For
    detector/local stages it is only an observable upper bound, not a successful
    cross-camera association. Identity equality is evaluated only for declared
    global stages, because local/detector IDs are camera-scoped or artificial.
    """
    if not 0 <= start_s < stop_s or not 0 < threshold <= 1 or not 0 <= minimum_score <= 1:
        raise ValueError('Invalid diagnostic interval/threshold')
    if type(endpoint_frames) is not int or endpoint_frames < 1:
        raise ValueError('Endpoint window must contain a positive integer frame count')
    full_prefix_gt = _validated_rows(prefix_gt, scene, 0., stop_s)
    gt = [r for r in full_prefix_gt if r['time_s'] >= start_s]
    stages = {name: _validated_rows(rows, scene, start_s, stop_s) for name, rows in stage_rows.items()}
    matches = {name: match_rows(gt, rows, threshold) for name, rows in stages.items()}
    references, exclusions = development_references(full_prefix_gt, routes, start_s, stop_s)
    raw_grouped = defaultdict(list)
    if raw_all_scores is not None:
        for row in _validated_rows(raw_all_scores, scene, start_s, stop_s):
            raw_grouped[(row['camera'], row['frame'])].append(row)
    summaries = {}
    for name, rows in stages.items():
        table = []
        for camera in sorted(map(int, scene['cameras'])):
            n_gt = sum(r['camera'] == camera for r in gt)
            n_pred = sum(r['camera'] == camera for r in rows)
            matched = sum(key[0] == camera for key in matches[name])
            table.append({'camera': camera, 'gt_observations': n_gt, 'matched_gt_observations': matched,
                          'unmatched_gt_observations': n_gt-matched, 'annotation_recall': matched/n_gt if n_gt else None,
                          'prediction_observations': n_pred, 'unmatched_prediction_observations': n_pred-matched})
        matched = len(matches[name])
        summaries[name] = {'gt_observations': len(gt), 'matched_gt_observations': matched,
                           'annotation_recall': matched/len(gt) if gt else None,
                           'prediction_observations': len(rows), 'unmatched_prediction_observations': len(rows)-matched,
                           'per_camera': table, 'reference_handoffs': len(references),
                           'source_endpoint_covered': 0, 'destination_endpoint_covered': 0,
                           'both_endpoints_covered': 0, 'missing_source_only': 0,
                           'missing_destination_only': 0, 'missing_both_endpoints': 0,
                           'routes': []}
        if name in identity_stages:
            summaries[name].update({'same_global_identity': 0, 'global_identity_break': 0})
    route_counts = {name: defaultdict(Counter) for name in stages}
    examples = []
    for index, reference in enumerate(references):
        source, destination = reference['source'], reference['destination']
        source_end, destination_start = source[-1]['frame'], destination[0]['frame']
        endpoint_rows = {
            'source': [r for r in reversed(source) if source_end-r['frame'] < endpoint_frames and r['time_s'] >= start_s],
            'destination': [r for r in destination if r['frame']-destination_start < endpoint_frames and r['time_s'] >= start_s]}
        pair = (source[0]['camera'], destination[0]['camera'])
        example = {'reference_index': index, 'identity': reference['identity'], 'source': pair[0],
                   'destination': pair[1], 'source_exit_native_frame1': source_end+1,
                   'destination_entry_native_frame1': destination_start+1, 'gap_s': reference['gap_s'],
                   'stages': {}}
        if raw_all_scores is not None:
            example['raw_gate_evidence'] = {endpoint: [dict(_frame_evidence(row, {}),
                **_raw_gate_evidence(row, raw_grouped, threshold, minimum_score))
                for row in rows] for endpoint, rows in endpoint_rows.items()}
        for name in stages:
            summary, counts = summaries[name], route_counts[name][pair]
            counts['reference_handoffs'] += 1
            evidence = {endpoint: [_frame_evidence(row, matches[name]) for row in rows]
                        for endpoint, rows in endpoint_rows.items()}
            source_match = next((r for r in evidence['source'] if r['matched']), None)
            destination_match = next((r for r in evidence['destination'] if r['matched']), None)
            outcome = ('both_endpoints_covered' if source_match and destination_match else
                       'missing_source_only' if destination_match else 'missing_destination_only'
                       if source_match else 'missing_both_endpoints')
            for metric, increment in [('source_endpoint_covered', bool(source_match)),
                                      ('destination_endpoint_covered', bool(destination_match)), (outcome, True)]:
                summary[metric] += int(increment)
                counts[metric] += int(increment)
            identity_outcome = None
            if name in identity_stages and source_match and destination_match:
                identity_outcome = ('same_global_identity' if source_match['prediction_id'] == destination_match['prediction_id']
                                    else 'global_identity_break')
                summary[identity_outcome] += 1
                counts[identity_outcome] += 1
            example['stages'][name] = {'outcome': outcome, 'identity_outcome': identity_outcome,
                                       'source_match': source_match, 'destination_match': destination_match,
                                       'endpoint_frames': evidence}
        examples.append(example)
    for name in summaries:
        summaries[name]['routes'] = [{'source': pair[0], 'destination': pair[1], **dict(counts)}
                                     for pair, counts in sorted(route_counts[name].items())]
    return {'version': 'cityflow_endpoint_diagnosis_v0_6', 'scenario': scene.get('scenario'),
            'development_interval_s': [start_s, stop_s], 'iou_threshold': threshold,
            'endpoint_native_frames': endpoint_frames, 'minimum_supplied_detection_score': minimum_score,
            'reference_handoffs': len(references), 'reference_exclusions': exclusions,
            'stages': summaries, 'reference_outcomes': examples,
            'cohort_rule': 'All-camera prefix contiguous GT topology before route filtering; positive gaps; >=5 observations per span; same fit_prefix start-frame and stop-minus-0.1s censor gates.',
            'limits': ['Development prefix only; this is not an independent test result.',
                       'Only multi-camera vehicles are annotated. Unmatched predictions are not declared false positives.',
                       'Raw/local endpoint box coverage is an upper bound on observable handoffs, not an identity association score.',
                       'Raw gate candidate evidence precedes one-to-one assignment; eligible candidates can lose assignment through competition.',
                       'Native clock uses supplied offsets; skipped/stuck frames are not repaired.']}


def read_raw_stages(scene, start_s, stop_s, detector='mask_rcnn', minimum_score=.1):
    if detector not in ('mask_rcnn', 'yolo3', 'ssd512'):
        raise ValueError('Only supplied raw detector files accepted')
    all_scores, above_score, with_roi, inputs = [], [], [], {}
    for camera_text, meta in scene['cameras'].items():
        camera = int(camera_text)
        path = Path(meta['video']).parent / 'det' / f'det_{detector}.txt'
        roi = load_roi(Path(meta['roi']))
        if roi.shape != (meta['height'], meta['width']):
            raise ValueError('ROI dimensions disagree with the camera')
        inputs[str(path)] = file_hash(path)
        inputs[str(Path(meta['roi']))] = file_hash(Path(meta['roi']))
        if meta.get('roi_sha256') != inputs[str(Path(meta['roi']))]:
            raise ValueError('Official ROI changed after scene freeze')
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split(',')
            if len(fields) != 10 or int(fields[1]) != -1:
                raise ValueError(f'Not a raw supplied detector row: {path}:{number}')
            frame = int(fields[0])-1
            timestamp = event_time_us(meta, frame)/1e6
            x, y, w, h, score = map(float, fields[2:7])
            if not np.isfinite([x,y,w,h,score]).all() or w <= 0 or h <= 0 or not 0 <= score <= 1:
                raise ValueError(f'Invalid raw detection: {path}:{number}')
            if not start_s <= timestamp < stop_s:
                continue
            row = {'camera': camera, 'id': number, 'frame': frame, 'time_s': timestamp,
                   'box': [x,y,x+w,y+h], 'score': score}
            row['roi_accepted'] = roi_accepts(row['box'], roi)
            all_scores.append(row)
            if score >= minimum_score:
                above_score.append(row)
                if row['roi_accepted']:
                    with_roi.append(row)
    return all_scores, {'supplied_before_roi': above_score, 'supplied_after_roi': with_roi}, inputs


def read_cache_rows(scene, cache_path, start_s, stop_s):
    rows = []
    with DetectionCache(cache_path, readonly=True) as cache:
        cached_scene = cache.get_meta('scene')
        if canonical(cached_scene) != canonical(scene):
            raise ValueError('Cache does not match the declared scene')
        for camera_text, meta in scene['cameras'].items():
            camera = int(camera_text)
            if canonical(cache.get_meta(f'camera:{camera}')) != canonical(meta):
                raise ValueError('Cache camera metadata does not match the declared scene')
            begin, end = local_frame_range(meta, start_s, stop_s)
            for frame in range(begin, end):
                detections, _, _ = cache.frame(camera, frame)
                for index, detection in enumerate(detections):
                    rows.append({'camera': camera, 'frame': frame, 'id': index,
                                 'box': detection['box'], 'score': detection['score']})
        detector = cache.get_meta('detector')
    return rows, detector


def read_prediction_rows(path, scene, start_s, stop_s):
    # Import lazily: matching diagnostics do not require the benchmark evaluator.
    from cityflow.evaluation import read_submission
    table = read_submission(path, cameras=map(int, scene['cameras']), allow_float_boxes=True)
    rows = []
    for values in table.itertuples(index=False, name=None):
        camera, identity, frame1, x, y, w, h, _, _ = values
        rows.append({'camera': int(camera), 'id': int(identity), 'frame': int(frame1)-1,
                     'box': [float(x),float(y),float(x+w),float(y+h)]})
    return _validated_rows(rows, scene, start_s, stop_s)


def render_markdown(report):
    rows = ['# CityFlow development endpoint diagnosis', '',
            f"Scenario: {report['scenario']}; development interval [{report['development_interval_s'][0]:g}, {report['development_interval_s'][1]:g}) seconds.", '',
            f"Eligible reference handoffs: {report['reference_handoffs']}. IoU gate: {report['iou_threshold']:g}; endpoint window: {report['endpoint_native_frames']} native frames.", '',
            '| Stage | GT boxes matched | Annotation recall | Source covered | Destination covered | Both endpoints covered |',
            '|---|---:|---:|---:|---:|---:|']
    for name, stage in report['stages'].items():
        recall = f"{100*stage['annotation_recall']:.2f}%" if stage['annotation_recall'] is not None else 'n/a'
        rows.append(f"| {name} | {stage['matched_gt_observations']}/{stage['gt_observations']} | {recall} | {stage['source_endpoint_covered']} | {stage['destination_endpoint_covered']} | {stage['both_endpoints_covered']} |")
    rows.extend(['', '## Per-camera annotation recall', '', '| Stage | Camera | Matched GT boxes | Annotation recall | Unmatched prediction observations |', '|---|---:|---:|---:|---:|'])
    for name, stage in report['stages'].items():
        for camera in stage['per_camera']:
            recall = f"{100*camera['annotation_recall']:.2f}%" if camera['annotation_recall'] is not None else 'n/a'
            rows.append(f"| {name} | {camera['camera']} | {camera['matched_gt_observations']}/{camera['gt_observations']} | {recall} | {camera['unmatched_prediction_observations']} |")
    rows.extend(['', '## Interpretation limits', ''])
    rows.extend(f'- {limit}' for limit in report['limits'])
    rows.extend(['', 'Complete per-reference endpoint frame, box, match, score and ROI evidence is in `report.json`.', ''])
    return '\n'.join(rows)


def generate_report(scene_path, output, *, start_s=0., stop_s=60., routes=((7,9),(8,6)),
                    cache_path=None, local_run=None, global_run=None, detector='mask_rcnn', minimum_score=.1):
    scene_path, output = Path(scene_path), Path(output)
    scene = json.loads(scene_path.read_text())
    if scene.get('scenario') != 'S02' or not 0 <= start_s < stop_s <= 60:
        raise ValueError('v0.6 endpoint diagnosis is restricted to the declared S02 development prefix [0,60)')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty diagnostic output directory')
    gt = annotation_rows(scene, 0., stop_s)
    raw_all, stages, inputs = read_raw_stages(scene, start_s, stop_s, detector, minimum_score)
    inputs[str(scene_path)] = file_hash(scene_path)
    for meta in scene['cameras'].values():
        path = Path(meta['video']).parent / 'gt' / 'gt.txt'
        inputs[str(path)] = file_hash(path)
    cache_provenance = None
    if cache_path is not None:
        cache_path = Path(cache_path)
        stages['cached_detections'], cache_provenance = read_cache_rows(scene, cache_path, start_s, stop_s)
        inputs[str(cache_path)] = file_hash(cache_path)
    for run, name, filename in ((local_run, 'local_tracker', 'local_predictions.txt'),
                                (global_run, 'global_assignment', 'predictions.txt')):
        if run is not None:
            path = Path(run)/filename
            manifest = Path(run)/'manifest.json'
            if not manifest.is_file():
                raise ValueError('Diagnostic tracker runs need a completed signed manifest')
            manifest_data = json.loads(manifest.read_text())
            if manifest_data.get('status') != 'complete':
                raise ValueError('Diagnostic tracker runs must be complete')
            config = manifest_data.get('options', {}).get('config', {})
            if (config.get('replay_start_s') != start_s or config.get('stop_s') != stop_s or
                    config.get('scenario') != scene['scenario']):
                raise ValueError('Local/global run must replay exactly the declared development prefix')
            if manifest_data.get('artifact_hashes', {}).get(filename) != file_hash(path):
                raise ValueError('Diagnostic tracking input changed after run completion')
            stages[name] = read_prediction_rows(path, scene, start_s, stop_s)
            inputs[str(path)] = file_hash(path)
            inputs[str(manifest)] = file_hash(manifest)
    report = diagnose(scene, gt, stages, routes=routes, start_s=start_s, stop_s=stop_s,
                      raw_all_scores=raw_all, minimum_score=minimum_score)
    report['cache_provenance'] = cache_provenance
    report['proof'] = {'input_sha256': inputs, 'implementation_sha256': file_hash(Path(__file__)),
                       'gt_reader_interval_s': [0., stop_s], 'read_suffix_ground_truth': False,
                       'routes': [list(r) for r in routes]}
    source_root = Path(__file__).parents[1]
    report['proof']['dependency_sha256'] = {str(relative): file_hash(source_root/relative)
        for relative in (Path('cityflow/data.py'), Path('cityflow/replay.py'), Path('rhd/cache.py'), Path('rhd/common.py'))}
    report['proof']['content_signature'] = digest(report)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'report.json', report)
    (output/'report.md').write_text(render_markdown(report), encoding='utf-8')
    write_json(output/'proof.json', {'report_sha256': file_hash(output/'report.json'),
                                    'markdown_sha256': file_hash(output/'report.md'), **report['proof']})
    return report
