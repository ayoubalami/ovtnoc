"""Select an S05 holdout from timing metadata and a strictly bounded GT prefix.

The reader stops before reading an identity/box field of the first out-of-prefix
row. It does not import the evaluation annotation reader, which reads whole GT
files before filtering. No detector, tracker, association or suffix scores enter
this selection. The final perception choice remains pending development results.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cityflow.data import contiguous_spans, read_pairs
from rhd.common import canonical, digest, file_hash, point_in_polygon, write_json


def read_prefix(path, meta, stop_s=120.):
    """Read sorted MOT rows only while offset+(frame1-1)/fps < stop_s."""
    rows, previous, seen = [], 0, set()
    prefix_hash = hashlib.sha256()
    first_unread_frame1 = None
    with Path(path).open('r', encoding='utf-8-sig', newline='') as source:
        while True:
            token = ''
            while True:
                char = source.read(1)
                if not char:
                    if token:
                        raise ValueError('Truncated MOT frame column')
                    break
                if char == ',':
                    break
                token += char
            if not char:
                break
            frame1 = int(token)
            time_s = meta['timestamp_offset_s']+(frame1-1)/meta['fps']
            if time_s >= stop_s:
                first_unread_frame1 = frame1
                break  # No identity or box column beyond the prefix is read.
            if frame1 < previous or frame1 < 1 or frame1 > meta['frames']:
                raise ValueError(f'Invalid/unsorted calibration frame in {path}')
            previous = frame1
            remainder = source.readline()
            prefix_hash.update((token+','+remainder).encode('utf-8'))
            values = remainder.rstrip('\r\n').split(',')
            if len(values) != 9:
                raise ValueError(f'Invalid MOT prefix row in {path}')
            identity = int(values[0]); x, y, width, height = map(float, values[1:5])
            if not np.isfinite([x, y, width, height]).all() or min(width, height) <= 0:
                raise ValueError('Invalid prefix box')
            key = (identity, frame1)
            if key in seen:
                raise ValueError('Duplicate prefix identity/frame')
            seen.add(key)
            rows.append({'camera': meta['camera'], 'id': identity,
                         'frame': frame1-1, 'time_s': time_s,
                         'box': [x, y, x+width, y+height]})
    return rows, {'file': str(Path(path).resolve()), 'rows_read': len(rows),
                  'interpreted_prefix_sha256': prefix_hash.hexdigest(),
                  'first_outside_frame_column_only': first_unread_frame1,
                  'last_interpreted_frame1': previous or None,
                  'stop_s_exclusive': stop_s}


def native_metadata(dataset):
    root = Path(dataset).resolve(); scenario = 'S05'
    timestamps = root/'cam_timestamp'/f'{scenario}.txt'
    framecounts = root/'cam_framenum'/f'{scenario}.txt'
    offsets = read_pairs(timestamps, float); counts = read_pairs(framecounts, int)
    cameras = {}
    for video in sorted((root/'validation'/scenario).glob('c*/vdo.avi')):
        camera = int(video.parent.name[1:]); cap = cv2.VideoCapture(str(video))
        try:
            if not cap.isOpened():
                raise ValueError(f'Unreadable video: {video}')
            meta = {'camera': camera, 'fps': cap.get(cv2.CAP_PROP_FPS),
                    'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'timestamp_offset_s': offsets[camera], 'video': str(video),
                    'video_bytes': video.stat().st_size,
                    'roi': str(video.parent/'roi.jpg'),
                    'roi_sha256': file_hash(video.parent/'roi.jpg')}
        finally:
            cap.release()
        if abs(meta['fps']-10.) > .001 or counts[camera] != meta['frames'] or offsets[camera] != 0.:
            raise ValueError('S05 metadata changed; revise the declared timing plan')
        cameras[str(camera)] = meta
    if set(map(int, cameras)) != set(offsets) or set(offsets) != set(counts):
        raise ValueError('Missing S05 video/clock/framecount')
    return {'dataset_root': str(root), 'split': 'validation', 'scenario': scenario,
            'cameras': cameras, 'timestamps_sha256': file_hash(timestamps),
            'framecounts_sha256': file_hash(framecounts),
            'location_map': str(root/'cam_loc'/'S0345.png'),
            'location_map_sha256': file_hash(root/'cam_loc'/'S0345.png'),
            'readme_sha256': file_hash(root/'ReadMe.txt'),
            'clock': 'offset_s + zero_based_local_frame/native_fps; integer microseconds',
            'timing_limit': 'Supplied start offsets only; transmission skips/stuck frames are not repaired.'}


def _point(row, scene):
    meta = scene['cameras'][str(row['camera'])]; x1, _, x2, y2 = row['box']
    return np.array([(x1+x2)/(2*meta['width']), y2/meta['height']])


def _zone(points, padding=.04):
    points = np.asarray(points); lo = np.maximum(points.min(0)-padding, 0)
    hi = np.minimum(points.max(0)+padding, 1)
    if np.any(hi <= lo):
        raise ValueError('Degenerate prefix endpoint zone')
    return [[float(lo[0]), float(lo[1])], [float(hi[0]), float(lo[1])],
            [float(hi[0]), float(hi[1])], [float(lo[0]), float(hi[1])]]


def fit_prefix_routes(rows, scene, stop_s=120., minimum_samples=5):
    pairs, exclusions = defaultdict(list), Counter()
    for identity, visits in contiguous_spans(rows).items():
        for source, destination in zip(visits, visits[1:]):
            if source[0]['camera'] == destination[0]['camera']:
                exclusions['same_camera'] += 1; continue
            if destination[0]['time_s'] <= source[-1]['time_s']:
                exclusions['overlap_or_nonpositive_gap'] += 1; continue
            if min(len(source), len(destination)) < 5:
                exclusions['short_span'] += 1; continue
            if (source[0]['frame'] == 0 or destination[0]['frame'] == 0 or
                    destination[-1]['time_s'] >= stop_s-.1 or source[-1]['time_s'] >= stop_s-.1):
                exclusions['prefix_boundary_censored'] += 1; continue
            pairs[(source[0]['camera'], destination[0]['camera'])].append((identity, source, destination))
    routes, statistics = [], []
    for (source_camera, destination_camera), samples in sorted(pairs.items()):
        stats = {'source': source_camera, 'destination': destination_camera,
                 'usable_prefix_transitions': len(samples)}
        if len(samples) < minimum_samples:
            stats['status'] = 'insufficient_transitions'; statistics.append(stats); continue
        exit_polygon = _zone([_point(s[-1], scene) for _, s, _ in samples])
        entry_polygon = _zone([_point(d[0], scene) for _, _, d in samples])
        velocities = []
        for _, source, _ in samples:
            prior, last = source[max(0, len(source)-6)], source[-1]
            velocities.append((_point(last, scene)-_point(prior, scene))/(last['time_s']-prior['time_s']))
        direction = np.median(velocities, axis=0)
        direction = direction/np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-12 else np.zeros(2)
        travel, crossing_records = [], []
        for identity, source, destination in samples:
            for index, row in enumerate(source):
                if index == 0:
                    continue
                delta = _point(row, scene)-_point(source[max(0, index-5)], scene)
                if point_in_polygon(_point(row, scene), exit_polygon) and (
                        np.linalg.norm(direction) == 0 or float(delta@direction) > 0):
                    elapsed = destination[0]['time_s']-row['time_s']
                    if elapsed > 0 and point_in_polygon(_point(destination[0], scene), entry_polygon):
                        travel.append(elapsed)
                        crossing_records.append({'id': identity, 'source_frame1': row['frame']+1,
                                                 'destination_frame1': destination[0]['frame']+1,
                                                 'travel_s': elapsed})
                    break
        stats['fitted_crossings'] = len(travel)
        if len(travel) < minimum_samples:
            stats['status'] = 'insufficient_crossings'; statistics.append(stats); continue
        lo, hi = np.quantile(travel, [.05, .95]); window = [max(0., float(lo)-.1), float(hi)+.1]
        routes.append({'source': source_camera, 'destination': destination_camera,
                       'exit_polygon': exit_polygon, 'entry_polygon': entry_polygon,
                       'direction': direction.tolist(), 'travel_time_s': window})
        stats.update({'status': 'supported', 'travel_time_s': window,
                      'median_travel_s': float(np.median(travel)), 'crossings': crossing_records})
        statistics.append(stats)
    return routes, statistics, dict(exclusions)


def select_four(scene, routes, statistics):
    supports = {(s['source'], s['destination']): s['fitted_crossings']
                for s in statistics if s['status'] == 'supported'}
    feasible = []
    for cameras in combinations(sorted(map(int, scene['cameras'])), 4):
        selected = [r for r in routes if r['source'] in cameras and r['destination'] in cameras]
        if not selected:
            continue
        coverage = min(scene['cameras'][str(c)]['frames']/10 for c in cameras)
        stop_s = float(min(360., math.floor(coverage))); start_s = stop_s-180.
        warmup = start_s-120.; maximum_prior = max(r['travel_time_s'][1] for r in selected)
        if warmup < maximum_prior+3.8:
            continue
        support = sum(supports[(r['source'], r['destination'])] for r in selected)
        feasible.append({'cameras': list(cameras), 'routes': selected,
                         'fitted_prefix_crossings': support, 'supported_routes': len(selected),
                         'replay_start_s': 120., 'score_start_s': start_s, 'stop_s': stop_s,
                         'warmup_s': warmup, 'maximum_route_upper_s': maximum_prior})
    if not feasible:
        raise ValueError('No four-camera subset meets the prefix support/warmup requirements')
    # Deterministic tie rule: greater support, then more routes, then smaller IDs.
    feasible.sort(key=lambda x: (-x['fitted_prefix_crossings'], -x['supported_routes'], x['cameras']))
    return feasible[0], [{k:v for k,v in item.items() if k != 'routes'} for item in feasible[:10]]


def render_prefix_audit(scene, selected, output):
    canvas = Image.new('RGB', (1280, 780), 'white'); draw = ImageDraw.Draw(canvas)
    for position, camera in enumerate(selected['cameras']):
        meta = scene['cameras'][str(camera)]; frame = 300
        cap = cv2.VideoCapture(meta['video'])
        try:
            for _ in range(frame):
                if not cap.grab():
                    raise ValueError('Missing prefix audit frame')
            ok, pixels = cap.read()
            if not ok:
                raise ValueError('Missing prefix audit frame')
        finally:
            cap.release()
        for route in selected['routes']:
            for cam_key, poly_key, color in [('source', 'exit_polygon', (0,0,255)),
                                              ('destination', 'entry_polygon', (0,255,0))]:
                if route[cam_key] != camera:
                    continue
                poly = np.rint(np.asarray(route[poly_key])*[meta['width'], meta['height']]).astype(np.int32)
                cv2.polylines(pixels, [poly], True, color, 4)
                cv2.putText(pixels, f"{route['source']}->{route['destination']}", tuple(poly[0]),
                            cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2)
        x, y = (position%2)*640, (position//2)*390
        canvas.paste(Image.fromarray(pixels[:,:,::-1]).resize((640,360)), (x,y+25))
        draw.text((x+10,y+6), f'c{camera:03d}: prefix t=30.0 s; RED exit / GREEN entry', fill='black')
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scene = native_metadata(args.dataset); rows, prefix_manifest = [], {}
    for camera, meta in scene['cameras'].items():
        camera_rows, evidence = read_prefix(Path(meta['video']).parent/'gt'/'gt.txt', meta)
        rows.extend(camera_rows); prefix_manifest[camera] = evidence
    routes, statistics, exclusions = fit_prefix_routes(rows, scene)
    selected, shortlist = select_four(scene, routes, statistics)
    for camera in selected['cameras']:
        meta = scene['cameras'][str(camera)]; meta['video_sha256'] = file_hash(Path(meta['video']))
    plan = {'version': 'cityflow_holdout_v0_6',
            'status': 'holdout_scene_routes_and_intervals_frozen_pending_development_model_selection',
            'dataset_root': scene['dataset_root'], 'split': 'validation', 'scenario': 'S05',
            'calibration_interval_s': [0.,120.], **selected,
            'prefix_topology_cameras': sorted(map(int, scene['cameras'])),
            'prefix_gt_rows_read': len(rows), 'prefix_read_manifest': prefix_manifest,
            'selection_rule': 'Exactly four cameras; maximize fitted positive-gap prefix crossings, then number of supported routes, then lexicographically smaller camera IDs. Score 180 seconds ending at min(360, floor(shortest camera duration)); require warmup >= maximum route upper +3.8 seconds.',
            'minimum_calibration_transitions_and_crossings': 5,
            'calibration_rules': 'Full-scene contiguous prefix GT topology precedes route/subset selection; positive visibility gap; spans >=5 observations; exclude recording/prefix censored spans; endpoint extrema +/-0.04 normalized padding; median five-frame velocity; first directional exit-zone crossing; travel quantiles 0.05/0.95 +/-0.1 seconds.',
            'all_prefix_route_statistics': statistics, 'prefix_exclusions': exclusions,
            'feasible_subset_shortlist': shortlist, 'scene_metadata_signature': digest(scene),
            'method_configuration': None,
            'development_policy': 'Choose perception/tracker settings using only S02 prefix development evidence. Freeze the selected variant and comparison controls before S05 replay or suffix evaluation. S02 suffix is already observed and cannot be used as an untouched confirmation set.',
            'suffix_access': 'No S05 GT identity/box field at or after 120 seconds read by this builder. No S05 method scores computed. Whole GT evaluation reader must not run until final configuration is frozen.',
            'future_evaluation_topology': 'Load full S05 GT across all 19 cameras after freeze, construct topology before selecting the four-camera routes and score interval; do not invent transitions by filtering out intervening cameras.',
            'scope': 'New physical location relative to S02 and RoundaboutHD, with disclosed labeled S05 prefix calibration. S03/S04/S05 share physical camera-location map S0345; S05 is not an unseen physical site relative to S03/S04.',
            'clock_limit': scene['timing_limit']}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/'holdout_plan.json'
    if target.exists():
        stored = json.loads(target.read_text())
        if canonical(stored) != canonical(plan):
            raise ValueError('Frozen holdout plan differs; use a new version')
        freeze = json.loads((args.output/'holdout_freeze.json').read_text())
        for name, expected in freeze['artifact_sha256'].items():
            if file_hash(args.output/name) != expected:
                raise ValueError(f'Frozen holdout artifact changed: {name}')
        print(f'Verified frozen holdout plan: {target}', flush=True); return
    write_json(target, plan); write_json(args.output/'holdout_scene_metadata.json', scene)
    render_prefix_audit(scene, selected, args.output/'holdout_prefix_zones.png')
    write_json(args.output/'holdout_freeze.json', {
        'status': plan['status'], 'builder_sha256': file_hash(Path(__file__)),
        'artifact_sha256': {name:file_hash(args.output/name) for name in (
            'holdout_plan.json','holdout_scene_metadata.json','holdout_prefix_zones.png')},
        'no_suffix_scores': True, 'prefix_identity_box_fields_only': True})
    print(json.dumps({'plan': str(target), 'cameras': selected['cameras'],
                      'routes': [[r['source'],r['destination']] for r in selected['routes']],
                      'prefix_crossings': selected['fitted_prefix_crossings'],
                      'replay': [selected['replay_start_s'],selected['stop_s']],
                      'score': [selected['score_start_s'],selected['stop_s']]}), flush=True)


if __name__ == '__main__':
    main()
