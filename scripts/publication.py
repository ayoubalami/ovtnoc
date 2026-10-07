"""Portable entry point for the unchanged, frozen experiment drivers.

Run from the repository root. Preparation rebinds paths and signs the relocated
configuration. It does not fit geometry or inspect evaluation annotations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
if 'YOLO_CONFIG_DIR' not in os.environ:
    settings = ROOT/'work/ultralytics'
    settings.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings)
sys.path.insert(0, str(ROOT/'src'))
from rhd.common import canonical, digest, file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bind(value, dataset):
    if isinstance(value, dict):
        return {bind(k, dataset): bind(v, dataset) for k, v in value.items()}
    if isinstance(value, list):
        return [bind(v, dataset) for v in value]
    if isinstance(value, str) and value.startswith('${CITYFLOW_ROOT}'):
        return str(dataset / value[len('${CITYFLOW_ROOT}'):].lstrip('/'))
    return value


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if canonical(read(path)) != canonical(value):
            raise ValueError(f'Prepared input changed: {path}. Use a new --work folder.')
    else:
        write_json(path, value)


def invoke(script, *args):
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT/'src') + os.pathsep + str(ROOT/'scripts')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    command = [sys.executable, '-B', str(ROOT/'scripts'/script), *map(str, args)]
    print('Running:', subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def prepare_cityflow(dataset, work):
    from cache_cityflow_yolo_v0_6 import verify_scene_assets
    from run_cityflow_holdout_v0_6 import derive_config
    dataset = dataset.resolve()
    templates = ROOT/'configs/cityflow'
    full = bind(read(templates/'holdout_scene_metadata.json'), dataset)
    selected = bind(read(templates/'selected_scene.json'), dataset)
    plan = bind(read(templates/'holdout_plan.json'), dataset)
    selection = read(templates/'selection.json')
    # Integrity checks read native input assets, never suffix identity labels.
    verify_scene_assets(selected)
    checks = [(dataset/'cam_timestamp/S05.txt', full['timestamps_sha256']),
              (dataset/'cam_framenum/S05.txt', full['framecounts_sha256']),
              (Path(full['location_map']), full['location_map_sha256']),
              (dataset/'ReadMe.txt', full['readme_sha256'])]
    for meta in full['cameras'].values():
        if not Path(meta['video']).is_file():
            raise FileNotFoundError(meta['video'])
        checks.append((Path(meta['roi']), meta['roi_sha256']))
    for path, expected in checks:
        if file_hash(path) != expected:
            raise ValueError(f'Frozen dataset asset differs: {path}')
    plan['scene_metadata_signature'] = digest(full)
    cfg = derive_config(plan, selection, selected, full)
    protocol = work/'protocols/cityflow'
    for name, value in [('holdout_plan.json', plan), ('holdout_scene_metadata.json', full),
                        ('selected_scene.json', selected), ('selection.json', selection)]:
        immutable(protocol/name, value)
    artifacts = {name: file_hash(protocol/name) for name in
                 ['holdout_plan.json', 'holdout_scene_metadata.json', 'selected_scene.json', 'selection.json']}
    immutable(protocol/'holdout_freeze.json', {
        'status': plan['status'], 'artifact_sha256': artifacts,
        'builder_sha256': file_hash(Path(__file__)),
        'no_suffix_scores': True, 'prefix_identity_box_fields_only': True,
        'purpose': 'Reproduction of previously published frozen geometry, with relocated input paths. '
                   'This is not a new prospective holdout selection.',
        'template_sha256': {p.name: file_hash(p) for p in sorted(templates.glob('*.json'))}})
    immutable(protocol/'replay_config.json', cfg)
    return protocol


def cityflow(args):
    work = args.work.resolve()
    protocol = prepare_cityflow(args.dataset, work)
    if args.stage == 'prepare':
        print(f'Prepared portable CityFlow inputs: {protocol}')
        return
    cache = work/'caches/cityflow_S05.sqlite'
    validation = work/'caches/cityflow_S05.validation.json'
    if args.stage in ('cache', 'all'):
        require_model(args.model)
        invoke('cache_cityflow_mask_interval_v0_6.py', '--scene', protocol/'selected_scene.json',
               '--model', args.model.resolve(), '--start-s', 120, '--stop-s', 340, '--output', cache)
        invoke('verify_cityflow_mask_interval_v0_6.py', '--scene', protocol/'selected_scene.json',
               '--cache', cache, '--model', args.model.resolve(), '--output', validation)
    if args.stage in ('run', 'all'):
        if not cache.is_file() or not validation.is_file():
            raise FileNotFoundError('Build and verify the cache first with --stage cache.')
        invoke('run_cityflow_holdout_v0_6.py', '--plan', protocol/'holdout_plan.json',
               '--selected-scene', protocol/'selected_scene.json', '--selection', protocol/'selection.json',
               '--cache', cache, '--cache-validation', validation, '--output', work/'runs/cityflow_S05',
               '--published-baseline-script', ROOT/'scripts/run_cityflow_published_v0_6.py')


def require_model(model):
    from vehicle_reid_support import MODEL_SHA256
    if model is None:
        raise ValueError('Provide --model with the downloaded vehicle-reid-0001 ONNX file.')
    if not model.is_file() or file_hash(model) != MODEL_SHA256:
        raise ValueError('Vehicle ReID model is missing or its SHA256 differs from the paper.')


def validate_roundabout_cache(dataset, config, cache_path):
    """Check all retained observations against the raw supplied detection archive."""
    import numpy as np
    from benchmark_support import ArchiveSource
    from rhd.cache import DetectionCache
    from vehicle_reid_support import MODEL_SHA256, in_regions
    counts = {}
    source = ArchiveSource(dataset, config)
    try:
        with DetectionCache(cache_path, readonly=True) as cache:
            detector = cache.get_meta('detector')
            if detector['reid_model_sha256'] != MODEL_SHA256:
                raise ValueError('Wrong ReID model in the cache')
            cache.validate_range(config['cameras'], config['start_frame'], config['stop_frame'], config['fps'])
            for camera in config['cameras']:
                meta = cache.get_meta(f'camera:{camera}')
                regions = detector['appearance_regions'][str(camera)]
                count = 0
                for frame in range(config['start_frame'], config['stop_frame']):
                    raw = source.frame(camera, frame)
                    actual, detector_ms, appearance_ms = cache.frame(camera, frame)
                    if len(raw) != len(actual) or detector_ms != 0 or not np.isfinite(appearance_ms) or appearance_ms < 0:
                        raise ValueError('Invalid raw observations or timing')
                    for a, b in zip(actual, raw):
                        if any(a[k] != b[k] for k in ['box', 'score', 'class_id']):
                            raise ValueError('Raw detection box, confidence or class changed')
                        feature = np.asarray(a['feature'])
                        selected = in_regions(a['box'], meta['width'], meta['height'], regions)
                        if feature.shape != (512,) or not np.isfinite(feature).all() or a['reid_computed'] != selected:
                            raise ValueError('Invalid observed descriptor or extraction region')
                        norm = float(np.linalg.norm(feature))
                        if (selected and abs(norm-1) > 1e-6) or (not selected and norm != 0):
                            raise ValueError('Invalid unit/zero descriptor contract')
                    count += len(actual)
                counts[str(camera)] = count
    finally:
        source.close()
    return {'status': 'complete', 'raw_detection_rows': counts, 'cache_sha256': file_hash(cache_path),
            'validator_sha256': file_hash(Path(__file__))}


def roundabout(args):
    from rhd.labels import prepare_labels
    from rhd.vision import discover
    dataset, work = args.dataset.resolve(), args.work.resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    cfg = read(ROOT/f'configs/roundabout/{args.cohort}.json')
    geometry_name = 'all4' if args.cohort == 'all4' else 'pair'
    geometry = read(ROOT/f'configs/roundabout/cache_geometry_{geometry_name}.json')
    folder = work/'protocols/roundabout'
    immutable(folder/f'{args.cohort}.json', cfg)
    immutable(folder/f'cache_geometry_{geometry_name}.json', geometry)
    if args.stage == 'prepare':
        # This step only checks file layout; label conversion is deferred to offline scoring.
        cameras = set(dict(discover(dataset)))
        if not set(cfg['cameras']) <= cameras:
            raise ValueError('Required RoundaboutHD camera videos are missing')
        print(f'Prepared {args.cohort}: [{cfg["start_frame"]},{cfg["stop_frame"]}) at {cfg["fps"]} FPS')
        return
    cache = work/f'caches/roundabout_{args.cohort}.sqlite'
    validation = cache.with_suffix('.validation.json')
    if args.stage in ('cache', 'all'):
        require_model(args.model)
        invoke('cache_vehicle_range.py', '--dataset', dataset, '--config', folder/f'cache_geometry_{geometry_name}.json',
               '--model', args.model.resolve(), '--start-frame', cfg['start_frame'],
               '--stop-frame', cfg['stop_frame'], '--output', cache)
        immutable(validation, validate_roundabout_cache(dataset, cfg, cache))
    if args.stage in ('run', 'all'):
        if not cache.is_file() or not validation.is_file():
            raise FileNotFoundError('Build and verify the cache first with --stage cache.')
        if read(validation).get('cache_sha256') != file_hash(cache):
            raise ValueError('Cache changed after validation')
        labels = work/'labels/roundabout'
        prepare_labels(dataset, labels, source_frame_base=1)
        output = work/f'runs/roundabout_{args.cohort}'
        invoke('run_roundabout_corrected_v0_6.py', '--config', folder/f'{args.cohort}.json',
               '--cache', cache, '--mcvt-gt', labels/'multicam_zero_based.txt',
               '--sct-gt', labels/'sct_zero_based.txt', '--output', output)
        if args.baseline:
            invoke('evaluate_published_association_v0_4.py', '--config', folder/f'{args.cohort}.json',
                   '--cache', cache, '--ground-truth', labels/'multicam_zero_based.txt',
                   '--local-run', output/'delay_k0_n1', '--output', work/f'runs/roundabout_{args.cohort}_baseline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='dataset_kind', required=True)
    for name, function in [('cityflow', cityflow), ('roundabout', roundabout)]:
        p = sub.add_parser(name)
        p.add_argument('--dataset', type=Path, required=True)
        p.add_argument('--work', type=Path, default=ROOT/'work')
        p.add_argument('--model', type=Path)
        p.add_argument('--stage', choices=['prepare', 'cache', 'run', 'all'], default='prepare')
        p.set_defaults(function=function)
        if name == 'roundabout':
            p.add_argument('--cohort', choices=['validation', 'test', 'all4'], default='validation')
            p.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()
