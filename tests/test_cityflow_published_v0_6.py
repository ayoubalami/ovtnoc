"""Synthetic original-camera/native-clock contracts for the S05 association adapter."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_cityflow_published_v0_6 import (
    CAMERA_ALIASES, Track, config_from_artifacts, preflight, run_baseline,
)
from cityflow.replay import local_frame_range, native_time_us
from rhd.cache import DetectionCache
from rhd.common import canonical, digest, file_hash, write_json
from rhd.evaluation import read_rows


def artifacts():
    selection = {'status':'frozen_after_development',
        'development':{'scenario':'S02','interval_s':[0.,60.]},
        'config':{'tracker':{'track_buffer':30}, 'association':{'kind':'bounded','confirmation_frames':8},
                  'communication':{'delay_s':0.}, 'output_filter':{'kind':'none'},
                  'seed':7,'fps':10.,'perception':'synthetic shared pretrained cache'}}
    plan = {'scenario':'S05', 'split':'validation','calibration_interval_s':[0.,.1],
        'cameras':list(CAMERA_ALIASES),'replay_start_s':.1,'score_start_s':4.,'stop_s':8.,
        'routes':[{'source':29,'destination':33,'travel_time_s':[0.,6.]}]}
    scene = {'scenario':'S05','split':'validation','dataset_root':'synthetic_unused_data',
        'cameras':{str(c):{'camera':c,'fps':10.,'frames':80,'width':100,'height':100,
                         'timestamp_offset_s':offset}
                   for c,offset in zip(CAMERA_ALIASES,[0.,.061,.421,.660])}}
    return selection,plan,scene


def stream(tmp_path, *, overlap=False, short_camera=False, no_descriptor=False):
    selection, plan, scene = artifacts()
    if short_camera:
        scene['cameras']['29']['frames'] = 5
    cfg = config_from_artifacts(selection,plan,scene)
    visible = {(29,f) for f in range(2,6)}
    if short_camera:
        visible |= {(34,f) for f in range(3,7)}
    elif overlap:
        visible |= {(34,1)}
    else:
        visible |= {(33,f) for f in range(22,26)}
        visible |= {(34,f) for f in range(72,75)}
    feature = np.zeros(512)
    if not no_descriptor:
        feature[0] = 1.
    cache_path = tmp_path/'cache.sqlite'
    decisions,rows = [],[]
    with DetectionCache(cache_path) as cache:
        cache.ensure_meta('scene',scene)
        cache.ensure_meta('detector',{'descriptor':'vehicle-reid-0001_512_v1','kind':'synthetic'})
        for camera_text,meta in scene['cameras'].items():
            camera = int(camera_text)
            cache.ensure_meta(f'camera:{camera}',meta)
            for local_frame in range(*local_frame_range(meta,cfg['replay_start_s'],cfg['stop_s'])):
                frame1 = local_frame+1
                detections=[]
                if (camera,frame1) in visible:
                    detections=[{'box':[10.,10.,20.,20.],'score':.9,'class_id':2,'feature':feature.tolist()}]
                    rows.append(f'{camera} 1 {frame1} 10.000000 10.000000 10.000000 10.000000 -1 -1\n')
                cache.add_frame(camera,local_frame,detections)
                decisions.append({'camera':camera,'frame':frame1,'detect':True,
                                  'time_s':native_time_us(meta,local_frame)/1e6})
        cache.commit()
    local = tmp_path/'local'
    local.mkdir()
    decisions.sort(key=lambda r:(r['time_s'],r['camera']))
    (local/'decisions.jsonl').write_text(''.join(canonical(d)+'\n' for d in decisions))
    (local/'local_predictions.txt').write_text(''.join(rows))
    summary = {**{k:cfg[k] for k in ('cameras','scenario','split','replay_start_s','score_start_s','stop_s')},
        'policy':'full','detector_call_fraction':1., 'native_camera_metadata':scene['cameras'],
        'detector_calls':len(decisions),'opportunities':len(decisions),
        'calls_by_camera':{str(c):sum(d['camera']==c for d in decisions) for c in CAMERA_ALIASES}}
    write_json(local/'summary.json',summary)
    options={'config':cfg,'cache_sha256':file_hash(cache_path),'source_signature':'synthetic source'}
    names=('decisions.jsonl','local_predictions.txt','summary.json')
    write_json(local/'manifest.json',{'status':'complete','options':options,'signature':digest(options),
        'artifact_hashes':{n:file_hash(local/n) for n in names}})
    return cfg,cache_path,local


def test_inventory_aliases_are_internal_and_export_original_one_based_frames(tmp_path):
    cfg,cache,local = stream(tmp_path)
    native,derived = tmp_path/'native',tmp_path/'derived'
    native_summary=run_baseline(cfg,cache,local,native)
    derived_summary=run_baseline(cfg,cache,local,derived,'development_route')
    native_rows,derived_rows=read_rows(native/'predictions.txt'),read_rows(derived/'predictions.txt')
    assert {r['camera'] for r in native_rows}=={29,33,34}
    assert not {6,7,8,9}&{r['camera'] for r in native_rows}
    assert {r['id'] for r in native_rows if r['camera'] in (29,33)}=={1}
    assert {r['id'] for r in native_rows if r['camera']==34}=={2}
    assert {r['id'] for r in derived_rows}=={1}
    spatial=lambda records:sorted((r['camera'],r['frame'],tuple(r['box'])) for r in records)
    assert spatial(read_rows(local/'local_predictions.txt'))==spatial(native_rows)==spatial(derived_rows)
    assert native_summary['retention_s']==3
    assert derived_summary['retention_s']==7
    assert native_summary['match_threshold']==derived_summary['match_threshold']==.65
    assert native_summary['original_to_native_alias']=={str(k):v for k,v in CAMERA_ALIASES.items()}
    assert native_summary['prediction_rows_without_descriptor_history']==0
    assert file_hash(native/'decisions.jsonl')==file_hash(local/'decisions.jsonl')
    assert run_baseline(cfg,cache,local,native)==native_summary


def test_asynchronous_active_source_exclusion_and_immediate_availability(tmp_path):
    cfg,cache,local=stream(tmp_path,overlap=True)
    out=tmp_path/'baseline'
    run_baseline(cfg,cache,local,out)
    rows=read_rows(out/'predictions.txt')
    assert {r['id'] for r in rows if r['camera']==29}=={1}
    assert {r['id'] for r in rows if r['camera']==34}=={2}
    availability=[json.loads(s) for s in (out/'availability.jsonl').read_text().splitlines()]
    with DetectionCache(cache,readonly=True) as detections:
        for a in availability:
            meta=detections.get_meta(f"camera:{a['camera']}")
            assert a['frame']==a['local_frame_zero_based']+1
            assert a['published_at_s']==a['time_s']==native_time_us(meta,a['frame']-1)/1e6


def test_native_video_end_retires_last_track_without_a_fake_frame(tmp_path):
    cfg,cache,local=stream(tmp_path,short_camera=True)
    values=preflight(cfg,cache,local,'native')
    assert values[3][500_000]==[]
    out=tmp_path/'baseline'
    summary=run_baseline(cfg,cache,local,out)
    rows=read_rows(out/'predictions.txt')
    assert {r['id'] for r in rows if r['camera'] in (29,34)}=={1}
    assert max(r['frame'] for r in rows if r['camera']==29)==5
    assert summary['detector_calls']==json.loads((local/'summary.json').read_text())['detector_calls']


def test_zero_descriptor_does_not_become_a_historical_identity_match(tmp_path):
    cfg,cache,local=stream(tmp_path,no_descriptor=True)
    out=tmp_path/'baseline'
    summary=run_baseline(cfg,cache,local,out)
    rows=read_rows(out/'predictions.txt')
    assert summary['actual_descriptor_observations_aligned']==0
    assert summary['prediction_rows_without_descriptor_history']==len(rows)
    assert len({r['id'] for r in rows})==3


def test_unfrozen_or_wrong_development_selection_and_scope_are_rejected():
    selection,plan,scene=artifacts()
    selection['status']='pending'
    with pytest.raises(ValueError,match='frozen'):
        config_from_artifacts(selection,plan,scene)
    selection['status']='frozen_after_development'
    selection['development']['interval_s']=[0.,120.]
    with pytest.raises(ValueError,match='only S02'):
        config_from_artifacts(selection,plan,scene)
    selection['development']['interval_s']=[0.,60.]
    plan['calibration_interval_s']=[0.,2.]
    with pytest.raises(ValueError,match='before holdout'):
        config_from_artifacts(selection,plan,scene)
    plan['calibration_interval_s']=[0.,.1]
    plan['cameras']=[6,7,8,9]
    with pytest.raises(ValueError,match='inventory'):
        config_from_artifacts(selection,plan,scene)


def test_no_ground_truth_file_is_needed_even_when_config_has_nonexistent_dataset(tmp_path):
    cfg,cache,local=stream(tmp_path)
    assert not Path(cfg['dataset']).exists()
    preflight_result=run_baseline(cfg,cache,local,tmp_path/'unused',preflight_only=True)
    assert preflight_result['status']=='preflight_passed'
    assert not (tmp_path/'unused').exists()
    assert preflight_result['options']['upstream_commit']=='f920c396aaa820e231bd3fb7ec12d563ce75c396'


def test_changed_local_config_or_outputs_are_rejected(tmp_path):
    cfg,cache,local=stream(tmp_path)
    wrong=copy.deepcopy(cfg)
    wrong['tracker']['track_buffer']=10
    with pytest.raises(ValueError,match='tracker/output filter'):
        preflight(wrong,cache,local,'native')
    out=tmp_path/'baseline'
    run_baseline(cfg,cache,local,out)
    (out/'predictions.txt').write_text('tampered\n')
    with pytest.raises(ValueError,match='Changed completed'):
        run_baseline(cfg,cache,local,out)
    (local/'decisions.jsonl').write_text('tampered\n')
    with pytest.raises(ValueError,match='Shared local artifact changed'):
        preflight(cfg,cache,local,'native')


def test_cache_mutation_and_signed_local_manifest_mutation_are_rejected(tmp_path):
    cfg,cache,local=stream(tmp_path)
    with DetectionCache(cache) as changed:
        changed.ensure_meta('unrelated_metadata',{'mutation':True})
    with pytest.raises(ValueError,match='cache differs'):
        preflight(cfg,cache,local,'native')
    manifest=json.loads((local/'manifest.json').read_text())
    manifest['options']['config']['seed']=99
    write_json(local/'manifest.json',manifest)
    with pytest.raises(ValueError,match='options/signature'):
        preflight(cfg,cache,local,'native')


def test_track_alias_and_best_feature_use_native_real_observation_history():
    # Use verified upstream best-feature implementation via a synthetic function.
    track=Track(35,1,lambda instance,mode:instance.obs_history[-1][3])
    assert track.camera==35 and track.cam=='c009'
    feature=np.zeros(512);feature[4]=2.
    assert track.observe(123_000,[1.,2.,11.,12.],.9,feature)
    assert len(track.obs_history)==1 and track.obs_history[0][0]==123_000
    assert track.get_feature()[4]==1.
    with pytest.raises(ValueError,match='Unknown'):
        Track(6,1,None)
