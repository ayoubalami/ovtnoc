"""Holdout annotation-access and full-topology provenance gates."""
import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_cityflow_holdout_v0_6 as driver
from rhd.common import digest, write_json


def selection():
    return {'status':'frozen_after_development',
        'development':{'scenario':'S02','interval_s':[0,60]},
        'config':{'tracker':{},'association':{'minimum_observations':1,'source_retention_frames':8},
                  'communication':{},'output_filter':{},'seed':7,'fps':10.,'perception':'fixed'},
        'detector_provenance':{'kind':'native_yolo','descriptor':'vehicle-reid-0001',
                               'detector_assets':{'weights_sha256':'fixed'}}}


def test_missing_freeze_blocks_annotation_reader(tmp_path, monkeypatch):
    def forbidden(*_):
        pytest.fail('Annotation reader reached before freeze')
    monkeypatch.setattr(driver,'annotation_rows',forbidden)
    with pytest.raises(ValueError,match='requires study_freeze'):
        driver.load_evaluation_inputs(tmp_path,'missing',{}, {})


def test_tampered_freeze_blocks_annotation_reader(tmp_path, monkeypatch):
    signature = driver.freeze_before_annotations(tmp_path,{'config':'fixed'})
    write_json(tmp_path/'study_freeze.json',{'status':driver.STATUS,'signature':signature,
                                         'signed_inputs':{'config':'changed'}})
    def forbidden(*_):
        pytest.fail('Annotation reader reached through tampered freeze')
    monkeypatch.setattr(driver,'annotation_rows',forbidden)
    with pytest.raises(ValueError,match='freeze signature changed'):
        driver.load_evaluation_inputs(tmp_path,signature,{}, {})


def test_new_signature_cannot_overwrite_existing_freeze(tmp_path):
    signature = driver.freeze_before_annotations(tmp_path,{'config':'fixed'})
    assert driver.freeze_before_annotations(tmp_path,{'config':'fixed'}) == signature
    with pytest.raises(ValueError,match='use a new output'):
        driver.freeze_before_annotations(tmp_path,{'config':'retuned_after_test'})


def test_freeze_refuses_unrecognized_existing_artifacts(tmp_path):
    (tmp_path/'old_result.json').write_text('{}')
    with pytest.raises(ValueError,match='must be empty'):
        driver.freeze_before_annotations(tmp_path,{'config':'fixed'})


def test_observed_suffix_cannot_be_declared_as_development():
    chosen = selection(); driver.validate_selection(chosen)
    chosen['development']['interval_s'] = [120,212]
    with pytest.raises(ValueError,match='S02'):
        driver.validate_selection(chosen)


def test_delay_controls_reject_changed_source_retention_or_observations():
    for key, value in [('minimum_observations',2),('source_retention_frames',4)]:
        chosen = selection(); chosen['config']['association'][key] = value
        with pytest.raises(ValueError,match='Delay controls require'):
            driver.validate_selection(chosen)


def test_artifact_hash_cannot_escape_study_directory(tmp_path):
    study = tmp_path/'study'; study.mkdir()
    outside = tmp_path/'outside.txt'; outside.write_text('outside')
    with pytest.raises(ValueError,match='changed or missing'):
        driver.verify_artifacts(study,{'../outside.txt':'irrelevant'})


def native_scene(cameras):
    return {'scenario':'S05','cameras':{str(c):{'fps':10.,'frames':100,'timestamp_offset_s':0.}
                                       for c in cameras}}


def row(camera, frame, identity=7):
    return {'camera':camera,'frame':frame,'id':identity,'box':[10.,10.,20.,20.]}


def test_omitted_intervening_camera_cannot_invent_reference_handoff():
    scene = native_scene([29,33,36])
    gt = [row(29,0),row(29,1),row(36,3),row(36,4),row(33,6),row(33,7)]
    predictions = [row(29,0,1),row(29,1,1),row(33,6,1),row(33,7,1)]
    cfg = {'routes':[{'source':29,'destination':33}],
           'replay_start_s':0.,'score_start_s':.5,'stop_s':2.}
    result = driver.score_handoffs(cfg,scene,gt,predictions)
    assert result['reference_handoffs'] == 0
    assert result['topology']['full_gt_cross_camera_pairs'] == 2
    assert result['topology']['excluded_reference_pairs']['unsupported_route'] == 1
    # Filtering before topology would wrongly introduce a supported reference.
    wrong = driver.score_handoffs(cfg,scene,[r for r in gt if r['camera'] != 36],predictions)
    assert wrong['reference_handoffs'] == 1


def test_official_clocks_remain_selected_while_handoff_scene_is_full(tmp_path, monkeypatch):
    directory = tmp_path/'K0'; directory.mkdir()
    (directory/'predictions.txt').write_text('29 1 1 10 10 10 10 -1 -1\n')
    captured = {}
    def export(_source, destination, **_):
        destination.write_text('export')
    def evaluate(**kwargs):
        captured['clocks'] = kwargs['camera_clocks']
        return {}
    def handoffs(_cfg, scene, gt, predictions, availability):
        captured['topology'] = scene
        return {}
    monkeypatch.setattr(driver,'export_official_predictions',export)
    monkeypatch.setattr(driver,'evaluate_predictions',evaluate)
    monkeypatch.setattr(driver,'score_handoffs',handoffs)
    cfg = {'dataset':str(tmp_path),'split':'validation','cameras':[29,33],
           'score_start_s':.5,'stop_s':2.}
    scene = native_scene([29,33,36])
    driver.score_run(cfg,scene,directory,tmp_path/'cohort.txt',[],tmp_path/'ROIs')
    assert set(captured['clocks']) == {29,33}
    assert set(captured['topology']['cameras']) == {'29','33','36'}
