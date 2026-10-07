"""Development diagnostics test actual attribution, topology and identity scopes."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cityflow_dev_v0_6.diagnostics import (
    diagnose, generate_report, match_rows, read_raw_stages, render_markdown,
)
from rhd.common import file_hash


@pytest.fixture
def scene():
    return {'scenario': 'S02', 'cameras': {
        str(c): {'camera': c, 'fps': 10., 'frames': 1000, 'timestamp_offset_s': 0.,
                 'width': 100, 'height': 100} for c in (6,7,8,9)}}


def row(camera, frame, identity=1, box=None, **kwargs):
    return {'camera': camera, 'frame': frame, 'id': identity, 'time_s': frame/10.,
            'box': box or [10.,10.,20.,20.], **kwargs}


def reference():
    return [row(7, f) for f in range(1, 21)] + [row(9, f) for f in range(31, 51)]


def test_assignment_is_one_to_one_and_unmatched_predictions_are_not_false_positives(scene):
    gt = [row(7, 1, 1), row(7, 1, 2)]
    pred = [row(7, 1, 10), row(7, 1, 11, box=[60.,60.,70.,70.])]
    report = diagnose(scene, gt, {'detector': pred})
    stage = report['stages']['detector']
    assert stage['matched_gt_observations'] == 1
    assert stage['annotation_recall'] == .5
    assert stage['unmatched_prediction_observations'] == 1
    assert not any('false_positive' in key for key in stage)
    assert len(match_rows(gt, pred)) == 1


def test_topology_contains_intervening_unconfigured_camera_before_route_filter(scene):
    gt = [row(7, f) for f in range(1, 6)] + [row(8, f) for f in range(11, 16)] + [row(9, f) for f in range(21, 26)]
    report = diagnose(scene, gt, {}, routes=[(7,9)])
    assert report['reference_handoffs'] == 0
    assert report['reference_exclusions']['unconfigured_route'] == 2


def test_boundary_censored_spans_are_excluded_consistently_with_fit_prefix(scene):
    start_censored = [row(7, f) for f in range(5)] + [row(9, f) for f in range(11, 16)]
    end_censored = [row(7, f, 2) for f in range(1, 6)] + [row(9, f, 2) for f in range(595, 600)]
    report = diagnose(scene, start_censored+end_censored, {})
    assert report['reference_handoffs'] == 0
    assert report['reference_exclusions']['prefix_censored'] == 2


def test_short_spans_and_visibility_overlap_are_excluded(scene):
    gt = [row(7, f) for f in range(1, 11)] + [row(9, f) for f in range(5, 15)]
    gt += [row(7, f, 2) for f in range(1, 5)] + [row(9, f, 2) for f in range(21, 30)]
    report = diagnose(scene, gt, {})
    assert report['reference_exclusions']['overlap'] == 1
    assert report['reference_exclusions']['short_span'] == 1


def test_endpoint_fifteen_native_frame_boundaries_and_per_frame_proof(scene):
    gt = reference()
    pred = [row(7, 6, 100), row(9, 45, 200)]
    report = diagnose(scene, gt, {'local_tracker': pred})
    stage = report['stages']['local_tracker']
    assert stage['both_endpoints_covered'] == 1
    assert 'same_global_identity' not in stage
    evidence = report['reference_outcomes'][0]['stages']['local_tracker']
    assert evidence['source_match']['native_frame1'] == 7
    assert evidence['destination_match']['native_frame1'] == 46
    assert len(evidence['endpoint_frames']['source']) == len(evidence['endpoint_frames']['destination']) == 15
    assert evidence['identity_outcome'] is None
    too_early = diagnose(scene, gt, {'detector': [row(7,5),row(9,45)]})
    too_late = diagnose(scene, gt, {'detector': [row(7,6),row(9,46)]})
    assert too_early['stages']['detector']['missing_source_only'] == 1
    assert too_late['stages']['detector']['missing_destination_only'] == 1


def test_ids_are_only_compared_for_declared_global_stage(scene):
    pred = [row(7,20,123), row(9,31,456)]
    report = diagnose(scene, reference(), {'local_tracker':pred, 'global_assignment':pred})
    assert report['stages']['global_assignment']['global_identity_break'] == 1
    assert report['reference_outcomes'][0]['stages']['global_assignment']['identity_outcome'] == 'global_identity_break'
    assert report['reference_outcomes'][0]['stages']['local_tracker']['identity_outcome'] is None
    confirmed = diagnose(scene, reference(), {'global_assignment':[row(7,20,123),row(9,31,123)]})
    assert confirmed['stages']['global_assignment']['same_global_identity'] == 1


def test_raw_score_roi_gate_evidence_does_not_assert_causality_after_assignment(scene):
    raw = [row(7,20,10, score=.09, roi_accepted=True),
           row(9,31,11, score=.9, roi_accepted=False)]
    report = diagnose(scene, reference(), {'after_roi': []}, raw_all_scores=raw)
    evidence = report['reference_outcomes'][0]['raw_gate_evidence']
    assert evidence['source'][0]['gate_evidence'] == 'all_iou_candidates_below_confidence'
    assert evidence['destination'][0]['gate_evidence'] == 'all_above_confidence_iou_candidates_rejected_by_roi'
    assert evidence['source'][1]['gate_evidence'] == 'no_raw_iou_candidate'
    raw[0]['score'] = .1
    report = diagnose(scene, reference(), {'after_roi': []}, raw_all_scores=raw)
    assert report['reference_outcomes'][0]['raw_gate_evidence']['source'][0]['gate_evidence'] == 'eligible_raw_iou_candidate_exists'
    assert report['stages']['after_roi']['missing_both_endpoints'] == 1


def test_clock_timestamp_and_duplicate_input_corruption_are_rejected(scene):
    corrupted = row(7,1)
    corrupted['time_s'] = 100.
    with pytest.raises(ValueError, match='timestamp'):
        diagnose(scene, [corrupted], {})
    with pytest.raises(ValueError, match='Duplicate'):
        diagnose(scene, [row(7,1),row(7,1)], {})
    with pytest.raises(ValueError, match='positive integer'):
        diagnose(scene, [], {}, endpoint_frames=0)


def test_prefix_labels_after_stop_are_excluded_and_interval_is_half_open(scene):
    gt = reference()+[row(7,600,2)]
    report = diagnose(scene, gt, {'raw':[row(7,600)]})
    assert report['stages']['raw']['gt_observations'] == 40
    assert report['stages']['raw']['prediction_observations'] == 0


def test_raw_reader_respects_score_boundary_and_image_only_roi(tmp_path):
    folder = tmp_path/'validation'/'S02'/'c007'
    (folder/'det').mkdir(parents=True)
    roi = np.full((30,30),255,dtype=np.uint8)
    roi[20,20] = 0
    Image.fromarray(roi).save(folder/'roi.png')
    raw = folder/'det'/'det_mask_rcnn.txt'
    raw.write_text('1,-1,1,1,5,5,0.1,-1,-1,-1\n1,-1,20,20,5,5,0.9,-1,-1,-1\n2,-1,1,1,5,5,0.099,-1,-1,-1\n')
    scene = {'scenario':'S02','cameras': {'7': {'camera':7,'fps':10.,'frames':100,'timestamp_offset_s':0.,
        'width':30,'height':30,'video':str(folder/'vdo.avi'),'roi':str(folder/'roi.png'),
        'roi_sha256':file_hash(folder/'roi.png')}}}
    all_scores, stages, inputs = read_raw_stages(scene,0.,1.)
    assert len(all_scores) == 3
    assert len(stages['supplied_before_roi']) == 2
    assert len(stages['supplied_after_roi']) == 1
    assert stages['supplied_after_roi'][0]['score'] == .1
    assert str(raw) in inputs
    Image.fromarray(np.full((30,30),255,dtype=np.uint8)).save(folder/'roi.png')
    with pytest.raises(ValueError, match='ROI changed'):
        read_raw_stages(scene,0.,1.)


def test_cli_report_guard_prevents_inspecting_suffix_gt(tmp_path):
    scene_path = tmp_path/'scene.json'
    scene_path.write_text(json.dumps({'scenario':'S02'}))
    with pytest.raises(ValueError, match='restricted'):
        generate_report(scene_path,tmp_path/'diagnosis',stop_s=61.)
    scene_path.write_text(json.dumps({'scenario':'S05'}))
    with pytest.raises(ValueError, match='restricted'):
        generate_report(scene_path,tmp_path/'diagnosis')


def test_markdown_contains_limits_and_correct_denominators(scene):
    report = diagnose(scene, reference(), {'raw':[row(7,20),row(9,31)]})
    text = render_markdown(report)
    assert '2/40' in text and '5.00%' in text
    assert 'Unmatched predictions are not declared false positives' in text
    assert 'not an identity association score' in text
