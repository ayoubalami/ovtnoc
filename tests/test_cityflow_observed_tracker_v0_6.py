"""Actual ByteTrack activation and causal measured-box adapter boundaries."""
import numpy as np
import pytest
from cityflow_dev_v0_6.observed_tracker import ObservedBoxTracker
from cityflow_dev_v0_6 import replay


def detection(box):
    return [dict(box=box,score=.95,class_id=2,feature=[1.]+[0.]*511)]


def test_measurement_box_and_descriptor_keep_byte_track_identity():
    tracker = ObservedBoxTracker(10,100,100,{}, {'kind':'none'})
    first = tracker.step(detection([10.,10.,50.,60.]),0)[0]
    actual = [8.,10.,60.,62.]
    second = tracker.step(detection(actual),1)[0]
    assert second['local_id'] == first['local_id']
    assert second['box'] == actual
    assert second['point'] == [.34,.62]
    np.testing.assert_array_equal(second['observation_feature'],[1.]+[0.]*511)
    prediction = tracker.step(None,2)[0]
    assert prediction['age_since_detection'] == 1
    assert not np.any(prediction['observation_feature'])


def test_no_track_is_created_from_low_score_unactivated_observation():
    tracker = ObservedBoxTracker(10,100,100,{}, {'kind':'none'})
    low = detection([10.,10.,50.,60.]); low[0]['score'] = .15
    assert tracker.step(low,0) == []


def test_original_index_survives_high_and_low_confidence_partitions():
    tracker = ObservedBoxTracker(10,100,100,{}, {'kind':'none'})
    # A low-score observation precedes the high-score car in the full list.
    wrong = detection([70.,10.,90.,40.])[0]; wrong['score'] = .15
    right = detection([10.,10.,50.,60.])[0]
    right['feature'] = [0.,1.]+[0.]*510
    first = tracker.step([wrong,right],0)[0]
    assert first['box'] == right['box']
    np.testing.assert_array_equal(first['observation_feature'],right['feature'])
    # Next the same car matches the low-score stage at original index 1.
    unrelated = detection([70.,10.,90.,40.])[0]
    right['score'] = .15
    second = next(t for t in tracker.step([unrelated,right],1) if t['local_id'] == first['local_id'])
    assert second['box'] == right['box']
    np.testing.assert_array_equal(second['observation_feature'],right['feature'])


def test_native_adapter_restores_globals_after_failure(monkeypatch):
    original = replay.legacy.CameraTracker
    def fail(*_):
        assert replay.legacy.CameraTracker is ObservedBoxTracker
        assert replay.legacy.source_signature is replay.source_signature
        raise ValueError('interrupted')
    monkeypatch.setattr(replay.legacy,'run',fail)
    with pytest.raises(ValueError,match='interrupted'):
        replay.run({'tracker_box_mode':'observed'},None,None)
    assert replay.legacy.CameraTracker is original
    assert replay.legacy.source_signature is replay._BASE_SIGNATURE
