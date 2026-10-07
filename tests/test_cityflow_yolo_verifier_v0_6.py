"""Reject evidence corruption at the raw/ROI/appearance verification boundary."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_cityflow_yolo_cache_v0_6 import validate_frame


def fixture():
    raw = [dict(box=[1.,1.,5.,5.],score=.7,class_id=2)]
    observed = [{**raw[0],'feature':[1.]+[0.]*511,'reid_computed':True}]
    return raw,observed,np.full((10,10),255,dtype=np.uint8)


def test_raw_box_changed_after_feature_extraction_is_rejected():
    raw,observed,roi = fixture(); observed[0]['box'] = [1.,1.,6.,5.]
    with pytest.raises(ValueError,match='box/score/class'):
        validate_frame(raw,observed,roi)


def test_zero_or_uncomputed_descriptor_is_rejected():
    raw,observed,roi = fixture(); observed[0]['feature'] = [0.]*512
    with pytest.raises(ValueError,match='nonzero'):
        validate_frame(raw,observed,roi)
    observed[0]['feature'] = [1.]+[0.]*511; observed[0]['reid_computed'] = False
    with pytest.raises(ValueError,match='Malformed'):
        validate_frame(raw,observed,roi)


def test_roi_removal_and_explicit_empty_frame():
    raw,observed,roi = fixture(); roi[:] = 0
    assert validate_frame(raw,[],roi)['outside_roi'] == 1
    assert validate_frame([],[],roi)['retained'] == 0
