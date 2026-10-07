"""Nonzero native interval and interrupted/resumed frame bookkeeping."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cache_cityflow_mask_interval_v0_6 import next_frame
from rhd.cache import DetectionCache


def test_nonzero_interval_resumes_committed_rows(tmp_path):
    with DetectionCache(tmp_path/'cache.sqlite') as cache:
        assert next_frame(cache,29,1200,3400) == 1200
        cache.add_frame(29,1200,[]);cache.add_frame(29,1201,[]);cache.commit()
        assert next_frame(cache,29,1200,3400) == 1202
    with DetectionCache(tmp_path/'cache.sqlite',readonly=True) as cache:
        assert next_frame(cache,29,1200,3400) == 1202


@pytest.mark.parametrize('frames',[[1199],[1201],[1200,1202],[1200,1201,1202]])
def test_holes_or_rows_outside_frozen_interval_rejected(tmp_path,frames):
    with DetectionCache(tmp_path/'cache.sqlite') as cache:
        for frame in frames:cache.add_frame(29,frame,[])
        cache.commit()
        with pytest.raises(ValueError,match='missing/extra'):
            next_frame(cache,29,1200,1202)
