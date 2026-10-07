"""Guard the untouched S05 suffix against label parsing during selection."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_cityflow_holdout_v0_6 import read_prefix


def metadata():
    return {'camera':29,'fps':10.,'frames':3545,'timestamp_offset_s':0.}


def test_prefix_reader_stops_before_invalid_suffix_identity(tmp_path):
    path = tmp_path/'gt.txt'
    path.write_text('1,7,1,2,3,4,1,-1,-1,-1\n2,7,2,3,3,4,1,-1,-1,-1\n'
                    '3,DO_NOT_PARSE_THIS_SUFFIX_ID_OR_BOX\n')
    rows, evidence = read_prefix(path, metadata(), stop_s=.2)
    assert len(rows) == 2
    assert [r['frame'] for r in rows] == [0,1]
    assert evidence['first_outside_frame_column_only'] == 3
    assert evidence['last_interpreted_frame1'] == 2


def test_duplicate_prefix_key_is_rejected(tmp_path):
    path = tmp_path/'gt.txt'
    path.write_text('1,7,1,2,3,4,1,-1,-1,-1\n1,7,2,3,3,4,1,-1,-1,-1\n')
    with pytest.raises(ValueError, match='Duplicate'):
        read_prefix(path, metadata())


def test_unsorted_prefix_frame_is_rejected(tmp_path):
    path = tmp_path/'gt.txt'
    path.write_text('2,7,1,2,3,4,1,-1,-1,-1\n1,8,2,3,3,4,1,-1,-1,-1\n')
    with pytest.raises(ValueError, match='unsorted'):
        read_prefix(path, metadata())


def test_native_offset_is_respected(tmp_path):
    path = tmp_path/'gt.txt'
    path.write_text('1,7,1,2,3,4,1,-1,-1,-1\n2,NO_SUFFIX_LABELS\n')
    meta = metadata(); meta['timestamp_offset_s'] = .061
    rows, evidence = read_prefix(path, meta, stop_s=.16)
    assert rows[0]['time_s'] == .061
    assert evidence['first_outside_frame_column_only'] == 2
