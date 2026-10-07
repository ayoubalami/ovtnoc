from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from paired_handoff_statistics_v0_6 import compare


def row(identity,frame,success):
    return dict(gt_id=identity,source_camera=29,destination_camera=33,
        source_last_native_frame_1based=frame,destination_first_native_frame_1based=frame+20,
        outcome='confirmed_handoffs' if success else 'identity_breaks')


def test_vehicle_resampling_keeps_multiple_visits_together_and_reports_losses():
    base = [row(1,100,False),row(1,200,False),row(2,300,True)]
    other = [row(1,100,True),row(1,200,True),row(2,300,False)]
    result = compare(base,other)
    assert result['vehicle_clusters'] == 2
    assert result['newly_confirmed'] == 2 and result['lost_confirmations'] == 1
    assert result['interval_95_pp'] == [-100.,100.]


def test_reference_mismatch_and_duplicates_rejected():
    with pytest.raises(ValueError,match='different reference'):
        compare([row(1,100,True)],[row(1,101,True)])
    with pytest.raises(ValueError,match='Repeated'):
        compare([row(1,100,True)]*2,[row(1,100,True)])
