"""Descriptor correction preserves causal motion-filtered local geometry."""
from rhd.vision import CameraTracker
from cityflow_dev_v0_6.roundabout_replay import FilteredIndexedCameraTracker


def test_local_motion_filtered_geometry_is_unchanged():
    settings = dict(kind='causal_motion',window_frames=4,minimum_observations=2,displacement_fraction=.005)
    stream = []
    for frame in range(4):
        stream.append([dict(box=[70.,10.,90.,40.],score=.15,class_id=2,feature=[1.]+[0.]*511),
                       dict(box=[10.+frame,10.,50.+frame,60.],score=.95,class_id=2,feature=[0.,1.]+[0.]*510)])
    old = CameraTracker(10,100,100,{},settings)
    old_rows = [old.step(ds,frame) for frame,ds in enumerate(stream)]
    new = FilteredIndexedCameraTracker(10,100,100,{},settings)
    new_rows = [new.step(ds,frame) for frame,ds in enumerate(stream)]
    for left,right in zip(old_rows,new_rows):
        assert len(left) == len(right)
        for a,b in zip(left,right):
            assert {k:v for k,v in a.items() if k not in ('feature','observation_feature')} == {
                k:v for k,v in b.items() if k not in ('feature','observation_feature')}
            assert b['observation_feature'] == [0.,1.]+[0.]*510
