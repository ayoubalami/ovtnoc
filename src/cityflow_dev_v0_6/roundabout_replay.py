"""Correct descriptor indices while preserving frozen RoundaboutHD geometry."""
from pathlib import Path
from rhd import runner as legacy
from rhd.vision import CameraTracker
from rhd.common import digest,file_hash
from .observed_tracker import IndexedBYTETracker,IndexedCameraTracker

_BASE_SIGNATURE = legacy.source_signature


class FilteredIndexedCameraTracker(IndexedCameraTracker):
    def __init__(self,fps,width,height,options,output_filter=None):
        CameraTracker.__init__(self,fps,width,height,options,output_filter)
        self.tracker = IndexedBYTETracker(self.tracker.args,frame_rate=fps)

    def step(self,detections,frame):
        snapshots = super().step(detections,frame)
        live = {track.track_id for track in self.tracker.tracked_stracks+self.tracker.lost_stracks}
        return self.output_filter.apply(snapshots,frame,live)


def source_signature():
    folder = Path(__file__).parent
    return digest(dict(unchanged_core=_BASE_SIGNATURE(),adapter=file_hash(Path(__file__)),
                       indexed_tracker=file_hash(folder/'observed_tracker.py')))


def run(*args,**kwargs):
    tracker,signature = legacy.CameraTracker,legacy.source_signature
    if signature is not _BASE_SIGNATURE:
        raise RuntimeError('Corrected Roundabout replay must run serially')
    legacy.CameraTracker = FilteredIndexedCameraTracker;legacy.source_signature = source_signature
    try:
        return legacy.run(*args,**kwargs)
    finally:
        legacy.CameraTracker = tracker;legacy.source_signature = signature
