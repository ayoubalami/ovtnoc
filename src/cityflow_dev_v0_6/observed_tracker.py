"""Use actual measurement boxes for observed tracks, retaining ByteTrack IDs/KF.

This avoids aspect-ratio smoothing suppressing accurate detector boxes at camera
boundaries. Prediction-only ticks retain the original Kalman box. No labels enter
the tracker and no new ID is produced independently of ByteTrack activation.
"""
import numpy as np
from rhd.vision import CameraTracker
from rhd.common import unit_vector
from ultralytics.trackers.byte_tracker import BYTETracker, STrack


class IndexedBYTETracker(BYTETracker):
    def init_track(self, results, img=None):
        if len(results) == 0:
            return []
        if results.id is None:
            raise ValueError('Original detection indices must survive confidence partitioning')
        boxes = np.concatenate([results.xywh,np.asarray(results.id).reshape(-1,1)],axis=-1)
        return [STrack(box,score,category) for box,score,category in zip(boxes,results.conf,results.cls)]


class IndexedCameraTracker(CameraTracker):
    """Preserve observation provenance through both ByteTrack confidence stages."""
    def __init__(self, fps, width, height, options, output_filter=None):
        if output_filter is not None and output_filter.get('kind','none') != 'none':
            raise ValueError('Indexed observation experiment retains all activated vehicles')
        super().__init__(fps,width,height,options,output_filter)
        self.tracker = IndexedBYTETracker(self.tracker.args,frame_rate=fps)

    def step(self,detections,frame):
        from ultralytics.engine.results import Boxes
        if detections is None:
            self.tracker.frame_id += 1
            pool = self.tracker.joint_stracks(self.tracker.tracked_stracks,self.tracker.lost_stracks)
            self.tracker.multi_predict(pool)
        else:
            # Boxes' seventh-column form preserves an index through slicing.
            # This temporary id field is a detection index, not a tracking ID.
            values = np.asarray([d['box']+[index,d['score'],d['class_id']]
                                 for index,d in enumerate(detections)],dtype=np.float32).reshape(-1,7)
            self.tracker.update(Boxes(values,(self.height,self.width)))
            for track in self.tracker.tracked_stracks:
                if not track.is_activated:
                    continue
                index = int(track.idx)
                if not 0 <= index < len(detections):
                    raise ValueError('Original observation index out of range')
                current = unit_vector(detections[index]['feature'])
                self.observation_features[track.track_id] = current
                old = self.features.get(track.track_id)
                self.features[track.track_id] = unit_vector(.8*old+.2*current) if old is not None else current
                self.last_observed[track.track_id] = frame
        snapshots = []
        for track in self.tracker.tracked_stracks:
            if not track.is_activated:
                continue
            box = np.asarray(track.xyxy,dtype=float)
            box[[0,2]] = np.clip(box[[0,2]],0,self.width)
            box[[1,3]] = np.clip(box[[1,3]],0,self.height)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            age = frame-self.last_observed.get(track.track_id,frame)
            feature = self.features.get(track.track_id,np.zeros(512))
            snapshots.append(dict(local_id=int(track.track_id),box=box.tolist(),score=float(track.score),
                feature=feature.tolist(),observation_feature=(self.observation_features.get(track.track_id,np.zeros(512))
                    if age == 0 else np.zeros_like(feature)).tolist(),
                point=[(box[0]+box[2])/(2*self.width),box[3]/self.height],
                velocity=[float(track.mean[4]/self.width),float(track.mean[5]/self.height)],
                uncertainty=float(np.trace(track.covariance[:2,:2])/max((box[2]-box[0])**2,1)),
                age_since_detection=age))
        live_ids = {track.track_id for track in self.tracker.tracked_stracks+self.tracker.lost_stracks}
        for mapping in (self.features,self.observation_features,self.last_observed):
            for key in list(mapping):
                if key not in live_ids:
                    del mapping[key]
        return snapshots


class ObservedBoxTracker(IndexedCameraTracker):
    def __init__(self, fps, width, height, options, output_filter=None):
        if output_filter is not None and output_filter.get('kind','none') != 'none':
            raise ValueError('Observed-box experiment retains all activated vehicles')
        super().__init__(fps,width,height,options,output_filter)

    def step(self, detections, frame):
        snapshots = super().step(detections,frame)
        if detections is None:
            return snapshots
        indices = {int(track.track_id):int(track.idx) for track in self.tracker.tracked_stracks
                   if track.is_activated}
        for snapshot in snapshots:
            index = indices[snapshot['local_id']]
            if not 0 <= index < len(detections) or snapshot['age_since_detection'] != 0:
                raise ValueError('Activated observation lacks a current detection index')
            observation = detections[index]
            box = np.asarray(observation['box'],dtype=float).copy()
            box[[0,2]] = np.clip(box[[0,2]],0,self.width)
            box[[1,3]] = np.clip(box[[1,3]],0,self.height)
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Invalid current detector observation box')
            snapshot['box'] = box.tolist()
            snapshot['point'] = [(box[0]+box[2])/(2*self.width),box[3]/self.height]
            snapshot['score'] = float(observation['score'])
        return snapshots
