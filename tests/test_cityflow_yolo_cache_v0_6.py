"""Native-frame, GT-free modern detector cache and interrupted-resume contracts."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest

from cityflow.replay import local_frame_range
from rhd.cache import DetectionCache
from rhd.common import canonical, file_hash


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
spec = spec_from_file_location("cityflow_yolo_cache_v0_6", SCRIPTS / "cache_cityflow_yolo_v0_6.py")
builder = module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.mark.parametrize("fps,offset,start,stop", [(10., 0., 0., .5), (10., .061, .1, .401),
    (8., .05, .175, .426), (10., .061, 100., 110.), (10., .0000005, 0., .4000005)])
def test_interval_matches_native_replay_clock(fps, offset, start, stop):
    meta = {"fps": fps, "timestamp_offset_s": offset, "frames": 16}
    assert builder.frame_range(meta, start, stop) == local_frame_range(meta, start, stop)


@pytest.mark.parametrize("start,stop", [(-1, 1), (1, 1), (2, 1), (float("nan"), 1), (0, float("inf"))])
def test_invalid_intervals_fail(start, stop):
    with pytest.raises(ValueError, match="interval"):
        builder.frame_range({"fps": 10., "timestamp_offset_s": 0., "frames": 20}, start, stop)


def test_roi_removes_only_image_corners_and_preserves_vehicle_classes():
    roi = np.full((12, 16), 255, np.uint8)
    roi[3, 3] = 254
    raw = [{"box": [.8, .8, 3.9, 3.9], "score": .9, "class_id": 2},
           {"box": [5., 5., 10., 10.], "score": .8, "class_id": 7}]
    retained, counts = builder.retained_observations(raw, roi)
    assert retained == [raw[1]] and retained[0] is not raw[1]
    assert counts == {"raw_post_confidence_nms_class_filter": 2, "outside_roi": 1, "retained": 1}
    assert "feature" not in raw[1]


@pytest.mark.parametrize("change", [{"class_id": 0}, {"score": float("nan")}, {"box": [2, 2, 1, 1]}])
def test_invalid_detector_outputs_cannot_be_cached(change):
    raw = {"box": [1., 1., 4., 4.], "score": .9, "class_id": 2, **change}
    with pytest.raises(ValueError, match="raw vehicle"):
        builder.retained_observations([raw], np.full((12, 16), 255, np.uint8))


def _scene(tmp_path, count=6):
    folder = tmp_path / "dataset/validation/S02/c006"
    folder.mkdir(parents=True)
    video = folder / "vdo.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10., (16, 12))
    if not writer.isOpened():
        pytest.skip("MJPG encoder unavailable")
    try:
        for frame in range(count):
            writer.write(np.full((12, 16, 3), frame*30, dtype=np.uint8))
    finally:
        writer.release()
    roi = folder / "roi.jpg"
    mask = np.full((12, 16), 255, np.uint8); mask[3, 3] = 254
    Image.fromarray(mask).save(roi, format="PNG")
    # Annotation and MTSC paths intentionally contain invalid data. The builder
    # must consume native images and ROI without opening these files.
    for name in ("gt/gt.txt", "mtsc/mtsc_mask_rcnn.txt"):
        path = folder / name; path.parent.mkdir(exist_ok=True)
        path.write_text("MUST NOT READ IDENTITY ANNOTATIONS")
    root = tmp_path / "dataset"
    for name in ("cam_timestamp", "cam_framenum"):
        (root / name).mkdir()
    timestamp = root / "cam_timestamp/S02.txt"; timestamp.write_text("c006 0.0\n")
    framecount = root / "cam_framenum/S02.txt"; framecount.write_text(f"c006 {count}\n")
    meta = {"camera": 6, "fps": 10., "frames": count, "width": 16, "height": 12,
            "timestamp_offset_s": 0., "video": str(video), "video_sha256": file_hash(video),
            "roi": str(roi), "roi_sha256": file_hash(roi)}
    return {"dataset_root": str(root), "split": "validation", "scenario": "S02",
            "cameras": {"6": meta}, "timestamps_sha256": file_hash(timestamp),
            "framecounts_sha256": file_hash(framecount)}


class Detector:
    def __init__(self, fail_at=None):
        self.seen = []
        self.fail_at = fail_at

    def __call__(self, image):
        value = int(image[0, 0, 0]); self.seen.append(value)
        if value == self.fail_at:
            raise RuntimeError("Simulated interruption")
        if value == 150:
            return []
        return [{"box": [5., 5., 10., 10.], "score": .9, "class_id": 2},
                {"box": [.8, .8, 3.9, 3.9], "score": .8, "class_id": 7}]


class ReID:
    def __init__(self):
        self.seen = []

    def descriptor(self, image, box):
        value = int(image[0, 0, 0]); self.seen.append((value, box))
        vector = np.zeros(512); vector[value] = 1.
        return vector.tolist()


def _initialize(cache, scene):
    cache.ensure_meta("scene", scene)
    for camera, meta in scene["cameras"].items():
        cache.ensure_meta(f"camera:{camera}", meta)


def test_nonzero_interval_decodes_exact_native_frames_and_keeps_raw_rows(tmp_path):
    scene = _scene(tmp_path)
    builder.verify_scene_assets(scene)
    output = tmp_path / "observations.sqlite"
    detector, reid = Detector(), ReID()
    ranges = {"6": builder.frame_range(scene["cameras"]["6"], .2, .6)}
    assert ranges == {"6": (2, 6)}
    with DetectionCache(output) as cache:
        _initialize(cache, scene)
        report = builder.extract(scene, cache, detector, reid, ranges, [6], output, progress_every=2)
        assert report["status"] == "complete"
        assert [r[0] for r in cache.db.execute("SELECT frame FROM frames ORDER BY frame")] == [2, 3, 4, 5]
        assert detector.seen == [60, 90, 120, 150]
        assert [value for value, _ in reid.seen] == [60, 90, 120]
        for frame, value in ((2, 60), (3, 90), (4, 120)):
            ds, det_ms, reid_ms = cache.frame(6, frame)
            assert len(ds) == 1 and ds[0]["feature"][value] == 1.
            assert ds[0]["reid_computed"] is True
            assert det_ms >= 0 and reid_ms >= 0
            raw = json.loads(cache.db.execute("SELECT payload FROM raw_frames WHERE camera=6 AND frame=?", (frame,)).fetchone()[0])
            assert len(raw) == 2 and all("feature" not in row for row in raw)
        assert cache.frame(6, 5)[0] == []
        assert report["camera_statistics"]["6"]["counts"] == {
            "raw_post_confidence_nms_class_filter": 6, "retained": 3,
            "nonzero_descriptors": 3, "outside_roi": 3}


def test_interrupted_resume_skips_only_committed_prefix_without_duplicates(tmp_path):
    scene = _scene(tmp_path); output = tmp_path / "resume.sqlite"
    ranges = {"6": (0, 6)}
    with DetectionCache(output) as cache:
        _initialize(cache, scene)
        with pytest.raises(RuntimeError, match="interruption"):
            builder.extract(scene, cache, Detector(fail_at=90), ReID(), ranges, [6], output, progress_every=2)
        assert builder.next_interval_frame(cache, 6, 0, 6) == 2  # uncommitted frame 2 rolled back
        resumed = Detector()
        report = builder.extract(scene, cache, resumed, ReID(), ranges, [6], output, progress_every=2)
        assert resumed.seen == [60, 90, 120, 150]
        assert report["status"] == "complete"
        assert cache.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0] == 6
        assert cache.db.execute("SELECT COUNT(*) FROM raw_frames").fetchone()[0] == 6


def test_invocation_profile_limit_is_resumable_and_preserves_interval(tmp_path):
    scene = _scene(tmp_path); output = tmp_path / "profile.sqlite"
    ranges = {"6": (2, 6)}
    with DetectionCache(output) as cache:
        _initialize(cache, scene)
        first = builder.extract(scene, cache, Detector(), ReID(), ranges, [6], output,
                                progress_every=50, max_frames_per_camera=1)
        assert first["status"] == "partial" and first["camera_statistics"]["6"]["next_local_frame"] == 3
        resumed = Detector()
        second = builder.extract(scene, cache, resumed, ReID(), ranges, [6], output)
        assert second["status"] == "complete" and resumed.seen == [90, 120, 150]


@pytest.mark.parametrize("tamper", ["gap", "raw_gap", "extra_frame"])
def test_resume_rejects_gaps_unmatched_raw_rows_and_extra_intervals(tmp_path, tamper):
    scene = _scene(tmp_path); output = tmp_path / "tamper.sqlite"
    ranges = {"6": (2, 6)}
    with DetectionCache(output) as cache:
        _initialize(cache, scene)
        builder.extract(scene, cache, Detector(), ReID(), ranges, [6], output)
        if tamper == "gap":
            cache.db.execute("DELETE FROM frames WHERE frame=3")
            cache.db.execute("DELETE FROM raw_frames WHERE frame=3")
        elif tamper == "raw_gap":
            cache.db.execute("DELETE FROM raw_frames WHERE frame=3")
        else:
            cache.add_frame(6, 0, [])
            cache.db.execute("INSERT INTO raw_frames VALUES (6,0,'[]')")
        cache.commit()
        with pytest.raises(ValueError, match="gaps"):
            builder.extract(scene, cache, Detector(), ReID(), ranges, [6], output)


def test_scene_asset_verification_rejects_changed_roi_and_clock_source(tmp_path):
    scene = _scene(tmp_path)
    builder.verify_scene_assets(scene)
    Path(scene["dataset_root"], "cam_timestamp/S02.txt").write_text("c006 0.1\n")
    with pytest.raises(ValueError, match="Scene source changed"):
        builder.verify_scene_assets(scene)


def test_weight_sources_require_existing_local_files_and_hash_every_export_asset(tmp_path):
    with pytest.raises(FileNotFoundError, match="local"):
        builder.detector_assets(tmp_path / "yolo11m.pt")
    source = tmp_path / "yolo11m.pt"; source.write_bytes(b"trusted local checkpoint")
    pt = builder.detector_assets(source)
    assert pt["source_checkpoint_sha256"] == file_hash(source)
    model = tmp_path / "yolo11m_openvino_model"; model.mkdir()
    for name in ("yolo11m.xml", "yolo11m.bin", "metadata.yaml", "export_provenance.json"):
        (model / name).write_text(name)
    with pytest.raises(ValueError, match="source .pt"):
        builder.detector_assets(model)
    ov = builder.detector_assets(model, source)
    assert set(ov["file_sha256"]) == {"yolo11m.xml", "yolo11m.bin", "metadata.yaml", "export_provenance.json"}
    before = ov["file_sha256"]["yolo11m.bin"]
    (model / "yolo11m.bin").write_text("changed model")
    assert builder.detector_assets(model, source)["file_sha256"]["yolo11m.bin"] != before


def test_changed_signed_inference_settings_or_source_cannot_resume(tmp_path):
    path = tmp_path / "signed.sqlite"
    with DetectionCache(path) as cache:
        original = {"source_sha256": "source A", "confidence": .1, "image_size": 960}
        cache.ensure_meta("detector", original)
        for changed in ({**original, "image_size": 640}, {**original, "source_sha256": "source B"},
                        {**original, "confidence": .2}):
            with pytest.raises(ValueError, match="provenance mismatch"):
                cache.ensure_meta("detector", changed)


def test_pytorch_predictor_thread_reset_is_corrected_after_initialization(tmp_path, monkeypatch):
    """Ultralytics changes CPU threads during first predictor setup."""
    checkpoint = tmp_path / "yolo11m.pt"; checkpoint.write_bytes(b"local")
    thread_state = {"count": 4}
    torch = SimpleNamespace(set_num_threads=lambda n: thread_state.update(count=n),
                            get_num_threads=lambda: thread_state["count"],
                            get_num_interop_threads=lambda: 2, manual_seed=lambda _: None)
    class FakeYOLO:
        def __init__(self, model, task):
            assert model == str(checkpoint.resolve()) and task == "detect"
            self.initialized = False

        def predict(self, **kwargs):
            assert kwargs["device"] == "cpu" and kwargs["half"] is False
            assert kwargs["rect"] is False and kwargs["classes"] == [2, 3, 5, 7]
            if not self.initialized:
                thread_state["count"] = 8
                self.initialized = True
            empty = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([])))
            return [SimpleNamespace(boxes=SimpleNamespace(xyxy=empty, conf=empty, cls=empty))]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    detector = builder.LocalYOLO(checkpoint, 4, {"seed": 7, "image_size": 960,
        "confidence": .1, "nms_iou": .7, "maximum_detections": 300})
    assert detector.runtime["torch_intraop_threads"] == 4
    assert thread_state["count"] == 4
    assert detector(np.zeros((12,16,3), np.uint8)) == []
    assert thread_state["count"] == 4
