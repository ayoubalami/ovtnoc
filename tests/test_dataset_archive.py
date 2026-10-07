import zipfile

import pytest

from rhd.dataset import find_ground_truth, prepare_archive, safe_destination
from rhd.labels import prepare_labels
from rhd.evaluation import evaluate


def test_archive_extracts_tracking_data_and_resumes_without_rewriting(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("RoundaboutHD/imagesc001/video.mp4", b"test-video")
        z.writestr("RoundaboutHD/Multi_CAM_Ground_Turth.txt", b"labels")
        z.writestr("RoundaboutHD/RoundaboutHD_Reid_subset/train_images/test.jpg", b"image")
    output = tmp_path / "output"
    result = prepare_archive(archive, output)
    assert result["files"] == 2
    assert find_ground_truth(output / "RoundaboutHD").name == "Multi_CAM_Ground_Turth.txt"
    target = output / "RoundaboutHD/imagesc001/video.mp4"
    before = target.stat().st_mtime_ns
    prepare_archive(archive, output)
    assert target.stat().st_mtime_ns == before
    assert not (output / "RoundaboutHD/RoundaboutHD_Reid_subset").exists()
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="preserving"):
        prepare_archive(archive, output)


@pytest.mark.parametrize("member", ["../escape.txt", "C:/escape.txt", "/escape.txt", "RoundaboutHD/a/../../escape.txt"])
def test_archive_paths_cannot_escape_output(tmp_path, member):
    with pytest.raises(ValueError, match="Unsafe"):
        safe_destination(tmp_path, member)


def test_annotation_filename_ambiguity_is_explicit(tmp_path):
    for name in ("Multi_CAM_Ground_Truth.txt", "Multi_CAM_Ground_Turth.txt"):
        (tmp_path / name).write_text("labels")
    with pytest.raises(ValueError, match="Both"):
        find_ground_truth(tmp_path)


def test_label_preparation_shifts_frames_and_preserves_conflicts(tmp_path):
    source = tmp_path / "dataset"
    sct = source / "imagesc001/SCT/SCT_GT.txt"
    sct.parent.mkdir(parents=True)
    sct.write_text("1 1 10 20 30 40 2\n")
    raw = "1 5 1 10 20 20 20 -1 -1\n1 5 1 11 20 20 20 -1 -1\n"
    (source / "Multi_CAM_Ground_Turth.txt").write_text(raw)
    report = prepare_labels(source, tmp_path / "prepared", 1)
    assert report["multicam"]["duplicate_extra_rows"] == 1
    assert report["multicam"]["frame_bounds"][1] == (0, 0)
    assert report["multicam"]["duplicate_details"][0]["identical"] is False
    assert (source / "Multi_CAM_Ground_Turth.txt").read_text() == raw
    assert (tmp_path / "prepared/sct_zero_based.txt").read_text().startswith("1 1000001 0 ")
    assert len((tmp_path / "prepared/multicam_zero_based.txt").read_text().splitlines()) == 2
    # JSON resumes retain the same fingerprint without rewriting prepared files.
    assert prepare_labels(source, tmp_path / "prepared", 1)["signature"] == report["signature"]


def test_sct_labels_do_not_claim_cross_camera_identity_metrics(tmp_path):
    rows = tmp_path / "sct.txt"
    rows.write_text("1 1 0 10 20 20 20 -1 -1\n")
    metrics = evaluate(rows, rows, [1], 0, 1, 15, label_scope="sct")
    assert metrics["local_identity"]["IDF1"] == 1
    assert "global_identity" not in metrics
    assert "handoffs" not in metrics
