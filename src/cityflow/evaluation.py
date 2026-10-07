"""Versioned CityFlowV2 evaluator bridge; the dataset evaluator stays unchanged.

The benchmark score follows the verified AIC22 evaluator's ROI, singleton-ID
and camera/frame concatenation rules. Supplemental scores retain single-camera
IDs, making fresh-ID fallbacks visible. No inference or association occurs here.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import importlib.metadata
import inspect
import math
import os
from pathlib import Path
import shutil
import tarfile
import zipfile

import motmetrics as mm
import numpy as np
import pandas as pd
from PIL import Image

OFFICIAL_SHA256 = "bfb89ebe3404c91903e379dc60d2aa5ef99584326891d14331be63d156e7dee4"
COLUMNS = ["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]
INTEGER_COLUMNS = COLUMNS[:7]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class _ImageCompatibility:
    TRANSPOSE = Image.Transpose.TRANSPOSE

    def __getattr__(self, name):
        return getattr(Image, name)


class _NumpyCompatibility:
    @staticmethod
    def asfarray(value):
        return np.asarray(value, dtype=float)

    def __getattr__(self, name):
        return getattr(np, name)


@contextmanager
def _motmetrics_compatibility():
    """Scope the removed NumPy alias to motmetrics' unchanged IoU function."""
    original = mm.distances.iou_matrix
    original_utils = mm.utils.iou_matrix
    namespace = dict(mm.distances.__dict__)
    namespace["np"] = _NumpyCompatibility()
    exec(compile(inspect.getsource(original), "<motmetrics-iou-compatibility>", "exec"), namespace)
    mm.distances.iou_matrix = namespace["iou_matrix"]
    mm.utils.iou_matrix = namespace["iou_matrix"]
    try:
        yield
    finally:
        mm.distances.iou_matrix = original
        mm.utils.iou_matrix = original_utils


def load_official_evaluator(path):
    """Compile exact official functions, omitting only unused module imports."""
    path = Path(path)
    if sha256(path) != OFFICIAL_SHA256:
        raise ValueError("CityFlow evaluator differs from the frozen AIC22 source")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"getData", "readData", "print_results", "eval"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in functions} != names:
        raise ValueError("Missing official evaluator functions")
    namespace = {"os": os, "zipfile": zipfile, "tarfile": tarfile,
                 "np": np, "pd": pd, "mm": mm, "Image": _ImageCompatibility()}
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    native_eval = next(n for n in functions if n.name == "eval")
    internal_names = {"removeOutliersROI", "removeOutliersSingleCam", "removeRepetition", "compare_dataframes_mtmc"}
    internals = [n for n in native_eval.body if isinstance(n, ast.FunctionDef) and n.name in internal_names]
    if {n.name for n in internals} != internal_names:
        raise ValueError("Missing official evaluator internals")
    namespace["mh"] = mm.metrics.create()
    exec(compile(ast.fix_missing_locations(ast.Module(body=internals, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def read_submission(path, *, cameras=None, frame_ranges=None, allow_float_boxes=False):
    """Require the official nine columns, original camera IDs and 1-based frames."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing submission file: {path}")
    if path.stat().st_size == 0:
        return pd.DataFrame({name: pd.Series(dtype="int64") for name in COLUMNS})
    table = pd.read_csv(path, sep=r"\s+|,", engine="python", header=None)
    if table.shape[1] != 9:
        raise ValueError("CityFlow tracking rows must have exactly nine columns")
    table.columns = COLUMNS
    try:
        table = table.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Non-numeric CityFlow row") from exc
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError("Non-finite CityFlow row")
    for column in (INTEGER_COLUMNS[:3] if allow_float_boxes else INTEGER_COLUMNS):
        if (table[column] != np.rint(table[column])).any():
            raise ValueError(f"CityFlow {column} must be integral")
        table[column] = table[column].astype("int64")
    if (table.CameraId < 1).any() or (table.FrameId < 1).any() or (table.Id < 0).any():
        raise ValueError("Use positive original camera/frame indices and non-negative IDs")
    if (table.Width <= 0).any() or (table.Height <= 0).any():
        raise ValueError("Non-positive CityFlow box size")
    if table.duplicated(["CameraId", "Id", "FrameId"]).any():
        raise ValueError("Duplicate camera/identity/frame key")
    if cameras is not None and not set(table.CameraId.unique()).issubset(set(cameras)):
        raise ValueError("Rows outside the declared camera cohort")
    if frame_ranges is not None:
        for camera, rows in table.groupby("CameraId"):
            bounds = frame_ranges.get(int(camera), frame_ranges.get(str(int(camera))))
            if bounds is None or not ((rows.FrameId >= bounds[0]) & (rows.FrameId < bounds[1])).all():
                raise ValueError("Rows outside declared 1-based [start,stop) frame ranges")
    # Preserve integer coordinate scalars during iterrows even when world
    # coordinates use float columns. Native ROI array indices must be integers.
    return table.astype(object)


def export_official_predictions(source, destination, *, cameras=None):
    """Truncate box coordinates identically for all methods, without expansion."""
    table = read_submission(source, cameras=cameras, allow_float_boxes=True)
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Official export already exists; use a new artifact path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for row in table.itertuples(index=False, name=None):
            camera, identity, frame, x, y, width, height, _, _ = row
            values = [int(value) for value in (camera, identity, frame, x, y, width, height)]
            if values[5] <= 0 or values[6] <= 0:
                raise ValueError("Integer export produced a non-positive box size")
            stream.write(" ".join(map(str, values)) + " -1 -1\n")
    return {"input_sha256": sha256(source), "export_sha256": sha256(destination),
            "rows": len(table), "coordinate_rule": "int(x), int(y), int(width), int(height), truncation toward zero; no box expansion"}


def frame_ranges_for_interval(camera_clocks, start_s, stop_s):
    """Exact native frame bounds for offset+(frame1-1)/fps in [start,stop)."""
    start, stop = Fraction(str(start_s)), Fraction(str(stop_s))
    if not 0 <= start < stop:
        raise ValueError("Invalid common-clock scoring interval")
    ranges = {}
    for camera, clock in camera_clocks.items():
        fps = Fraction(str(clock["fps"]))
        raw_offset = clock.get("offset_s", clock.get("timestamp_offset_s"))
        if raw_offset is None:
            raise ValueError("Missing native camera timestamp offset")
        offset = Fraction(str(raw_offset))
        if fps <= 0:
            raise ValueError("Non-positive camera FPS")
        first = max(1, math.ceil((start - offset) * fps) + 1)
        last = max(1, math.ceil((stop - offset) * fps) + 1)
        if "frames" in clock:
            first, last = min(first, int(clock["frames"]) + 1), min(last, int(clock["frames"]) + 1)
        ranges[int(camera)] = [first, last]
    return ranges


def select_frame_ranges(table, frame_ranges):
    mask = np.zeros(len(table), dtype=bool)
    for camera, bounds in frame_ranges.items():
        mask |= ((table.CameraId == int(camera)) & (table.FrameId >= bounds[0]) &
                 (table.FrameId < bounds[1])).to_numpy(dtype=bool)
    return table.loc[mask].copy()


def prepare_rois(dataset_root, split, scenario, cameras, output_directory):
    """Stage only the supplied dataset masks into the native evaluator layout."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("Invalid CityFlow split")
    output = Path(output_directory)
    files = {}
    for camera in sorted(cameras):
        source = Path(dataset_root) / split / scenario / f"c{camera:03d}" / "roi.jpg"
        if not source.is_file():
            raise ValueError(f"Missing dataset ROI: {source}")
        destination = output / split / f"c{camera:03d}" / "roi.jpg"
        expected = sha256(source)
        if destination.exists():
            if sha256(destination) != expected:
                raise ValueError("Existing staged ROI differs from the dataset")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        files[str(camera)] = {"source": str(source), "sha256": expected}
    return files


def _empty_prediction_metrics(ground_truth):
    """Native concatenation/IoU scoring extended to a zero-prediction boundary."""
    parts, offset = [], 0
    for camera in sorted(ground_truth.CameraId.unique()):
        part = ground_truth.loc[ground_truth.CameraId == camera,
                                ["FrameId", "Id", "X", "Y", "Width", "Height"]].copy()
        last = int(part.FrameId.max())
        part.FrameId += offset
        parts.append(part.set_index(["FrameId", "Id"]))
        offset += last
    hypotheses = pd.DataFrame(columns=["X", "Y", "Width", "Height"],
                             index=pd.MultiIndex.from_arrays([[], []], names=["FrameId", "Id"]))
    accumulator = mm.utils.compare_to_groundtruth(pd.concat(parts), hypotheses, "iou")
    metrics = list(mm.metrics.motchallenge_metrics) + ["num_frames", "idfp", "idfn", "idtp"]
    return mm.metrics.create().compute(accumulator, metrics=metrics, name="MultiCam")


def _metrics_dict(summary):
    result = {}
    for name, value in summary.iloc[-1].items():
        if pd.isna(value):
            result[name] = None
        elif isinstance(value, (int, np.integer)):
            result[name] = int(value)
        else:
            result[name] = float(value)
    return result


def evaluate_predictions(*, official_eval_path, ground_truth_path, prediction_path,
                         roi_directory, split, cameras, frame_ranges=None,
                         score_interval_s=None, camera_clocks=None):
    """Evaluate frozen outputs; this API accepts no model or association options.

    ``ground_truth_path`` must already describe the declared cohort. For a
    scenario-specific score, select the scenario from the official GT offline.
    ``frame_ranges`` applies equally to GT and predictions as a strict guard,
    never as an inference filter. Metric values are ratios, not percentages.
    """
    namespace = load_official_evaluator(official_eval_path)
    ground_truth = read_submission(ground_truth_path, cameras=cameras, frame_ranges=frame_ranges)
    predictions = read_submission(prediction_path, cameras=cameras, frame_ranges=frame_ranges)
    score_ranges = None
    if score_interval_s is not None:
        if camera_clocks is None or set(map(int, camera_clocks)) != set(cameras):
            raise ValueError("Scoring requires exact clocks for all declared cameras")
        score_ranges = frame_ranges_for_interval(camera_clocks, *score_interval_s)
        ground_truth = select_frame_ranges(ground_truth, score_ranges)
        predictions = select_frame_ranges(predictions, score_ranges)
    if ground_truth.empty:
        raise ValueError("Evaluation cohort has no ground truth")
    if set(ground_truth.CameraId.unique()) != set(cameras):
        raise ValueError("Ground truth is missing a declared camera")
    roi_hashes = {}
    for camera in sorted(cameras):
        path = Path(roi_directory) / split / f"c{camera:03d}" / "roi.jpg"
        if not path.is_file():
            raise ValueError(f"Missing staged ROI: {path}")
        roi_hashes[str(camera)] = sha256(path)
    if predictions.empty:
        roi_predictions = predictions.copy()
    else:
        roi_predictions = namespace["removeOutliersROI"](predictions.copy(), dstype=split,
                                                        roidir=str(roi_directory))
    benchmark_predictions = namespace["removeRepetition"](
        namespace["removeOutliersSingleCam"](roi_predictions.copy()))

    def score(table):
        if table.empty:
            return _metrics_dict(_empty_prediction_metrics(ground_truth))
        return _metrics_dict(namespace["compare_dataframes_mtmc"](ground_truth.copy(), table.copy()))

    with _motmetrics_compatibility():
        if not benchmark_predictions.empty:
            benchmark = _metrics_dict(namespace["eval"](ground_truth.copy(), predictions.copy(),
                                      dstype=split, roidir=str(roi_directory)))
        else:
            benchmark = score(benchmark_predictions)
        roi_only = score(roi_predictions)
        unfiltered = score(predictions)
    return {"dataset": "CityFlowV2 / AIC22 Track1", "cameras": sorted(cameras), "split": split,
            "official_benchmark": benchmark, "roi_only_identity": roi_only,
            "unfiltered_identity": unfiltered,
            "filter_counts": {"input_prediction_rows": len(predictions),
                              "rows_after_official_roi": len(roi_predictions),
                              "rows_after_official_singleton_filter": len(benchmark_predictions),
                              "rows_removed_by_roi": len(predictions) - len(roi_predictions),
                              "rows_removed_by_singleton_filter": len(roi_predictions) - len(benchmark_predictions),
                              "input_prediction_ids": int(predictions.Id.nunique()),
                              "benchmark_prediction_ids": int(benchmark_predictions.Id.nunique()),
                              "ground_truth_rows": len(ground_truth)},
            "provenance": {"official_eval_sha256": OFFICIAL_SHA256,
                           "ground_truth_sha256": sha256(ground_truth_path),
                           "prediction_sha256": sha256(prediction_path), "roi_sha256": roi_hashes,
                           "frame_ranges_1based_exclusive": frame_ranges,
                           "score_interval_s": score_interval_s,
                           "score_native_frame_ranges_1based_exclusive": score_ranges,
                           "camera_clocks": camera_clocks,
                           "lap_solver": mm.lap.default_solver,
                           "packages": {n: importlib.metadata.version(n) for n in
                                        ("numpy", "pandas", "Pillow", "motmetrics", "scipy")}},
            "compatibility": ["Exact verified evaluator functions are AST-loaded; unused pytrec_eval import is omitted.",
                              "Pillow TRANSPOSE is supplied through a local Image proxy.",
                              "NumPy asfarray is supplied only to the unchanged motmetrics IoU function, then restored.",
                              "The native empty-prediction failure is extended to zero true positives using identical camera concatenation and IoU scoring.",
                              "Input duplicates are rejected before the native keep-first rule can hide them."],
            "interpretation": "Official scoring removes single-camera predicted IDs. Report supplemental ROI-only and unfiltered global ID metrics alongside it; scores alone do not establish full published-method reproduction."}
