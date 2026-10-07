"""Export native-video illustrations of measured CityFlow handoff outcomes.

Annotations are used only for post-experiment illustration. Endpoint continuity
does not establish which physical source an association decision selected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from cityflow.data import event_time_us
from cityflow.evaluation import read_submission
from rhd.common import file_hash, write_json
from rhd.evaluation import spatial_matches

CATEGORIES = {"confirmed":"confirmed_handoffs", "identity_break":"identity_breaks", "missed":"missed_handoffs"}
TITLES = {"confirmed":"Confirmed endpoint continuity", "identity_break":"Broken endpoint continuity", "missed":"Missing endpoint evidence"}


def select_examples(metrics):
    """Earliest destination entry per observed category; missing categories stay explicit."""
    handoffs = metrics["handoffs"]
    selected = []
    for category, outcome in CATEGORIES.items():
        candidates = [r for r in handoffs["reference_outcomes"] if r["outcome"] == outcome]
        if not candidates:
            selected.append({"category":category, "present":False,
                             "reason":"No measured reference outcome in this category."})
            continue
        reference = min(candidates, key=lambda r:(r["destination_start_time_s"], r["source_end_time_s"],
                        r["gt_id"], r["source_camera"], r["destination_camera"],
                        r["source_last_native_frame_1based"], r["destination_first_native_frame_1based"]))
        source_id,destination_id = reference["source_predicted_id"],reference["destination_predicted_id"]
        if ((category == "confirmed" and (source_id is None or source_id != destination_id)) or
            (category == "identity_break" and (source_id is None or destination_id is None or source_id == destination_id)) or
            (category == "missed" and source_id is not None and destination_id is not None)):
            raise ValueError("Metric category contradicts recorded endpoint IDs")
        selected.append({"category":category, "present":True, "candidate_count":len(candidates),
                         "reference":reference})
    return selected


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _annotation_frames(scene, requested):
    result, hashes = {}, {}
    for camera in sorted({c for c,_ in requested}):
        path = Path(scene["cameras"][str(camera)]["video"]).parent/"gt"/"gt.txt"
        hashes[str(camera)] = {"path":str(path.resolve()), "sha256":file_hash(path)}
        seen = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            fields = line.split(",")
            frame, identity = int(fields[0]), int(fields[1])
            if (camera,frame) not in requested:
                continue
            key = camera,frame,identity
            if key in seen:
                raise ValueError("Duplicate GT row in selected illustration frame")
            seen.add(key)
            x,y,w,h = map(float, fields[2:6])
            if not np.isfinite([x,y,w,h]).all() or min(w,h) <= 0:
                raise ValueError("Invalid GT box in selected illustration frame")
            result.setdefault((camera,frame),[]).append({"camera":camera,"frame":frame-1,
                                                       "id":identity,"box":[x,y,x+w,y+h]})
    return result, hashes


def _panels(example, endpoint_frames=15):
    reference = example["reference"]
    result = []
    for role in ("source","destination"):
        endpoint = reference["source_last_native_frame_1based" if role == "source" else
                             "destination_first_native_frame_1based"]
        matched = reference[f"{role}_matched_native_frame_1based"]
        identity = reference[f"{role}_predicted_id"]
        if (matched is None) != (identity is None):
            raise ValueError("Metric matched frame and predicted ID disagree")
        if type(endpoint) is not int or endpoint < 1 or (matched is not None and
                (type(matched) is not int or matched < 1)):
            raise ValueError("Metric frame columns must be positive native integers")
        if matched is not None:
            distance = endpoint-matched if role == "source" else matched-endpoint
            if not 0 <= distance < endpoint_frames:
                raise ValueError("Reported match lies outside its native endpoint window")
        result.append({"role":role, "camera":reference[f"{role}_camera"],
                       "gt_id":reference["gt_id"], "gt_endpoint_frame_1based":endpoint,
                       "rendered_frame_1based":matched if matched is not None else endpoint,
                       "matched_frame_1based":matched, "reported_prediction_id":identity})
    return result


def verify_panels(examples, scene, prediction_path, availability_path=None, endpoint_frames=15):
    """Verify a recorded prediction against the GT at the exact displayed frame."""
    selected = [e for e in examples if e["present"]]
    requested = {(p["camera"],p["rendered_frame_1based"]) for e in selected for p in _panels(e,endpoint_frames)}
    gt_frames, hashes = _annotation_frames(scene, requested)
    table = read_submission(prediction_path, cameras=map(int,scene["cameras"]), allow_float_boxes=True)
    prediction_frames = {}
    for r in table.itertuples(index=False,name=None):
        camera,identity,frame,x,y,w,h,_,_ = r
        key = int(camera),int(frame)
        if key in requested:
            prediction_frames.setdefault(key,[]).append({"camera":int(camera),"frame":int(frame)-1,
                                                        "id":int(identity),"box":[float(x),float(y),float(x+w),float(y+h)]})
    availability = {}
    if availability_path is not None and Path(availability_path).is_file():
        for line in Path(availability_path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if (r["camera"],r["frame"]) not in requested:
                continue
            key = r["camera"],r["frame"],r["global_id"]
            if key in availability:
                raise ValueError("Duplicate availability row for selected illustration")
            observed = event_time_us(scene["cameras"][str(r["camera"])],r["frame"]-1)/1e6
            published = r["published_at_s"]
            if not np.isfinite(published) or published < observed-1.1e-6:
                raise ValueError("Invalid publication time in selected illustration")
            availability[key] = max(0.,published-observed)
    for example in selected:
        panels = _panels(example,endpoint_frames)
        for panel in panels:
            key = panel["camera"],panel["rendered_frame_1based"]
            gt, predictions = gt_frames.get(key,[]), prediction_frames.get(key,[])
            reference_gt = next((r for r in gt if r["id"] == panel["gt_id"]),None)
            if reference_gt is None:
                raise ValueError("Reference GT is absent from the reported native frame")
            gt_to_pred, _ = spatial_matches(gt,predictions,.5)
            matched = gt_to_pred.get((key[0],key[1]-1,panel["gt_id"]))
            if matched != panel["reported_prediction_id"]:
                raise ValueError("Exact-frame IoU match does not reproduce the recorded metric endpoint")
            prediction = next((r for r in predictions if r["id"] == matched),None)
            panel["gt_box_xyxy"] = reference_gt["box"]
            panel["prediction_box_xyxy"] = prediction["box"] if prediction else None
            panel["time_s"] = event_time_us(scene["cameras"][str(key[0])],key[1]-1)/1e6
            panel["publication_wait_s"] = availability.get((key[0],key[1],matched)) if matched is not None else None
            if panel["role"] == "destination":
                recorded = example["reference"]["destination_publication_wait_s"]
                actual = panel["publication_wait_s"]
                if (recorded is None) != (actual is None) or (recorded is not None and abs(recorded-actual) > 1.1e-6):
                    raise ValueError("Recorded matched publication wait does not reproduce availability")
        example["panels"] = panels
    return requested, hashes


def read_exact_frames(scene, requested):
    """Advance from native frame zero, grabbing preceding frames and reading requested ones."""
    result, videos = {}, {}
    for camera in sorted({c for c,_ in requested}):
        meta = scene["cameras"][str(camera)]
        path = Path(meta["video"])
        expected = meta["video_sha256"]
        actual = file_hash(path)
        if actual != expected:
            raise ValueError("Source video differs from frozen scene metadata")
        wanted = {f for c,f in requested if c == camera}
        if any(type(f) is not int or not 1 <= f <= meta["frames"] for f in wanted):
            raise ValueError("Illustration frame outside native video coverage")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Unreadable source video: {path}")
            if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != meta["frames"] or abs(capture.get(cv2.CAP_PROP_FPS)-meta["fps"]) > .001:
                raise ValueError("Video metadata differs from frozen scene")
            for frame1 in range(1,max(wanted)+1):
                if frame1 in wanted:
                    ok, pixels = capture.read()
                    if not ok or pixels is None:
                        raise ValueError(f"Cannot decode camera {camera}, frame {frame1}")
                    if abs(capture.get(cv2.CAP_PROP_POS_FRAMES)-frame1) > .1:
                        raise ValueError("Decoder position disagrees with sequential counter")
                    if pixels.shape[:2] != (meta["height"],meta["width"]):
                        raise ValueError("Decoded frame dimensions disagree with frozen scene")
                    result[(camera,frame1)] = pixels
                elif not capture.grab():
                    raise ValueError(f"Video ended before camera {camera}, frame {frame1}")
                if frame1 % 1500 == 0:
                    print(f"Sequentially read camera {camera} through frame {frame1}",flush=True)
        finally:
            capture.release()
        videos[str(camera)] = {"path":str(path.resolve()),"sha256":actual,"bytes":path.stat().st_size,
                               "fps":meta["fps"],"native_frames":meta["frames"],
                               "decode_rule":"No seeking: sequential grab from frame zero; read each requested native frame."}
    return result,videos


def _crop(box, width, height):
    x,y,x2,y2 = box
    cw,ch = min(width,max(160.,2*(x2-x),2*(y2-y)*16/9)),min(height,max(100.,2*(y2-y)))
    ch = min(height,max(ch,cw*9/16))
    left = max(0.,min(width-cw,(x+x2-cw)/2));top = max(0.,min(height-ch,(y+y2-ch)/2))
    return [int(left),int(top),int(np.ceil(left+cw)),int(np.ceil(top+ch))]


def _boxes(axis,panel,left=0,top=0):
    for key,color,style in (("gt_box_xyxy","#00e676","--"),("prediction_box_xyxy","#039be5","-")):
        box = panel[key]
        if box is not None:
            x,y,x2,y2 = box
            axis.add_patch(Rectangle((x-left,y-top),x2-x,y2-y,fill=False,
                                    edgecolor=color,linestyle=style,
                                    linewidth=3.0 if key == "gt_box_xyxy" else 1.8))


def render_example(example,frames,output,method):
    fig,axes = plt.subplots(2,2,figsize=(13.5,8.5),gridspec_kw={"height_ratios":[1.15,1.]})
    fig.suptitle(f"{TITLES[example['category']]} - custom handoff metric",fontsize=17,y=.985)
    for column,panel in enumerate(example["panels"]):
        pixels = frames[(panel["camera"],panel["rendered_frame_1based"])]
        rgb = cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)
        height,width = pixels.shape[:2]
        axes[0,column].imshow(rgb);_boxes(axes[0,column],panel)
        prediction = panel["reported_prediction_id"]
        prediction_label = str(prediction) if prediction is not None else "no endpoint match"
        wait = panel["publication_wait_s"]
        wait_label = f"publication wait {wait:.3f} s" if wait is not None else "publication wait unavailable"
        axes[0,column].set_title(f"{panel['role'].capitalize()} evidence: camera {panel['camera']}\n"
            f"Native frame {panel['rendered_frame_1based']} | scene time {panel['time_s']:.3f} s\n"
            f"GT {panel['gt_id']} | predicted ID {prediction_label}\n{wait_label}",fontsize=11,pad=8)
        left,top,right,bottom = _crop(panel["gt_box_xyxy"],width,height)
        axes[1,column].imshow(rgb[top:bottom,left:right],interpolation="nearest")
        _boxes(axes[1,column],panel,left,top)
        axes[1,column].set_title("Crop enlargement of the same original frame",fontsize=11,pad=7)
        for axis in axes[:,column]:
            axis.set_axis_off()
        panel["crop_bounds_xyxy"] = [left,top,right,bottom]
        panel["native_frame_dimensions"] = [width,height]
    fig.text(.5,.042,f"{method} | dashed green: GT | solid blue: exact-frame IoU-matched prediction\n"
             "Original video pixels with annotations and crop enlargement. This figure does not identify a wrong physical source.",
             ha="center",va="center",fontsize=10)
    fig.subplots_adjust(top=.80,bottom=.11,left=.025,right=.98,wspace=.12,hspace=.21)
    fig.savefig(output/f"{example['category']}.png",dpi=160,facecolor="white")
    plt.close(fig)


def _caption(example,scenario,method):
    r = example["reference"]
    start = f"{method}, CityFlowV2 {scenario}, GT vehicle {r['gt_id']}, camera {r['source_camera']} to {r['destination_camera']}. "
    if example["category"] == "confirmed":
        state = f"Matched endpoint prediction IDs agree ({r['source_predicted_id']} to {r['destination_predicted_id']}). "
    elif example["category"] == "identity_break":
        state = f"Matched endpoint prediction IDs differ ({r['source_predicted_id']} to {r['destination_predicted_id']}). This demonstrates an endpoint continuity failure; a wrong selected physical source is not established. "
    else:
        state = "At least one endpoint has no spatial match within the declared native-frame window. This does not distinguish detector, tracker, filter, or association causes. "
    frames = "; ".join(f"{p['role']}: displayed native frame {p['rendered_frame_1based']} (literal GT endpoint {p['gt_endpoint_frame_1based']})" for p in example["panels"])
    wait = r["destination_publication_wait_s"]
    return start+state+frames+". "+(f"Matched destination publication wait: {wait:.3f} s. " if wait is not None else "Matched destination publication wait is unavailable. ")+"Green dashed boxes are GT; blue boxes are one-to-one IoU matches at threshold 0.5. Illustrative examples are not additional independent trials."


def export_examples(metrics_path,scene_path,output):
    metrics_path,scene_path,output = Path(metrics_path),Path(scene_path),Path(output)
    if output.exists():
        raise ValueError("Use a new output directory; existing example artifacts are preserved")
    metrics,scene = _read_json(metrics_path),_read_json(scene_path)
    if metrics["handoffs"]["scope"]["scenario"] != scene["scenario"]:
        raise ValueError("Metrics and scene describe different scenarios")
    if metrics["handoffs"]["scope"]["iou_threshold"] != .5:
        raise ValueError("This verified exporter requires the frozen IoU=0.5 endpoint rule")
    run = metrics_path.parent
    predictions = run/"predictions.txt";availability = run/"availability.jsonl"
    manifest_path = run/"manifest.json"
    method = run.name
    method_options = None
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("status") != "complete":
            raise ValueError("Cannot illustrate incomplete inference")
        expected = manifest.get("artifact_hashes",{}).get("predictions.txt")
        if expected is not None and file_hash(predictions) != expected:
            raise ValueError("Prediction file differs from completed inference manifest")
        method_options = manifest.get("options",{}).get("config",{}).get("association")
        if method_options and "confirmation_frames" in method_options:
            method = f"{run.name} (K={method_options['confirmation_frames']})"
    examples = select_examples(metrics)
    requested,gt_hashes = verify_panels(examples,scene,predictions,availability,
                                     metrics["handoffs"]["scope"]["endpoint_native_frames"])
    frames,videos = read_exact_frames(scene,requested)
    output.mkdir(parents=True)
    (output/"source_frames").mkdir()
    frame_records = []
    for (camera,frame),pixels in sorted(frames.items()):
        path = output/"source_frames"/f"camera{camera:03d}_native_frame{frame:06d}.png"
        if not cv2.imwrite(str(path),pixels):
            raise ValueError("Cannot preserve decoded source frame")
        frame_records.append({"camera":camera,"native_frame_1based":frame,"local_frame_zero_based":frame-1,
                              "time_s":event_time_us(scene["cameras"][str(camera)],frame-1)/1e6,
                              "file":str(path.relative_to(output)),"png_sha256":file_hash(path),
                              "decoded_bgr_sha256":hashlib.sha256(pixels.tobytes()).hexdigest()})
    captions = ["# CityFlowV2 recorded handoff examples","",
                "Examples illustrate custom endpoint outcomes; they do not establish a wrong physical association source. "
                "Frames are one-based native video frames. Matched endpoint-window frames are displayed when available, "
                "otherwise the literal GT endpoint is displayed.",""]
    for example in examples:
        captions += [f"## {example['category'].replace('_',' ').capitalize()}",""]
        if not example["present"]:
            captions += [example["reason"],""]
            continue
        render_example(example,frames,output,method)
        example["caption"] = _caption(example,scene["scenario"],method)
        captions += [f"![{example['category']}]({example['category']}.png)","",example["caption"],""]
    (output/"captions.md").write_text("\n".join(captions),encoding="utf-8")
    report = {"status":"complete","dataset_id":f"CityFlowV2/AIC22 Track1/{scene['scenario']}",
              "method":method,"method_association_options":method_options,"run_directory":str(run.resolve()),
              "scope":metrics["handoffs"]["scope"],"examples":examples,"source_frames":frame_records,
              "source_videos":videos,"ground_truth_files":gt_hashes,
              "provenance":{"metrics_path":str(metrics_path.resolve()),"metrics_sha256":file_hash(metrics_path),
                            "scene_path":str(scene_path.resolve()),"scene_sha256":file_hash(scene_path),
                            "predictions_sha256":file_hash(predictions),
                            "availability_sha256":file_hash(availability) if availability.is_file() else None,
                            "run_manifest_sha256":file_hash(manifest_path) if manifest_path.is_file() else None,
                            "script_sha256":file_hash(Path(__file__)),"opencv_version":cv2.__version__,
                            "matplotlib_version":matplotlib.__version__},
              "selection_rule":"Earliest destination entry, then source end, GT ID, cameras and native endpoint frames, per measured category. Missing categories remain explicit.",
              "pixel_rule":"Original decoded pixels; drawing boxes/text and arranging full-frame/crop panels only. Source PNGs preserve untouched decoded frames.",
              "interpretation":"Custom endpoint continuity states only. No wrong physical source claim is inferred from differing IDs or a missing endpoint.",
              "artifact_sha256":{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    write_json(output/"examples.json",report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics",type=Path,required=True)
    parser.add_argument("--scene",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report = export_examples(args.metrics,args.scene,args.output)
    print(json.dumps({"output":str(args.output),"method":report["method"],
                      "categories":{e['category']:e['present'] for e in report['examples']}},indent=2))


if __name__ == "__main__":
    main()
