#!/usr/bin/env python3

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import depthai as dai

from common import (
    DEFAULT_FISHEYE_SOCKETS,
    detect_aprilgrid,
    detect_apriltag_ids_in_image,
    detect_single_apriltag,
    draw_detection,
    ensure_dir,
    frame_timestamp_ms,
    make_camera_pipeline,
    make_grid,
    normalize_socket_names,
    save_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SINGLE_TAG_IMAGE = REPO_ROOT / "data" / "calibration" / "apriltag_test2.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture synchronized calibration images from a DepthAI fisheye module.",
        allow_abbrev=False,
    )
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--in", dest="dataset_kind", action="store_const", const="intrinsics", default="intrinsics", help="Capture an intrinsics dataset. Default.")
    dataset_group.add_argument("--ex", dest="dataset_kind", action="store_const", const="extrinsics", help="Capture an extrinsics dataset.")

    parser.add_argument("--module", choices=["head", "left_wrist", "right_wrist", "handle"], default=None, help="Camera module being calibrated.")
    parser.add_argument("--head", dest="module", action="store_const", const="head", help="Shortcut for --module head.")
    parser.add_argument("--left-wrist", "--left_wrist", dest="module", action="store_const", const="left_wrist", help="Shortcut for --module left_wrist.")
    parser.add_argument("--right-wrist", "--right_wrist", dest="module", action="store_const", const="right_wrist", help="Shortcut for --module right_wrist.")
    parser.add_argument("--handle", dest="module", action="store_const", const="handle", help="Shortcut for --module handle.")
    parser.add_argument("--output", default="", help="Dataset output folder. Defaults to data/calibration/{intrinsics_dataset|extrinsics_dataset}/{module}_dataset.")
    parser.add_argument("--sockets", nargs="+", default=DEFAULT_FISHEYE_SOCKETS, help="Camera sockets to capture. Default: CAM_A CAM_B CAM_C CAM_D.")
    parser.add_argument("--width", type=int, default=1920, help="Camera output width.")
    parser.add_argument("--height", type=int, default=1200, help="Camera output height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Camera FPS.")
    parser.add_argument("--max-skew-ms", type=float, default=50.0, help="Max timestamp spread accepted for one capture.")
    parser.add_argument("--device", default="", help="Optional DepthAI MX ID.")
    parser.add_argument("--detect-every", type=int, default=15, help="Run board detection every N GUI loops.")
    parser.add_argument("--detect-scale", type=float, default=1.0, help="Scale used for live board detection overlays.")
    parser.add_argument("--robust-detection", action="store_true", help="Try slower fallback AprilTag detection passes when the fast preview path misses.")
    parser.add_argument("--debug-detection", action="store_true", help="Print live per-camera board detection diagnostics.")
    parser.add_argument(
        "--headless-test-seconds",
        type=float,
        default=0.0,
        help="Start the camera pipeline, collect frames without GUI or board detection, then exit.",
    )

    parser.add_argument("--target", choices=["aprilgrid", "single_tag"], default="aprilgrid", help="Calibration target type.")
    parser.add_argument("--aprilgrid-rows", type=int, default=6, help="AprilGrid tag rows.")
    parser.add_argument("--aprilgrid-cols", type=int, default=6, help="AprilGrid tag columns.")
    parser.add_argument("--tag-size", type=float, default=0.0352, help="AprilTag black square size in meters. Use the physical single tag size in single_tag mode.")
    parser.add_argument("--tag-spacing", type=float, default=0.3, help="Kalibr AprilGrid tagSpacing ratio. Default: 0.3.")
    parser.add_argument("--tag-id", type=int, default=None, help="Target AprilTag ID for single_tag mode. Defaults to the ID detected in --tag-image.")
    parser.add_argument("--tag-image", default=str(DEFAULT_SINGLE_TAG_IMAGE), help="Reference single-tag image used to infer --tag-id.")
    parser.add_argument("--start-id", type=int, default=0, help="First AprilTag ID. Default: 0.")
    parser.add_argument("--end-id", type=int, default=35, help="Last AprilTag ID. Default: 35.")
    parser.add_argument("--tag-family", default="DICT_APRILTAG_36H11", help="AprilTag dictionary. Default: DICT_APRILTAG_36H11.")
    parser.add_argument("--min-tags", type=int, default=6, help="Minimum AprilTags required per camera detection.")
    args = parser.parse_args()
    if args.module is None:
        args.module = "head"
    return args


def resolve_single_tag_id(args):
    if args.tag_id is not None:
        return int(args.tag_id)
    ids = detect_apriltag_ids_in_image(args.tag_image, args.tag_family)
    if len(ids) != 1:
        raise RuntimeError(f"Expected exactly one AprilTag in {args.tag_image}, detected IDs: {ids}")
    return int(ids[0])


def default_output(args):
    if args.output:
        return Path(args.output)
    dataset_folder = "extrinsics_dataset" if args.dataset_kind == "extrinsics" else "intrinsics_dataset"
    return REPO_ROOT / "data" / "calibration" / dataset_folder / f"{args.module}_dataset"


def detect_for_overlay(frame, args):
    scale = max(0.1, min(1.0, args.detect_scale))
    detect_frame = frame
    if scale != 1.0:
        detect_frame = cv2.resize(frame, None, fx=scale, fy=scale)

    if args.target == "single_tag":
        detection = detect_single_apriltag(
            detect_frame,
            tag_id=args.resolved_tag_id,
            tag_size_m=args.tag_size,
            dictionary_name=args.tag_family,
            robust=args.robust_detection,
        )
    else:
        detection = detect_aprilgrid(
            detect_frame,
            rows=args.aprilgrid_rows,
            cols=args.aprilgrid_cols,
            tag_size_m=args.tag_size,
            tag_spacing_m=args.tag_spacing * args.tag_size,
            start_id=args.start_id,
            end_id=args.end_id,
            dictionary_name=args.tag_family,
            min_tags=args.min_tags,
            robust=args.robust_detection,
        )

    if detection is not None and scale != 1.0:
        detection["image_points"] = detection["image_points"] / scale
        for key in ("mean_tag_edge_px", "min_tag_edge_px", "max_tag_edge_px"):
            if key in detection:
                detection[key] = float(detection[key]) / scale
    return detection


def gray_stats(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mean, stddev = cv2.meanStdDev(gray)
    return float(mean[0][0]), float(stddev[0][0])


def detection_debug_line(name, frame, detection):
    gray_mean, gray_std = gray_stats(frame)
    if detection is None:
        return f"{name}: no board | gray_mean={gray_mean:.1f} gray_std={gray_std:.1f}"

    tag_ids = detection.get("tag_ids", [])
    shown_ids = ",".join(str(tag_id) for tag_id in tag_ids[:12])
    if len(tag_ids) > 12:
        shown_ids += ",..."
    return (
        f"{name}: tags={detection.get('tag_count', 0)} "
        f"corners={detection.get('corner_count', len(detection['image_points']))} "
        f"edge_mean={float(detection.get('mean_tag_edge_px', 0.0)):.1f}px "
        f"edge_min={float(detection.get('min_tag_edge_px', 0.0)):.1f}px "
        f"gray_mean={gray_mean:.1f} gray_std={gray_std:.1f} "
        f"detector={detection.get('detector', '')} ids=[{shown_ids}]"
    )


def main():
    args = parse_args()
    args.resolved_tag_id = resolve_single_tag_id(args) if args.target == "single_tag" else None
    socket_names = normalize_socket_names(args.sockets)
    output = ensure_dir(default_output(args))
    images_dir = ensure_dir(output / "images")

    board_info = {
        "type": args.target,
        "aprilgrid_rows": args.aprilgrid_rows,
        "aprilgrid_cols": args.aprilgrid_cols,
        "tag_size_m": args.tag_size,
        "tag_spacing_ratio": args.tag_spacing,
        "tag_spacing_m": args.tag_spacing * args.tag_size,
        "tag_id": args.resolved_tag_id,
        "tag_image": str(Path(args.tag_image)),
        "start_id": args.start_id,
        "end_id": args.end_id,
        "tag_family": args.tag_family,
        "min_tags": args.min_tags,
    }
    save_json(
        output / "dataset.json",
        {
            "schema": "calibrate.capture_dataset.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_kind": args.dataset_kind,
            "module": args.module,
            "sockets": socket_names,
            "image_size": [args.width, args.height],
            "board": board_info,
        },
    )

    pipeline, streams = make_camera_pipeline(socket_names, args.width, args.height, args.fps)
    device_args = []
    if args.device:
        success, info = dai.Device.getDeviceByMxId(args.device)
        if not success:
            raise RuntimeError(f"Device not found: {args.device}")
        device_args.append(info)

    if args.headless_test_seconds <= 0:
        print("Press SPACE to save a synchronized sample. Press q to quit.")
        print(f"Dataset kind: {args.dataset_kind}")
        print(f"Cameras: {' '.join(socket_names)}")
        print(f"Saving dataset to: {Path(output).resolve()}")

    latest = {}
    last_detections = {}
    loop_count = 0
    sample_id = 0
    try:
        with dai.Device(pipeline, *device_args) as device:
            queues = {
                name: device.getOutputQueue(stream_name, maxSize=1, blocking=False)
                for name, stream_name in streams.items()
            }

            if args.headless_test_seconds > 0:
                deadline = time.monotonic() + args.headless_test_seconds
                counts = {name: 0 for name in socket_names}
                while time.monotonic() < deadline:
                    for name, queue in queues.items():
                        msg = queue.tryGet()
                        if msg is not None:
                            counts[name] += 1
                            latest[name] = {
                                "timestamp_ms": frame_timestamp_ms(msg),
                                "sequence": msg.getSequenceNum(),
                            }
                    time.sleep(0.002)

                print("Headless pipeline test result:")
                ok = True
                for name in socket_names:
                    count = counts[name]
                    ok = ok and count > 0
                    print(f"  {name}: {count} frames")
                if len(latest) == len(socket_names):
                    timestamps = [latest[name]["timestamp_ms"] for name in socket_names]
                    print(f"  latest timestamp skew: {max(timestamps) - min(timestamps):.2f} ms")
                if not ok:
                    raise RuntimeError("At least one camera produced zero frames.")
                return

            while True:
                loop_count += 1
                for name, queue in queues.items():
                    msg = queue.tryGet()
                    if msg is not None:
                        latest[name] = {
                            "frame": msg.getCvFrame(),
                            "timestamp_ms": frame_timestamp_ms(msg),
                            "sequence": msg.getSequenceNum(),
                        }

                overlay_frames = {}
                for name in socket_names:
                    item = latest.get(name)
                    if item is None:
                        overlay_frames[name] = None
                        continue
                    if loop_count % max(1, args.detect_every) == 0:
                        last_detections[name] = detect_for_overlay(item["frame"], args)
                        if args.debug_detection:
                            print(detection_debug_line(name, item["frame"], last_detections[name]), flush=True)
                    overlay_frames[name] = draw_detection(
                        item["frame"],
                        last_detections.get(name),
                        "aprilgrid",
                    )

                grid = make_grid(overlay_frames, socket_names)
                cv2.putText(
                    grid,
                    "SPACE save | q quit | keep board still while saving",
                    (18, grid.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(f"{' '.join(socket_names)} fisheye calibration capture", grid)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key != ord(" "):
                    continue

                if any(name not in latest for name in socket_names):
                    print("Skip: waiting for all cameras.")
                    continue

                timestamps = [latest[name]["timestamp_ms"] for name in socket_names]
                skew_ms = max(timestamps) - min(timestamps)
                if skew_ms > args.max_skew_ms:
                    print(f"Skip: camera timestamp skew {skew_ms:.2f} ms > {args.max_skew_ms:.2f} ms")
                    continue

                found = sum(1 for detection in last_detections.values() if detection is not None)

                sample_id += 1
                sample_name = f"sample_{sample_id:06d}"
                sample_dir = ensure_dir(images_dir / sample_name)
                metadata = {
                    "sample": sample_name,
                    "timestamp_skew_ms": skew_ms,
                    "cameras": {},
                }

                for name in socket_names:
                    image_path = sample_dir / f"{name}.png"
                    cv2.imwrite(str(image_path), latest[name]["frame"])
                    detection = last_detections.get(name)
                    metadata["cameras"][name] = {
                        "image": str(image_path.relative_to(output)),
                        "timestamp_ms": latest[name]["timestamp_ms"],
                        "sequence": latest[name]["sequence"],
                        "board_detected": detection is not None,
                        "corners": 0 if detection is None else int(len(detection["image_points"])),
                    }
                    if detection is not None:
                        metadata["cameras"][name]["image_points"] = detection["image_points"].reshape(-1, 2).tolist()
                        metadata["cameras"][name]["object_points"] = detection["object_points"].reshape(-1, 3).tolist()
                        metadata["cameras"][name]["ids"] = detection["ids"].reshape(-1).tolist()
                        metadata["cameras"][name]["tag_ids"] = detection["tag_ids"]
                        metadata["cameras"][name]["tag_count"] = detection["tag_count"]
                        metadata["cameras"][name]["detector"] = detection.get("detector", "")
                        metadata["cameras"][name]["mean_tag_edge_px"] = detection.get("mean_tag_edge_px", 0.0)
                        metadata["cameras"][name]["min_tag_edge_px"] = detection.get("min_tag_edge_px", 0.0)
                        metadata["cameras"][name]["max_tag_edge_px"] = detection.get("max_tag_edge_px", 0.0)

                save_json(sample_dir / "metadata.json", metadata)
                print(f"Saved {sample_name}: skew={skew_ms:.2f} ms, board visible in {found}/{len(socket_names)} cameras")
    except RuntimeError as ex:
        print(f"Capture stopped because the DepthAI stream failed: {ex}")
        raise
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
