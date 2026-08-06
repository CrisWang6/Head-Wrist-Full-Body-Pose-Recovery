#!/usr/bin/env python3
"""Record one or two externally triggered OAK modules with CAM_A and CAM_D.

The program first runs both cameras freely for preview.  Press SPACE after the
view and storage location have been checked.  It then restarts the OAK pipeline
in external-trigger mode, opens all queues, and prints [ARMED].  Only send the
external 30 Hz trigger after that message appears.

The raw high-quality MJPEG streams contain no preview frames.  timestamps.csv uses exposure
END as device_timestamp_us so it can be consumed directly by the calibration
scripts.  trigger_events.csv records the first trigger inferred from the first
externally triggered frame's exposure-start OAK timestamp.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import cv2
import depthai as dai
import numpy as np


WIDTH = 1920
HEIGHT = 1200
FPS = 30.0
EXPOSURE_US = 3_000
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 400
DEFAULT_ISO = 800
DEFAULT_MJPEG_QUALITY = 95
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\hand\Desktop\oak_ad_recordings")
WINDOW_NAME = "OAK CAM_A/D external-trigger recorder"

CAMERAS = {
    "CAM_A": {
        "socket": dai.CameraBoardSocket.CAM_A,
        "side": "left",
    },
    "CAM_D": {
        "socket": dai.CameraBoardSocket.CAM_D,
        "side": "right",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"session root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--mxid", help="optional OAK MXID")
    parser.add_argument(
        "--external-count",
        type=int,
        choices=(1, 2),
        default=1,
        help="number of OAK boards to record (default: 1)",
    )
    parser.add_argument(
        "--mxids",
        nargs="+",
        help="two MXIDs in external_01/external_02 order",
    )
    parser.add_argument("--iso", type=int, default=DEFAULT_ISO)
    parser.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_MJPEG_QUALITY,
        help="MJPEG quality from 1 to 100 (default: 95)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="seconds after the first trigger; 0 records until Q",
    )
    parser.add_argument(
        "--no-avi",
        action="store_true",
        help="keep only .mjpeg files; otherwise also remux to .avi",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the DepthAI API and pipeline construction without a device",
    )
    return parser.parse_args()


def timestamp_us(packet, offset=None) -> int | None:
    try:
        value = (
            packet.getTimestampDevice()
            if offset is None
            else packet.getTimestampDevice(offset)
        )
        return int(round(value.total_seconds() * 1_000_000.0))
    except Exception:
        return None


def packet_bytes(packet) -> bytes:
    data = packet.getData()
    return data.tobytes() if hasattr(data, "tobytes") else bytes(data)


def is_complete_jpeg(data: bytes) -> bool:
    return data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")


def configure_camera(
    camera,
    camera_name: str,
    iso: int,
    external_trigger: bool,
) -> None:
    camera.setBoardSocket(CAMERAS[camera_name]["socket"])
    camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
    camera.setVideoSize(WIDTH, HEIGHT)
    camera.setPreviewSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
    camera.setPreviewKeepAspectRatio(False)
    camera.setInterleaved(False)
    camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    camera.setFps(FPS)
    camera.initialControl.setManualExposure(EXPOSURE_US, iso)
    if external_trigger:
        camera.initialControl.setFrameSyncMode(
            dai.CameraControl.FrameSyncMode.INPUT
        )


def make_preview_pipeline(iso: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    for camera_name in CAMERAS:
        camera = pipeline.createColorCamera()
        configure_camera(camera, camera_name, iso, external_trigger=False)
        output = pipeline.createXLinkOut()
        output.setStreamName(f"{camera_name}_preview")
        camera.preview.link(output.input)
    return pipeline


def make_record_pipeline(iso: int, quality: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    for camera_name in CAMERAS:
        camera = pipeline.createColorCamera()
        configure_camera(camera, camera_name, iso, external_trigger=True)

        encoder = pipeline.createVideoEncoder()
        encoder.setDefaultProfilePreset(
            FPS, dai.VideoEncoderProperties.Profile.MJPEG
        )
        encoder.setQuality(quality)
        camera.video.link(encoder.input)

        encoded_output = pipeline.createXLinkOut()
        encoded_output.setStreamName(f"{camera_name}_mjpeg")
        encoder.bitstream.link(encoded_output.input)

        preview_output = pipeline.createXLinkOut()
        preview_output.setStreamName(f"{camera_name}_preview")
        camera.preview.link(preview_output.input)
    return pipeline


def find_device_info(mxid: str | None):
    if mxid is None:
        return None
    available = dai.Device.getAllAvailableDevices()
    for info in available:
        candidate = (
            info.getMxId() if hasattr(info, "getMxId") else info.getDeviceId()
        )
        if candidate == mxid:
            return info
    found = [
        info.getMxId() if hasattr(info, "getMxId") else info.getDeviceId()
        for info in available
    ]
    raise RuntimeError(f"Could not find MXID={mxid}; available devices: {found}")


def open_device(pipeline: dai.Pipeline, mxid: str | None) -> dai.Device:
    info = find_device_info(mxid)
    return dai.Device(pipeline) if info is None else dai.Device(pipeline, info)


def device_mxid(device: dai.Device) -> str:
    info = device.getDeviceInfo()
    return info.getMxId() if hasattr(info, "getMxId") else info.getDeviceId()


def info_mxid(info) -> str:
    return info.getMxId() if hasattr(info, "getMxId") else info.getDeviceId()


def resolve_dual_bindings(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.mxids:
        selected = list(args.mxids)
    else:
        selected = sorted(
            info_mxid(info) for info in dai.Device.getAllAvailableDevices()
        )[:2]
    if len(selected) != 2:
        available = sorted(
            info_mxid(info) for info in dai.Device.getAllAvailableDevices()
        )
        raise RuntimeError(
            "Two-board mode requires two available OAK devices; "
            f"selected={selected}, available={available}"
        )
    if selected[0] == selected[1]:
        raise RuntimeError("The same OAK MXID cannot be assigned twice")
    return list(zip(("external_01", "external_02"), selected))


def open_dual_device(pipeline: dai.Pipeline, mxid: str) -> dai.Device:
    deadline = time.monotonic() + 15.0
    while True:
        try:
            info = find_device_info(mxid)
            return dai.Device(pipeline, info)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)


def unique_session_path(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    return candidate


class RecordingSession:
    timestamp_fields = [
        "camera",
        "side",
        "frame_index",
        "sequence",
        "timestamp_reference",
        "device_timestamp_us",
        "packet_device_timestamp_us",
        "exposure_start_device_timestamp_us",
        "exposure_middle_device_timestamp_us",
        "exposure_end_device_timestamp_us",
        "host_receive_monotonic_ns",
        "payload_bytes",
        "jpeg_valid",
    ]
    trigger_fields = [
        "event",
        "camera",
        "sequence",
        "oak_device_timestamp_us",
        "timestamp_source",
        "exposure_start_device_timestamp_us",
        "exposure_end_device_timestamp_us",
        "host_receive_monotonic_ns",
    ]

    def __init__(
        self,
        output_root: Path,
        quality: int,
        iso: int,
        remux_avi: bool,
        directory: Path | None = None,
    ) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        self.directory = directory or unique_session_path(output_root)
        self.directory.mkdir(parents=True, exist_ok=directory is not None)
        self.quality = quality
        self.iso = iso
        self.remux_avi = remux_avi
        self.created_at = datetime.now().astimezone()
        self.armed_at: str | None = None
        self.first_trigger_at: str | None = None
        self.device_id: str | None = None
        self.status = "preview"
        self.counts = {camera: 0 for camera in CAMERAS}
        self.bytes_written = {camera: 0 for camera in CAMERAS}
        self.first_packets: dict[str, dict[str, Any]] = {}
        self.summary_trigger_written = False
        self.video_paths = {
            camera: self.directory
            / (
                f"{config['side']}_{camera}_{WIDTH}x{HEIGHT}_"
                f"{int(FPS)}fps.mjpeg"
            )
            for camera, config in CAMERAS.items()
        }
        self.video_files: dict[str, BinaryIO] = {
            camera: path.open("wb", buffering=4 * 1024 * 1024)
            for camera, path in self.video_paths.items()
        }
        self.timestamps_path = self.directory / "timestamps.csv"
        self.timestamps_file: TextIO = self.timestamps_path.open(
            "w", newline="", encoding="utf-8"
        )
        self.timestamps_writer = csv.DictWriter(
            self.timestamps_file, fieldnames=self.timestamp_fields
        )
        self.timestamps_writer.writeheader()
        self.trigger_path = self.directory / "trigger_events.csv"
        self.trigger_file: TextIO = self.trigger_path.open(
            "w", newline="", encoding="utf-8"
        )
        self.trigger_writer = csv.DictWriter(
            self.trigger_file, fieldnames=self.trigger_fields
        )
        self.trigger_writer.writeheader()
        self.flush()
        self.write_metadata()

    def arm(self, device_id: str) -> None:
        self.device_id = device_id
        self.status = "armed_waiting_for_external_trigger"
        self.armed_at = datetime.now().astimezone().isoformat()
        self.trigger_writer.writerow(
            {
                "event": "recording_armed_host_event",
                "camera": "",
                "sequence": "",
                "oak_device_timestamp_us": "",
                "timestamp_source": "host_only",
                "exposure_start_device_timestamp_us": "",
                "exposure_end_device_timestamp_us": "",
                "host_receive_monotonic_ns": time.perf_counter_ns(),
            }
        )
        self.flush()
        self.write_metadata()

    def write_packet(self, camera: str, packet) -> None:
        host_ns = time.perf_counter_ns()
        raw = packet_bytes(packet)
        sequence = int(packet.getSequenceNum())
        packet_ts = timestamp_us(packet)
        exposure_start = timestamp_us(packet, dai.CameraExposureOffset.START)
        exposure_middle = timestamp_us(packet, dai.CameraExposureOffset.MIDDLE)
        exposure_end = timestamp_us(packet, dai.CameraExposureOffset.END)
        # On old firmware that does not expose END, keep the recording usable and
        # make the fallback visible in the CSV.
        alignment_timestamp = (
            exposure_end if exposure_end is not None else packet_ts
        )
        timestamp_reference = (
            "exposure_end"
            if exposure_end is not None
            else "packet_timestamp_fallback"
        )
        frame_index = self.counts[camera]
        jpeg_valid = is_complete_jpeg(raw)

        self.video_files[camera].write(raw)
        self.counts[camera] += 1
        self.bytes_written[camera] += len(raw)
        self.timestamps_writer.writerow(
            {
                "camera": camera,
                "side": CAMERAS[camera]["side"],
                "frame_index": frame_index,
                "sequence": sequence,
                "timestamp_reference": timestamp_reference,
                "device_timestamp_us": alignment_timestamp,
                "packet_device_timestamp_us": packet_ts,
                "exposure_start_device_timestamp_us": exposure_start,
                "exposure_middle_device_timestamp_us": exposure_middle,
                "exposure_end_device_timestamp_us": exposure_end,
                "host_receive_monotonic_ns": host_ns,
                "payload_bytes": len(raw),
                "jpeg_valid": int(jpeg_valid),
            }
        )

        if camera not in self.first_packets:
            trigger_timestamp = (
                exposure_start if exposure_start is not None else packet_ts
            )
            trigger_source = (
                "first_frame_exposure_start"
                if exposure_start is not None
                else "first_frame_packet_timestamp_fallback"
            )
            first = {
                "event": "first_frame_after_external_trigger",
                "camera": camera,
                "sequence": sequence,
                "oak_device_timestamp_us": trigger_timestamp,
                "timestamp_source": trigger_source,
                "exposure_start_device_timestamp_us": exposure_start,
                "exposure_end_device_timestamp_us": exposure_end,
                "host_receive_monotonic_ns": host_ns,
            }
            self.first_packets[camera] = first
            self.trigger_writer.writerow(first)
            self.trigger_file.flush()
            if not jpeg_valid:
                print(
                    f"[WARNING] {camera} first encoded packet is not a complete "
                    "JPEG image; inspect the raw MJPEG stream."
                )

        if (
            not self.summary_trigger_written
            and len(self.first_packets) == len(CAMERAS)
        ):
            usable = [
                row
                for row in self.first_packets.values()
                if row["oak_device_timestamp_us"] is not None
            ]
            if not usable:
                raise RuntimeError(
                    "The first CAM_A/D packets contained no OAK device timestamp"
                )
            earliest = min(
                usable,
                key=lambda row: row["oak_device_timestamp_us"],
            )
            self.trigger_writer.writerow(
                {
                    "event": "first_external_trigger",
                    "camera": earliest["camera"],
                    "sequence": earliest["sequence"],
                    "oak_device_timestamp_us": earliest[
                        "oak_device_timestamp_us"
                    ],
                    "timestamp_source": (
                        "minimum_first_frame_exposure_start_across_CAM_A_CAM_D"
                    ),
                    "exposure_start_device_timestamp_us": earliest[
                        "exposure_start_device_timestamp_us"
                    ],
                    "exposure_end_device_timestamp_us": earliest[
                        "exposure_end_device_timestamp_us"
                    ],
                    "host_receive_monotonic_ns": earliest[
                        "host_receive_monotonic_ns"
                    ],
                }
            )
            self.summary_trigger_written = True
            self.status = "recording"
            self.first_trigger_at = datetime.now().astimezone().isoformat()
            self.trigger_file.flush()
            print(
                "[FIRST_TRIGGER] "
                f"OAK={earliest['oak_device_timestamp_us']} us "
                f"source={earliest['camera']} exposure_start"
            )
            self.write_metadata()

        if self.counts[camera] % int(FPS) == 0:
            self.flush()

    def flush(self) -> None:
        for handle in self.video_files.values():
            handle.flush()
        self.timestamps_file.flush()
        self.trigger_file.flush()

    def write_metadata(self, **extra) -> None:
        metadata = {
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "armed_at": self.armed_at,
            "first_trigger_at": self.first_trigger_at,
            "device_mxid": self.device_id,
            "resolution": [WIDTH, HEIGHT],
            "fps": FPS,
            "manual_exposure_us": EXPOSURE_US,
            "manual_iso": self.iso,
            "codec": "MJPEG",
            "mjpeg_quality": self.quality,
            "external_trigger": {
                "CAM_A": "FrameSyncMode.INPUT",
                "CAM_D": "FrameSyncMode.INPUT",
                "first_trigger_timestamp_definition": (
                    "minimum CAM_A/D first-frame exposure-start OAK timestamp"
                ),
            },
            "frame_alignment_timestamp": (
                "timestamps.csv device_timestamp_us = exposure END OAK timestamp"
            ),
            "videos": {
                camera: self.video_paths[camera].name for camera in CAMERAS
            },
            "frame_counts": self.counts,
            "bytes_written": self.bytes_written,
            "timestamps_csv": self.timestamps_path.name,
            "trigger_events_csv": self.trigger_path.name,
            **extra,
        }
        (self.directory / "recording.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def close(self, status: str, error: str | None = None) -> list[Path]:
        self.status = status
        self.flush()
        for handle in self.video_files.values():
            handle.close()
        self.timestamps_file.close()
        self.trigger_file.close()
        avi_paths = self.remux() if self.remux_avi else []
        self.write_metadata(
            completed_at=datetime.now().astimezone().isoformat(),
            error=error,
            avi_files=[path.name for path in avi_paths],
        )
        return avi_paths

    def remux(self) -> list[Path]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            print("[INFO] ffmpeg not found; keeping raw .mjpeg files only.")
            return []
        outputs = []
        for camera, source in self.video_paths.items():
            if self.counts[camera] == 0:
                continue
            destination = source.with_suffix(".avi")
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "mjpeg",
                    "-framerate",
                    str(FPS),
                    "-i",
                    str(source),
                    "-c:v",
                    "copy",
                    str(destination),
                ],
                check=False,
            )
            if result.returncode == 0 and destination.exists():
                outputs.append(destination)
            else:
                print(f"[WARNING] ffmpeg could not remux {source.name}.")
        return outputs


def compose_preview(
    frames: dict[str, np.ndarray],
    state: str,
    counts: dict[str, int] | None = None,
) -> np.ndarray:
    blank = np.zeros((PREVIEW_HEIGHT, PREVIEW_WIDTH, 3), dtype=np.uint8)
    panels = []
    for camera, config in CAMERAS.items():
        frame = frames.get(camera, blank).copy()
        cv2.putText(
            frame,
            f"{config['side'].upper()}  {camera}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panels.append(frame)
    canvas = np.hstack(panels)
    if state == "preview":
        message = "PREVIEW + STORAGE READY   SPACE: ARM   Q: CANCEL"
        color = (0, 220, 255)
    elif state == "armed":
        message = "ARMED - NOW ENABLE EXTERNAL TRIGGER   Q: STOP"
        color = (0, 165, 255)
    else:
        count_text = " ".join(
            f"{camera}={counts[camera]}" for camera in CAMERAS
        )
        message = f"RECORDING {count_text}   Q: STOP"
        color = (0, 0, 255)
    cv2.rectangle(
        canvas,
        (0, PREVIEW_HEIGHT - 50),
        (canvas.shape[1], PREVIEW_HEIGHT),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        canvas,
        message,
        (24, PREVIEW_HEIGHT - 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def preview_until_armed(
    args: argparse.Namespace,
    latest_frames: dict[str, np.ndarray],
) -> bool:
    pipeline = make_preview_pipeline(args.iso)
    with open_device(pipeline, args.mxid) as device:
        queues = {
            camera: device.getOutputQueue(
                f"{camera}_preview", maxSize=1, blocking=False
            )
            for camera in CAMERAS
        }
        print("[PREVIEW] Storage files are open. Press SPACE to arm.")
        while True:
            for camera, queue in queues.items():
                packet = queue.tryGet()
                if packet is not None:
                    latest_frames[camera] = packet.getCvFrame()
            cv2.imshow(
                WINDOW_NAME,
                compose_preview(latest_frames, "preview"),
            )
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                return True
            if key in (ord("q"), ord("Q"), 27):
                return False
            time.sleep(0.001)


def record_external_triggered(
    args: argparse.Namespace,
    session: RecordingSession,
    latest_frames: dict[str, np.ndarray],
) -> None:
    pipeline = make_record_pipeline(args.iso, args.quality)
    with open_device(pipeline, args.mxid) as device:
        encoded_queues = {
            camera: device.getOutputQueue(
                f"{camera}_mjpeg", maxSize=200, blocking=False
            )
            for camera in CAMERAS
        }
        preview_queues = {
            camera: device.getOutputQueue(
                f"{camera}_preview", maxSize=1, blocking=False
            )
            for camera in CAMERAS
        }
        session.arm(device_mxid(device))
        print(
            "[ARMED] Both MJPEG files, CSV files, device queues and external "
            "trigger modes are ready."
        )
        print("[ARMED] You may enable the external 30 Hz trigger now.")

        first_trigger_host_time: float | None = None
        last_status = time.perf_counter()
        while True:
            received = False
            for camera, queue in encoded_queues.items():
                while True:
                    packet = queue.tryGet()
                    if packet is None:
                        break
                    received = True
                    session.write_packet(camera, packet)
            if received and first_trigger_host_time is None:
                first_trigger_host_time = time.perf_counter()

            for camera, queue in preview_queues.items():
                packet = queue.tryGet()
                if packet is not None:
                    latest_frames[camera] = packet.getCvFrame()

            state = "recording" if session.summary_trigger_written else "armed"
            cv2.imshow(
                WINDOW_NAME,
                compose_preview(latest_frames, state, session.counts),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return
            if (
                args.duration > 0
                and first_trigger_host_time is not None
                and time.perf_counter() - first_trigger_host_time
                >= args.duration
            ):
                return
            now = time.perf_counter()
            if now - last_status >= 1.0:
                print(
                    f"[STATUS] CAM_A={session.counts['CAM_A']} "
                    f"CAM_D={session.counts['CAM_D']} frames"
                )
                last_status = now
            time.sleep(0.001)


def compose_dual_preview(
    latest_frames: dict[str, dict[str, np.ndarray]],
    state: str,
    sessions: dict[str, RecordingSession] | None = None,
) -> np.ndarray:
    rows = []
    for module_label in ("external_01", "external_02"):
        counts = sessions[module_label].counts if sessions else None
        row = compose_preview(latest_frames[module_label], state, counts)
        cv2.putText(
            row,
            module_label,
            (row.shape[1] - 190, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        rows.append(row)
    return np.vstack(rows)


def preview_until_armed_dual(
    args: argparse.Namespace,
    bindings: list[tuple[str, str]],
    latest_frames: dict[str, dict[str, np.ndarray]],
) -> bool:
    with ExitStack() as stack:
        queues = {}
        for module_label, mxid in bindings:
            device = stack.enter_context(
                open_dual_device(make_preview_pipeline(args.iso), mxid)
            )
            queues[module_label] = {
                camera: device.getOutputQueue(
                    f"{camera}_preview", maxSize=1, blocking=False
                )
                for camera in CAMERAS
            }
        print("[PREVIEW] Both boards and storage files are ready. Press SPACE to arm.")
        while True:
            for module_label, module_queues in queues.items():
                for camera, queue in module_queues.items():
                    packet = queue.tryGet()
                    if packet is not None:
                        latest_frames[module_label][camera] = packet.getCvFrame()
            cv2.imshow(
                WINDOW_NAME,
                compose_dual_preview(latest_frames, "preview"),
            )
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                return True
            if key in (ord("q"), ord("Q"), 27):
                return False
            time.sleep(0.001)


def record_external_triggered_dual(
    args: argparse.Namespace,
    bindings: list[tuple[str, str]],
    sessions: dict[str, RecordingSession],
    latest_frames: dict[str, dict[str, np.ndarray]],
) -> None:
    with ExitStack() as stack:
        devices = {}
        encoded_queues = {}
        preview_queues = {}
        for module_label, mxid in bindings:
            device = stack.enter_context(
                open_dual_device(
                    make_record_pipeline(args.iso, args.quality), mxid
                )
            )
            devices[module_label] = device
            encoded_queues[module_label] = {
                camera: device.getOutputQueue(
                    f"{camera}_mjpeg", maxSize=200, blocking=False
                )
                for camera in CAMERAS
            }
            preview_queues[module_label] = {
                camera: device.getOutputQueue(
                    f"{camera}_preview", maxSize=1, blocking=False
                )
                for camera in CAMERAS
            }

        for module_label, device in devices.items():
            sessions[module_label].arm(device_mxid(device))
        print(
            "[ARMED] Both boards' MJPEG files, CSV files, device queues and "
            "external trigger modes are ready."
        )
        print("[ARMED] You may enable the external 30 Hz trigger now.")

        first_trigger_host_time: float | None = None
        last_status = time.perf_counter()
        while True:
            received = False
            for module_label, module_queues in encoded_queues.items():
                for camera, queue in module_queues.items():
                    while True:
                        packet = queue.tryGet()
                        if packet is None:
                            break
                        received = True
                        sessions[module_label].write_packet(camera, packet)
            if received and first_trigger_host_time is None:
                first_trigger_host_time = time.perf_counter()

            for module_label, module_queues in preview_queues.items():
                for camera, queue in module_queues.items():
                    packet = queue.tryGet()
                    if packet is not None:
                        latest_frames[module_label][camera] = packet.getCvFrame()

            state = (
                "recording"
                if any(s.summary_trigger_written for s in sessions.values())
                else "armed"
            )
            cv2.imshow(
                WINDOW_NAME,
                compose_dual_preview(latest_frames, state, sessions),
            )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return
            if (
                args.duration > 0
                and first_trigger_host_time is not None
                and time.perf_counter() - first_trigger_host_time
                >= args.duration
            ):
                return
            now = time.perf_counter()
            if now - last_status >= 1.0:
                status = " | ".join(
                    f"{label}: CAM_A={session.counts['CAM_A']} "
                    f"CAM_D={session.counts['CAM_D']}"
                    for label, session in sessions.items()
                )
                print(f"[STATUS] {status} frames")
                last_status = now
            time.sleep(0.001)


def validate_args(args: argparse.Namespace) -> None:
    if args.iso <= 0:
        raise SystemExit("--iso must be positive")
    if not 1 <= args.quality <= 100:
        raise SystemExit("--quality must be between 1 and 100")
    if args.duration < 0:
        raise SystemExit("--duration cannot be negative")
    if args.external_count == 1 and args.mxids:
        raise SystemExit("Single-board mode uses --mxid, not --mxids")
    if args.external_count == 2 and args.mxid:
        raise SystemExit("Two-board mode uses --mxids, not --mxid")
    if args.external_count == 2 and args.mxids and len(args.mxids) != 2:
        raise SystemExit("Two-board mode requires exactly two values after --mxids")
    if EXPOSURE_US >= int(round(1_000_000.0 / FPS)):
        raise RuntimeError("Exposure must be shorter than one frame period")


def run_single(args: argparse.Namespace) -> int:
    make_preview_pipeline(args.iso)
    make_record_pipeline(args.iso, args.quality)
    if args.check:
        print(
            f"OK: DepthAI {dai.__version__}; CAM_A/D {WIDTH}x{HEIGHT} "
            f"@ {FPS:g} fps; exposure={EXPOSURE_US} us; "
            f"MJPEG quality={args.quality}."
        )
        return 0

    output_root = args.output_dir.expanduser().resolve()
    session = RecordingSession(
        output_root,
        args.quality,
        args.iso,
        remux_avi=not args.no_avi,
    )
    latest_frames: dict[str, np.ndarray] = {}
    error: str | None = None
    final_status = "cancelled_before_arming"
    print(f"[SESSION] {session.directory}")
    print(
        f"[CONFIG] CAM_A/D {WIDTH}x{HEIGHT} @ {FPS:g} fps, "
        f"exposure={EXPOSURE_US / 1000:g} ms, ISO={args.iso}, "
        f"MJPEG quality={args.quality}"
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME, PREVIEW_WIDTH * len(CAMERAS), PREVIEW_HEIGHT
    )
    try:
        if preview_until_armed(args, latest_frames):
            record_external_triggered(args, session, latest_frames)
            final_status = (
                "complete"
                if session.summary_trigger_written
                else "stopped_before_first_trigger"
            )
    except KeyboardInterrupt:
        final_status = "interrupted"
        print("\n[STOP] Ctrl+C received; closing files safely.")
    except BaseException as exc:
        final_status = "error"
        error = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {error}")
        raise
    finally:
        session.close(final_status, error)
        cv2.destroyAllWindows()
        print(
            f"[SAVED] {session.directory} | "
            f"CAM_A={session.counts['CAM_A']} "
            f"CAM_D={session.counts['CAM_D']} frames"
        )
    return 0


def run_dual(args: argparse.Namespace) -> int:
    make_preview_pipeline(args.iso)
    make_record_pipeline(args.iso, args.quality)
    if args.check:
        print(
            f"OK: DepthAI {dai.__version__}; two external OAK boards, "
            f"CAM_A/D {WIDTH}x{HEIGHT} @ {FPS:g} fps; "
            f"exposure={EXPOSURE_US} us; MJPEG quality={args.quality}."
        )
        return 0

    bindings = resolve_dual_bindings(args)
    output_root = args.output_dir.expanduser().resolve()
    run_directory = unique_session_path(output_root)
    run_directory.mkdir(parents=True)
    sessions = {
        module_label: RecordingSession(
            output_root,
            args.quality,
            args.iso,
            remux_avi=not args.no_avi,
            directory=run_directory / module_label,
        )
        for module_label, _ in bindings
    }
    (run_directory / "external_modules.json").write_text(
        json.dumps(
            {
                "external_count": 2,
                "modules": {
                    module_label: {"device_mxid": mxid}
                    for module_label, mxid in bindings
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    latest_frames: dict[str, dict[str, np.ndarray]] = {
        module_label: {} for module_label, _ in bindings
    }
    error: str | None = None
    final_status = "cancelled_before_arming"
    print(f"[SESSION] {run_directory}")
    for module_label, mxid in bindings:
        print(
            f"[DEVICE] {module_label}={mxid} -> "
            f"{sessions[module_label].directory}"
        )
    print(
        f"[CONFIG] CAM_A/D {WIDTH}x{HEIGHT} @ {FPS:g} fps, "
        f"exposure={EXPOSURE_US / 1000:g} ms, ISO={args.iso}, "
        f"MJPEG quality={args.quality}"
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        PREVIEW_WIDTH * len(CAMERAS),
        PREVIEW_HEIGHT * 2,
    )
    try:
        if preview_until_armed_dual(args, bindings, latest_frames):
            record_external_triggered_dual(
                args, bindings, sessions, latest_frames
            )
            final_status = (
                "complete"
                if all(s.summary_trigger_written for s in sessions.values())
                else "stopped_before_first_trigger"
            )
    except KeyboardInterrupt:
        final_status = "interrupted"
        print("\n[STOP] Ctrl+C received; closing files safely.")
    except BaseException as exc:
        final_status = "error"
        error = f"{type(exc).__name__}: {exc}"
        print(f"[ERROR] {error}")
        raise
    finally:
        for session in sessions.values():
            session.close(final_status, error)
        cv2.destroyAllWindows()
        for module_label, session in sessions.items():
            print(
                f"[SAVED] {module_label}: {session.directory} | "
                f"CAM_A={session.counts['CAM_A']} "
                f"CAM_D={session.counts['CAM_D']} frames"
            )
    return 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.external_count == 1:
        return run_single(args)
    return run_dual(args)


if __name__ == "__main__":
    raise SystemExit(main())
