#!/usr/bin/env python3
# coding=utf-8
"""Four-camera AR0234 preview with ISP mode and an optional RAW host mode.

The default uses the OAK ISP for normal color and sends a 640x400 stream.
Pass --raw-host to send packed RAW10 and process it on this PC instead.
Neither mode configures an external trigger.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

import cv2
import depthai as dai
import numpy as np


CAMERAS = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}

SOCKET_ALIASES = {
    "RGB": "CAM_A",
    "LEFT": "CAM_B",
    "RIGHT": "CAM_C",
}

BAYER_TO_BGR = {
    "BG": cv2.COLOR_BayerBG2BGR,
    "GB": cv2.COLOR_BayerGB2BGR,
    "RG": cv2.COLOR_BayerRG2BGR,
    "GR": cv2.COLOR_BayerGR2BGR,
}


def unpack_raw10(data: np.ndarray, width: int, height: int, stride: int) -> np.ndarray:
    """Unpack MIPI RAW10 (four pixels in five bytes) into a uint16 image."""
    packed = np.asarray(data, dtype=np.uint8).reshape(-1)
    groups = (width + 3) // 4
    row_bytes = groups * 5
    if stride <= 0:
        stride = row_bytes
    required = stride * height
    if packed.size < required or stride < row_bytes:
        raise ValueError(
            f"invalid RAW10 payload: {packed.size} bytes, need {required}; "
            f"stride={stride}, row_bytes={row_bytes}"
        )

    # Strip any padding at the end of each row, then unpack all rows at once.
    src = packed[:required].reshape(height, stride)[:, :row_bytes]
    src = src.reshape(height, groups, 5)
    out = np.empty((height, groups, 4), dtype=np.uint16)
    out[:, :, 0] = (src[:, :, 0].astype(np.uint16) << 2) | (src[:, :, 4] & 0x03)
    out[:, :, 1] = (src[:, :, 1].astype(np.uint16) << 2) | ((src[:, :, 4] >> 2) & 0x03)
    out[:, :, 2] = (src[:, :, 2].astype(np.uint16) << 2) | ((src[:, :, 4] >> 4) & 0x03)
    out[:, :, 3] = (src[:, :, 3].astype(np.uint16) << 2) | ((src[:, :, 4] >> 6) & 0x03)
    return out.reshape(height, groups * 4)[:, :width]


def raw_to_bgr(
    packet: dai.ImgFrame,
    bayer: str,
    host_awb: bool,
    gamma_lut: np.ndarray,
    preview_width: int,
) -> np.ndarray:
    width = packet.getWidth()
    height = packet.getHeight()
    # DepthAI v2 unpacks packed RAW10 in native C++ inside getCvFrame(). This is
    # substantially faster than unpacking millions of pixels with Python/NumPy.
    raw10 = packet.getCvFrame()
    if raw10.ndim != 2 or raw10.shape != (height, width):
        data = packet.getData()
        if hasattr(packet, "getStride"):
            stride = packet.getStride()
        else:
            if height <= 0 or data.size % height != 0:
                raise ValueError(
                    f"cannot infer RAW stride: payload={data.size}, height={height}"
                )
            stride = data.size // height
        raw10 = unpack_raw10(data, width, height, stride)

    # Debayer on the host. Keep the result in its native 10-bit range.
    bgr10 = cv2.cvtColor(raw10, BAYER_TO_BGR[bayer])
    preview_height = max(1, round(preview_width * height / width))
    if bgr10.shape[1] != preview_width:
        bgr10 = cv2.resize(
            bgr10,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )

    if host_awb:
        # Lightweight gray-world white balance. Statistics use a sparse sample;
        # the actual channel scaling is still applied to the full-resolution PC frame.
        means = bgr10[::16, ::16].reshape(-1, 3).mean(axis=0)
        target = float(means.mean())
        gains = np.clip(target / np.maximum(means, 1.0), 0.5, 2.5)
        bgr10 = np.clip(
            bgr10.astype(np.float32) * gains.reshape(1, 1, 3),
            0,
            1023,
        ).astype(np.uint16)

    # RAW is linear and looks unnaturally dark on a normal display. Apply the
    # requested display gamma on the host through a small 10-bit lookup table.
    return gamma_lut[bgr10]


def isp_to_bgr(packet: dai.ImgFrame, preview_width: int) -> np.ndarray:
    """Convert the ISP's YUV frame to host BGR and enforce preview width."""
    frame = packet.getCvFrame()
    if frame.shape[1] == preview_width:
        return frame
    height = max(1, round(preview_width * frame.shape[0] / frame.shape[1]))
    return cv2.resize(frame, (preview_width, height), interpolation=cv2.INTER_AREA)


def fit_cell(frame: np.ndarray, width: int, height: int, label: str) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return canvas


def make_mosaic(frames: dict[str, np.ndarray], cell_width: int) -> np.ndarray:
    cell_height = max(1, round(cell_width * 5 / 8))
    cells = []
    for name in CAMERAS:
        frame = frames.get(name)
        if frame is None:
            frame = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        cells.append(fit_cell(frame, cell_width, cell_height, name))
    return np.vstack((np.hstack(cells[:2]), np.hstack(cells[2:])))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Four-camera AR0234 preview; OAK ISP mode is the default.",
    )
    parser.add_argument("--fps", type=float, default=60.0, help="free-running sensor FPS")
    parser.add_argument(
        "--bayer",
        choices=tuple(BAYER_TO_BGR),
        default="GR",
        help="sensor Bayer order; change this if colors look wrong",
    )
    parser.add_argument("--cell-width", type=int, default=640, help="mosaic cell width")
    parser.add_argument(
        "--raw-host",
        action="store_true",
        help="bypass ISP and use the previous packed-RAW host-processing mode",
    )
    parser.add_argument(
        "--isp-scale",
        type=int,
        default=3,
        help="ISP downscale denominator; 3 converts 1920x1200 to 640x400",
    )
    parser.add_argument(
        "--no-host-awb",
        action="store_true",
        help="disable host-side automatic white balance",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=2.2,
        help="host display gamma; 2.2 gives a normal color preview",
    )
    args = parser.parse_args()

    if args.fps <= 0 or args.cell_width <= 0 or args.gamma <= 0 or args.isp_scale <= 0:
        raise SystemExit("--fps, --cell-width, --gamma and --isp-scale must be positive")

    gamma_lut = np.rint(
        np.power(np.arange(1024, dtype=np.float32) / 1023.0, 1.0 / args.gamma) * 255.0
    ).astype(np.uint8)

    latest: dict[str, np.ndarray] = {}
    current_bayer = args.bayer
    received = {name: 0 for name in CAMERAS}
    errors = {name: 0 for name in CAMERAS}
    report_started = time.monotonic()

    # This project currently uses DepthAI 2.x. Open the device first so its
    # connected sockets can be inspected, then upload a legacy v2 pipeline.
    with dai.Device() as device:
        connected = {
            SOCKET_ALIASES.get(feature.socket.name, feature.socket.name)
            for feature in device.getConnectedCameraFeatures()
        }
        pipeline = dai.Pipeline()
        # Avoid splitting these very large RAW frames into small XLink chunks.
        # This is the high-throughput setting recommended by cam_test.py too.
        pipeline.setXLinkChunkSize(0)
        stream_names = []

        for name, socket in CAMERAS.items():
            if name not in connected:
                print(f"{name}: not connected")
                continue
            cam = pipeline.createColorCamera()
            cam.setBoardSocket(socket)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
            cam.setFps(float(args.fps))

            xout = pipeline.createXLinkOut()
            xout.setStreamName(name)
            if args.raw_host:
                cam.setRawOutputPacked(True)
                cam.raw.link(xout.input)
            else:
                # AR0234 is 1920x1200. 1/3 produces a compact 640x400 ISP
                # stream with the device tuning, AWB, CCM, gamma, and denoise.
                cam.setIspScale(1, args.isp_scale)
                cam.isp.link(xout.input)
            stream_names.append(name)

        if not stream_names:
            print("No CAM_A..CAM_D cameras were detected.")
            return

        try:
            mode = f"RAW host, Bayer {current_bayer}" if args.raw_host else f"OAK ISP 1/{args.isp_scale}"
            print(f"Starting free-run: {args.fps:g} FPS, {mode}")
            print("Press Q to quit." + (" Press B to cycle Bayer order." if args.raw_host else ""))
            device.startPipeline(pipeline)
            queues = {
                name: device.getOutputQueue(name=name, maxSize=1, blocking=False)
                for name in stream_names
            }
            with suppress(Exception):
                print(f"USB speed: {device.getUsbSpeed().name}")

            executor = ThreadPoolExecutor(max_workers=len(queues))
            while True:
                futures = {}
                for name, queue in queues.items():
                    packet = queue.tryGet()
                    if packet is None:
                        continue
                    if args.raw_host:
                        futures[name] = executor.submit(
                            raw_to_bgr,
                            packet,
                            current_bayer,
                            not args.no_host_awb,
                            gamma_lut,
                            args.cell_width,
                        )
                    else:
                        futures[name] = executor.submit(
                            isp_to_bgr,
                            packet,
                            args.cell_width,
                        )

                for name, future in futures.items():
                    try:
                        latest[name] = future.result()
                        received[name] += 1
                    except (ValueError, cv2.error) as exc:
                        errors[name] += 1
                        if errors[name] <= 3:
                            print(f"{name}: RAW conversion failed: {exc}")

                if latest:
                    cv2.imshow("AR0234 RAW - host processed", make_mosaic(latest, args.cell_width))

                now = time.monotonic()
                elapsed = now - report_started
                if elapsed >= 2.0:
                    rates = "  ".join(
                        f"{name}={received[name] / elapsed:.1f} fps"
                        for name in queues
                    )
                    print(f"Host receive: {rates}")
                    for name in received:
                        received[name] = 0
                    report_started = now

                key = cv2.waitKey(1) & 0xFF
                if args.raw_host and key in (ord("b"), ord("B")):
                    modes = tuple(BAYER_TO_BGR)
                    current_bayer = modes[(modes.index(current_bayer) + 1) % len(modes)]
                    print(f"Bayer order: {current_bayer}")
                elif key in (ord("q"), ord("Q")):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            if "executor" in locals():
                executor.shutdown(wait=True, cancel_futures=True)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
