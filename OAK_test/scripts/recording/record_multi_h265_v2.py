#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import depthai as dai


SOCKETS = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
}
VIDEO_SIZE = (1920, 1200)
RESOLUTION = dai.ColorCameraProperties.SensorResolution.THE_1200_P
FRAME_SYNC_INPUT_CAMERAS = {"CAM_A"}
EXTERNAL_TRIGGER_CAMERAS = {"CAM_B", "CAM_C"}


class PinctrlGPIO:
    def __init__(
        self,
        start_pin: int,
        stop_pin: int,
        ready_pin: int,
        active_low: bool = True,
        enabled: bool = True,
    ) -> None:
        self.start_pin = int(start_pin)
        self.stop_pin = int(stop_pin)
        self.ready_pin = int(ready_pin)
        self.active_low = bool(active_low)
        self.enabled = bool(enabled)
        self.pinctrl = shutil.which("pinctrl")
        if self.enabled and self.pinctrl is None:
            raise RuntimeError("pinctrl was not found; cannot use GPIO buttons/output")

    def setup(self) -> None:
        if not self.enabled:
            return
        pull = "pu" if self.active_low else "pd"
        self._run("set", str(self.start_pin), "ip", pull)
        self._run("set", str(self.stop_pin), "ip", pull)
        self._run("set", str(self.ready_pin), "op", "pn", "dl")

    def cleanup(self) -> None:
        self.set_ready(False)

    def set_ready(self, high: bool) -> None:
        if not self.enabled:
            return
        self._run("set", str(self.ready_pin), "op", "pn", "dh" if high else "dl")
        print(f"[GPIO] GPIO{self.ready_pin}={'HIGH' if high else 'LOW'}", flush=True)

    def wait_for_start(self) -> None:
        if not self.enabled:
            return
        print(
            f"[GPIO] Waiting for start button on GPIO{self.start_pin} "
            f"({'LOW' if self.active_low else 'HIGH'} active).",
            flush=True,
        )
        self._wait_for_press(self.start_pin)
        print("[GPIO] Start button pressed.", flush=True)

    def stop_pressed(self) -> bool:
        if not self.enabled:
            return False
        return self._is_pressed(self.stop_pin)

    def _wait_for_press(self, pin: int) -> None:
        while True:
            if self._is_pressed(pin):
                time.sleep(0.03)
                if self._is_pressed(pin):
                    while self._is_pressed(pin):
                        time.sleep(0.03)
                    return
            time.sleep(0.02)

    def _is_pressed(self, pin: int) -> bool:
        level = self._level(pin)
        return level == 0 if self.active_low else level == 1

    def _level(self, pin: int) -> int:
        result = self._run("lev", str(pin), capture=True)
        text = result.stdout.strip()
        if text not in {"0", "1"}:
            raise RuntimeError(f"unexpected pinctrl level for GPIO{pin}: {text!r}")
        return int(text)

    def _run(self, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.pinctrl, *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
        )


@dataclass
class Module:
    index: int
    mxid: str
    device: dai.Device
    queues: dict[str, object]
    files: dict[str, object]
    counts: dict[str, int]
    sizes: dict[str, int]
    records: list[dict]


def device_id(info_or_device) -> str:
    if hasattr(info_or_device, "getMxId"):
        return info_or_device.getMxId()
    if hasattr(info_or_device, "getDeviceId"):
        return info_or_device.getDeviceId()
    return getattr(info_or_device, "mxid", "")


def parse_cameras(text: str) -> list[str]:
    out = []
    for item in text.split(","):
        token = item.strip().upper()
        if not token:
            continue
        if not token.startswith("CAM_"):
            token = f"CAM_{token}"
        if token not in SOCKETS:
            raise SystemExit(f"unsupported camera: {item}")
        out.append(token)
    if not out:
        raise SystemExit("no cameras selected")
    return out


def apply_camera_sync_mode(cam_name: str, control: dai.CameraControl, sync_mode: str) -> None:
    if sync_mode == "none":
        print(f"{cam_name}: FSYNC disabled; internal clock mode.")
        return
    if sync_mode != "fsync_input":
        raise ValueError(f"unsupported sync_mode: {sync_mode}")
    if cam_name in EXTERNAL_TRIGGER_CAMERAS:
        control.setExternalTrigger(1, 0)
        print(f"{cam_name}: continuous external trigger requested with setExternalTrigger(1, 0).")
    else:
        control.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
        print(f"{cam_name}: continuous FSYNC INPUT mode requested with setFrameSyncMode(INPUT).")


def make_pipeline(cameras: list[str], fps: float, bitrate_kbps: int, sync_mode: str) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    for cam_name in cameras:
        cam = pipeline.createColorCamera()
        cam.setBoardSocket(SOCKETS[cam_name])
        cam.setResolution(RESOLUTION)
        cam.setVideoSize(*VIDEO_SIZE)
        cam.setFps(float(fps))
        apply_camera_sync_mode(cam_name, cam.initialControl, sync_mode)

        enc = pipeline.createVideoEncoder()
        enc.setDefaultProfilePreset(float(fps), dai.VideoEncoderProperties.Profile.H265_MAIN)
        enc.setBitrateKbps(int(bitrate_kbps))
        enc.setKeyframeFrequency(max(1, int(round(fps))))

        xout = pipeline.createXLinkOut()
        xout.setStreamName(f"{cam_name}_h265")
        cam.video.link(enc.input)
        enc.bitstream.link(xout.input)
    return pipeline


def open_modules(args, cameras: list[str], session: Path) -> tuple[contextlib.ExitStack, list[Module]]:
    infos = dai.Device.getAllAvailableDevices()
    by_id = {device_id(info): info for info in infos}
    missing = [mxid for mxid in args.mxids if mxid not in by_id]
    if missing:
        raise RuntimeError(f"missing requested devices {missing}; available={sorted(by_id)}")

    stack = contextlib.ExitStack()
    modules: list[Module] = []
    try:
        for index, mxid in enumerate(args.mxids):
            print(f"[INIT] opening module {index + 1}/{len(args.mxids)} {mxid}", flush=True)
            pipeline = make_pipeline(cameras, args.fps, args.bitrate_kbps, args.sync_mode)
            device = stack.enter_context(dai.Device(pipeline, by_id[mxid]))
            print(f"[INIT] opened {mxid} usb={device.getUsbSpeed().name}", flush=True)
            queues = {
                cam: device.getOutputQueue(f"{cam}_h265", maxSize=args.queue_size, blocking=False)
                for cam in cameras
            }
            files = {
                cam: open(session / f"module{index + 1:02d}_{mxid[-8:]}_{cam}.h265", "wb", buffering=1024 * 1024)
                for cam in cameras
            }
            modules.append(
                Module(
                    index=index,
                    mxid=mxid,
                    device=device,
                    queues=queues,
                    files=files,
                    counts={cam: 0 for cam in cameras},
                    sizes={cam: 0 for cam in cameras},
                    records=[],
                )
            )
            if args.module_settle > 0:
                print(f"[INIT] settle {args.module_settle:g}s", flush=True)
                time.sleep(args.module_settle)
    except Exception:
        stack.close()
        raise
    return stack, modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mxids", nargs="+", required=True)
    parser.add_argument("--cameras", default="A,B,C")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bitrate-kbps", type=int, default=12000)
    parser.add_argument("--queue-size", type=int, default=30)
    parser.add_argument("--module-settle", type=float, default=3.0)
    parser.add_argument("--sync-mode", choices=["fsync_input", "none"], default="none")
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "Desktop" / "record")
    parser.add_argument("--start-pin", type=int, default=4)
    parser.add_argument("--stop-pin", type=int, default=17)
    parser.add_argument("--ready-pin", type=int, default=18)
    parser.add_argument("--button-active-high", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--no-gpio", action="store_true")
    args = parser.parse_args()

    cameras = parse_cameras(args.cameras)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    session = args.out_dir / time.strftime("%m%d_%H%M%S")
    session.mkdir()

    print("DepthAI", dai.__version__)
    print("Session", session)
    print("OAK-side H265; Raspberry Pi writes bitstream only.")
    gpio = PinctrlGPIO(
        args.start_pin,
        args.stop_pin,
        args.ready_pin,
        active_low=not args.button_active_high,
        enabled=not args.no_gpio,
    )
    gpio.setup()
    gpio.set_ready(False)
    if args.auto_start:
        print("[GPIO] --auto-start enabled; not waiting for GPIO4.", flush=True)
    else:
        gpio.wait_for_start()

    stack = None
    modules: list[Module] = []
    start = time.perf_counter()
    last = start
    stop_reason = "unknown"
    try:
        stack, modules = open_modules(args, cameras, session)
        gpio.set_ready(True)
        print("[READY] Pipelines initialized and GPIO18 is HIGH. Recording loop starts now.", flush=True)
        start = time.perf_counter()
        last = start
        while time.perf_counter() - start < args.duration:
            now = time.perf_counter()
            if gpio.stop_pressed():
                stop_reason = f"GPIO{args.stop_pin}_pressed"
                break
            for module in modules:
                for cam, queue in module.queues.items():
                    while True:
                        pkt = queue.tryGet()
                        if pkt is None:
                            break
                        data = bytes(pkt.getData())
                        module.files[cam].write(data)
                        module.counts[cam] += 1
                        module.sizes[cam] += len(data)
                        module.records.append(
                            {
                                "module": module.index + 1,
                                "mxid": module.mxid,
                                "camera": cam,
                                "seq": int(pkt.getSequenceNum()),
                                "device_ts_ms": pkt.getTimestampDevice().total_seconds() * 1000,
                                "host_ts_ms": now * 1000,
                                "bytes": len(data),
                            }
                        )
            if now - last >= 1.0:
                total_packets = sum(sum(m.counts.values()) for m in modules)
                total_mb = sum(sum(m.sizes.values()) for m in modules) / 1024 / 1024
                print(f"[REC] {now - start:.1f}s packets={total_packets} MB={total_mb:.2f}", flush=True)
                last = now
            time.sleep(0.001)
        else:
            stop_reason = f"duration_{args.duration:g}s"
    finally:
        elapsed = time.perf_counter() - start
        for module in modules:
            for file_obj in module.files.values():
                file_obj.close()
        if stack is not None:
            stack.close()
        gpio.cleanup()

    rows = [row for module in modules for row in module.records]
    with (session / "timestamps.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["module", "mxid", "camera", "seq", "device_ts_ms", "host_ts_ms", "bytes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "depthai": dai.__version__,
        "session": str(session),
        "mxids": args.mxids,
        "cameras": cameras,
        "sync_mode": args.sync_mode,
        "stop_reason": stop_reason,
        "elapsed_s": elapsed,
        "modules": [
            {
                "module": module.index + 1,
                "mxid": module.mxid,
                "counts": module.counts,
                "sizes": module.sizes,
                "fps": {cam: module.counts[cam] / elapsed for cam in cameras},
            }
            for module in modules
        ],
    }
    (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
