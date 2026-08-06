#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import shutil
import subprocess
import threading
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
DEFAULT_MANUAL_EXPOSURE_US = 12_000
DEFAULT_MANUAL_ISO = 150


class PinctrlGPIO:
    def __init__(
        self,
        start_pin: int,
        stop_pin: int,
        trigger_pin: int,
        status_pin: int,
        active_low: bool = True,
        enabled: bool = True,
    ) -> None:
        self.start_pin = int(start_pin)
        self.stop_pin = int(stop_pin)
        self.trigger_pin = int(trigger_pin)
        self.status_pin = int(status_pin)
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
        self._run("set", str(self.trigger_pin), "op", "pn", "dl")
        self._run("set", str(self.status_pin), "op", "pn", "dl")

    def cleanup(self) -> None:
        self.set_trigger(False)
        self.set_status(False)

    def set_trigger(self, high: bool) -> None:
        if not self.enabled:
            return
        self._run("set", str(self.trigger_pin), "op", "pn", "dh" if high else "dl")
        print(f"[GPIO] trigger GPIO{self.trigger_pin}={'HIGH' if high else 'LOW'}", flush=True)

    def set_status(self, high: bool) -> None:
        if not self.enabled:
            return
        self._run("set", str(self.status_pin), "op", "pn", "dh" if high else "dl")
        print(f"[GPIO] status GPIO{self.status_pin}={'HIGH' if high else 'LOW'}", flush=True)

    def wait_for_start(self, message: str) -> None:
        if not self.enabled:
            return
        print(
            f"[GPIO] {message} Waiting for GPIO{self.start_pin} "
            f"({'LOW' if self.active_low else 'HIGH'} active).",
            flush=True,
        )
        self._wait_for_press(self.start_pin)
        print("[GPIO] Start button pressed.", flush=True)

    def stop_pressed(self) -> bool:
        if not self.enabled:
            return False
        return self._is_pressed(self.stop_pin)

    def stop_pressed_confirmed(self, hold_s: float) -> bool:
        if not self.stop_pressed():
            return False
        deadline = time.monotonic() + max(0.0, hold_s)
        while time.monotonic() < deadline:
            time.sleep(0.01)
            if not self.stop_pressed():
                return False
        return True

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


class StatusBlinker:
    def __init__(self, gpio: PinctrlGPIO, interval_s: float = 1.0) -> None:
        self.gpio = gpio
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start_initializing(self) -> None:
        if not self.gpio.enabled:
            return
        self.stop()
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_initializing, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.stop_event.set()
        self.thread.join(timeout=2.0)
        self.thread = None

    def _sleep_or_stop(self, seconds: float) -> bool:
        return self.stop_event.wait(seconds)

    def _run_initializing(self) -> None:
        # One status cycle per second: two quick flashes, then idle.
        while not self.stop_event.is_set():
            cycle_start = time.monotonic()
            for _ in range(2):
                self.gpio.set_status(True)
                if self._sleep_or_stop(0.12):
                    return
                self.gpio.set_status(False)
                if self._sleep_or_stop(0.12):
                    return
            remaining = self.interval_s - (time.monotonic() - cycle_start)
            if remaining > 0 and self._sleep_or_stop(remaining):
                return


@dataclass
class Module:
    index: int
    mxid: str
    device: dai.Device
    queues: dict[str, object]
    files: dict[str, object]
    imu_available: bool
    imu_status: str
    imu_queue: object | None
    imu_file: object | None
    imu_writer: csv.DictWriter | None
    counts: dict[str, int]
    sizes: dict[str, int]
    imu_count: int
    imu_error_count: int
    imu_last_error: str | None
    records: list[dict]
    first_frame_events: list[dict]


def module_camera_key(module: Module, cam: str) -> str:
    return f"module{module.index + 1:02d}_{module.mxid[-8:]}_{cam}"


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
    control.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
    print(f"{cam_name}: FSYNC INPUT mode requested with setFrameSyncMode(INPUT).")


def apply_camera_control_options(
    control: dai.CameraControl,
    exposure_us: int | None,
    iso: int,
    white_balance_k: int | None,
) -> None:
    if exposure_us is not None:
        control.setManualExposure(int(exposure_us), int(iso))
    if white_balance_k is not None:
        control.setManualWhiteBalance(int(white_balance_k))


def create_imu_pipeline(pipeline: dai.Pipeline, imu_rate: int) -> None:
    imu = pipeline.create(dai.node.IMU)
    imu.enableFirmwareUpdate(False)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, int(imu_rate))
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, int(imu_rate))
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(20)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("imu")
    imu.out.link(xout.input)


def make_imu_only_pipeline(imu_rate: int) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    create_imu_pipeline(pipeline, imu_rate)
    return pipeline


def make_pipeline(
    cameras: list[str],
    fps: float,
    bitrate_kbps: int,
    sync_mode: str,
    manual_exposure_us: int | None,
    manual_iso: int,
    manual_white_balance_k: int | None,
    enable_imu: bool,
    imu_rate: int,
) -> dai.Pipeline:
    pipeline = dai.Pipeline()
    for cam_name in cameras:
        cam = pipeline.createColorCamera()
        cam.setBoardSocket(SOCKETS[cam_name])
        cam.setResolution(RESOLUTION)
        cam.setVideoSize(*VIDEO_SIZE)
        cam.setFps(float(fps))
        apply_camera_sync_mode(cam_name, cam.initialControl, sync_mode)
        apply_camera_control_options(
            cam.initialControl,
            manual_exposure_us,
            manual_iso,
            manual_white_balance_k,
        )

        enc = pipeline.createVideoEncoder()
        enc.setDefaultProfilePreset(float(fps), dai.VideoEncoderProperties.Profile.H265_MAIN)
        enc.setBitrateKbps(int(bitrate_kbps))
        enc.setKeyframeFrequency(max(1, int(round(fps))))

        xout = pipeline.createXLinkOut()
        xout.setStreamName(f"{cam_name}_h265")
        cam.video.link(enc.input)
        enc.bitstream.link(xout.input)
    if enable_imu:
        create_imu_pipeline(pipeline, imu_rate)
    return pipeline


def write_packet_data(file_obj, packet) -> int:
    data = packet.getData()
    if hasattr(data, "tofile"):
        before = file_obj.tell()
        data.tofile(file_obj)
        return file_obj.tell() - before
    raw = bytes(data)
    file_obj.write(raw)
    return len(raw)


def drain_queues(modules: list[Module]) -> dict[str, int]:
    drained: dict[str, int] = {}
    for module in modules:
        for cam, queue in module.queues.items():
            key = module_camera_key(module, cam)
            count = 0
            while queue.tryGet() is not None:
                count += 1
            drained[key] = count
    return drained


def drain_imu_queues(modules: list[Module]) -> dict[str, int]:
    drained: dict[str, int] = {}
    for module in modules:
        if module.imu_queue is None:
            continue
        key = f"module{module.index + 1:02d}_{module.mxid[-8:]}_imu"
        count = 0
        while module.imu_queue.tryGet() is not None:
            count += 1
        drained[key] = count
    return drained


def seconds_ms(delta) -> float:
    return delta.total_seconds() * 1000.0


def timestamp_ms(packet, offset=None) -> float | None:
    try:
        if offset is None:
            return seconds_ms(packet.getTimestampDevice())
        return seconds_ms(packet.getTimestampDevice(offset))
    except Exception:
        return None


def write_video_record(module: Module, cam: str, pkt, data_size: int, host_now: float) -> None:
    seq = int(pkt.getSequenceNum())
    device_ts_ms = timestamp_ms(pkt)
    exposure_start_ts_ms = timestamp_ms(pkt, dai.CameraExposureOffset.START)
    exposure_middle_ts_ms = timestamp_ms(pkt, dai.CameraExposureOffset.MIDDLE)
    exposure_end_ts_ms = timestamp_ms(pkt, dai.CameraExposureOffset.END)
    host_ts_ms = host_now * 1000
    module.counts[cam] += 1
    module.sizes[cam] += data_size
    if module.counts[cam] == 1:
        module.first_frame_events.append(
            {
                "event": "first_frame_after_trigger",
                "module": module.index + 1,
                "mxid": module.mxid,
                "camera": cam,
                "seq": seq,
                "device_ts_ms": device_ts_ms,
                "exposure_start_ts_ms": exposure_start_ts_ms,
                "exposure_middle_ts_ms": exposure_middle_ts_ms,
                "exposure_end_ts_ms": exposure_end_ts_ms,
                "host_ts_ms": host_ts_ms,
            }
        )
    module.records.append(
        {
            "module": module.index + 1,
            "mxid": module.mxid,
            "camera": cam,
            "seq": seq,
            "device_ts_ms": device_ts_ms,
            "exposure_start_ts_ms": exposure_start_ts_ms,
            "exposure_middle_ts_ms": exposure_middle_ts_ms,
            "exposure_end_ts_ms": exposure_end_ts_ms,
            "host_ts_ms": host_ts_ms,
            "bytes": data_size,
        }
    )


def write_imu_records(module: Module, host_now: float, raise_errors: bool = True) -> int:
    if module.imu_queue is None or module.imu_writer is None:
        return 0
    written = 0
    while True:
        try:
            imu_data = module.imu_queue.tryGet()
        except Exception as exc:
            module.imu_error_count += 1
            module.imu_last_error = f"{type(exc).__name__}: {exc}"
            if raise_errors:
                raise
            module.imu_status = f"disabled after runtime error: {module.imu_last_error}"
            module.imu_available = False
            module.imu_queue = None
            print(
                f"[IMU_ERROR] module{module.index + 1:02d} {module.mxid}: "
                f"{module.imu_last_error}; disabling IMU logging for this module and continuing video.",
                flush=True,
            )
            return written
        if imu_data is None:
            break
        for imu_packet in imu_data.packets:
            accel = imu_packet.acceleroMeter
            gyro = imu_packet.gyroscope
            module.imu_writer.writerow(
                {
                    "module": module.index + 1,
                    "mxid": module.mxid,
                    "host_ts_ms": host_now * 1000,
                    "accel_seq": int(accel.getSequenceNum()),
                    "accel_device_ts_ms": seconds_ms(accel.getTimestampDevice()),
                    "ax_m_s2": accel.x,
                    "ay_m_s2": accel.y,
                    "az_m_s2": accel.z,
                    "gyro_seq": int(gyro.getSequenceNum()),
                    "gyro_device_ts_ms": seconds_ms(gyro.getTimestampDevice()),
                    "gx_rad_s": gyro.x,
                    "gy_rad_s": gyro.y,
                    "gz_rad_s": gyro.z,
                }
            )
            module.imu_count += 1
            written += 1
    return written


def wait_for_first_packets(
    modules: list[Module],
    timeout_s: float,
    required: str,
) -> dict[str, int]:
    required = required.lower()
    if timeout_s <= 0 or required == "none":
        return {}

    expected = {
        (module.index, cam): module_camera_key(module, cam)
        for module in modules
        for cam in module.queues
    }
    seen: dict[tuple[int, str], int] = {}
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        now = time.perf_counter()
        for module in modules:
            for cam, queue in module.queues.items():
                pkt = queue.tryGet()
                if pkt is None:
                    continue
                data_size = write_packet_data(module.files[cam], pkt)
                write_video_record(module, cam, pkt, data_size, now)
                seen[(module.index, cam)] = seen.get((module.index, cam), 0) + 1
        if required == "any" and seen:
            break
        if required == "all" and len(seen) == len(expected):
            break
        time.sleep(0.001)

    return {name: seen.get(key, 0) for key, name in expected.items()}


def detect_required_devices(args) -> dict[str, object]:
    infos = dai.Device.getAllAvailableDevices()
    by_id = {device_id(info): info for info in infos}
    missing = [mxid for mxid in args.mxids if mxid not in by_id]
    if missing:
        raise RuntimeError(f"missing requested devices {missing}; available={sorted(by_id)}")
    print(f"[PREFLIGHT] detected required devices: {', '.join(args.mxids)}", flush=True)
    return by_id


def probe_imu_devices(args, by_id: dict[str, object]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if args.no_imu:
        for mxid in args.mxids:
            result[mxid] = {"available": False, "status": "disabled by --no-imu"}
        return result

    for mxid in args.mxids:
        print(f"[PREFLIGHT] probing IMU on {mxid} for {args.imu_preflight_timeout:g}s", flush=True)
        count = 0
        imu_name = None
        status = "no IMU packets received"
        try:
            pipeline = make_imu_only_pipeline(args.imu_rate)
            with dai.Device(pipeline, by_id[mxid]) as device:
                try:
                    imu_name = str(device.getConnectedIMU())
                except Exception as exc:
                    imu_name = f"unavailable: {exc}"
                queue = device.getOutputQueue("imu", maxSize=args.imu_queue_size, blocking=False)
                deadline = time.monotonic() + args.imu_preflight_timeout
                while time.monotonic() < deadline:
                    imu_data = queue.tryGet()
                    if imu_data is None:
                        time.sleep(0.005)
                        continue
                    count += len(imu_data.packets)
                    if count > 0:
                        break
            if count > 0:
                status = f"ok, first {count} packet(s), connected IMU={imu_name}"
        except Exception as exc:
            status = f"{type(exc).__name__}: {exc}"
        available = count > 0
        if not available and args.require_imu:
            raise RuntimeError(f"IMU preflight failed on {mxid}: {status}")
        print(
            f"[PREFLIGHT] IMU {mxid}: {'available' if available else 'unavailable'} ({status})",
            flush=True,
        )
        result[mxid] = {"available": available, "status": status, "connected_imu": imu_name}
    return result


def create_session(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = out_dir / time.strftime("%m%d_%H%M%S")
    suffix = 1
    while session.exists():
        session = out_dir / f"{time.strftime('%m%d_%H%M%S')}_{suffix:02d}"
        suffix += 1
    session.mkdir()
    return session


def open_modules(
    args,
    cameras: list[str],
    session: Path,
    by_id: dict[str, object],
    imu_preflight: dict[str, dict],
) -> tuple[contextlib.ExitStack, list[Module]]:
    stack = contextlib.ExitStack()
    modules: list[Module] = []
    try:
        for index, mxid in enumerate(args.mxids):
            print(f"[INIT] opening module {index + 1}/{len(args.mxids)} {mxid}", flush=True)
            pipeline = make_pipeline(
                cameras,
                args.fps,
                args.bitrate_kbps,
                args.sync_mode,
                args.manual_exposure_us,
                args.manual_iso,
                args.manual_wb_k,
                bool(imu_preflight.get(mxid, {}).get("available", False)),
                args.imu_rate,
            )
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
            imu_queue = None
            imu_file = None
            imu_writer = None
            imu_available = bool(imu_preflight.get(mxid, {}).get("available", False))
            imu_status = str(imu_preflight.get(mxid, {}).get("status", "not probed"))
            if imu_available:
                imu_queue = device.getOutputQueue("imu", maxSize=args.imu_queue_size, blocking=False)
                imu_file = open(session / f"module{index + 1:02d}_{mxid[-8:]}_imu.csv", "w", newline="", encoding="utf-8")
                imu_writer = csv.DictWriter(
                    imu_file,
                    fieldnames=[
                        "module",
                        "mxid",
                        "host_ts_ms",
                        "accel_seq",
                        "accel_device_ts_ms",
                        "ax_m_s2",
                        "ay_m_s2",
                        "az_m_s2",
                        "gyro_seq",
                        "gyro_device_ts_ms",
                        "gx_rad_s",
                        "gy_rad_s",
                        "gz_rad_s",
                    ],
                )
                imu_writer.writeheader()
            modules.append(
                Module(
                    index=index,
                    mxid=mxid,
                    device=device,
                    queues=queues,
                    files=files,
                    imu_available=imu_available,
                    imu_status=imu_status,
                    imu_queue=imu_queue,
                    imu_file=imu_file,
                    imu_writer=imu_writer,
                    counts={cam: 0 for cam in cameras},
                    sizes={cam: 0 for cam in cameras},
                    imu_count=0,
                    imu_error_count=0,
                    imu_last_error=None,
                    records=[],
                    first_frame_events=[],
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
    parser.add_argument("--duration", type=float, default=0.0, help="recording duration in seconds; <=0 records until stop button")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bitrate-kbps", type=int, default=12000)
    parser.add_argument("--queue-size", type=int, default=30)
    parser.add_argument("--module-settle", type=float, default=3.0)
    parser.add_argument("--pre-trigger-settle", type=float, default=2.0)
    parser.add_argument("--trigger-warmup", type=float, default=0.0)
    parser.add_argument("--sync-mode", choices=["fsync_input", "none"], default="fsync_input")
    parser.add_argument("--manual-exposure-us", type=int, default=DEFAULT_MANUAL_EXPOSURE_US)
    parser.add_argument("--manual-iso", type=int, default=DEFAULT_MANUAL_ISO)
    parser.add_argument("--manual-wb-k", type=int, default=None)
    parser.add_argument("--imu-rate", type=int, default=60)
    parser.add_argument("--imu-queue-size", type=int, default=80)
    parser.add_argument("--imu-preflight-timeout", type=float, default=1.5)
    parser.add_argument("--require-imu", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument(
        "--allow-imu-dropouts",
        action="store_true",
        help="continue video recording if an IMU stream fails; default is strict/fail-fast",
    )
    parser.add_argument("--out-dir", type=Path, default=Path.home() / "Desktop" / "record")
    parser.add_argument("--start-pin", type=int, default=4)
    parser.add_argument("--stop-pin", type=int, default=17)
    parser.add_argument("--stop-hold-s", type=float, default=0.25, help="GPIO stop must stay active this long")
    parser.add_argument("--trigger-pin", type=int, default=18)
    parser.add_argument("--status-pin", type=int, default=23)
    parser.add_argument("--button-active-high", action="store_true")
    parser.add_argument("--auto-start", action="store_true")
    parser.add_argument("--skip-init-button", action="store_true")
    parser.add_argument("--no-gpio", action="store_true")
    args = parser.parse_args()

    if args.fps != 30:
        print(f"[warning] requested fps={args.fps:g}; hardware-sync recording is tuned for 30 fps.", flush=True)
    if args.manual_exposure_us is not None and args.manual_exposure_us <= 0:
        raise SystemExit("--manual-exposure-us must be positive")
    if args.manual_iso <= 0:
        raise SystemExit("--manual-iso must be positive")
    if args.imu_rate <= 0:
        raise SystemExit("--imu-rate must be positive")
    if args.imu_preflight_timeout <= 0:
        raise SystemExit("--imu-preflight-timeout must be positive")
    frame_period_us = 1_000_000.0 / args.fps if args.fps > 0 else 0
    if args.manual_exposure_us is not None and args.manual_exposure_us >= frame_period_us:
        print(
            f"[warning] manual exposure {args.manual_exposure_us}us is >= "
            f"frame period {frame_period_us:.0f}us at {args.fps:g}fps.",
            flush=True,
        )

    cameras = parse_cameras(args.cameras)

    print("DepthAI", dai.__version__)
    print("OAK-side H265; Raspberry Pi writes bitstream only.")
    print(
        "IMU logging: "
        f"{'disabled' if args.no_imu else f'raw accel+gyro at {args.imu_rate} Hz per module'}"
    )
    gpio = PinctrlGPIO(
        args.start_pin,
        args.stop_pin,
        args.trigger_pin,
        args.status_pin,
        active_low=not args.button_active_high,
        enabled=not args.no_gpio,
    )
    gpio.setup()
    gpio.set_trigger(False)
    gpio.set_status(False)
    status_blinker = StatusBlinker(gpio)
    if args.auto_start:
        print("[GPIO] --auto-start enabled; not waiting for the two GPIO4 presses.", flush=True)
    elif args.skip_init_button:
        print("[GPIO] --skip-init-button enabled; initializing immediately, second GPIO4 press is still required.", flush=True)
    else:
        gpio.wait_for_start("Press once to initialize cameras.")

    stack = None
    session: Path | None = None
    modules: list[Module] = []
    start = time.perf_counter()
    last = start
    stop_reason = "unknown"
    trigger_delay_ms: float | None = None
    error_text: str | None = None
    trigger_events: list[dict] = []
    try:
        print("[GPIO] Device preflight/initialization started; GPIO23 double-blinks once per second.", flush=True)
        status_blinker.start_initializing()
        by_id = detect_required_devices(args)
        imu_preflight = probe_imu_devices(args, by_id)
        session = create_session(args.out_dir)
        print("Session", session)
        stack, modules = open_modules(args, cameras, session, by_id, imu_preflight)
        expected_streams = len(args.mxids) * len(cameras)
        ready_streams = sum(len(module.queues) for module in modules)
        ready_files = sum(len(module.files) for module in modules)
        if ready_streams != expected_streams or ready_files != expected_streams:
            raise RuntimeError(
                f"recording not fully armed: expected_streams={expected_streams}, "
                f"ready_streams={ready_streams}, ready_files={ready_files}"
            )
        if args.pre_trigger_settle > 0:
            print(
                f"[ARMING] all queues/files are open; settling {args.pre_trigger_settle:g}s "
                "before enabling external trigger.",
                flush=True,
            )
            time.sleep(args.pre_trigger_settle)
        drained = drain_queues(modules)
        drained_imu = drain_imu_queues(modules)
        drained_text = ", ".join(f"{name}={count}" for name, count in drained.items())
        print(f"[ARMING] cleared queued startup packets before trigger: {drained_text}", flush=True)
        if drained_imu:
            drained_imu_text = ", ".join(f"{name}={count}" for name, count in drained_imu.items())
            print(f"[ARMING] cleared queued startup IMU packets before trigger: {drained_imu_text}", flush=True)
        print(
            f"[RECORDING_ARMED] {len(modules)} modules, {ready_streams} queues, "
            f"{ready_files} files are ready.",
            flush=True,
        )
        status_blinker.stop()
        gpio.set_status(True)
        if args.auto_start:
            print("[GPIO] --auto-start enabled; enabling trigger without second GPIO4 press.", flush=True)
        else:
            gpio.wait_for_start(
                f"Cameras initialized; GPIO{args.status_pin} LED is on. "
                "Connect/enable external sync wiring, then press again to start trigger."
            )
        drain_queues(modules)
        drain_imu_queues(modules)
        gpio.set_trigger(True)
        trigger_high_ts = time.perf_counter()
        trigger_delay_ms = 0.0
        trigger_events.append(
            {
                "event": "trigger_enable_high",
                "module": "",
                "mxid": "",
                "camera": "",
                "seq": "",
                "device_ts_ms": "",
                "exposure_start_ts_ms": "",
                "exposure_middle_ts_ms": "",
                "exposure_end_ts_ms": "",
                "host_ts_ms": trigger_high_ts * 1000,
            }
        )
        print(
            f"[READY] {'trigger GPIO is HIGH' if not args.no_gpio else 'GPIO disabled'}; "
            "timed recording starts immediately.",
            flush=True,
        )
        if args.trigger_warmup > 0:
            time.sleep(args.trigger_warmup)
        start = time.perf_counter()
        last = start
        last_blink = start
        blink_on = True
        gpio.set_status(True)
        record_until_stop = args.duration <= 0
        if record_until_stop:
            print(f"[RECORDING] recording starts now; press GPIO{args.stop_pin} to stop.", flush=True)
        else:
            print(f"[RECORDING] timed recording starts now for {args.duration:g}s.", flush=True)
        while record_until_stop or time.perf_counter() - start < args.duration:
            now = time.perf_counter()
            if gpio.stop_pressed_confirmed(args.stop_hold_s):
                stop_reason = f"GPIO{args.stop_pin}_pressed"
                gpio.set_trigger(False)
                break
            for module in modules:
                write_imu_records(module, now, raise_errors=args.allow_imu_dropouts is False)
                for cam, queue in module.queues.items():
                    while True:
                        pkt = queue.tryGet()
                        if pkt is None:
                            break
                        data_size = write_packet_data(module.files[cam], pkt)
                        write_video_record(module, cam, pkt, data_size, now)
            if now - last_blink >= 1.0:
                blink_on = not blink_on
                gpio.set_status(blink_on)
                last_blink = now
            if now - last >= 1.0:
                total_packets = sum(sum(m.counts.values()) for m in modules)
                total_mb = sum(sum(m.sizes.values()) for m in modules) / 1024 / 1024
                print(f"[REC] {now - start:.1f}s packets={total_packets} MB={total_mb:.2f}", flush=True)
                last = now
            time.sleep(0.001)
        else:
            stop_reason = "stop_button_required" if record_until_stop else f"duration_{args.duration:g}s"
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        if stop_reason == "unknown":
            stop_reason = "exception"
        print(f"[ERROR] {error_text}", flush=True)
    finally:
        status_blinker.stop()
        elapsed = time.perf_counter() - start
        for module in modules:
            write_imu_records(module, time.perf_counter(), raise_errors=False)
            for file_obj in module.files.values():
                file_obj.close()
            if module.imu_file is not None:
                module.imu_file.close()
        if stack is not None:
            stack.close()
        gpio.cleanup()

    if session is None:
        print(
            json.dumps(
                {
                    "depthai": dai.__version__,
                    "mxids": args.mxids,
                    "cameras": cameras,
                    "stop_reason": stop_reason,
                    "error": error_text,
                    "session": None,
                },
                indent=2,
            )
        )
        if error_text is not None:
            raise SystemExit(1)
        return

    rows = [row for module in modules for row in module.records]
    trigger_rows = [row for module in modules for row in module.first_frame_events]
    trigger_rows = sorted(trigger_rows, key=lambda row: (row.get("host_ts_ms") or 0, row.get("module") or 0, row.get("camera") or ""))
    trigger_rows = [*trigger_events, *trigger_rows]
    with (session / "timestamps.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "module",
            "mxid",
            "camera",
            "seq",
            "device_ts_ms",
            "exposure_start_ts_ms",
            "exposure_middle_ts_ms",
            "exposure_end_ts_ms",
            "host_ts_ms",
            "bytes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (session / "trigger_events.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "event",
            "module",
            "mxid",
            "camera",
            "seq",
            "device_ts_ms",
            "exposure_start_ts_ms",
            "exposure_middle_ts_ms",
            "exposure_end_ts_ms",
            "host_ts_ms",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trigger_rows)

    summary = {
        "depthai": dai.__version__,
        "session": str(session),
        "mxids": args.mxids,
        "cameras": cameras,
        "sync_mode": args.sync_mode,
        "manual_exposure_us": args.manual_exposure_us,
        "manual_iso": args.manual_iso,
        "manual_wb_k": args.manual_wb_k,
        "imu_enabled": not args.no_imu,
        "allow_imu_dropouts": args.allow_imu_dropouts,
        "imu_rate": None if args.no_imu else args.imu_rate,
        "imu_preflight_timeout": None if args.no_imu else args.imu_preflight_timeout,
        "stop_reason": stop_reason,
        "error": error_text,
        "trigger_delay_ms": trigger_delay_ms,
        "trigger_events_csv": str(session / "trigger_events.csv"),
        "elapsed_s": elapsed,
        "modules": [
            {
                "module": module.index + 1,
                "mxid": module.mxid,
                "counts": module.counts,
                "sizes": module.sizes,
                "fps": {cam: module.counts[cam] / elapsed for cam in cameras},
                "imu_available": module.imu_available,
                "imu_status": module.imu_status,
                "imu_count": module.imu_count,
                "imu_fps": module.imu_count / elapsed if module.imu_available else None,
                "imu_error_count": module.imu_error_count,
                "imu_last_error": module.imu_last_error,
            }
            for module in modules
        ],
    }
    (session / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if error_text is not None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
