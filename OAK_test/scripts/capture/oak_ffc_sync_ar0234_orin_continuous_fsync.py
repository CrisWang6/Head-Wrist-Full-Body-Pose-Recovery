# coding=utf-8
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import depthai as dai

FPS = 30
RECORD_SECONDS = 10.0
RECORD_DIR = Path("recordings")
DEFAULT_MANUAL_EXPOSURE_US = 10_000
DEFAULT_MANUAL_ISO = 150

DEVICE_RECORD_MODES = {
    "device_h265": {
        "profile": dai.VideoEncoderProperties.Profile.H265_MAIN,
        "suffix": "h265",
        "extension": "h265",
    },
    "device_mjpeg": {
        "profile": dai.VideoEncoderProperties.Profile.MJPEG,
        "suffix": "mjpeg",
        "extension": "mjpeg",
    },
}

cam_list = {
    "CAM_A": {"color": True, "res": "1200", "video_res": "1200"},
    "CAM_B": {"color": True, "res": "1200", "video_res": "1200"},
    "CAM_C": {"color": True, "res": "1200", "video_res": "1200"},
    "CAM_D": {"color": True, "res": "1200", "video_res": "1200"},
}

mono_res_opts = {
    "400": dai.MonoCameraProperties.SensorResolution.THE_400_P,
    "480": dai.MonoCameraProperties.SensorResolution.THE_480_P,
    "720": dai.MonoCameraProperties.SensorResolution.THE_720_P,
    "800": dai.MonoCameraProperties.SensorResolution.THE_800_P,
    "1200": dai.MonoCameraProperties.SensorResolution.THE_1200_P,
}

color_res_opts = {
    "720": dai.ColorCameraProperties.SensorResolution.THE_720_P,
    "800": dai.ColorCameraProperties.SensorResolution.THE_800_P,
    "1080": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
    "1200": dai.ColorCameraProperties.SensorResolution.THE_1200_P,
    "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
    "5mp": dai.ColorCameraProperties.SensorResolution.THE_5_MP,
    "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    "48mp": dai.ColorCameraProperties.SensorResolution.THE_48_MP,
}

color_video_size_opts = {
    "720": (1280, 720),
    "800": (1280, 800),
    "1080": (1920, 1080),
    "1200": (1920, 1200),
    "4k": (3840, 2160),
}

cam_socket_to_name = {
    "RGB": "CAM_A",
    "LEFT": "CAM_B",
    "RIGHT": "CAM_C",
    "CAM_A": "CAM_A",
    "CAM_B": "CAM_B",
    "CAM_C": "CAM_C",
    "CAM_D": "CAM_D",
}

cam_socket_opts = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}

FRAME_SYNC_INPUT_CAMERAS = {"CAM_A", "CAM_B", "CAM_C", "CAM_D"}


@dataclass
class PacketInfo:
    seq: int
    ts_s: float


@dataclass
class FpsStats:
    frames: int = 0
    last_frames: int = 0
    fps: float = 0.0
    last_seq: int | None = None
    last_ts_s: float | None = None

    def add(self, info: PacketInfo) -> None:
        self.frames += 1
        self.last_seq = info.seq
        self.last_ts_s = info.ts_s

    def update_rate(self, elapsed: float) -> None:
        if elapsed <= 0:
            return
        self.fps = (self.frames - self.last_frames) / elapsed
        self.last_frames = self.frames


@dataclass
class FrameGrouper:
    camera_names: list[str]
    fps: float
    group_by: str = "timestamp"
    log_interval: float = 1.0
    groups: dict[int, dict[str, PacketInfo]] = field(default_factory=dict)
    last_log_ts: float = field(default_factory=time.monotonic)
    complete_groups: int = 0
    last_complete_groups: int = 0
    incomplete_dropped_groups: int = 0
    last_incomplete_dropped_groups: int = 0
    spread_ms_values: list[float] = field(default_factory=list)
    stats: dict[str, FpsStats] = field(init=False)

    def __post_init__(self) -> None:
        self.stats = {name: FpsStats() for name in self.camera_names}

    def add(self, cam_name: str, packet: dai.ImgFrame) -> None:
        info = PacketInfo(
            seq=int(packet.getSequenceNum()),
            ts_s=packet.getTimestampDevice().total_seconds(),
        )
        self.stats[cam_name].add(info)
        key = self._key(info)
        self.groups.setdefault(key, {})[cam_name] = info
        self._flush_complete(key)
        self._cleanup(key)

    def _key(self, info: PacketInfo) -> int:
        if self.group_by == "sequence":
            return info.seq
        return round(info.ts_s * self.fps)

    def _flush_complete(self, key: int) -> None:
        group = self.groups.get(key)
        if group is None:
            return
        if any(name not in group for name in self.camera_names):
            return

        self.complete_groups += 1
        self.spread_ms_values.append(self._spread_ms(group))
        now = time.monotonic()
        if self.log_interval > 0 and now - self.last_log_ts >= self.log_interval:
            self._print_group(key, group)
            self._print_fps(now)
            self.last_log_ts = now
        del self.groups[key]

    def _print_group(self, key: int, group: dict[str, PacketInfo]) -> None:
        ordered = [(name, group[name]) for name in self.camera_names]
        timestamps = [info.ts_s for _, info in ordered]
        min_ts = min(timestamps)
        spread_ms = self._spread_ms(group)
        seq_text = "/".join(str(info.seq) for _, info in ordered)
        offset_text = ", ".join(f"{name}={((info.ts_s - min_ts) * 1000.0):.3f}ms" for name, info in ordered)
        print(
            f"[GROUP] key={key} by={self.group_by}, seq={seq_text}, "
            f"spread={spread_ms:.3f} ms, offsets: {offset_text}"
        )

    def _spread_ms(self, group: dict[str, PacketInfo]) -> float:
        timestamps = [info.ts_s for info in group.values()]
        return (max(timestamps) - min(timestamps)) * 1000.0

    def _print_fps(self, now: float) -> None:
        elapsed = now - self.last_log_ts
        if elapsed <= 0:
            return

        for stat in self.stats.values():
            stat.update_rate(elapsed)

        complete_fps = (self.complete_groups - self.last_complete_groups) / elapsed
        incomplete_rate = (self.incomplete_dropped_groups - self.last_incomplete_dropped_groups) / elapsed
        fps_text = ", ".join(
            f"{name}={self.stats[name].fps:.2f}"
            for name in self.camera_names
        )
        last_seq_text = "/".join(
            "n/a" if self.stats[name].last_seq is None else str(self.stats[name].last_seq)
            for name in self.camera_names
        )
        print(
            f"[FPS] recv: {fps_text}, complete_groups={complete_fps:.2f}/s, "
            f"incomplete_dropped={incomplete_rate:.2f}/s, totals: "
            f"complete={self.complete_groups}, incomplete_dropped={self.incomplete_dropped_groups}, "
            f"last_seq={last_seq_text}"
        )
        self.last_complete_groups = self.complete_groups
        self.last_incomplete_dropped_groups = self.incomplete_dropped_groups

    def _cleanup(self, latest_key: int) -> None:
        stale_before = latest_key - max(4, int(self.fps))
        stale_keys = [key for key in self.groups if key < stale_before]
        for key in stale_keys:
            group = self.groups.pop(key)
            if group:
                self.incomplete_dropped_groups += 1

    def print_summary(self) -> None:
        print("[SYNC_SUMMARY]")
        print(
            f"  groups complete={self.complete_groups}, "
            f"incomplete_dropped={self.incomplete_dropped_groups}, "
            f"pending={len(self.groups)}"
        )
        for name in self.camera_names:
            stat = self.stats[name]
            print(
                f"  {name}: frames={stat.frames}, "
                f"last_seq={stat.last_seq if stat.last_seq is not None else 'n/a'}, "
                f"last_ts={stat.last_ts_s:.6f}s" if stat.last_ts_s is not None
                else f"  {name}: frames={stat.frames}, last_seq=n/a, last_ts=n/a"
            )
        if not self.spread_ms_values:
            print("  spread: no complete groups")
            return

        values = sorted(self.spread_ms_values)
        mean = sum(values) / len(values)
        print(
            "  spread_ms: "
            f"min={values[0]:.3f}, mean={mean:.3f}, "
            f"p50={self._percentile(values, 50):.3f}, "
            f"p95={self._percentile(values, 95):.3f}, "
            f"p99={self._percentile(values, 99):.3f}, "
            f"max={values[-1]:.3f}"
        )

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: float) -> float:
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return sorted_values[0]
        rank = (len(sorted_values) - 1) * percentile / 100.0
        lower = int(rank)
        upper = min(lower + 1, len(sorted_values) - 1)
        fraction = rank - lower
        return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def apply_camera_control_options(
    control: dai.CameraControl,
    exposure_us: int | None,
    iso: int,
    white_balance_k: int | None,
) -> None:
    if exposure_us is not None:
        control.setManualExposure(exposure_us, iso)
    if white_balance_k is not None:
        control.setManualWhiteBalance(white_balance_k)


def apply_camera_sync_mode(cam_name: str, control: dai.CameraControl, sync_mode: str) -> None:
    if sync_mode == "fsync_input":
        if cam_name not in FRAME_SYNC_INPUT_CAMERAS:
            print(f"{cam_name}: defaulting to setFrameSyncMode(INPUT) for external FSYNC.")
        control.setFrameSyncMode(dai.CameraControl.FrameSyncMode.INPUT)
        print(f"{cam_name}: continuous FSYNC INPUT mode requested with setFrameSyncMode(INPUT).")
        return

    if sync_mode == "none":
        print(f"{cam_name}: FSYNC disabled; camera runs from internal clock.")
        return

    raise ValueError(f"unsupported sync_mode: {sync_mode}")


def create_pipeline(
    device_record_mode: str | None = None,
    sync_mode: str = "fsync_input",
    manual_exposure_us: int | None = None,
    manual_iso: int = 100,
    manual_white_balance_k: int | None = None,
):
    pipeline = dai.Pipeline()
    device_record = DEVICE_RECORD_MODES.get(device_record_mode)
    cam = {}
    xout = {}
    enc = {}
    for cam_name, cam_props in cam_list.items():
        xout[cam_name] = pipeline.create(dai.node.XLinkOut)
        if device_record is not None:
            xout[cam_name].setStreamName(f"{cam_name}_{device_record['suffix']}")
        else:
            xout[cam_name].setStreamName(cam_name)
        if cam_props["color"]:
            cam[cam_name] = pipeline.create(dai.node.ColorCamera)
            cam[cam_name].setResolution(color_res_opts[cam_props["res"]])
            if device_record is not None:
                video_res = cam_props.get("video_res", cam_props["res"])
                video_size = color_video_size_opts.get(video_res)
                if video_size is not None:
                    cam[cam_name].setVideoSize(*video_size)
                enc[cam_name] = pipeline.create(dai.node.VideoEncoder)
                enc[cam_name].setDefaultProfilePreset(
                    float(FPS),
                    device_record["profile"],
                )
                enc[cam_name].setKeyframeFrequency(max(1, int(FPS)))
                cam[cam_name].video.link(enc[cam_name].input)
                enc[cam_name].bitstream.link(xout[cam_name].input)
            else:
                cam[cam_name].isp.link(xout[cam_name].input)
        else:
            cam[cam_name] = pipeline.createMonoCamera()
            cam[cam_name].setResolution(mono_res_opts[cam_props["res"]])
            if device_record is not None:
                enc[cam_name] = pipeline.create(dai.node.VideoEncoder)
                enc[cam_name].setDefaultProfilePreset(
                    float(FPS),
                    device_record["profile"],
                )
                enc[cam_name].setKeyframeFrequency(max(1, int(FPS)))
                cam[cam_name].out.link(enc[cam_name].input)
                enc[cam_name].bitstream.link(xout[cam_name].input)
            else:
                cam[cam_name].out.link(xout[cam_name].input)
        cam[cam_name].setBoardSocket(cam_socket_opts[cam_name])
        cam[cam_name].setFps(FPS)
        apply_camera_sync_mode(cam_name, cam[cam_name].initialControl, sync_mode)
        apply_camera_control_options(
            cam[cam_name].initialControl,
            manual_exposure_us,
            manual_iso,
            manual_white_balance_k,
        )

#     script = pipeline.create(dai.node.Script)
#     script.setProcessor(dai.ProcessorType.LEON_CSS)
#     script.setScript(
#         """# coding=utf-8
# import time
# import GPIO
#
# # Script static arguments
# fps = %f
#
# calib = Device.readCalibration2().getEepromData()
# prodName  = calib.productName
# boardName = calib.boardName
# boardRev  = calib.boardRev
#
# node.warn(f'Product name  : {prodName}')
# node.warn(f'Board name    : {boardName}')
# node.warn(f'Board revision: {boardRev}')
#
# revision = -1
# # Very basic parsing here, TODO improve
# if len(boardRev) >= 2 and boardRev[0] == 'R':
#     revision = int(boardRev[1])
# node.warn(f'Parsed revision number: {revision}')
#
# # Defaults for OAK-FFC-4P older revisions (<= R5)
# GPIO_FSIN_2LANE = 41  # COM_AUX_IO2
# GPIO_FSIN_4LANE = 40
# GPIO_FSIN_MODE_SELECT = 6  # Drive 1 to tie together FSIN_2LANE and FSIN_4LANE
#
# if revision >= 6:
#     GPIO_FSIN_2LANE = 41  # still COM_AUX_IO2, no PWM capable
#     GPIO_FSIN_4LANE = 42  # also not PWM capable
#     GPIO_FSIN_MODE_SELECT = 38  # Drive 1 to tie together FSIN_2LANE and FSIN_4LANE
# # Note: on R7 GPIO_FSIN_MODE_SELECT is pulled up, driving high isn't necessary (but fine to do)
#
# # GPIO initialization
# GPIO.setup(GPIO_FSIN_2LANE, GPIO.OUT)
# GPIO.write(GPIO_FSIN_2LANE, 0)
#
# GPIO.setup(GPIO_FSIN_4LANE, GPIO.IN)
#
# GPIO.setup(GPIO_FSIN_MODE_SELECT, GPIO.OUT)
# GPIO.write(GPIO_FSIN_MODE_SELECT, 1)
#
# period = 1 / fps
# active = 0.001
#
# node.warn(f'FPS: {fps}  Period: {period}')
#
# withInterrupts = False
# if withInterrupts:
#     node.critical(f'[TODO] FSYNC with timer interrupts (more precise) not implemented')
# else:
#     overhead = 0.003  # Empirical, TODO add thread priority option!
#     while True:
#         GPIO.write(GPIO_FSIN_2LANE, 1)
#         time.sleep(active)
#         GPIO.write(GPIO_FSIN_2LANE, 0)
#         time.sleep(period - active - overhead)
# """ % (FPS)
#     )

    return pipeline


def make_recording_dir(base_dir: Path, record_mode: str) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_name = record_mode.removeprefix("device_")
    out_dir = base_dir / f"orin_continuous_fsync_{mode_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def resolve_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG_BIN = resolve_ffmpeg()


def convert_h265_to_mp4(h265_path: Path, mp4_path: Path, fps: float) -> bool:
    input_path = h265_path
    cleanup_path = None
    vps_offset = find_first_h265_vps_offset(h265_path)
    if vps_offset is None:
        print(
            f"[warning] {h265_path.name} does not contain an obvious HEVC VPS "
            "near the start; MP4 conversion may fail."
        )
    elif vps_offset > 0:
        cleanup_path = h265_path.with_suffix(".clean.h265")
        copy_file_from_offset(h265_path, cleanup_path, vps_offset)
        input_path = cleanup_path
        print(
            f"[info] {h265_path.name} starts before VPS; converting from byte offset {vps_offset}."
        )

    cmd = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-framerate",
        f"{fps:g}",
        "-f",
        "hevc",
        "-i",
        str(input_path),
        "-c",
        "copy",
        "-tag:v",
        "hvc1",
        str(mp4_path),
    ]
    print(f"Converting {h265_path.name} -> {mp4_path.name}")
    result = subprocess.run(cmd, check=False)
    if cleanup_path is not None:
        cleanup_path.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"[warning] ffmpeg conversion failed for {h265_path}")
        return False
    return True


def convert_mjpeg_to_mkv(mjpeg_path: Path, mkv_path: Path, fps: float) -> bool:
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-fflags",
        "+genpts",
        "-framerate",
        f"{fps:g}",
        "-f",
        "mjpeg",
        "-i",
        str(mjpeg_path),
        "-c",
        "copy",
        str(mkv_path),
    ]
    print(f"Converting {mjpeg_path.name} -> {mkv_path.name}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[warning] ffmpeg conversion failed for {mjpeg_path}")
        return False
    return True


def find_first_h265_vps_offset(h265_path: Path, scan_bytes: int = 8_000_000) -> int | None:
    data = h265_path.read_bytes()[:scan_bytes]
    start = 0
    while True:
        idx = data.find(b"\x00\x00\x00\x01", start)
        if idx < 0 or idx + 5 >= len(data):
            break
        nal_type = (data[idx + 4] >> 1) & 0x3F
        if nal_type == 32:
            return idx
        start = idx + 4
    return None


def copy_file_from_offset(src_path: Path, dst_path: Path, offset: int) -> None:
    with src_path.open("rb") as src, dst_path.open("wb") as dst:
        src.seek(offset)
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def write_packet_data(file_obj, packet) -> None:
    data = packet.getData()
    if hasattr(data, "tofile"):
        data.tofile(file_obj)
    else:
        file_obj.write(bytes(data))


def drain_output_queues(output_queues: dict[str, dai.DataOutputQueue]) -> dict[str, int]:
    drained = {name: 0 for name in output_queues}
    for name, queue in output_queues.items():
        while True:
            packet = queue.tryGet()
            if packet is None:
                break
            drained[name] += 1
    return drained


def wait_for_fsync_start(output_queues: dict[str, dai.DataOutputQueue]) -> None:
    print("[FSYNC_ARMED] Pipeline and output queues are ready.")
    print("[FSYNC_ARMED] Keep FSYNC disabled until this prompt, then enable/connect FSYNC.")
    try:
        input("[FSYNC_ARMED] Press Enter after FSYNC is enabled to start recording/logging...")
    except EOFError:
        print("[FSYNC_ARMED] stdin is not interactive; continuing without waiting.")
    drained = drain_output_queues(output_queues)
    drained_text = ", ".join(f"{name}={count}" for name, count in drained.items())
    print(f"[FSYNC_ARMED] Cleared queued startup packets: {drained_text}")


def main():
    global cam_list, FPS

    parser = argparse.ArgumentParser(
        description="Continuous FSYNC-input AR0234 sync test on Orin/OAK-FFC.",
    )
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--queue-size", type=int, default=8)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=RECORD_SECONDS,
        help="recording duration in seconds when --record-mode is not none",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=RECORD_DIR,
        help="directory for encoded recordings and converted container files",
    )
    parser.add_argument(
        "--record-mode",
        choices=["none", "device_h265", "device_mjpeg"],
        default="none",
        help="none previews/logs raw frames; device_h265/device_mjpeg use DepthAI VideoEncoder",
    )
    parser.add_argument(
        "--sync-mode",
        choices=["fsync_input", "none"],
        default="fsync_input",
        help="fsync_input uses FrameSyncMode.INPUT on CAM_A/CAM_B/CAM_C/CAM_D; none runs cameras from internal clocks",
    )
    parser.add_argument(
        "--cameras",
        default="A,B,C,D",
        help="comma-separated camera sockets to use, e.g. A,D or CAM_A,CAM_D",
    )
    parser.add_argument(
        "--group-by",
        choices=["timestamp", "sequence"],
        default="timestamp",
        help="group frames before calculating sync spread",
    )
    parser.add_argument(
        "--group-log-interval",
        type=float,
        default=1.0,
        help="seconds between grouped sync log lines; <=0 disables live group/fps logs",
    )
    parser.add_argument(
        "--record-progress-interval",
        type=float,
        default=1.0,
        help="seconds between recording progress lines; <=0 disables progress logs",
    )
    parser.add_argument(
        "--raw-log",
        action="store_true",
        help="also print every packet timestamp as it arrives",
    )
    parser.add_argument(
        "--wait-for-fsync",
        action="store_true",
        help="wait for Enter after queues are ready so FSYNC can be enabled cleanly",
    )
    parser.add_argument(
        "--no-wait-for-fsync",
        action="store_true",
        help="skip the default FSYNC wait in encoded recording mode",
    )
    parser.add_argument(
        "--manual-exposure-us",
        type=int,
        default=DEFAULT_MANUAL_EXPOSURE_US,
        help="fixed exposure time in microseconds; default enables manual exposure",
    )
    parser.add_argument(
        "--manual-iso",
        type=int,
        default=DEFAULT_MANUAL_ISO,
        help="ISO value used with manual exposure, usually 100..1600",
    )
    parser.add_argument(
        "--manual-wb-k",
        type=int,
        default=None,
        help="set fixed white balance color temperature in kelvins",
    )
    args = parser.parse_args()
    FPS = args.fps
    record_encoded = args.record_mode != "none"
    device_record = DEVICE_RECORD_MODES.get(args.record_mode)
    if args.wait_for_fsync and args.no_wait_for_fsync:
        raise SystemExit("--wait-for-fsync and --no-wait-for-fsync cannot be used together")
    should_wait_for_fsync = args.wait_for_fsync or (
        args.sync_mode == "fsync_input" and record_encoded and not args.no_wait_for_fsync
    )
    if args.manual_exposure_us is not None and args.manual_exposure_us <= 0:
        raise SystemExit("--manual-exposure-us must be positive")
    if args.manual_iso <= 0:
        raise SystemExit("--manual-iso must be positive")
    frame_period_us = 1_000_000.0 / FPS if FPS > 0 else 0
    if args.manual_exposure_us is not None and args.manual_exposure_us >= frame_period_us:
        print(
            f"[warning] manual exposure {args.manual_exposure_us}us is >= "
            f"frame period {frame_period_us:.0f}us at {FPS:g}fps; "
            "use a shorter exposure to avoid frame timing issues."
        )
    if FFMPEG_BIN is None and record_encoded:
        raise SystemExit(
            "ffmpeg was not found in PATH and imageio-ffmpeg is not installed; "
            "cannot convert encoded recordings"
        )
    requested_cameras = []
    for item in args.cameras.split(","):
        token = item.strip().upper()
        if not token:
            continue
        if not token.startswith("CAM_"):
            token = f"CAM_{token}"
        if token not in cam_list:
            raise SystemExit(f"unsupported camera '{item}'. Use A,B,C,D or CAM_A,...")
        requested_cameras.append(token)
    if not requested_cameras:
        raise SystemExit("--cameras selected no cameras")

    # # 创建 DepthAI 设备配置对象
    # config = dai.Device.Config()
    #
    # # 设置 GPIO 引脚 6 为输出模式，初始状态为高电平
    # config.board.gpio[38] = dai.BoardConfig.GPIO(
    #     dai.BoardConfig.GPIO.OUTPUT, dai.BoardConfig.GPIO.Level.HIGH,
    # )
    # # 设置 OpenVINO 版本号
    # # config.version = dai.OpenVINO.VERSION_2021_4

    # 创建 DepthAI 设备对象
    with dai.Device() as device:
        print(device.getIrDrivers())
        print(device.setIrFloodLightIntensity(0))
        print(device.setIrLaserDotProjectorIntensity(1000))
        # 获取连接到设备上的相机列表，输出相机名称、分辨率、支持的颜色类型等信息
        print("Connected cameras:")
        sensor_names = {}  # type: dict[str, str]
        for p in device.getConnectedCameraFeatures():
            # 输出相机信息
            print(
                f" -socket {p.socket.name:6}: {p.sensorName:6} {p.width:4} x {p.height:4} focus:",
                end="",
            )
            print("auto " if p.hasAutofocus else "fixed", "- ", end="")
            supported_types = [color_type.name for color_type in p.supportedTypes]
            print(*supported_types)

            # 更新相机属性表
            cam_name = cam_socket_to_name[p.socket.name]
            sensor_names[cam_name] = p.sensorName

        # 仅保留设备已连接的相机
        for cam_name in cam_list:
            if cam_name not in sensor_names:
                print(f"{cam_name} is not connected !")

        cam_list = {
            name: cam_list[name]
            for name in cam_list
            if name in sensor_names and name in requested_cameras
        }
        if not cam_list:
            raise SystemExit("none of the requested cameras are connected")

        # 开始执行给定的管道
        device.startPipeline(
            create_pipeline(
                device_record_mode=args.record_mode if record_encoded else None,
                sync_mode=args.sync_mode,
                manual_exposure_us=args.manual_exposure_us,
                manual_iso=args.manual_iso,
                manual_white_balance_k=args.manual_wb_k,
            )
        )

        # 创建相机输出队列
        output_queues = {}
        out_dir = make_recording_dir(args.record_dir, args.record_mode) if record_encoded else None
        encoded_files = {}
        encoded_paths = {}
        for cam_name in cam_list:
            stream_name = f"{cam_name}_{device_record['suffix']}" if record_encoded else cam_name
            output_queues[cam_name] = device.getOutputQueue(
                name=stream_name, maxSize=args.queue_size, blocking=False,
            )
            if record_encoded:
                encoded_path = out_dir / f"{cam_name}.{device_record['extension']}"
                encoded_paths[cam_name] = encoded_path
                encoded_files[cam_name] = encoded_path.open("wb")
            elif not args.no_show:
                cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cam_name, 640, 480)
        #inQ = device.getInputQueue("in")

        def trigger():
            buffer = dai.Buffer()
            buffer.setData([1])
            inQ.send(buffer)

        if not record_encoded and not should_wait_for_fsync:
            time.sleep(1)
        # trigger()  # Inital trigger
        group_order = [name for name in ("CAM_A", "CAM_B", "CAM_C", "CAM_D") if name in output_queues]
        live_group_log_interval = 0.0 if record_encoded else args.group_log_interval
        frame_grouper = FrameGrouper(
            group_order,
            float(FPS),
            group_by=args.group_by,
            log_interval=live_group_log_interval,
        )
        if args.sync_mode == "fsync_input":
            print(
                "Continuous FSYNC INPUT mode enabled: "
                "CAM_A/CAM_B/CAM_C/CAM_D use setFrameSyncMode(INPUT). "
                f"External FSYNC frequency should match FPS: {FPS:g} Hz."
            )
        else:
            print("FSYNC disabled. Cameras are running from internal clocks.")
        print(
            f"Manual exposure enabled: exposure={args.manual_exposure_us}us, "
            f"ISO={args.manual_iso}"
        )
        if args.manual_wb_k is not None:
            print(f"Manual white balance enabled: {args.manual_wb_k}K")
        if record_encoded:
            print(
                f"Recording encoded video only, no preview. Duration: {args.duration:g}s, "
                f"mode={args.record_mode}"
            )
            print(f"Output directory: {out_dir}")
            print("Live sync logs are disabled during recording; summary prints at the end.")
        else:
            print(
                f"Grouped sync logging enabled: group_by={args.group_by}, "
                f"cameras={','.join(group_order)}, interval={args.group_log_interval:g}s"
            )
        if should_wait_for_fsync:
            wait_for_fsync_start(output_queues)

        # 循环读取并显示视频流
        start_ts = time.monotonic()
        saved_packets = {name: 0 for name in output_queues}
        last_progress_ts = start_ts
        try:
            while not device.isClosed():
                if record_encoded and time.monotonic() - start_ts >= args.duration:
                    break

                frame_list = []
                for cam_name in cam_list:
                    if record_encoded:
                        while True:
                            packet = output_queues[cam_name].tryGet()
                            if packet is None:
                                break
                            write_packet_data(encoded_files[cam_name], packet)
                            try:
                                frame_grouper.add(cam_name, packet)
                            except AttributeError:
                                pass
                            saved_packets[cam_name] += 1
                        continue

                    packet = output_queues[cam_name].tryGet()
                    if packet is not None:
                        frame_grouper.add(cam_name, packet)
                        if args.raw_log:
                            print(cam_name + ":", packet.getTimestampDevice())
                        # 获取视频帧并添加到帧列表中
                        frame_list.append((cam_name, packet.getCvFrame()))

                now = time.monotonic()
                if (
                    record_encoded
                    and args.record_progress_interval > 0
                    and now - last_progress_ts >= args.record_progress_interval
                ):
                    elapsed = now - start_ts
                    count_text = ", ".join(
                        f"{name}={saved_packets[name]}" for name in output_queues
                    )
                    print(f"[REC] {elapsed:.1f}/{args.duration:g}s packets: {count_text}")
                    last_progress_ts = now

                if frame_list and args.raw_log:
                    print("-------------------------------")
                if frame_list and not args.no_show:
                    # 显示视频帧
                    for cam_name, frame in frame_list:
                        cv2.imshow(cam_name, frame)

                # 等待用户按下 "q" 键，退出循环并关闭窗口
                key = cv2.waitKey(1) if not args.no_show else 255
                if key == ord("q"):
                    break
                # if key == ord("c"):
                #    trigger()
        finally:
            for file_obj in encoded_files.values():
                file_obj.close()
        cv2.destroyAllWindows()
        frame_grouper.print_summary()

        if record_encoded:
            print("Recording finished.")
            for cam_name, count in saved_packets.items():
                print(f"{cam_name}: encoded packets saved={count}, output={encoded_paths[cam_name]}")

            converted = 0
            for cam_name in output_queues:
                if args.record_mode == "device_h265":
                    h265_path = out_dir / f"{cam_name}.h265"
                    mp4_path = out_dir / f"{cam_name}.mp4"
                    if h265_path.exists() and h265_path.stat().st_size > 0:
                        if convert_h265_to_mp4(h265_path, mp4_path, float(FPS)):
                            converted += 1
                    else:
                        print(f"[warning] {h265_path} is empty; skipping mp4 conversion")
                elif args.record_mode == "device_mjpeg":
                    mjpeg_path = out_dir / f"{cam_name}.mjpeg"
                    mkv_path = out_dir / f"{cam_name}.mkv"
                    if mjpeg_path.exists() and mjpeg_path.stat().st_size > 0:
                        if convert_mjpeg_to_mkv(mjpeg_path, mkv_path, float(FPS)):
                            converted += 1
                    else:
                        print(f"[warning] {mjpeg_path} is empty; skipping mkv conversion")
            print(f"Container conversion finished: {converted}/{len(output_queues)} files")


if __name__ == "__main__":
    main()
