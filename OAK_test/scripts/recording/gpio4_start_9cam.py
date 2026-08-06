#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


DEFAULT_MXIDS = [
    "19443010A1D45D2E00",
    "194430109113652E00",
    "194430105141782E00",
]


def run_pinctrl(*args: str, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["pinctrl", *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture else subprocess.DEVNULL,
    )


def pin_level(pin: int) -> int:
    result = run_pinctrl("lev", str(pin), capture=True)
    text = result.stdout.strip()
    if text not in {"0", "1"}:
        raise RuntimeError(f"unexpected pinctrl level for GPIO{pin}: {text!r}")
    return int(text)


def is_pressed(pin: int, active_low: bool) -> bool:
    level = pin_level(pin)
    return level == 0 if active_low else level == 1


def wait_for_press(pin: int, active_low: bool) -> None:
    while True:
        if is_pressed(pin, active_low):
            time.sleep(0.03)
            if is_pressed(pin, active_low):
                while is_pressed(pin, active_low):
                    time.sleep(0.03)
                return
        time.sleep(0.05)


def set_output(pin: int, high: bool) -> None:
    run_pinctrl("set", str(pin), "op", "pn", "dh" if high else "dl")


def blink_status(pin: int, count: int, interval_s: float) -> None:
    for _ in range(count):
        set_output(pin, True)
        time.sleep(interval_s)
        set_output(pin, False)
        time.sleep(interval_s)


def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S "))
        f.write(text)
        f.write("\n")


def build_record_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        "bash",
        "-lc",
        "source /home/whr/miniforge3/etc/profile.d/conda.sh && "
        "conda activate camtest && "
        "cd /home/whr/HearWristCam && "
        "exec python test_code/head_wrist_record/9cam_record.py "
        "--mxids "
        + " ".join(args.mxids)
        + " "
        f"--cameras {args.cameras} "
        f"--duration {args.duration:g} "
        f"--fps {args.fps:g} "
        f"--bitrate-kbps {args.bitrate_kbps} "
        f"--module-settle {args.module_settle:g} "
        f"--pre-trigger-settle {args.pre_trigger_settle:g} "
        f"--trigger-warmup {args.trigger_warmup:g} "
        f"--sync-mode {args.sync_mode} "
        f"--out-dir {args.out_dir} "
        f"--start-pin {args.start_pin} "
        f"--stop-pin {args.stop_pin} "
        f"--stop-hold-s {args.stop_hold_s:g} "
        f"--trigger-pin {args.trigger_pin} "
        f"--status-pin {args.status_pin} "
        f"--manual-exposure-us {args.manual_exposure_us} "
        f"--manual-iso {args.manual_iso} "
        f"--imu-rate {args.imu_rate} "
        "--skip-init-button",
    ]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-pin", type=int, default=4)
    parser.add_argument("--stop-pin", type=int, default=17)
    parser.add_argument("--stop-hold-s", type=float, default=0.25)
    parser.add_argument("--trigger-pin", type=int, default=18)
    parser.add_argument("--status-pin", type=int, default=23)
    parser.add_argument("--button-active-high", action="store_true")
    parser.add_argument("--mxids", nargs="+", default=DEFAULT_MXIDS)
    parser.add_argument("--cameras", default="A,B,C")
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--bitrate-kbps", type=int, default=12000)
    parser.add_argument("--module-settle", type=float, default=3.0)
    parser.add_argument("--pre-trigger-settle", type=float, default=2.0)
    parser.add_argument("--trigger-warmup", type=float, default=0.0)
    parser.add_argument("--sync-mode", choices=["fsync_input", "none"], default="fsync_input")
    parser.add_argument("--manual-exposure-us", type=int, default=12_000)
    parser.add_argument("--manual-iso", type=int, default=150)
    parser.add_argument("--imu-rate", type=int, default=60)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/whr/Desktop/record"))
    parser.add_argument("--runner-log", type=Path, default=Path("/home/whr/Desktop/record/gpio4_9cam_runner.log"))
    args = parser.parse_args()

    active_low = not args.button_active_high
    pull = "pu" if active_low else "pd"
    run_pinctrl("set", str(args.start_pin), "ip", pull)
    run_pinctrl("set", str(args.stop_pin), "ip", pull)
    run_pinctrl("set", str(args.trigger_pin), "op", "pn", "dl")
    run_pinctrl("set", str(args.status_pin), "op", "pn", "dl")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[GPIO_LISTENER] ready: GPIO{args.start_pin} starts 9cam; "
        f"GPIO{args.status_pin}=status, GPIO{args.trigger_pin}=trigger.",
        flush=True,
    )
    while True:
        wait_for_press(args.start_pin, active_low)
        print("[GPIO_LISTENER] start button pressed; launching 9cam_record.py", flush=True)
        log_line(args.runner_log, "start button pressed; launching 9cam_record.py")
        set_output(args.status_pin, True)
        time.sleep(0.15)
        set_output(args.status_pin, False)
        with args.runner_log.open("a", encoding="utf-8") as log:
            log.write(time.strftime("%Y-%m-%d %H:%M:%S [GPIO_LISTENER] child output begins\n"))
            log.flush()
            proc = subprocess.Popen(build_record_command(args), stdout=log, stderr=subprocess.STDOUT)
            ret = proc.wait()
            log.write(time.strftime(f"%Y-%m-%d %H:%M:%S [GPIO_LISTENER] child exited with code {ret}\n"))
        print(f"[GPIO_LISTENER] 9cam_record.py exited with code {ret}; listening again.", flush=True)
        log_line(args.runner_log, f"9cam_record.py exited with code {ret}; listening again")
        run_pinctrl("set", str(args.trigger_pin), "op", "pn", "dl")
        run_pinctrl("set", str(args.status_pin), "op", "pn", "dl")
        if ret != 0:
            blink_status(args.status_pin, count=6, interval_s=0.12)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
