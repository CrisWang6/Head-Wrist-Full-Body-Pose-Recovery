#!/usr/bin/env python3

import argparse
from pathlib import Path

import depthai as dai

from common import ensure_dir, load_json, normalize_socket_names, save_json, socket_from_name


REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_CONFIG_DIR = REPO_ROOT / "configs" / "calibration" / "device"


def parse_args():
    parser = argparse.ArgumentParser(description="Create or flash DepthAI EEPROM calibration from an intrinsics JSON.")
    parser.add_argument("calibration", help="Intrinsics calibration JSON to export.")
    parser.add_argument("--device", default="", help="Optional DepthAI MX ID.")
    parser.add_argument("--output", default=str(DEVICE_CONFIG_DIR / "depthai_calibration.json"), help="DepthAI calibration JSON to write.")
    parser.add_argument("--flash", action="store_true", help="Actually flash the calibration to EEPROM.")
    parser.add_argument("--imu-json", default="", help="Optional IMU extrinsics JSON with R and t_m fields.")
    parser.add_argument("--imu-dest", default="", help="Camera socket for IMU extrinsics destination. Defaults to reference camera.")
    return parser.parse_args()


def backup_path(device):
    return DEVICE_CONFIG_DIR / f"backup_calib_{device.getMxId()}.json"


def maybe_read_device_calibration(device):
    try:
        return device.readCalibration2()
    except Exception:
        return dai.CalibrationHandler()


def set_camera_data(calib, calibration):
    for name, cam in calibration["cameras"].items():
        socket = socket_from_name(name)
        width, height = cam["image_size"]
        calib.setCameraType(socket, dai.CameraModel.Fisheye)
        calib.setCameraIntrinsics(socket, cam["K"], int(width), int(height))
        calib.setDistortionCoefficients(socket, cam["D"])
        if "fov_deg" in cam:
            calib.setFov(socket, float(cam["fov_deg"]))


def set_camera_extrinsics(calib, calibration):
    for pair in calibration.get("extrinsics", {}).values():
        src = socket_from_name(pair["src"])
        dst = socket_from_name(pair["dst"])
        calib.setCameraExtrinsics(src, dst, pair["R"], pair["t_m"])


def set_imu_extrinsics(calib, calibration, imu_json_path, imu_dest):
    imu_data = None
    if imu_json_path:
        imu_data = load_json(imu_json_path)
    elif calibration.get("imu_extrinsics"):
        imu_data = calibration["imu_extrinsics"]

    if not imu_data:
        return False

    dest_name = imu_dest or imu_data.get("dest") or calibration["reference_camera"]
    calib.setImuExtrinsics(socket_from_name(dest_name), imu_data["R"], imu_data["t_m"])
    return True


def main():
    args = parse_args()
    calibration = load_json(args.calibration)
    normalize_socket_names(calibration["cameras"].keys())

    device_args = []
    if args.device:
        success, info = dai.Device.getDeviceByMxId(args.device)
        if not success:
            raise RuntimeError(f"Device not found: {args.device}")
        device_args.append(info)

    with dai.Device(*device_args) as device:
        backup = backup_path(device)
        try:
            device.readCalibration2().eepromToJsonFile(str(backup))
            print(f"Backed up current calibration to: {backup.resolve()}")
        except Exception as ex:
            print(f"Could not back up current calibration: {ex}")

        calib = maybe_read_device_calibration(device)
        set_camera_data(calib, calibration)
        set_camera_extrinsics(calib, calibration)
        has_imu = set_imu_extrinsics(calib, calibration, args.imu_json, args.imu_dest)

        output = Path(args.output)
        ensure_dir(output.parent)
        calib.eepromToJsonFile(str(output))
        print(f"Wrote DepthAI calibration JSON: {output.resolve()}")
        if not has_imu:
            print("No IMU extrinsics were written. Provide --imu-json when you have measured/solved IMU pose.")

        if args.flash:
            device.flashCalibration2(calib)
            print("Flashed calibration to EEPROM.")
        else:
            print("Dry run only. Re-run with --flash to write EEPROM.")

        save_json(
            output.with_suffix(".summary.json"),
            {
                "source": str(Path(args.calibration).resolve()),
                "depthai_json": str(output.resolve()),
                "flashed": bool(args.flash),
                "imu_written": bool(has_imu),
            },
        )


if __name__ == "__main__":
    main()


