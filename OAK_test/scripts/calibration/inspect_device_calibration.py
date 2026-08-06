#!/usr/bin/env python3

import argparse
from pathlib import Path

import depthai as dai
import numpy as np

from common import DEFAULT_FISHEYE_SOCKETS, normalize_socket_names, save_json, socket_from_name, to_list


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect current EEPROM calibration for a 4-fisheye + IMU module.")
    parser.add_argument("--sockets", nargs="+", default=DEFAULT_FISHEYE_SOCKETS)
    parser.add_argument("--device", default="", help="Optional DepthAI MX ID.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    socket_names = normalize_socket_names(args.sockets)
    device_args = []
    if args.device:
        success, info = dai.Device.getDeviceByMxId(args.device)
        if not success:
            raise RuntimeError(f"Device not found: {args.device}")
        device_args.append(info)

    report = {
        "schema": "calibrate.device_calibration_report.v1",
        "cameras": {},
        "camera_extrinsics": {},
        "imu": {},
    }

    with dai.Device(*device_args) as device:
        print(f"MX ID: {device.getMxId()}")
        print(f"USB speed: {device.getUsbSpeed().name}")
        print(f"Connected IMU: {device.getConnectedIMU()}")
        try:
            print(f"IMU firmware: {device.getIMUFirmwareVersion()}")
            report["imu"]["firmware"] = str(device.getIMUFirmwareVersion())
        except Exception as ex:
            print(f"IMU firmware unavailable: {ex}")

        calib = device.readCalibration2()
        for name in socket_names:
            socket = socket_from_name(name)
            camera_report = {}
            print(f"\n{name}")
            try:
                K, width, height = calib.getDefaultIntrinsics(socket)
                D = calib.getDistortionCoefficients(socket)
                model = calib.getDistortionModel(socket)
                camera_report.update(
                    {
                        "image_size": [width, height],
                        "model": str(model).split(".")[-1],
                        "K": to_list(K),
                        "D": to_list(D),
                    }
                )
                print(f"  size: {width} x {height}")
                print(f"  model: {model}")
                print(f"  K:\n{np.asarray(K)}")
                print(f"  D: {np.asarray(D).reshape(-1)}")
            except Exception as ex:
                camera_report["error"] = str(ex)
                print(f"  calibration unavailable: {ex}")
            report["cameras"][name] = camera_report

        for src in socket_names:
            for dst in socket_names:
                if src == dst:
                    continue
                key = f"{src}_to_{dst}"
                try:
                    T = calib.getCameraExtrinsics(socket_from_name(src), socket_from_name(dst), False)
                    report["camera_extrinsics"][key] = to_list(T)
                    print(f"\n{key}:")
                    print(np.asarray(T))
                except Exception:
                    pass

        for name in socket_names:
            try:
                imu_to_cam = calib.getImuToCameraExtrinsics(socket_from_name(name), False)
                cam_to_imu = calib.getCameraToImuExtrinsics(socket_from_name(name), False)
                report["imu"][f"imu_to_{name}"] = to_list(imu_to_cam)
                report["imu"][f"{name}_to_imu"] = to_list(cam_to_imu)
                print(f"\nIMU_to_{name}:")
                print(np.asarray(imu_to_cam))
            except Exception as ex:
                report["imu"][f"imu_to_{name}_error"] = str(ex)

        if args.output:
            save_json(Path(args.output), report)
            print(f"\nSaved report: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()



