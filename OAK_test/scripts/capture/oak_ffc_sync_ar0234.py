# coding=utf-8
from __future__ import annotations

import time
import argparse
from contextlib import suppress

import cv2
import depthai as dai


FPS = 15

# DepthAI v3 requestOutput() uses an explicit output size. Keep this moderate
# for four-camera preview stability. Raise to (1280, 800) only after confirming
# stable USB3 and a stable four-camera pipeline.
OUTPUT_SIZE = (1280, 800)
USE_EXTERNAL_TRIGGER = True

CAM_LIST = {
    "CAM_A": {"color": True},
    "CAM_B": {"color": True},
    "CAM_C": {"color": True},
    "CAM_D": {"color": True},
}

CAM_SOCKET_TO_NAME = {
    "RGB": "CAM_A",
    "LEFT": "CAM_B",
    "RIGHT": "CAM_C",
    "CAM_A": "CAM_A",
    "CAM_B": "CAM_B",
    "CAM_C": "CAM_C",
    "CAM_D": "CAM_D",
}

CAM_SOCKET_OPTS = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
    "CAM_D": dai.CameraBoardSocket.CAM_D,
}


def print_discovered_devices() -> None:
    print("DepthAI device discovery:")

    discovery_calls = (
        ("available", dai.Device.getAllAvailableDevices),
        ("connected", dai.Device.getAllConnectedDevices),
        ("bootloader", dai.DeviceBootloader.getAllAvailableDevices),
    )
    found_any = False

    for label, discovery_call in discovery_calls:
        try:
            devices = discovery_call()
        except Exception as exc:
            print(f" - {label}: discovery failed: {exc}")
            continue

        if not devices:
            print(f" - {label}: none")
            continue

        found_any = True
        for device_info in devices:
            print(f" - {label}: {device_info}")

    if not found_any:
        print(
            " - no booted OAK device is available. If DepthAI logs mention "
            "X_LINK_BOOTLOADER, the device is stuck before DepthAI can boot it."
        )
        print(
            " - Windows Device Manager should normally show Luxonis Device. "
            "If it shows Luxonis Bootloader, fully power-cycle the OAK."
        )


def open_device() -> dai.Device:
    try:
        return dai.Device()
    except RuntimeError as default_error:
        print(f"Opening device with default USB speed failed: {default_error}")
        print_discovered_devices()
        print("Retrying with USB2 compatibility mode...")

    try:
        return dai.Device(maxUsbSpeed=dai.UsbSpeed.HIGH)
    except RuntimeError as usb2_error:
        print(f"Opening device with USB2 compatibility mode failed: {usb2_error}")
        print_discovered_devices()
        raise RuntimeError(
            "Could not open OAK device. If Windows lists it as Luxonis "
            "Bootloader or DepthAI logs mention X_LINK_BOOTLOADER, unplug OAK "
            "power and USB, wait 10 seconds, reconnect power first if present, "
            "then reconnect USB. Also make sure no other DepthAI program is "
            "using it."
        ) from usb2_error


def print_device_info(device: dai.Device) -> dict[str, str]:
    try:
        print(f"USB speed: {device.getUsbSpeed().name}")
    except Exception:
        pass

    try:
        ir_drivers = device.getIrDrivers()
        print(f"IR drivers: {ir_drivers}")
        if ir_drivers:
            print(f"IR flood off: {device.setIrFloodLightIntensity(0)}")
            print(f"IR laser: {device.setIrLaserDotProjectorIntensity(1000)}")
    except Exception as exc:
        print(f"IR control skipped: {exc}")

    print("Connected cameras:")
    sensor_names: dict[str, str] = {}
    for features in device.getConnectedCameraFeatures():
        print(
            f" -socket {features.socket.name:6}: "
            f"{features.sensorName:6} {features.width:4} x {features.height:4} "
            f"focus:{'auto ' if features.hasAutofocus else 'fixed'} - ",
            end="",
        )
        print(*[sensor_type.name for sensor_type in features.supportedTypes])

        cam_name = CAM_SOCKET_TO_NAME.get(features.socket.name)
        if cam_name is not None:
            sensor_names[cam_name] = features.sensorName

    return sensor_names


def create_outputs(
    pipeline: dai.Pipeline,
    active_cameras: dict[str, dict[str, bool]],
) -> dict[str, dai.Node.Output]:
    outputs: dict[str, dai.Node.Output] = {}

    for cam_name, cam_props in active_cameras.items():
        cam = pipeline.create(dai.node.Camera)

        if cam_props["color"]:
            cam.setSensorType(dai.CameraSensorType.COLOR)
            frame_type = (
                dai.ImgFrame.Type.BGR888i
                if pipeline.getDefaultDevice().getPlatform() == dai.Platform.RVC4
                else dai.ImgFrame.Type.BGR888p
            )
        else:
            cam.setSensorType(dai.CameraSensorType.MONO)
            frame_type = dai.ImgFrame.Type.GRAY8

        cam.build(CAM_SOCKET_OPTS[cam_name], sensorFps=float(FPS))
        # DepthAI v3 Camera.build(...) replaces the old setBoardSocket/setFps
        # calls. Use explicit external trigger control instead of FrameSyncMode.
        if USE_EXTERNAL_TRIGGER:
            cam.initialControl.setExternalTrigger(1, 0)

        outputs[cam_name] = cam.requestOutput(
            OUTPUT_SIZE,
            type=frame_type,
            fps=float(FPS),
        )

    return outputs


def main() -> None:
    global FPS, OUTPUT_SIZE, USE_EXTERNAL_TRIGGER

    parser = argparse.ArgumentParser(
        description="Preview synchronized AR0234 cameras on an OAK-FFC-4P device.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS,
        help="sensor/output FPS; keep 15 until the device is stable",
    )
    parser.add_argument(
        "--size",
        default=f"{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}",
        help="output size, for example 640x400 or 1280x800",
    )
    parser.add_argument(
        "--no-external-trigger",
        action="store_true",
        help="free-run cameras for basic bring-up without FSIN input",
    )
    args = parser.parse_args()

    try:
        width_text, height_text = args.size.lower().split("x", 1)
        OUTPUT_SIZE = (int(width_text), int(height_text))
    except ValueError as exc:
        raise SystemExit("--size must use WIDTHxHEIGHT, for example 640x400") from exc

    FPS = args.fps
    USE_EXTERNAL_TRIGGER = not args.no_external_trigger

    print(
        f"Starting with fps={FPS}, size={OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}, "
        f"external_trigger={USE_EXTERNAL_TRIGGER}"
    )

    should_stop = False

    try:
        device = open_device()
    except RuntimeError as exc:
        print(f"Could not open OAK device: {exc}")
        return

    with device:
        sensor_names = print_device_info(device)

        active_cameras = {
            name: CAM_LIST[name]
            for name in CAM_LIST
            if name in sensor_names
        }

        for cam_name in set(CAM_LIST).difference(sensor_names):
            print(f"{cam_name} is not connected!")

        if not active_cameras:
            print("No configured cameras are connected.")
            return

        with dai.Pipeline(device) as pipeline:
            outputs = create_outputs(pipeline, active_cameras)

            output_queues = {}
            for cam_name, output in outputs.items():
                output_queues[cam_name] = output.createOutputQueue(
                    maxSize=4,
                    blocking=False,
                )
                cv2.namedWindow(cam_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(cam_name, 640, 480)

            pipeline.start()
            time.sleep(1)

            try:
                while pipeline.isRunning() and not should_stop:
                    frame_list = []

                    for cam_name, queue in output_queues.items():
                        try:
                            packet = queue.tryGet()
                        except Exception as exc:
                            print(f"{cam_name} output queue error: {exc}")
                            should_stop = True
                            break

                        if packet is not None:
                            print(cam_name + ":", packet.getTimestampDevice())
                            frame_list.append((cam_name, packet.getCvFrame()))

                    if frame_list:
                        print("-------------------------------")
                        for cam_name, frame in frame_list:
                            cv2.imshow(cam_name, frame)

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        should_stop = True
                        break
            except KeyboardInterrupt:
                print("Interrupted by user.")
            finally:
                if pipeline.isRunning():
                    with suppress(Exception):
                        pipeline.stop()
                with suppress(Exception):
                    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
