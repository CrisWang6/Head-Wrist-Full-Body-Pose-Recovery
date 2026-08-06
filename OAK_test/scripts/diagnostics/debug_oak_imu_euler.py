from __future__ import annotations

import argparse
import math
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class OrientationSample:
    sequence: int
    timestamp_ms: int
    quaternion: tuple[float, float, float, float]
    euler_deg: tuple[float, float, float]
    accuracy: float | None
    fps: float
    source: str
    accel: tuple[float, float, float] | None = None
    gyro: tuple[float, float, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug OAK IMU rotation-vector quaternion and visualize Euler angles."
    )
    parser.add_argument("--rate", type=int, default=100, help="fused IMU report rate in Hz, default 100")
    parser.add_argument("--accel-rate", type=int, default=500, help="raw accelerometer report rate in Hz, default 500")
    parser.add_argument("--gyro-rate", type=int, default=400, help="raw gyroscope report rate in Hz, default 400")
    parser.add_argument("--batch", type=int, default=10, help="max IMU batch reports, default 10")
    parser.add_argument(
        "--mode",
        choices=("raw", "fused"),
        default="fused",
        help="raw reads accel+gyro and integrates a quaternion; fused reads BNO086 rotation vector",
    )
    parser.add_argument(
        "--sensor",
        choices=("rotation", "game", "arvr", "geomagnetic"),
        default="rotation",
        help="fused quaternion source: rotation uses magnetometer, game avoids magnetometer",
    )
    parser.add_argument("--mahony-kp", type=float, default=1.2, help="raw mode Mahony proportional gain")
    parser.add_argument("--mahony-ki", type=float, default=0.03, help="raw mode Mahony integral gain for gyro bias tracking")
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=300,
        help="raw mode stationary startup samples used to estimate gyro bias",
    )
    parser.add_argument(
        "--update-imu-firmware",
        action="store_true",
        help="update BNO08x IMU firmware, then exit; do not unplug the device while updating",
    )
    parser.add_argument(
        "--force-imu-firmware-update",
        action="store_true",
        help="force IMU firmware update even when DepthAI thinks it is current",
    )
    parser.add_argument("--width", type=int, default=960, help="visualization window width")
    parser.add_argument("--height", type=int, default=640, help="visualization window height")
    return parser.parse_args()


def open_device(dai: Any) -> Any:
    try:
        return dai.Device()
    except RuntimeError as exc:
        print(f"[oak imu] opening default USB speed failed: {exc}")
        print("[oak imu] retrying with USB2 compatibility mode")
    return dai.Device(maxUsbSpeed=dai.UsbSpeed.HIGH)


def sensor_kind(dai: Any, name: str) -> Any:
    sensors = {
        "rotation": dai.IMUSensor.ROTATION_VECTOR,
        "game": dai.IMUSensor.GAME_ROTATION_VECTOR,
        "arvr": dai.IMUSensor.ARVR_STABILIZED_ROTATION_VECTOR,
        "geomagnetic": dai.IMUSensor.GEOMAGNETIC_ROTATION_VECTOR,
    }
    return sensors[name]


def print_imu_info(device: Any) -> None:
    try:
        imu_type = device.getConnectedIMU()
        firmware = device.getIMUFirmwareVersion()
        embedded = device.getEmbeddedIMUFirmwareVersion()
        print(f"[oak imu] detected IMU type={imu_type}, firmware={firmware}, embedded={embedded}")
    except Exception as exc:
        print(f"[oak imu] IMU detection query failed: {exc}")


def update_imu_firmware(dai: Any, force: bool = False) -> int:
    device = open_device(dai)
    try:
        print_imu_info(device)
        print("[oak imu] WARNING: IMU firmware update can soft-brick the IMU if power/USB is interrupted.")
        print("[oak imu] Keep the device connected until the update finishes.")
        answer = input("Type UPDATE to start IMU firmware update, or anything else to cancel: ")
        if answer != "UPDATE":
            print("[oak imu] firmware update cancelled")
            return 1

        started = device.startIMUFirmwareUpdate(force)
        if not started:
            print("[oak imu] could not start IMU firmware update; make sure no IMU pipeline is running")
            return 1

        while True:
            finished, percentage = device.getIMUFirmwareUpdateStatus()
            print(f"[oak imu] firmware update: {percentage:.1f}%")
            if finished:
                if percentage == 100:
                    print("[oak imu] firmware update successful")
                    print("[oak imu] unplug/replug the device before reading IMU data again")
                    return 0
                print("[oak imu] firmware update failed")
                return 1
            time.sleep(1.0)
    finally:
        with suppress(Exception):
            device.close()


def create_imu_queue(
    dai: Any,
    rate: int,
    accel_rate: int,
    gyro_rate: int,
    batch: int,
    mode: str,
    sensor: Any,
) -> tuple[Any, Any, Any]:
    pipeline = dai.Pipeline()
    imu = pipeline.create(dai.node.IMU)
    imu.enableFirmwareUpdate(False)
    if mode == "raw":
        # Match Luxonis' IMU example: accel and gyro are enabled as separate reports.
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, accel_rate)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, gyro_rate)
    else:
        imu.enableIMUSensor(sensor, rate)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(max(1, batch))

    if hasattr(dai.node, "XLinkOut"):
        xlink_out = pipeline.create(dai.node.XLinkOut)
        xlink_out.setStreamName("imu")
        imu.out.link(xlink_out.input)
        device = dai.Device(pipeline)
        print_imu_info(device)
        with suppress(Exception):
            print(f"[oak imu] USB speed: {device.getUsbSpeed().name}")
        queue = device.getOutputQueue(name="imu", maxSize=max(50, batch * 4), blocking=False)
    else:
        queue = imu.out.createOutputQueue(maxSize=max(50, batch * 4), blocking=False)
        pipeline.start()
        device = pipeline.getDefaultDevice()
        print_imu_info(device)
        with suppress(Exception):
            print(f"[oak imu] USB speed: {device.getUsbSpeed().name}")
    return device, pipeline, queue


def normalize_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float, float, float]:
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    return x / norm, y / norm, z / norm, w / norm


def quaternion_to_euler_deg(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    x, y, z, w = normalize_quaternion(x, y, z, w)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def euler_deg_to_quaternion(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr = math.cos(math.radians(roll) * 0.5)
    sr = math.sin(math.radians(roll) * 0.5)
    cp = math.cos(math.radians(pitch) * 0.5)
    sp = math.sin(math.radians(pitch) * 0.5)
    cy = math.cos(math.radians(yaw) * 0.5)
    sy = math.sin(math.radians(yaw) * 0.5)
    return normalize_quaternion(
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def multiply_quaternion(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return normalize_quaternion(
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def inverse_quaternion(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = normalize_quaternion(*q)
    return -x, -y, -z, w


def gyro_delta_quaternion(gx: float, gy: float, gz: float, dt: float) -> tuple[float, float, float, float]:
    angle = math.sqrt(gx * gx + gy * gy + gz * gz) * dt
    if angle <= 1e-12:
        return 0.0, 0.0, 0.0, 1.0
    scale = math.sin(angle * 0.5) / max(math.sqrt(gx * gx + gy * gy + gz * gz), 1e-12)
    return normalize_quaternion(gx * scale, gy * scale, gz * scale, math.cos(angle * 0.5))


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    x, y, z, w = normalize_quaternion(x, y, z, w)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion(float(x), float(y), float(z), float(w))


def rotation_y_matrix(deg: float) -> np.ndarray:
    angle = math.radians(deg)
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


DISPLAY_IMU_TO_HEAD = rotation_y_matrix(-60.0)


def apply_display_head_frame(quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    rot_imu = quaternion_to_matrix(*quaternion).astype(np.float64)
    rot_head = DISPLAY_IMU_TO_HEAD @ rot_imu @ DISPLAY_IMU_TO_HEAD.T
    return matrix_to_quaternion(rot_head)


def report_timestamp_ms(report: Any) -> int:
    try:
        return int(report.getTimestampDevice().total_seconds() * 1000.0)
    except Exception:
        return int(time.time() * 1000)


def report_sequence(report: Any) -> int:
    try:
        return int(report.getSequenceNum())
    except Exception:
        return int(getattr(report, "sequence", 0))


class RawQuaternionEstimator:
    """Six-axis Mahony AHRS with stationary startup gyro-bias calibration."""

    def __init__(
        self,
        kp: float = 1.2,
        ki: float = 0.03,
        calibration_samples: int = 300,
    ) -> None:
        self.kp = max(0.0, kp)
        self.ki = max(0.0, ki)
        self.calibration_samples = max(0, calibration_samples)
        self.quaternion = (0.0, 0.0, 0.0, 1.0)
        self.last_timestamp_ms: int | None = None
        self.gyro_bias = (0.0, 0.0, 0.0)
        self.integral_error = (0.0, 0.0, 0.0)
        self._calib_count = 0
        self._gyro_sum = [0.0, 0.0, 0.0]
        self._accel_sum = [0.0, 0.0, 0.0]
        self.zero_reference_inv = (0.0, 0.0, 0.0, 1.0)

    @property
    def is_calibrating(self) -> bool:
        return self._calib_count < self.calibration_samples

    @property
    def calibration_progress(self) -> tuple[int, int]:
        return min(self._calib_count, self.calibration_samples), self.calibration_samples

    def update(
        self,
        accel: tuple[float, float, float],
        gyro: tuple[float, float, float],
        timestamp_ms: int,
    ) -> tuple[float, float, float, float]:
        if self.is_calibrating:
            self._accumulate_calibration(accel, gyro, timestamp_ms)
            return self.quaternion

        if self.last_timestamp_ms is None:
            self.quaternion = self._accel_initial_quaternion(accel)
            self.last_timestamp_ms = timestamp_ms
            return self.quaternion

        dt = max(0.0, min(0.1, (timestamp_ms - self.last_timestamp_ms) / 1000.0))
        self.last_timestamp_ms = timestamp_ms
        if dt <= 0.0:
            return self.quaternion

        gx = gyro[0] - self.gyro_bias[0]
        gy = gyro[1] - self.gyro_bias[1]
        gz = gyro[2] - self.gyro_bias[2]

        ax, ay, az = accel
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 1e-6:
            ax, ay, az = ax / norm, ay / norm, az / norm
            vx, vy, vz = self._estimated_gravity()

            ex = ay * vz - az * vy
            ey = az * vx - ax * vz
            ez = ax * vy - ay * vx

            ix, iy, iz = self.integral_error
            ix += self.ki * ex * dt
            iy += self.ki * ey * dt
            iz += self.ki * ez * dt
            self.integral_error = (ix, iy, iz)

            gx += self.kp * ex + ix
            gy += self.kp * ey + iy
            gz += self.kp * ez + iz

        self.quaternion = self._integrate_gyro(gx, gy, gz, dt)
        return self.quaternion

    def display_quaternion(self) -> tuple[float, float, float, float]:
        return multiply_quaternion(self.zero_reference_inv, self.quaternion)

    def zero_current_orientation(self) -> None:
        self.zero_reference_inv = inverse_quaternion(self.quaternion)

    def reset(self) -> None:
        self.quaternion = (0.0, 0.0, 0.0, 1.0)
        self.last_timestamp_ms = None
        self.gyro_bias = (0.0, 0.0, 0.0)
        self.integral_error = (0.0, 0.0, 0.0)
        self._calib_count = 0
        self._gyro_sum = [0.0, 0.0, 0.0]
        self._accel_sum = [0.0, 0.0, 0.0]
        self.zero_reference_inv = (0.0, 0.0, 0.0, 1.0)

    def _accumulate_calibration(
        self,
        accel: tuple[float, float, float],
        gyro: tuple[float, float, float],
        timestamp_ms: int,
    ) -> None:
        if self.calibration_samples <= 0:
            self.quaternion = self._accel_initial_quaternion(accel)
            self.last_timestamp_ms = timestamp_ms
            self._calib_count = 0
            return

        for index in range(3):
            self._gyro_sum[index] += gyro[index]
            self._accel_sum[index] += accel[index]
        self._calib_count += 1
        self.last_timestamp_ms = timestamp_ms

        if self._calib_count >= self.calibration_samples:
            inv = 1.0 / max(1, self._calib_count)
            self.gyro_bias = (
                self._gyro_sum[0] * inv,
                self._gyro_sum[1] * inv,
                self._gyro_sum[2] * inv,
            )
            mean_accel = (
                self._accel_sum[0] * inv,
                self._accel_sum[1] * inv,
                self._accel_sum[2] * inv,
            )
            self.quaternion = self._accel_initial_quaternion(mean_accel)
            print(
                "[oak imu] raw calibration done: "
                f"gyro_bias=({self.gyro_bias[0]:+.6f}, {self.gyro_bias[1]:+.6f}, {self.gyro_bias[2]:+.6f}) rad/s"
            )

    def _estimated_gravity(self) -> tuple[float, float, float]:
        qx, qy, qz, qw = self.quaternion
        return (
            2.0 * (qx * qz - qw * qy),
            2.0 * (qw * qx + qy * qz),
            qw * qw - qx * qx - qy * qy + qz * qz,
        )

    def _integrate_gyro(self, gx: float, gy: float, gz: float, dt: float) -> tuple[float, float, float, float]:
        qx, qy, qz, qw = self.quaternion
        half_dt = 0.5 * dt
        dx = (qw * gx + qy * gz - qz * gy) * half_dt
        dy = (qw * gy - qx * gz + qz * gx) * half_dt
        dz = (qw * gz + qx * gy - qy * gx) * half_dt
        dw = (-qx * gx - qy * gy - qz * gz) * half_dt
        return normalize_quaternion(qx + dx, qy + dy, qz + dz, qw + dw)

    @staticmethod
    def _accel_initial_quaternion(accel: tuple[float, float, float]) -> tuple[float, float, float, float]:
        ax, ay, az = accel
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        return euler_deg_to_quaternion(roll, pitch, 0.0)


def read_latest_fused_sample(queue: Any, fps_meter: "FpsMeter") -> OrientationSample | None:
    latest = None
    while True:
        try:
            data = queue.tryGet()
        except Exception as exc:
            print(f"[oak imu] queue closed or read failed: {exc}")
            return None
        if data is None:
            break
        for packet in data.packets:
            latest = packet.rotationVector

    if latest is None:
        return None

    quat = normalize_quaternion(
        float(latest.i),
        float(latest.j),
        float(latest.k),
        float(latest.real),
    )
    return OrientationSample(
        sequence=report_sequence(latest),
        timestamp_ms=report_timestamp_ms(latest),
        quaternion=quat,
        euler_deg=quaternion_to_euler_deg(*quat),
        accuracy=float(latest.rotationVectorAccuracy)
        if hasattr(latest, "rotationVectorAccuracy")
        else None,
        fps=fps_meter.tick(),
        source="fused",
    )


class OrientationZero:
    def __init__(self) -> None:
        self.reference_inv = (0.0, 0.0, 0.0, 1.0)

    def apply(self, quaternion: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return multiply_quaternion(self.reference_inv, quaternion)

    def zero(self, quaternion: tuple[float, float, float, float]) -> None:
        self.reference_inv = inverse_quaternion(quaternion)

    def reset(self) -> None:
        self.reference_inv = (0.0, 0.0, 0.0, 1.0)


def read_latest_raw_sample(
    queue: Any,
    fps_meter: "FpsMeter",
    estimator: RawQuaternionEstimator,
) -> OrientationSample | None:
    latest_sample = None
    while True:
        try:
            data = queue.tryGet()
        except Exception as exc:
            print(f"[oak imu] queue closed or read failed: {exc}")
            return None
        if data is None:
            break
        for packet in data.packets:
            accel_report = packet.acceleroMeter
            gyro_report = packet.gyroscope
            if accel_report is None or gyro_report is None:
                continue

            accel = (float(accel_report.x), float(accel_report.y), float(accel_report.z))
            gyro = (float(gyro_report.x), float(gyro_report.y), float(gyro_report.z))
            timestamp_ms = min(report_timestamp_ms(accel_report), report_timestamp_ms(gyro_report))
            estimator.update(accel, gyro, timestamp_ms)
            quat = estimator.display_quaternion()
            if estimator.is_calibrating:
                done, total = estimator.calibration_progress
                source = f"raw-calibrating {done}/{total}"
            else:
                source = "raw-mahony"
            latest_sample = OrientationSample(
                sequence=report_sequence(gyro_report),
                timestamp_ms=timestamp_ms,
                quaternion=quat,
                euler_deg=quaternion_to_euler_deg(*quat),
                accuracy=None,
                fps=0.0,
                source=source,
                accel=accel,
                gyro=gyro,
            )

    if latest_sample is None:
        return None
    latest_sample.fps = fps_meter.tick()
    return latest_sample


class FpsMeter:
    def __init__(self) -> None:
        self.last = time.monotonic()
        self.value = 0.0

    def tick(self) -> float:
        now = time.monotonic()
        dt = now - self.last
        self.last = now
        if dt > 0:
            inst = 1.0 / dt
            self.value = inst if self.value <= 0 else self.value * 0.9 + inst * 0.1
        return self.value


def draw_bar(canvas: np.ndarray, label: str, value: float, limit: float, y: int, color: tuple[int, int, int]) -> None:
    x0, x1 = 80, canvas.shape[1] - 70
    mid = (x0 + x1) // 2
    cv2.line(canvas, (x0, y), (x1, y), (70, 70, 70), 2)
    cv2.line(canvas, (mid, y - 16), (mid, y + 16), (120, 120, 120), 1)
    clipped = max(-limit, min(limit, value))
    end = int(mid + clipped / limit * (x1 - x0) / 2)
    cv2.line(canvas, (mid, y), (end, y), color, 12)
    cv2.circle(canvas, (end, y), 7, color, -1)
    cv2.putText(canvas, label, (24, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(canvas, f"{value:8.2f} deg", (x1 - 130, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230, 230, 230), 1)


def draw_history(canvas: np.ndarray, history: deque[tuple[float, float, float]], rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (65, 65, 65), 1)
    for frac in (0.25, 0.5, 0.75):
        yy = y + int(h * frac)
        cv2.line(canvas, (x, yy), (x + w, yy), (40, 40, 40), 1)
    cv2.putText(canvas, "Euler history (-180..180 deg)", (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)

    values = list(history)
    if len(values) < 2:
        return

    colors = ((80, 180, 255), (120, 230, 120), (255, 160, 90))
    for axis in range(3):
        pts = []
        for idx, sample in enumerate(values):
            xx = x + int(idx * w / max(1, len(values) - 1))
            yy = y + int((180.0 - max(-180.0, min(180.0, sample[axis]))) / 360.0 * h)
            pts.append((xx, yy))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, colors[axis], 2, cv2.LINE_AA)


def project_axis(
    origin: tuple[int, int],
    vector: np.ndarray,
    scale: float,
    view: str = "iso",
) -> tuple[int, int]:
    vx, vy, vz = vector.tolist()
    if view == "top":
        sx = origin[0] + int(vy * scale)
        sy = origin[1] + int(vx * scale)
    elif view == "side":
        sx = origin[0] + int(vy * scale)
        sy = origin[1] - int(vz * scale)
    else:
        sx = origin[0] + int((vy + 0.55 * vx) * scale)
        sy = origin[1] + int((0.35 * vx - vz) * scale)
    return sx, sy


def draw_arrow_with_label(
    canvas: np.ndarray,
    origin: tuple[int, int],
    vector: np.ndarray,
    scale: float,
    color: tuple[int, int, int],
    label: str,
    view: str,
    thickness: int = 3,
) -> None:
    end = project_axis(origin, vector, scale, view)
    cv2.arrowedLine(canvas, origin, end, color, thickness, cv2.LINE_AA, tipLength=0.16)
    cv2.circle(canvas, end, 4, color, -1)
    cv2.putText(canvas, label, (end[0] + 7, end[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def draw_orientation_axes(canvas: np.ndarray, quat: tuple[float, float, float, float]) -> None:
    x, y, z, w = quat
    rot = quaternion_to_matrix(x, y, z, w)
    height, width = canvas.shape[:2]
    origin = (width // 2, height - 120)
    scale = 125.0
    axes = (
        (rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32), (60, 80, 255), "+X depth"),
        (rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32), (80, 220, 80), "+Y horizontal"),
        (rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32), (255, 170, 70), "+Z up"),
    )
    cv2.putText(canvas, "Head-frame axes", (origin[0] - 105, origin[1] - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)
    cv2.line(canvas, (origin[0] - 150, origin[1]), (origin[0] + 150, origin[1]), (58, 58, 58), 1)
    cv2.line(canvas, (origin[0], origin[1] - 145), (origin[0], origin[1] + 35), (58, 58, 58), 1)
    cv2.ellipse(canvas, origin, (145, 42), 0, 0, 360, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.circle(canvas, origin, 5, (230, 230, 230), -1)
    for vector, color, label in axes:
        draw_arrow_with_label(canvas, origin, vector, scale, color, label, "iso", 4)

    legend_x = 24
    legend_y = height - 118
    for offset, (_, color, label) in enumerate(axes):
        y0 = legend_y + offset * 24
        cv2.line(canvas, (legend_x, y0), (legend_x + 30, y0), color, 5)
        cv2.putText(canvas, label, (legend_x + 42, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    draw_projection_view(canvas, rot, (width - 175, height - 178), "top", "Top: Y/X", "Y", "X")
    draw_projection_view(canvas, rot, (width - 175, height - 66), "side", "Side: Y/Z", "Y", "Z")


def draw_projection_view(
    canvas: np.ndarray,
    rot: np.ndarray,
    origin: tuple[int, int],
    view: str,
    title: str,
    horizontal_label: str,
    vertical_label: str,
) -> None:
    scale = 45.0
    cv2.rectangle(canvas, (origin[0] - 68, origin[1] - 48), (origin[0] + 88, origin[1] + 48), (55, 55, 55), 1)
    cv2.putText(canvas, title, (origin[0] - 64, origin[1] - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
    cv2.line(canvas, (origin[0] - 56, origin[1]), (origin[0] + 56, origin[1]), (65, 65, 65), 1)
    cv2.line(canvas, (origin[0], origin[1] - 36), (origin[0], origin[1] + 36), (65, 65, 65), 1)
    cv2.putText(canvas, f"+{horizontal_label}", (origin[0] + 59, origin[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    cv2.putText(canvas, f"+{vertical_label}", (origin[0] + 5, origin[1] - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    axes = (
        (rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32), (60, 80, 255), "X"),
        (rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32), (80, 220, 80), "Y"),
        (rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32), (255, 170, 70), "Z"),
    )
    for vector, color, label in axes:
        draw_arrow_with_label(canvas, origin, vector, scale, color, label, view, 2)


def render(
    sample: OrientationSample | None,
    history: deque[tuple[float, float, float]],
    size: tuple[int, int],
) -> np.ndarray:
    width, height = size
    canvas = np.full((height, width, 3), (24, 26, 29), dtype=np.uint8)
    cv2.putText(canvas, "OAK IMU head Euler debug", (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (240, 240, 240), 2)
    cv2.putText(canvas, "q: quit    r: zero pose    display frame: IMU Ry(+60 deg)", (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (170, 170, 170), 1)

    if sample is None:
        cv2.putText(canvas, "Waiting for IMU packets...", (24, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 180, 255), 2)
        return canvas

    roll, pitch, yaw = sample.euler_deg
    draw_bar(canvas, "Roll", roll, 180.0, 125, (80, 180, 255))
    draw_bar(canvas, "Pitch", pitch, 90.0, 175, (120, 230, 120))
    draw_bar(canvas, "Yaw", yaw, 180.0, 225, (255, 160, 90))

    qx, qy, qz, qw = sample.quaternion
    accuracy = "n/a" if sample.accuracy is None else f"{sample.accuracy:.4f} rad"
    cv2.putText(canvas, f"src={sample.source}  seq={sample.sequence}  t={sample.timestamp_ms} ms  fps={sample.fps:5.1f}  acc={accuracy}", (24, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(canvas, f"quat x={qx:+.5f} y={qy:+.5f} z={qz:+.5f} w={qw:+.5f}", (24, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    if sample.accel is not None and sample.gyro is not None:
        ax, ay, az = sample.accel
        gx, gy, gz = sample.gyro
        cv2.putText(canvas, f"acc m/s2 x={ax:+.3f} y={ay:+.3f} z={az:+.3f}   gyro rad/s x={gx:+.3f} y={gy:+.3f} z={gz:+.3f}", (24, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    draw_history(canvas, history, (24, 350, width - 48, 135))
    draw_orientation_axes(canvas, sample.quaternion)
    return canvas


def main() -> int:
    args = parse_args()
    if args.rate <= 0:
        raise ValueError("--rate must be positive")
    if args.accel_rate <= 0:
        raise ValueError("--accel-rate must be positive")
    if args.gyro_rate <= 0:
        raise ValueError("--gyro-rate must be positive")
    if args.calibration_samples < 0:
        raise ValueError("--calibration-samples cannot be negative")
    if args.width < 640 or args.height < 480:
        raise ValueError("--width/--height should be at least 640x480")

    try:
        import depthai as dai
    except ImportError as exc:
        raise RuntimeError("depthai is required; install project requirements first") from exc

    if args.update_imu_firmware:
        return update_imu_firmware(dai, args.force_imu_firmware_update)

    sensor = sensor_kind(dai, args.sensor)
    if args.mode == "raw":
        print(
            f"[oak imu] starting mode=raw, accel={args.accel_rate} Hz, "
            f"gyro={args.gyro_rate} Hz, batch={args.batch}"
        )
    else:
        print(f"[oak imu] starting mode=fused, sensor={sensor.name}, rate={args.rate} Hz, batch={args.batch}")
    device, pipeline, queue = create_imu_queue(
        dai,
        args.rate,
        args.accel_rate,
        args.gyro_rate,
        args.batch,
        args.mode,
        sensor,
    )

    history: deque[tuple[float, float, float]] = deque(maxlen=240)
    fps_meter = FpsMeter()
    raw_estimator = RawQuaternionEstimator(
        kp=args.mahony_kp,
        ki=args.mahony_ki,
        calibration_samples=args.calibration_samples,
    )
    fused_zero = OrientationZero()
    latest: OrientationSample | None = None
    latest_fused_raw_quaternion: tuple[float, float, float, float] | None = None
    first_wait_started = time.monotonic()
    no_packet_hint_printed = False

    try:
        while True:
            if args.mode == "raw":
                sample = read_latest_raw_sample(queue, fps_meter, raw_estimator)
            else:
                sample = read_latest_fused_sample(queue, fps_meter)
                if sample is not None:
                    latest_fused_raw_quaternion = sample.quaternion
                    quat = fused_zero.apply(latest_fused_raw_quaternion)
                    sample.quaternion = quat
                    sample.euler_deg = quaternion_to_euler_deg(*quat)
            if sample is not None:
                quat = apply_display_head_frame(sample.quaternion)
                sample.quaternion = quat
                sample.euler_deg = quaternion_to_euler_deg(*quat)
                latest = sample
                history.append(sample.euler_deg)
            elif not no_packet_hint_printed and time.monotonic() - first_wait_started > 3.0:
                no_packet_hint_printed = True
                print(
                    "[oak imu] no IMU packets received after 3s. "
                    "If you see 'IMU driver failed with error code 1' and firmware is 3.2.13, "
                    "run this script with --update-imu-firmware, then unplug/replug the OAK."
                )

            frame = render(latest, history, (args.width, args.height))
            cv2.imshow("OAK IMU Euler Debug", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                history.clear()
                if latest is not None:
                    if args.mode == "raw":
                        raw_estimator.zero_current_orientation()
                    else:
                        if latest_fused_raw_quaternion is not None:
                            fused_zero.zero(latest_fused_raw_quaternion)
                    quat = (0.0, 0.0, 0.0, 1.0)
                    latest.quaternion = quat
                    latest.euler_deg = (0.0, 0.0, 0.0)
                    latest.source = f"{latest.source}-zeroed"
                    print("[oak imu] current orientation set as zero")
                else:
                    raw_estimator.reset()
                    fused_zero.reset()
                    print("[oak imu] no current orientation yet; reset estimator state")
    finally:
        cv2.destroyAllWindows()
        with suppress(Exception):
            if pipeline.isRunning():
                pipeline.stop()
        with suppress(Exception):
            device.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
