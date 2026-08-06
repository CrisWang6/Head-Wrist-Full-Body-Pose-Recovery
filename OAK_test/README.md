# OAK Test

OAK/DepthAI acquisition, hardware synchronization, IMU recording, device diagnostics and calibration utilities for head/wrist multi-camera capture.

This repository stores source code, lightweight configs and final calibration parameters. It does not store raw recordings, calibration image dumps, ROS bags, logs, secrets or large generated artifacts.

## Project Layout

```text
OAK_test/
|-- scripts/
|   |-- capture/       # OAK FFC preview, sync and encoded capture
|   |-- recording/     # head + wrist multi-camera recording helpers
|   |-- calibration/   # calibration capture/export helpers
|   `-- diagnostics/   # IMU and timing diagnostics
|-- tools/
|   |-- device_ui/     # device discovery and GUI/stress-test tools
|   `-- kalibr/        # Kalibr helper scripts and custom wrappers
|-- configs/
|   `-- calibration/   # final reusable intrinsics/extrinsics
`-- docs/
```

## 2026-08 OAK A/D External Trigger Update

This repository now includes the newer CAM_A/CAM_D external-trigger recorder and related head/wrist recording utilities used by the 2026-08 pose-recovery experiments.

Main additions:

- `scripts/capture/capture_oak_ad_external_trigger.py`: previews first, then restarts the OAK pipeline in external-trigger mode; prints `[ARMED]` before the external trigger should be enabled.
- `scripts/recording/gpio4_start_9cam.py` and `scripts/recording/gpio4-start-9cam.service`: Raspberry Pi GPIO start helper and systemd service.
- `scripts/recording/9cam_record.py`, `scripts/recording/v2_multi_h265_record.py`, `scripts/recording/sync_imu_to_frames.py`: multi-camera recording and IMU/frame synchronization utilities.

## Confirmed Acquisition Notes

- CAM_A/CAM_D are recorded at 1920x1200 with device timestamps.
- Downstream alignment must use timestamp/trigger tables rather than raw H265 packet order.
- If any camera misses a trigger, strict synchronized processing should drop that trigger for every camera.
- A strict deleted-frame video written at 50 fps can look accelerated; source-time review videos should preserve the original trigger timeline when visual speed matters.
- Start the external trigger only after the recorder prints `[ARMED]`.

## Environment

Recommended runtime is Python 3.10+ with DepthAI, OpenCV, NumPy, SciPy, AprilTag utilities and FFmpeg available on the host system. Kalibr-related tooling should remain in its own ROS/Docker environment.

Install from source:

```bash
python -m pip install -e .
```

For GUI tools:

```bash
python -m pip install -e ".[gui]"
```
