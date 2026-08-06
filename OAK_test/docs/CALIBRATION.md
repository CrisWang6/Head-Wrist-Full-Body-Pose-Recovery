# 标定 calibrate

这个目录集中放标定相关内容：采集脚本、DepthAI EEPROM 导出工具、内外参数据集，以及已经标好的参数文件。

## 目录结构

- `capture_intrinsics_dataset.py`: 内参数据采集入口。
- `inspect_device_calibration.py`: 读取当前设备 EEPROM 标定。
- `export_depthai_calibration.py`: 从内参 JSON 生成 DepthAI calibration JSON，可选写入 EEPROM。
- `dataset/intrinsics_dataset/`: 内参数据集，包含 `head_dataset`、`left_wrist_dataset`、`right_wrist_dataset`、`handle_dataset`。
- `dataset/extrinsics_dataset/`: 外参数据集，保留旧外参采集数据。
- `parameters/intrinsics/`: 已标好的内参结果，包括 Kalibr 输出和脚本生成的 JSON。
- `parameters/extrinsics/`: 已标好的外参结果。
- `parameters/device/`: DepthAI EEPROM 备份、检查报告和导出的 DepthAI calibration JSON。

## 默认标定板

- AprilTag family: `tag36h11`
- grid: `6 x 6`
- tag size: `35.2 mm`
- Kalibr `tagSpacing`: `0.3`
- physical gap: `10.56 mm`
- start id: `0`
- end id: `35`

脚本里的 `--tag-spacing` 按 Kalibr 习惯表示 ratio，也就是白色间距 / tag 黑色方块边长。当前板子 `10.56 / 35.2 = 0.3`。

## 1. 采集内参数据

所有模组默认采集四路 `CAM_A CAM_B CAM_C CAM_D`：

```bash
python scripts/calibration/capture_intrinsics_dataset.py \
  --in \
  --left-wrist
```

头部、右腕或 ABCD 手柄分别使用：

```bash
python scripts/calibration/capture_intrinsics_dataset.py --module head
python scripts/calibration/capture_intrinsics_dataset.py --module right_wrist
python scripts/calibration/capture_intrinsics_dataset.py --module handle
```

如果采外参数据集，用 `--ex`。例如头部模组会保存到 `data/calibration/extrinsics_dataset/head_dataset`：

```bash
python scripts/calibration/capture_intrinsics_dataset.py \
  --ex \
  --head
```

如果只是想确认四路 DepthAI stream 能不能同时起来，不开 GUI、不做检测：

```bash
python scripts/calibration/capture_intrinsics_dataset.py \
  --module handle \
  --headless-test-seconds 5
```

窗口里绿色点和 tag id 表示 AprilGrid 检测成功。按空格保存同步样本，按 `q` 退出。即使所有相机都没有识别到 Tag，按空格仍会保存四路原始图像，并在样本的 `metadata.json` 中记录 `board_detected: false`。预览卡顿时可调大 `--detect-every`，识别困难时可临时加 `--robust-detection`。

脚本会优先使用 `pupil_apriltags` 检测 tag36h11；如果环境里没有这个包才退回 OpenCV `cv2.aruco`。你这类 Kalibr 风格 AprilGrid 建议使用 `camtest` 环境里的 `pupil_apriltags`。

建议每路至少采 35-60 组有效样本，覆盖中心、四角、边缘、近中远距离和不同倾斜姿态。

## 2. 已有参数位置

已标好的内参和 Kalibr 输出在：

```text
configs/calibration/intrinsics
```

已标好的外参和 Kalibr 输出在：

```text
configs/calibration/extrinsics
```

## 3. 导出 DepthAI calibration JSON

```bash
python scripts/calibration/export_depthai_calibration.py \
  configs/calibration/intrinsics/left_wrist/left_wrist_intrinsics_kalibr_omni_1920x1200.json \
  --output configs/calibration/device/depthai_left_wrist_calib.json
```

确认生成的 JSON 没问题后再写 EEPROM：

```bash
python scripts/calibration/export_depthai_calibration.py \
  configs/calibration/intrinsics/left_wrist/left_wrist_intrinsics_kalibr_omni_1920x1200.json \
  --output configs/calibration/device/depthai_left_wrist_calib.json \
  --flash
```
