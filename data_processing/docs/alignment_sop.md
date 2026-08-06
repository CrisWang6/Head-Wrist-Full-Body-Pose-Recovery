# Align Data SOP

适用目标：把一组 9cam 视频时间戳、3 个 OAK IMU CSV、动捕 FBX/BVH 源数据统一对齐，输出一个 30Hz 总表。

最终输出目录：

```text
<dataset>/aligned_data/
```

最终输出文件：

```text
aligned_30hz.csv
aligned_30hz_report.json
```

## 1. 输入源数据

每组数据目录应至少包含：

```text
timestamps.csv
summary.json
module01_*_CAM_A/B/C.h265 or .mp4
module02_*_CAM_A/B/C.h265 or .mp4
module03_*_CAM_A/B/C.h265 or .mp4
module01_*_imu.csv
module02_*_imu.csv
module03_*_imu.csv
*.fbx
```

IMU CSV 使用原始字段：

```text
module,mxid,host_ts_ms,
accel_seq,accel_device_ts_ms,ax_m_s2,ay_m_s2,az_m_s2,
gyro_seq,gyro_device_ts_ms,gx_rad_s,gy_rad_s,gz_rad_s
```

相机与 IMU、动捕对齐统一使用 `timestamps.csv` 中的原始 OAK 时间字段：

```text
device_ts_ms
```

`exposure_start_ts_ms`、`exposure_middle_ts_ms`、`exposure_end_ts_ms` 均由 `device_ts_ms` 和曝光时长推算，不作为独立有效时间源，不参与对齐，也不写入最终 `aligned_30hz.csv`。

## 2. 生成 30Hz 相机主表

从 `timestamps.csv` 生成按 `seq` 对齐的 9cam 宽表。

每个 `seq` 一行，每路相机保留：

```text
moduleXX_CAM_Y_device_ts_ms
```

模块和相机身份已包含在列名前缀中，因此不重复保存 `module_id`、`camera_name`。丢帧处理使用的 gap 标志、重建槽位时间等字段仅供脚本内部计算，不写入最终 CSV。

如果某一路相机中间丢帧，用 `device_ts_ms` 的约 33ms 帧周期检测 gap。短 gap 按普通丢帧处理：接近 66ms 时插入 1 行空值，保证后续帧不串位。长 gap 不展开：当 gap 超过 `2.5 * 相机帧周期`（30Hz 时约 83ms）时，视为采集中断/系统卡顿，只在报告中记录该长 gap 和估计缺失槽数，不把这一段展开成大量空行。

最终交付前要从尾部删除不满 9 路的残行，直到最后一个 `seq` 九路相机都有数据。

## 3. 合并 IMU 原始数据

对 30Hz 主表的每个 `seq`，每个 module 先计算该 module 三路相机 `device_ts_ms` 的中位数：

```text
cam_device_ts_ms = median(CAM_A, CAM_B, CAM_C)
```

然后在同 module 的 IMU CSV 中寻找最近的 `gyro_device_ts_ms` 行。

最终 `aligned_30hz.csv` 中直接复制该 IMU 原始行，不保存二次计算的 IMU 结果。

每个 module 的 IMU 输出字段统一加前缀：

```text
moduleXX_imu_<原始字段名>
```

## 4. 解析动捕源数据

FBX 如果是 ASCII 格式，Blender 可能无法直接导入。此时从 FBX 中解析：

```text
Model
AnimationCurveNode
AnimationCurve
Connections
```

根据骨骼父子层级和 local T/R 曲线做 FK，导出动捕源关节位姿表：

```text
mocap_joints_wide.csv
```

该表只作为 FBX 的表格化源数据表示，不使用腕部微分等二次结果作为最终输出字段。

## 5. IMU 与动捕时间对齐

时间搜索必须直接使用左右腕 IMU 原始 CSV 中的全部角速度样本：

```text
gyro_device_ts_ms, gx_rad_s, gy_rad_s, gz_rad_s
```

不要先把 IMU 降采样到 30Hz 相机 `seq`。原始采样是不等间隔的，必须保留每条 `gyro_device_ts_ms`；相机丢帧和补空也不能进入 IMU-mocap 时间搜索。

### 5.1 统一两块 IMU 的时间零点

对 module02、module03 分别取第一次触发时 CAM_A/B/C 的 `device_ts_ms` 中位数作为该模块的时钟原点：

```text
imu_elapsed_sec = (gyro_device_ts_ms - first_camera_device_ts_median_ms) / 1000
```

两块 OAK 的 device clock 数值不同，但第一次外部触发相同，因此这样可把左右腕 IMU 放到同一个采集时间轴。

### 5.2 用腕部局部坐标系角速度

从 mocap 左右腕世界四元数计算腕部自身坐标系角速度：

```text
dq_body = inverse(q[i]) * q[i+1]
mocap_body_gyro = log(dq_body) / dt
```

不能直接把 mocap 世界系角速度和 IMU 传感器系角速度按固定单轴比较。每只腕部允许拟合一个固定的三维旋转矩阵，把 mocap 腕部坐标映射到该 IMU 的传感器坐标；该旋转只描述安装方向，不参与时间缩放。

### 5.3 全局时间模型与搜索

整段数据只允许一个 `global_scale` 和一个 `global_offset_sec`：

```text
mocap_time_sec = global_offset_sec + global_scale * imu_elapsed_sec
```

搜索规则：

- module02、module03 必须共享同一组 `scale + offset`，左右手不能各用一组参数。
- 同时测试 `module02->LeftHand/module03->RightHand` 及左右互换，选择全局分数高者。
- 角速度模长只用于粗搜索候选；最终参数必须由左右腕三轴向量的整段相关性决定。
- `global_scale` 的中心由 mocap 与原始 IMU 时长比给出，默认在中心前后约 `0.025` 内搜索；不能固定复用其他数据集的比例。
- 初次三轴搜索后，必须对多个 60s 高动态窗口分别求残余 lag，并拟合 `residual_lag = intercept + slope * time`。用 `offset += intercept`、`scale += slope` 消除固定偏移和随时间漂移，再以原始 IMU 全采样率对 `scale + offset` 做二维精搜；不能只围绕粗搜索结果继续缩小步长。
- 将整段按 60s 切窗，偶数窗拟合坐标旋转并在奇数窗评分，再反向验证。
- 固定最终参数后，每个 60s 窗以不大于 `2ms` 的步长在小范围搜索残余 lag，用其 P90 作为时间精度指标。

当前 `0715/001` 原始时间戳结果：

```text
module02 -> LeftHand
module03 -> RightHand
global_scale = 0.9999835290
global_offset_sec = 0.804553
global three-axis vector correlation = 0.958502
held-window cross-validation correlation = 0.958230
60s local residual lag: median_abs = 2.0ms, P90 = 7.8ms, max = 8.0ms
```

本项目的接受标准：

- 整段三轴相关性和隔窗交叉验证相关性均应大于 `0.90`。
- 多个 60s 高动态窗口的残余 lag 应围绕 0 分布，P90 目标不超过 `50ms`。
- 若只在单个波段得到高相关、左右腕要求不同 offset、或交叉验证明显下降，则判定为错误周期匹配，不能输出 aligned 数据。

## 6. 合并动捕源数据

对每个 30Hz `seq`：

1. 对每个 module 取该 `seq` 三路相机 `device_ts_ms` 的中位数，减去第 5.1 节对应的时钟原点。
2. 使用统一的 `global_scale + global_offset_sec` 映射 module02 和 module03 的相机实际经过时间，并取两者平均作为目标 mocap 时间：

```text
mocap_time_sec_target
```

3. 在 `mocap_joints_wide.csv` 中找最近的 mocap 源行。
4. 如果目标 mocap 时间超出 mocap 源文件范围，`mocap_valid=0`，mocap 字段留空。
5. 如果有效，`mocap_valid=1`，直接复制最近 mocap 源行，字段统一加前缀：

```text
mocap_<原始字段名>
```

最终只保留源数据字段，不把腕部微分、相关系数曲线、叠图数据等中间结果写入 `aligned_30hz.csv`。

## 7. 裁剪共同有效时间段

最终 `aligned_30hz.csv` 只保留 camera、IMU、mocap 三套系统共同有效的时间范围。

规则：

```text
起点 = 三套系统中最晚开始的时间
终点 = 三套系统中最早结束的时间
```

也就是说：

- 如果 camera/IMU 比 mocap 早开始，删除 mocap 开始前的 camera/IMU 行。
- 如果 mocap 比 camera/IMU 早开始，删除 camera/IMU 开始前的 mocap 对应行。
- 结束时同理，只保留最早结束系统之前的共同区间。

实际实现时，先为每个 `seq` 标记：

```text
mocap_valid
```

然后只保留 `mocap_valid=1` 且 IMU/相机均有效的行。

裁剪完成后重新编号：

```text
seq = 0, 1, 2, ... N-1
```

原始相机序号只用于内部处理，不另存 `source_seq`；最终表直接使用重新编号后的 `seq`。

## 8. 最终目录整理

每组数据处理完成后，只保留：

```text
原始视频 h265/mp4
原始 IMU CSV
原始 timestamps.csv
summary.json
原始 mocap FBX/BVH
aligned_data/aligned_30hz.csv
aligned_data/aligned_30hz_report.json
```

删除：

```text
timestamps_by_seq*.csv
timestamps_by_seq*.json/txt
fbx_mocap_csv/
fbx_pose_export/
sync_analysis/
其它叠图、匹配报告、中间导出文件
```

## 9. 质量检查

生成 `aligned_30hz_report.json`，至少记录：

```text
rows
columns
mocap_valid_rows
mocap_invalid_rows
overlap_trim_removed_rows
global_scale
mocap nearest dt median/max_abs
missing_imu_count
source_files
```

合格标准：

- `missing_imu_count = 0`
- 有效 mocap 行的最近帧误差应小于等于 60Hz 半帧，约 `8.33ms`
- 最终 `aligned_30hz.csv` 中不应保留共同时间段外的空 mocap 行
- 裁剪后 `seq` 必须从 0 重新排序，最终 CSV 不保留 `source_seq`
- `global_scale` 必须在本组数据中重新搜索，不直接复用旧数据

