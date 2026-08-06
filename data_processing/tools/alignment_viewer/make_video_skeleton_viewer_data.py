import bisect
import csv
import json
from pathlib import Path


DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0715\001")
ALIGNED = DATASET / "aligned_data" / "aligned_30hz.csv"
TIMESTAMPS = DATASET / "timestamps.csv"
RAW_VIDEO = DATASET / "module01_D45D2E00_CAM_B.h265"
MOCAP = DATASET / "aligned_data" / "mocap_source_csv" / "mocap_joints_wide.csv"
REPORT = DATASET / "aligned_data" / "aligned_30hz_report.json"
OUT = Path(__file__).with_name("video_skeleton_data.js")
OUT_REPORT = Path(__file__).with_name("video_mocap_mapping_report.json")

VIDEO_MODULE = 1
VIDEO_CAMERA = "CAM_B"
MAX_PREVIEW_SECONDS = 60.0

JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "Head_End",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightFoot_End",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftFoot_End",
]

EDGES = [
    ["Hips", "Spine"],
    ["Spine", "Spine1"],
    ["Spine1", "Spine2"],
    ["Spine2", "Neck"],
    ["Neck", "Head"],
    ["Head", "Head_End"],
    ["Spine2", "RightShoulder"],
    ["RightShoulder", "RightArm"],
    ["RightArm", "RightForeArm"],
    ["RightForeArm", "RightHand"],
    ["Spine2", "LeftShoulder"],
    ["LeftShoulder", "LeftArm"],
    ["LeftArm", "LeftForeArm"],
    ["LeftForeArm", "LeftHand"],
    ["Hips", "RightUpLeg"],
    ["RightUpLeg", "RightLeg"],
    ["RightLeg", "RightFoot"],
    ["RightFoot", "RightFoot_End"],
    ["Hips", "LeftUpLeg"],
    ["LeftUpLeg", "LeftLeg"],
    ["LeftLeg", "LeftFoot"],
    ["LeftFoot", "LeftFoot_End"],
]


def fnum(value: str, digits: int):
    if value == "" or value is None:
        return None
    return round(float(value), digits)


def find_first_vps_offset(scan_bytes: int = 8_000_000) -> int:
    with RAW_VIDEO.open("rb") as f:
        data = f.read(scan_bytes)
    start = 0
    while True:
        idx = data.find(b"\x00\x00\x00\x01", start)
        if idx < 0 or idx + 5 >= len(data):
            raise RuntimeError(f"No HEVC VPS found near the start of {RAW_VIDEO}")
        if ((data[idx + 4] >> 1) & 0x3F) == 32:
            return idx
        start = idx + 4


def read_video_packet_mapping() -> tuple[dict[str, int], int, float, float]:
    rows = []
    with TIMESTAMPS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if int(row["module"]) == VIDEO_MODULE and row["camera"] == VIDEO_CAMERA:
                rows.append(row)
    rows.sort(key=lambda row: float(row["exposure_middle_ts_ms"]))

    vps_offset = find_first_vps_offset()
    cumulative_bytes = 0
    first_mp4_packet = None
    for i, row in enumerate(rows):
        packet_end = cumulative_bytes + int(row["bytes"])
        if cumulative_bytes <= vps_offset < packet_end:
            first_mp4_packet = i
            break
        cumulative_bytes = packet_end
    if first_mp4_packet is None:
        raise RuntimeError("The first VPS does not fall inside a timestamped CAM_B packet")

    mapping = {
        row["exposure_middle_ts_ms"]: i - first_mp4_packet
        for i, row in enumerate(rows)
        if i >= first_mp4_packet
    }
    camera_origin_ms = float(rows[0]["exposure_middle_ts_ms"])
    first_decodable_ms = float(rows[first_mp4_packet]["exposure_middle_ts_ms"])
    return mapping, first_mp4_packet, camera_origin_ms, first_decodable_ms


def load_mocap() -> tuple[list[float], list[list[float | None]]]:
    times = []
    poses = []
    with MOCAP.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            times.append(float(row["time_sec"]))
            pose = []
            for joint in JOINTS:
                pose.extend(
                    [
                        fnum(row.get(f"{joint}_world_x"), 2),
                        fnum(row.get(f"{joint}_world_y"), 2),
                        fnum(row.get(f"{joint}_world_z"), 2),
                    ]
                )
            poses.append(pose)
    return times, poses


def nearest_mocap_index(times: list[float], target: float) -> int:
    right = bisect.bisect_left(times, target)
    if right <= 0:
        return 0
    if right >= len(times):
        return len(times) - 1
    left = right - 1
    return left if abs(times[left] - target) <= abs(times[right] - target) else right


def main() -> None:
    packet_mapping, first_mp4_packet, camera_origin_ms, first_decodable_ms = read_video_packet_mapping()
    mocap_times, mocap_poses = load_mocap()
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    alignment = report["alignment_parameters"]
    global_scale = float(alignment["global_scale"])
    global_offset = float(alignment["global_mocap_offset_sec"])

    frames = []
    seq = []
    mocap_time = []
    mocap_target = []
    timeline_sec = []
    camera_exposure_ms = []
    video_frame_missing = []
    bounds = {
        "min_x": float("inf"),
        "max_x": float("-inf"),
        "min_y": float("inf"),
        "max_y": float("-inf"),
        "min_z": float("inf"),
        "max_z": float("-inf"),
    }

    with ALIGNED.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        first_camera_time = None
        for row in reader:
            slot_value = row.get(f"module{VIDEO_MODULE:02d}_{VIDEO_CAMERA}_slot_exposure_middle_ts_ms", "")
            if not slot_value:
                continue
            slot_time = float(slot_value)
            exposure_value = row.get(f"module{VIDEO_MODULE:02d}_{VIDEO_CAMERA}_exposure_middle_ts_ms", "")
            source_index = packet_mapping.get(exposure_value)
            if first_camera_time is None and source_index is None:
                continue
            if first_camera_time is None:
                first_camera_time = slot_time
            elapsed = (slot_time - first_camera_time) / 1000.0
            if elapsed > MAX_PREVIEW_SECONDS:
                break
            seq_i = int(row["seq"])
            seq.append(seq_i)
            target = global_offset + ((slot_time - camera_origin_ms) / 1000.0) * global_scale
            mocap_i = nearest_mocap_index(mocap_times, target)
            nearest_time = mocap_times[mocap_i]
            mocap_time.append(round(nearest_time, 6))
            mocap_target.append(round(target, 6))
            timeline_sec.append(round(elapsed, 6))
            camera_exposure_ms.append(round(slot_time, 3))
            video_frame_missing.append(source_index is None)

            pose = mocap_poses[mocap_i]
            for joint_i in range(len(JOINTS)):
                x, y, z = pose[joint_i * 3 : joint_i * 3 + 3]
                if x is not None:
                    bounds["min_x"] = min(bounds["min_x"], x)
                    bounds["max_x"] = max(bounds["max_x"], x)
                    bounds["min_y"] = min(bounds["min_y"], y)
                    bounds["max_y"] = max(bounds["max_y"], y)
                    bounds["min_z"] = min(bounds["min_z"], z)
                    bounds["max_z"] = max(bounds["max_z"], z)
            frames.append(pose)

    payload = {
        "dataset": str(DATASET),
        "videoLabel": "module01 CAM_B",
        "rows": len(frames),
        "seq": seq,
        "mocapTime": mocap_time,
        "mocapTarget": mocap_target,
        "timelineSec": timeline_sec,
        "cameraExposureMs": camera_exposure_ms,
        "videoFrameMissing": video_frame_missing,
        "joints": JOINTS,
        "edges": EDGES,
        "frames": frames,
        "bounds": bounds,
        "alignment": {
            "method": alignment["method"],
            "globalScale": alignment["global_scale"],
            "globalOffsetSec": alignment["global_mocap_offset_sec"],
            "globalVectorCorr": alignment["global_vector_corr"],
            "crossValidationCorr": alignment["cross_validation"]["mean_held_rotation_score"],
        },
        "mapping": {
            "module": VIDEO_MODULE,
            "camera": VIDEO_CAMERA,
            "mappedRows": len(timeline_sec),
            "previewSeconds": timeline_sec[-1] if timeline_sec else 0,
            "timeSource": f"module{VIDEO_MODULE:02d}_{VIDEO_CAMERA}_slot_exposure_middle_ts_ms",
            "firstMp4RawPacket": first_mp4_packet,
            "cameraOriginMs": camera_origin_ms,
            "firstDecodableExposureMs": first_decodable_ms,
            "trimmedCameraTimeMs": first_decodable_ms - camera_origin_ms,
            "note": "MP4 frames start at the first timestamp packet containing an HEVC VPS. Mocap poses are selected directly from module01 exposure time through the accepted global clock model.",
        },
    }
    OUT.write_text(
        "window.VIDEO_SKELETON_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    mapping_report = {
        "video": str(RAW_VIDEO),
        "timestamp_rows_before_first_decodable_frame": first_mp4_packet,
        "camera_origin_ms": camera_origin_ms,
        "first_decodable_exposure_ms": first_decodable_ms,
        "trimmed_camera_time_ms": first_decodable_ms - camera_origin_ms,
        "old_viewer_error": "The old viewer treated MP4 frame 0 as raw timestamp row 0.",
        "new_mapping": "MP4 frame 0 maps to the first raw packet containing the HEVC VPS; mocap time is derived directly from module01 exposure time.",
        "preview_rows": len(frames),
        "preview_duration_sec": payload["mapping"]["previewSeconds"],
    }
    OUT_REPORT.write_text(json.dumps(mapping_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"frames={len(frames)} preview_seconds={payload['mapping']['previewSeconds']:.6f}")
    print(json.dumps(mapping_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
