import bisect
import csv
import json
from pathlib import Path


DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0715\001")
ALIGNED = DATASET / "aligned_data" / "aligned_30hz.csv"
ALIGNED_REPORT = DATASET / "aligned_data" / "aligned_30hz_report.json"
GLOBAL_REPORT = DATASET / "aligned_data" / "global_imu_mocap_alignment_report.json"
MOCAP_BODY_GYRO = DATASET / "aligned_data" / "mocap_source_csv" / "wrist_mocap_body_gyro.csv"
OUT = Path(__file__).with_name("alignment_data.js")
MAX_PREVIEW_SECONDS = 60.0

JOINTS = [
    "Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "Head_End",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightUpLeg", "RightLeg", "RightFoot", "RightFoot_End",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftFoot_End",
]

EDGES = [
    ["Hips", "Spine"], ["Spine", "Spine1"], ["Spine1", "Spine2"],
    ["Spine2", "Neck"], ["Neck", "Head"], ["Head", "Head_End"],
    ["Spine2", "RightShoulder"], ["RightShoulder", "RightArm"],
    ["RightArm", "RightForeArm"], ["RightForeArm", "RightHand"],
    ["Spine2", "LeftShoulder"], ["LeftShoulder", "LeftArm"],
    ["LeftArm", "LeftForeArm"], ["LeftForeArm", "LeftHand"],
    ["Hips", "RightUpLeg"], ["RightUpLeg", "RightLeg"],
    ["RightLeg", "RightFoot"], ["RightFoot", "RightFoot_End"],
    ["Hips", "LeftUpLeg"], ["LeftUpLeg", "LeftLeg"],
    ["LeftLeg", "LeftFoot"], ["LeftFoot", "LeftFoot_End"],
]

SERIES = []
for module in (1, 2, 3):
    for axis in "xyz":
        SERIES.append({
            "key": f"module{module:02d}_imu_g{axis}_rad_s",
            "label": f"M{module} g{axis}",
            "group": "IMU raw",
            "kind": "imu",
        })
for wrist, short in (("LeftHand", "L"), ("RightHand", "R")):
    for axis in "xyz":
        SERIES.append({
            "key": f"mocap_{wrist}_body_g{axis}",
            "label": f"Mocap {short} body g{axis}",
            "group": "Mocap body",
            "kind": "mocap_body",
            "wrist": wrist,
            "axis": axis,
        })
for module, wrist in ((2, "LeftHand"), (3, "RightHand")):
    for axis in "xyz":
        SERIES.append({
            "key": f"mocap_to_module{module:02d}_g{axis}",
            "label": f"Mocap -> M{module} g{axis}",
            "group": "Mocap mapped to IMU",
            "kind": "mocap_mapped",
            "module": f"module{module:02d}",
            "wrist": wrist,
            "axis": axis,
        })

DEFAULT_VISIBLE = {
    "module02_imu_gz_rad_s",
    "mocap_to_module02_gz",
    "module03_imu_gz_rad_s",
    "mocap_to_module03_gz",
}


def fnum(value, digits=5):
    if value in ("", None):
        return None
    return round(float(value), digits)


def nearest_idx(values, target):
    index = bisect.bisect_left(values, target)
    if index <= 0:
        return 0
    if index >= len(values):
        return len(values) - 1
    return index - 1 if abs(values[index - 1] - target) <= abs(values[index] - target) else index


def read_mocap_body():
    times = []
    vectors = {"LeftHand": [], "RightHand": []}
    with MOCAP_BODY_GYRO.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            times.append(float(row["time_sec"]))
            for wrist in vectors:
                vectors[wrist].append([
                    float(row[f"{wrist}_body_gx_rad_s"]),
                    float(row[f"{wrist}_body_gy_rad_s"]),
                    float(row[f"{wrist}_body_gz_rad_s"]),
                ])
    return times, vectors


def row_vector_times_matrix(vector, matrix):
    return [sum(vector[k] * matrix[k][j] for k in range(3)) for j in range(3)]


def main():
    aligned_report = json.loads(ALIGNED_REPORT.read_text(encoding="utf-8"))
    global_report = json.loads(GLOBAL_REPORT.read_text(encoding="utf-8"))
    mocap_times, mocap_vectors = read_mocap_body()
    rotations = global_report["imu_to_mocap_body_rotations"]

    frames = []
    seq = []
    timeline = []
    mocap_time = []
    values = {spec["key"]: [] for spec in SERIES}
    bounds = {"min_x": float("inf"), "max_x": float("-inf"), "min_y": float("inf"),
              "max_y": float("-inf"), "min_z": float("inf"), "max_z": float("-inf")}

    first_target = None
    with ALIGNED.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            target = float(row["mocap_time_sec_target"])
            if first_target is None:
                first_target = target
            elapsed = target - first_target
            if elapsed > MAX_PREVIEW_SECONDS:
                break
            current_mocap_time = float(row["mocap_nearest_time_sec"])
            source_index = nearest_idx(mocap_times, current_mocap_time)
            body = {wrist: mocap_vectors[wrist][source_index] for wrist in mocap_vectors}
            mapped = {
                "module02": row_vector_times_matrix(body["LeftHand"], rotations["module02"]),
                "module03": row_vector_times_matrix(body["RightHand"], rotations["module03"]),
            }

            seq.append(int(row["seq"]))
            timeline.append(round(elapsed, 6))
            mocap_time.append(round(current_mocap_time, 6))
            pose = []
            for joint in JOINTS:
                point = [fnum(row.get(f"mocap_{joint}_world_{axis}"), 2) for axis in "xyz"]
                pose.extend(point)
                if point[0] is not None:
                    for axis, value in zip("xyz", point):
                        bounds[f"min_{axis}"] = min(bounds[f"min_{axis}"], value)
                        bounds[f"max_{axis}"] = max(bounds[f"max_{axis}"], value)
            frames.append(pose)

            for spec in SERIES:
                if spec["kind"] == "imu":
                    value = fnum(row.get(spec["key"]), 5)
                elif spec["kind"] == "mocap_body":
                    value = round(body[spec["wrist"]]["xyz".index(spec["axis"])], 5)
                else:
                    value = round(mapped[spec["module"]]["xyz".index(spec["axis"])], 5)
                values[spec["key"]].append(value)

    payload = {
        "dataset": str(DATASET),
        "rows": len(frames),
        "seq": seq,
        "timelineSec": timeline,
        "mocapTime": mocap_time,
        "joints": JOINTS,
        "edges": EDGES,
        "frames": frames,
        "series": [
            {
                "key": spec["key"],
                "label": spec["label"],
                "group": spec["group"],
                "defaultVisible": spec["key"] in DEFAULT_VISIBLE,
                "values": values[spec["key"]],
            }
            for spec in SERIES
        ],
        "alignment": {
            "globalScale": global_report["global_scale"],
            "globalOffsetSec": global_report["global_offset_sec"],
            "globalVectorCorr": global_report["global_vector_corr"],
            "crossValidationCorr": global_report["cross_validation"]["mean_held_rotation_score"],
            "localLagP90Ms": global_report["local_lag_validation"]["p90_abs_delta_ms"],
            "mocapNearestMaxMs": aligned_report["mocap_dt_ms_valid_rows"]["max_abs"],
        },
        "bounds": bounds,
    }
    OUT.write_text(
        "window.ALIGN_VIEWER_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")
    print(f"rows={len(frames)} seconds={timeline[-1]:.6f} series={len(SERIES)}")


if __name__ == "__main__":
    main()
