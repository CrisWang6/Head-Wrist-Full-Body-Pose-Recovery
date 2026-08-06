import csv
from pathlib import Path

import cv2


DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0715\001")
ALIGNED = DATASET / "aligned_data" / "aligned_30hz.csv"
TIMESTAMPS = DATASET / "timestamps.csv"
RAW_VIDEO = DATASET / "module01_D45D2E00_CAM_B.h265"
SRC_VIDEO = DATASET / "mp4" / "module01_D45D2E00_CAM_B.mp4"
OUT_DIR = Path(__file__).with_name("aligned_cam_b_frames")

MODULE = 1
CAMERA = "CAM_B"
FPS = 30.0
MAX_PREVIEW_SECONDS = 60.0
OUT_WIDTH = 640
JPEG_QUALITY = 78


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


def read_raw_frame_indices() -> tuple[dict[str, int], int, float, float]:
    rows = []
    with TIMESTAMPS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if int(row["module"]) == MODULE and row["camera"] == CAMERA:
                rows.append(row)
    rows.sort(key=lambda r: float(r["exposure_middle_ts_ms"]))

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

    index_by_exposure = {
        row["exposure_middle_ts_ms"]: i - first_mp4_packet
        for i, row in enumerate(rows)
        if i >= first_mp4_packet
    }
    camera_origin_ms = float(rows[0]["exposure_middle_ts_ms"])
    first_decodable_ms = float(rows[first_mp4_packet]["exposure_middle_ts_ms"])
    return index_by_exposure, first_mp4_packet, camera_origin_ms, first_decodable_ms


def read_aligned_targets(raw_index_by_exposure: dict[str, int]) -> list[int | None]:
    col = f"module{MODULE:02d}_{CAMERA}_exposure_middle_ts_ms"
    slot_col = f"module{MODULE:02d}_{CAMERA}_slot_exposure_middle_ts_ms"
    targets = []
    with ALIGNED.open(newline="", encoding="utf-8-sig") as f:
        first_slot = None
        for row in csv.DictReader(f):
            slot_value = row.get(slot_col, "")
            if not slot_value:
                continue
            slot = float(slot_value)
            exposure = row.get(col, "")
            source_index = raw_index_by_exposure.get(exposure)
            if first_slot is None and source_index is None:
                continue
            if first_slot is None:
                first_slot = slot
            if (slot - first_slot) / 1000.0 > MAX_PREVIEW_SECONDS:
                break
            targets.append(source_index)

    last = None
    for i, value in enumerate(targets):
        if value is None:
            targets[i] = last
        else:
            last = value
    first_valid = next((v for v in targets if v is not None), 0)
    return [first_valid if v is None else v for v in targets]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("frame_*.jpg"):
        old.unlink()

    raw_index_by_exposure, first_mp4_packet, camera_origin_ms, first_decodable_ms = read_raw_frame_indices()
    targets = read_aligned_targets(raw_index_by_exposure)
    cap = cv2.VideoCapture(str(SRC_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {SRC_VIDEO}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_h = int(round(src_h * OUT_WIDTH / src_w))
    current_idx = -1
    current_frame = None
    held = 0
    for seq_i, target in enumerate(targets):
        if current_frame is None or target > current_idx:
            while current_idx < target:
                ok, frame = cap.read()
                current_idx += 1
                if not ok:
                    break
                current_frame = frame
        else:
            held += 1
        if current_frame is None:
            raise RuntimeError("No source frame available")
        frame_out = cv2.resize(current_frame, (OUT_WIDTH, out_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT_DIR / f"frame_{seq_i:04d}.jpg"), frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if (seq_i + 1) % 300 == 0:
            print(f"written={seq_i + 1}/{len(targets)} source_idx={current_idx}", flush=True)
    cap.release()
    print(f"wrote {len(targets)} frames to {OUT_DIR}")
    print(f"timeline_seconds={MAX_PREVIEW_SECONDS:.3f} rows={len(targets)} size={OUT_WIDTH}x{out_h} held_previous_frames={held}")
    print(
        f"mp4_first_raw_packet={first_mp4_packet} "
        f"trimmed_camera_time_ms={first_decodable_ms - camera_origin_ms:.3f}"
    )


if __name__ == "__main__":
    main()
