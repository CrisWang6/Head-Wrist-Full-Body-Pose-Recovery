import csv
from pathlib import Path

import cv2


DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0715\001")
ALIGNED = DATASET / "aligned_data" / "aligned_30hz.csv"
TIMESTAMPS = DATASET / "timestamps.csv"
SRC_VIDEO = DATASET / "mp4" / "module01_D45D2E00_CAM_B.mp4"
OUT_VIDEO = Path(__file__).with_name("aligned_module01_CAM_B_preview.mp4")

MODULE = 1
CAMERA = "CAM_B"
OUT_FPS = 30.0
OUT_WIDTH = 960
MAX_PREVIEW_SECONDS = 60.0


def read_raw_frame_indices() -> dict[str, int]:
    rows = []
    with TIMESTAMPS.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if int(row["module"]) == MODULE and row["camera"] == CAMERA:
                rows.append(row)
    rows.sort(key=lambda r: float(r["exposure_middle_ts_ms"]))
    return {r["exposure_middle_ts_ms"]: i for i, r in enumerate(rows)}


def read_aligned_targets(raw_index_by_exposure: dict[str, int]) -> list[int | None]:
    col = f"module{MODULE:02d}_{CAMERA}_exposure_middle_ts_ms"
    targets = []
    with ALIGNED.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            exposure = row.get(col, "")
            targets.append(raw_index_by_exposure.get(exposure))

    last = None
    for i, value in enumerate(targets):
        if value is None:
            targets[i] = last
        else:
            last = value
    first_valid = next((v for v in targets if v is not None), 0)
    return [first_valid if v is None else v for v in targets]


def main() -> None:
    raw_index_by_exposure = read_raw_frame_indices()
    targets = read_aligned_targets(raw_index_by_exposure)
    max_frames = int(round(MAX_PREVIEW_SECONDS * OUT_FPS))
    targets = targets[:max_frames]
    cap = cv2.VideoCapture(str(SRC_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {SRC_VIDEO}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_h = int(round(src_h * OUT_WIDTH / src_w))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_VIDEO), fourcc, OUT_FPS, (OUT_WIDTH, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer: {OUT_VIDEO}")

    current_idx = -1
    current_frame = None
    written = 0
    missing_holds = 0
    for target in targets:
        if current_frame is None or target > current_idx:
            while current_idx < target:
                ok, frame = cap.read()
                current_idx += 1
                if not ok:
                    break
                current_frame = frame
        else:
            missing_holds += 1
        if current_frame is None:
            raise RuntimeError("No source frame available")
        frame_out = cv2.resize(current_frame, (OUT_WIDTH, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame_out)
        written += 1
        if written % 300 == 0:
            print(f"written={written}/{len(targets)} source_idx={current_idx}", flush=True)

    cap.release()
    writer.release()
    print(f"wrote {OUT_VIDEO}")
    print(f"frames={written} fps={OUT_FPS} seconds={written / OUT_FPS:.3f} size={OUT_WIDTH}x{out_h} held_previous_frames={missing_holds}")


if __name__ == "__main__":
    main()
