# 0806dataset — head RGB frames for EgoRear stage-1

Extracted from 0806 batch head elementary `.h265` streams using aligned
`aligned_30hz.csv` exposure timestamps and `timestamps.csv` packet sizes
(`H265CaptureReader` in joint_projection).

## Layout

```
frames/
  wu/0712_033709/CAM_A/{seq:06d}.jpg
  wu/0712_033709/CAM_D/{seq:06d}.jpg
  wrist/0712_032704/CAM_A/{seq:06d}.jpg
  wrist/0712_032704/CAM_D/{seq:06d}.jpg
  ankle/0712_033034/CAM_A/{seq:06d}.jpg
  ankle/0712_033034/CAM_D/{seq:06d}.jpg
manifest/
  wu.json
  wrist.json
  ankle.json
```

`{seq}` is the aligned timeline index from `aligned_30hz.csv` (strict temporal order).

## EgoRear stage-1 usage

Point `MultiViewHeatmapDataset.frame_root` at `frames/` and set each label NPZ
`source_render_dir` to the sample path, e.g. `wu/0712_033709`. Camera names in
labels should be `CAM_A` and `CAM_D`. Training resizes to `(456, 256)` by default.

## Re-run extraction

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sapiens2
python /home/gaoweijian/0806_batch/repo/test_code/joint_projection/extract_0806_head_frames.py \
  --batch-root /home/gaoweijian/0806_batch \
  --output-root /home/gaoweijian/0806dataset \
  --skip-existing

# Verify only (no decode):
python .../extract_0806_head_frames.py --verify-only
```

Native resolution JPG (1920x1200); resize at train time.
