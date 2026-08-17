# THuman2.1 Blender Asset Preparation

This document covers the short-test path for preparing THuman2.1 appearance assets before large-scale Isaac Sim generation.

## Inputs

Place the two archives under:

```text
data/thuman/archives/THuman2.1_Release_.7z
data/thuman/archives/THuman2.1_Release Smpl-X Paras_new.7z
```

These archives are data assets and should not be committed.

## Short Test

Prepare one subject:

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/prepare_thuman_blender_assets.py \
  --count 1 \
  --overwrite-extract \
  --overwrite-assets
```

Run a headless Blender import smoke test on one prepared subject:

```bash
SUBJECT_DIR=$(find smplx_models/thuman_appearances -mindepth 1 -maxdepth 1 -type d | sort | head -1)
blender -b --python scripts/blender_thuman_smoke_test.py -- \
  --asset-dir "$SUBJECT_DIR"
```

The smoke test writes `blender_smoke_test.json` in the prepared subject directory.

When the full scan archive is still transferring, extract one subject locally and copy it to:

```text
data/thuman/extracted_subset/scans/model/<subject>
```

Then run the same preparation against the extracted scan root:

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/prepare_thuman_blender_assets.py \
  --scan-root data/thuman/extracted_subset/scans \
  --smplx-archive "data/thuman/archives/THuman2.1_Release Smpl-X Paras_new.7z" \
  --subjects 0000 \
  --overwrite-assets \
  --strict-textures
```

## Full Preparation

After the short test passes, increase the count:

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/prepare_thuman_blender_assets.py \
  --count 100 \
  --overwrite-assets
```

The prepared appearance library can be used by render commands through:

```text
--appearance-root /home/gaoweijian/Simulation/smplx_models/thuman_appearances
```

## Notes

- The preparation script only extracts selected subject folders, so it is safe for quick validation.
- It uses `py7zr` when no system `7z` binary is available.
- The prepared library stores SMPL-X files plus the nearest THuman textured OBJ/MTL/image bundle when available.
