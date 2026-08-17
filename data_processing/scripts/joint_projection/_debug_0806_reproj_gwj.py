import json
import sys
from pathlib import Path

import numpy as np

JP = "/home/gaoweijian/0810_batch/repo/test_code/joint_projection"
sys.path.insert(0, JP)
from constants_0806_training import LABEL_NPZ_NAME, LIMB_ORDER
from render_stage3_dual_skeleton_yaw import load_playback_records

DS = Path("/home/gaoweijian/0806dataset")
_, _, pred = load_playback_records(DS / "eval/v31/stage3_aligned_test_3d_viz/skeleton_playback_stage3_test_pred.json")
lookup = {}
paths = {}
for limb in LIMB_ORDER:
    data = np.load(DS / "labels" / limb / LABEL_NPZ_NAME, allow_pickle=True)
    for fi, s in enumerate(data["source_aligned_seq"].astype(int)):
        lookup[int(s)] = limb
        paths[int(s)] = (str(data["image_paths"][fi, 0]), str(data["image_paths"][fi, 1]))

print("playback frames", len(pred))
print("first5 seq", [pred[i]["seq"] for i in range(5)])
print("limbs first5", [lookup.get(int(pred[i]["seq"])) for i in range(5)])

pre = json.loads((DS / "pre_limb_map.json").read_text())
ankle_rows = set()
for line in Path(pre["ankle"]).read_text().splitlines():
    if line.strip():
        ankle_rows.add(int(json.loads(line)["seq"]))
print("pre_limb ankle rows", len(ankle_rows))

for i in range(5):
    s = int(pred[i]["seq"])
    img_a = paths.get(s, ("", ""))[0]
    print(
        i,
        "seq",
        s,
        "limb",
        lookup.get(s),
        "pre",
        s in ankle_rows,
        "img_exists",
        Path(img_a).is_file() if img_a else False,
        "img",
        img_a[:80] if img_a else "",
    )

# aligned status check
import csv
from multiview_geometry import load_json, rigid_world_transform
from render_multiview_to_head import head_mocap_correction, resolve_repo_path

cfg = load_json(f"{JP}/configs/0806_ankle_dual_external_mocap.json")
head_cfg = cfg["head"]
data_root = Path(pre["ankle"]).parent.parent.parent
aligned_path = data_root / "aligned_data" / "aligned_30hz.csv"
with aligned_path.open(encoding="utf-8-sig") as f:
    aligned = {int(r["seq"]): r for r in csv.DictReader(f)}
prefix = head_cfg.get("rigid_prefix", "mocap_CH3_08")
import numpy as np
from constants_0806_training import LIMB_ORDER, LABEL_NPZ_NAME

split = np.load(DS / "splits/pack30_v31.npz")
test = split["test_indices"].astype(int)
print("test frames", len(test))
offsets = {}
off = 0
for limb in LIMB_ORDER:
    n = len(np.load(DS / "labels" / limb / LABEL_NPZ_NAME)["source_aligned_seq"])
    offsets[limb] = (off, off + n)
    off += n
print("offsets", offsets)
for gi in test[:10]:
    for limb, (a, b) in offsets.items():
        if a <= gi < b:
            print("gi", gi, "->", limb)
            break
