"""Patch train_pose3d.py to use global dataset index instead of seq dict."""
from pathlib import Path

p = Path("/home/gaoweijian/EgoRear_w_hand/experiments/stage3_pose3d/scripts/train_pose3d.py")
text = p.read_text(encoding="utf-8")

old_run = '''def run_epoch(model, loader, pose_by_sequence, device, joint_count, *, optimizer=None, max_steps=0):
    import torch

    training = optimizer is not None
    model.train(training)
    total_error = np.zeros(joint_count, dtype=np.float64)
    total_count = 0
    losses = []
    proposal_losses = []
    for step, batch in enumerate(loader, start=1):
        images = batch["img"].to(device, non_blocking=True).float()
        target = torch.as_tensor(
            np.stack([pose_by_sequence[int(seq)] for seq in batch["frame_idx"].tolist()]),
            device=device,
            dtype=torch.float32,
        )'''

new_run = '''def run_epoch(model, loader, pose_targets, device, joint_count, *, optimizer=None, max_steps=0):
    import torch

    training = optimizer is not None
    model.train(training)
    total_error = np.zeros(joint_count, dtype=np.float64)
    total_count = 0
    losses = []
    proposal_losses = []
    for step, batch in enumerate(loader, start=1):
        images = batch["img"].to(device, non_blocking=True).float()
        global_idx = batch["global_idx"].detach().cpu().numpy().astype(np.int64)
        target = torch.as_tensor(pose_targets[global_idx], device=device, dtype=torch.float32)'''

if old_run not in text:
    raise SystemExit("run_epoch block not found")
text = text.replace(old_run, new_run, 1)

old_dict = '''    pose_by_sequence = {
        int(sequence): pose_values[index]
        for index, sequence in enumerate(pose_frames)
        if pose_valid[index]
    }'''

new_dict = '''    pose_targets = pose_values.astype(np.float32)

    class _IndexedSubset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return int(self.indices.shape[0])

        def __getitem__(self, idx):
            global_idx = int(self.indices[idx])
            sample = dict(self.base[global_idx])
            sample["global_idx"] = np.int64(global_idx)
            return sample

    def _collate_with_global_idx(batch):
        global_idx = torch.as_tensor([int(item.pop("global_idx")) for item in batch])
        collated = torch_collate(batch)
        collated["global_idx"] = global_idx
        return collated'''

if old_dict not in text:
    raise SystemExit("pose_by_sequence block not found")
text = text.replace(old_dict, new_dict, 1)

text = text.replace(
    "train_loader = DataLoader(Subset(dataset, train_indices.tolist()), shuffle=True, **loader_args)",
    "train_loader = DataLoader(_IndexedSubset(dataset, train_indices), shuffle=True, **{k: v for k, v in loader_args.items() if k != 'collate_fn'}, collate_fn=_collate_with_global_idx)",
)
text = text.replace(
    "val_loader = DataLoader(Subset(dataset, val_indices.tolist()), shuffle=False, **loader_args)",
    "val_loader = DataLoader(_IndexedSubset(dataset, val_indices), shuffle=False, **{k: v for k, v in loader_args.items() if k != 'collate_fn'}, collate_fn=_collate_with_global_idx)",
)
text = text.replace("pose_by_sequence", "pose_targets")
text = text.replace(
    "from torch.utils.data import DataLoader, Subset",
    "from torch.utils.data import DataLoader, Subset\n    import torch",
)

p.write_text(text, encoding="utf-8")
print("patched", p)
