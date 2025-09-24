#!/usr/bin/env python
# Backend/model.py — DeepLabv3+ (baseline) and SegFormer-B2 (improved)
# Repo defaults: Backend/images, Backend/masks_multi, Backend/splits

import argparse
from pathlib import Path
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- Albumentations for augmentations ---
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    raise SystemExit("Install albumentations & opencv: pip install albumentations opencv-python")

# ===== Paths relative to repo =====
ROOT = Path(__file__).resolve().parents[1]
BROOT = ROOT / "Backend"
DEF_IMAGES = BROOT / "images"
DEF_MASKS = BROOT / "masks_multi"
DEF_SPLITS = BROOT / "splits"
DEF_OVERLAYS = BROOT / "overlays"
DEF_METRICS = BROOT / "docs" / "metrics"

PALETTE = {
    0: (0, 0, 0),        # background
    1: (0, 255, 0),      # healthy
    2: (255, 0, 0),      # bleached
    3: (255, 255, 0),    # dead
}

# =====================
# Dataset + transforms
# =====================
def make_train_tfms(size=512):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ColorJitter(0.3, 0.3, 0.3, 0.2, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.MotionBlur(p=0.2), A.GaussianBlur(p=0.2), A.GaussNoise(p=0.2),
        # ✅ fixed API (quality_range is a tuple, compression_type is str)
        A.ImageCompression(quality_range=(60, 100), compression_type="jpeg", p=0.3),
        A.Resize(size, size),
        ToTensorV2(),
    ])

def make_val_tfms(size=512):
    return A.Compose([A.Resize(size, size), ToTensorV2()])

class ReefSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, ids, transform):
        self.img_dir, self.mask_dir = Path(img_dir), Path(mask_dir)
        self.ids = [i.strip() for i in ids if i.strip()]
        self.t = transform

    def __len__(self): return len(self.ids)

    def _read_image(self, stem: str):
        for ext in [".jpg", ".png"]:
            p = self.img_dir / f"{stem}{ext}"
            if p.exists():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise FileNotFoundError(f"Image not found for {stem} in {self.img_dir}")

    def _read_mask(self, stem: str):
        p = self.mask_dir / f"{stem}.png"
        if not p.exists():
            raise FileNotFoundError(f"Mask not found for {stem}.png in {self.mask_dir}")
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, i):
        stem = self.ids[i]
        img = self._read_image(stem)
        mask = self._read_mask(stem).astype(np.int64)
        out = self.t(image=img, mask=mask)
        return out["image"].float() / 255.0, out["mask"].long(), stem

# ==============
# Models
# ==============
def make_deeplab(num_classes: int):
    from torchvision.models.segmentation import deeplabv3_resnet50
    return deeplabv3_resnet50(weights=None, num_classes=num_classes)

def make_segformer_b2(num_classes: int):
    try:
        from segformer_pytorch import Segformer  # pip install segformer-pytorch timm
    except ImportError:
        raise ImportError("Install segformer: pip install segformer-pytorch timm")
    return Segformer(
        dims=(64, 128, 320, 512), heads=8, ff_expansion=4,
        reduction_ratio=2, num_layers=2, decoder_dim=256, num_classes=num_classes
    )

def select_model(name: str, num_classes: int):
    name = name.lower()
    if name in {"deeplab", "deeplabv3"}:
        return make_deeplab(num_classes)
    if name in {"segformer", "segformer-b2", "b2"}:
        return make_segformer_b2(num_classes)
    raise ValueError(f"Unknown model {name}")

# ==============
# Loss + metrics
# ==============
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-5): super().__init__(); self.eps = eps
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        targets_oh = torch.nn.functional.one_hot(targets, probs.shape[1]).permute(0,3,1,2).float()
        dims = (0,2,3)
        num = 2 * (probs * targets_oh).sum(dims)
        den = probs.sum(dims) + targets_oh.sum(dims) + self.eps
        return 1 - (num / den).mean()

def compute_confusion(pred, gt, n_classes=4, ignore=255):
    mask = (gt != ignore)
    pred = pred[mask]; gt = gt[mask]
    cm = np.bincount(gt*n_classes + pred, minlength=n_classes*n_classes)
    return cm.reshape(n_classes, n_classes)

def miou_from_cm(cm: np.ndarray) -> Tuple[float, np.ndarray]:
    iou = np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-9)
    return float(np.nanmean(iou)), iou

def bleaching_percentage(mask_pred: np.ndarray) -> float:
    coral = (mask_pred==1) | (mask_pred==2) | (mask_pred==3)
    bleached = (mask_pred==2)
    total = int(coral.sum())
    return float(bleached.sum()*100.0/total) if total>0 else 0.0

# ============
# Training/Eval
# ============
def train_one_epoch(model, dl, opt, ce, dice, device="cuda"):
    model.train()
    total=0.0
    for x,y,_ in dl:
        x,y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, dict) and "out" in out: out = out["out"]
        loss = 0.5*ce(out,y) + 0.5*dice(out,y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * x.size(0)
    return total / len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl, device="cuda", num_classes=4) -> Dict:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    bleaches: List[float] = []
    for x,y,_ in dl:
        x = x.to(device)
        out = model(x)
        if isinstance(out, dict) and "out" in out: out = out["out"]
        pred = torch.argmax(out, dim=1).cpu().numpy()
        y_np = y.numpy()
        for pi, gi in zip(pred, y_np):
            cm += compute_confusion(pi, gi, n_classes=num_classes)
            bleaches.append(bleaching_percentage(pi))
    miou_mean, ious = miou_from_cm(cm)
    return {"mIoU": float(miou_mean),
            "per_class_IoU": [float(x) for x in ious],
            "bleach_pct_mean": float(np.mean(bleaches) if bleaches else 0.0)}

def save_metrics(metrics: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f: json.dump(metrics, f, indent=2)

# ========
# CLI main
# ========
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="deeplab",
                   choices=["deeplab","segformer","segformer-b2","b2"])
    p.add_argument("--images", type=str, default=str(DEF_IMAGES))
    p.add_argument("--masks",  type=str, default=str(DEF_MASKS))
    p.add_argument("--splits", type=str, default=str(DEF_SPLITS))
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--out-metrics", type=str, default=str(DEF_METRICS))
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ids = (Path(args.splits)/"train.txt").read_text().splitlines()
    val_ids   = (Path(args.splits)/"val.txt").read_text().splitlines()

    ds_tr = ReefSegDataset(args.images, args.masks, train_ids, make_train_tfms(args.tile_size))
    ds_va = ReefSegDataset(args.images, args.masks, val_ids,   make_val_tfms(args.tile_size))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = select_model(args.model, args.num_classes).to(device)

    ce, dice = nn.CrossEntropyLoss(ignore_index=255), DiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, outdir = -1.0, Path(args.out_metrics)
    outdir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, dl_tr, opt, ce, dice, device)
        metrics = evaluate(model, dl_va, device=device, num_classes=args.num_classes)
        sched.step()

        print(f"[{epoch:03d}] loss={loss:.4f} mIoU={metrics['mIoU']:.4f} "
              f"bleach%~{metrics['bleach_pct_mean']:.2f}")

        save_metrics(metrics, outdir / f"{args.model}_epoch{epoch:03d}.json")
        if metrics["mIoU"] > best:
            best = metrics["mIoU"]
            torch.save(model.state_dict(), BROOT / f"{args.model}_best.pt")

    print(f"Best mIoU={best:.4f} → saved weights to {BROOT}")

if __name__ == "__main__":
    main()
