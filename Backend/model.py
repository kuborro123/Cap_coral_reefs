"""
DeepLabV3+ (ResNet-50) training for coral bleaching segmentation — fully annotated.
You will see clear progress bars during training/validation and periodic visual outputs.

Labels expected in masks (grayscale, uint8):
  0 = background
  1 = healthy coral
  2 = bleached coral
  (3 = dead coral)  # optional — set NUM_CLASSES accordingly

HOW TO POINT TO YOUR DATA (READ THIS):
  1) Put the path to YOUR CSV below under CONFIG: CSV_PATH.
     Your CSV must have at least columns: image_path, mask_multi_path.
     If your CSV already has absolute paths, you're done.
     If they are relative, you can set DATA_ROOT to the folder that contains the images and masks;
     the script will join DATA_ROOT / image_path, DATA_ROOT / mask_multi_path.

  2) If your CSV does NOT have a 'split' column (train/val/test), set AUTO_SPLIT=True below.
     Otherwise leave it False and the script will respect your split.

Run examples:
  python deeplab_resnet_training_annotated.py
  (it will use the CONFIG defaults you set below)

  or override from CLI:
  python deeplab_resnet_training_annotated.py --csv /path/to/your.csv --out runs/exp1 --epochs 30

Dependencies (install once):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install segmentation-models-pytorch albumentations opencv-python torchmetrics pandas pillow tqdm
"""

from __future__ import annotations

import argparse
from pathlib import Path

# --- Numerics & DL ---
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
from torchvision.transforms.functional import normalize as tv_normalize

# DeepLabV3+ from segmentation_models_pytorch
try:
    import segmentation_models_pytorch as smp
except Exception as e:
    raise ImportError("segmentation-models-pytorch is required. Install with: pip install segmentation-models-pytorch") from e

# =========================
# ====== CONFIG BLOCK =====
# =========================
# >>>>> EDIT THESE LINES <<<<<
CSV_PATH    = Path("reef_dataset_index_with_split.csv")  # <-- PUT YOUR CSV HERE
DATA_ROOT   = None  # e.g., Path("/data/coral") if CSV paths are relative to a base folder; else keep None
OUT_DIR     = Path("runs/exp_deeplab")
NUM_CLASSES = 3   # 3={bg,healthy,bleached}; set 4 if you have dead coral class
IMAGE_SIZE  = 512 # longest side resize, then pad to square
EPOCHS      = 20
BATCH_SIZE  = 6
NUM_WORKERS = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
MIXED_PRECISION = True  # uses torch.amp for speed/VRAM
AUTO_SPLIT   = False    # set True if your CSV lacks 'split' column
VAL_EVERY    = 1        # validate every N epochs
SAVE_PRED_EVERY = 2     # save a small grid of predictions every N epochs (visual QC)
SAMPLES_TO_SAVE = 8     # how many val images to visualize when saving
SEED = 42
# =========================

# ---------- CLI overrides (optional) ----------
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, default=None, help='Path to CSV (overrides CSV_PATH)')
parser.add_argument('--out', type=str, default=None, help='Output directory (overrides OUT_DIR)')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--size', type=int, default=None)
parser.add_argument('--num-classes', type=int, default=None)
parser.add_argument('--auto-split', action='store_true', help='Force auto split even if CSV has split')
args_cli = parser.parse_args()

if args_cli.csv is not None:
    CSV_PATH = Path(args_cli.csv)
if args_cli.out is not None:
    OUT_DIR = Path(args_cli.out)
if args_cli.epochs is not None:
    EPOCHS = args_cli.epochs
if args_cli.batch_size is not None:
    BATCH_SIZE = args_cli.batch_size
if args_cli.size is not None:
    IMAGE_SIZE = args_cli.size
if args_cli.num_classes is not None:
    NUM_CLASSES = args_cli.num_classes
if args_cli.auto_split:
    AUTO_SPLIT = True

OUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================
# ===== Helper: color map =====
# =============================
# Define colors for overlays (BGR for OpenCV drawing)
COLOR_MAP = {
    0: (0,   0,   0  ),   # background = black
    1: (0, 255,   0  ),   # healthy   = green
    2: (0,   0, 255 ),   # bleached  = red (BGR order)
    3: (0, 255, 255 ),   # dead      = yellow (optional)
}

def mask_to_color(mask: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert class-index mask (H,W) to color image (H,W,3) for visualization."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        color[mask == c] = COLOR_MAP.get(c, (255, 255, 255))
    return color


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, num_classes: int, alpha: float = 0.5) -> np.ndarray:
    """Blend colorized mask on top of the RGB image for quick visual QC."""
    color = mask_to_color(mask, num_classes)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, color, alpha, 0)
    return blended

# ===================================
# ===== Dataset & augmentations =====
# ===================================
import albumentations as A

class ReefSegDataset(Dataset):
    """Loads (image_path, mask_multi_path) pairs from the CSV, applies robust augmentations,
    and returns normalized tensors that DeepLab expects.
    """
    def __init__(self, df: pd.DataFrame, size: int, is_train: bool, num_classes: int, data_root: Path | None):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.is_train = is_train
        self.num_classes = num_classes
        self.data_root = data_root

        # Underwater-friendly augmentations
        if is_train:
            self.augs = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.ColorJitter(p=0.3),
                A.MotionBlur(p=0.2),
                A.GaussNoise(p=0.2),
                A.OneOf([
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    A.RandomBrightnessContrast(p=1.0),
                ], p=0.2),
            ])
        else:
            self.augs = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row['image_path'])
        mask_path = Path(row['mask_multi_path'])
        if self.data_root is not None:
            img_path = (self.data_root / img_path).resolve()
            mask_path = (self.data_root / mask_path).resolve()

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Ensure label IDs are within range
        mask = np.clip(mask, 0, self.num_classes - 1).astype(np.uint8)

        augmented = self.augs(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        # to torch tensors, normalized like ImageNet
        img_t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_t = tv_normalize(img_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask_t = torch.from_numpy(mask).long()
        return img_t, mask_t

# ===================================
# ===== Model, loss, optimizer  =====
# ===================================

def build_model(num_classes: int) -> nn.Module:
    """DeepLabV3+ with ResNet-50 encoder pre-trained on ImageNet."""
    model = smp.DeepLabV3Plus(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        classes=num_classes,
        activation=None,  # we return raw logits
    )
    return model

class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
    def forward(self, logits, targets):
        probs = logits.softmax(dim=1)
        onehot = F.one_hot(targets, num_classes=self.num_classes).permute(0,3,1,2).float()
        dims = (0,2,3)
        inter = torch.sum(probs * onehot, dims)
        denom = torch.sum(probs + onehot, dims)
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()

# ==============================
# ===== Utility: split, log =====
# ==============================

def maybe_split(df: pd.DataFrame) -> pd.DataFrame:
    if ('split' in df.columns) and not AUTO_SPLIT:
        return df
    # Deterministic 80/10/10 by hashing image_path
    def _split_row(p: str):
        h = abs(hash(p)) % 100
        if h < 80: return 'train'
        if h < 90: return 'val'
        return 'test'
    df = df.copy()
    df['split'] = df['image_path'].astype(str).apply(_split_row)
    return df


def save_val_samples(model: nn.Module, ds: Dataset, device: torch.device, out_dir: Path, num_classes: int, n: int = 8):
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        count = min(n, len(ds))
        for i in range(count):
            img_t, gt_t = ds[i]
            img_b = img_t.unsqueeze(0).to(device)
            logits = model(img_b)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            # back to numpy RGB for overlay
            img_np = (img_t.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # saving in BGR is okay

            gt_overlay   = overlay_mask(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), gt_t.numpy(), num_classes)
            pred_overlay = overlay_mask(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), pred,            num_classes)

            cv2.imwrite(str(out_dir / f"val_{i:03d}_image.jpg"), img_np)
            cv2.imwrite(str(out_dir / f"val_{i:03d}_gt_overlay.jpg"), cv2.cvtColor(gt_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f"val_{i:03d}_pred_overlay.jpg"), cv2.cvtColor(pred_overlay, cv2.COLOR_RGB2BGR))

# ====================
# ======  Main  ======
# ====================

def main():
    # Repro
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Load CSV ---
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}. Set CSV_PATH at top or pass --csv.")
    df = pd.read_csv(CSV_PATH)
    required = {"image_path", "mask_multi_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = maybe_split(df)

    # --- Build datasets/loaders ---
    train_df = df[df['split'] == 'train']
    val_df   = df[df['split'] == 'val']
    print(f"Loaded CSV with {len(df)} rows — train={len(train_df)}, val={len(val_df)}")

    train_ds = ReefSegDataset(train_df, size=IMAGE_SIZE, is_train=True,  num_classes=NUM_CLASSES, data_root=DATA_ROOT)
    val_ds   = ReefSegDataset(val_df,   size=IMAGE_SIZE, is_train=False, num_classes=NUM_CLASSES, data_root=DATA_ROOT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # --- Device & model ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = build_model(NUM_CLASSES).to(device)

    # --- Losses, metrics, optim ---
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(NUM_CLASSES)
    jaccard = MulticlassJaccardIndex(num_classes=NUM_CLASSES, average='none').to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scaler = torch.amp.GradScaler('cuda', enabled=MIXED_PRECISION)

    best_miou = -1.0
    (OUT_DIR / 'samples').mkdir(parents=True, exist_ok=True)

    # --- Training loop with progress bars ---
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} — TRAIN", leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                logits = model(imgs)
                loss = 0.7 * ce_loss(logits, masks) + 0.3 * dice_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=loss.item())  # live update in progress bar

        avg_train_loss = running_loss / len(train_ds)

        # --- Validation ---
        if epoch % VAL_EVERY == 0:
            model.eval()
            val_running = 0.0
            miou_sum = np.zeros(NUM_CLASSES, dtype=np.float64)
            batches = 0
            with torch.no_grad():
                for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} — VAL", leave=False):
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    logits = model(imgs)
                    vloss = 0.7 * ce_loss(logits, masks) + 0.3 * dice_loss(logits, masks)
                    val_running += vloss.item() * imgs.size(0)

                    preds = logits.argmax(dim=1)
                    miou = jaccard(preds, masks).cpu().numpy()  # per-class array
                    miou_sum += miou
                    batches += 1

            per_class_miou = miou_sum / max(1, batches)
            mean_miou = float(np.nanmean(per_class_miou))
            avg_val_loss = val_running / max(1, len(val_ds))

            # Scheduler step on metric
            scheduler.step(mean_miou)

            # Log to console and file
            log_line = (
                f"epoch={epoch} train_loss={avg_train_loss:.4f} "
                f"val_loss={avg_val_loss:.4f} mIoU={mean_miou:.4f} "
                f"per_class={np.round(per_class_miou,3).tolist()}"

            )
            print(log_line.strip())
            with open(OUT_DIR / 'log.txt', 'a') as f:
                f.write(log_line)

            # Save best checkpoint
            if mean_miou > best_miou:
                best_miou = mean_miou
                torch.save({
                    'model': model.state_dict(),
                    'num_classes': NUM_CLASSES,
                    'image_size': IMAGE_SIZE,
                }, OUT_DIR / 'best.pt')
                print(f"[✓] New best mIoU={best_miou:.4f}. Saved to {OUT_DIR / 'best.pt'}")

            # Periodic visual QC
            if (epoch % SAVE_PRED_EVERY) == 0:
                save_val_samples(model, val_ds, device, OUT_DIR / f'samples/epoch_{epoch:03d}', NUM_CLASSES, n=SAMPLES_TO_SAVE)
                print(f"Saved validation sample overlays to {OUT_DIR / f'samples/epoch_{epoch:03d}'}")

    print("Training complete.")

if __name__ == "__main__":
    main()
