"""
Pixel-level IoU & Dice evaluation for YOLOv8-seg runs.

Usage (examples at bottom):
    python eval_iou_dice.py --weights runs/baseline_s0/weights/best.pt --data data.yaml --split test
    python eval_iou_dice.py --weights runs/mild_s0/weights/best.pt     --data data.yaml --split test

Outputs a CSV (append/update) with:
    run, mIoU, IoU_hard, IoU_soft, IoU_other, mDice, Dice_hard, Dice_soft, Dice_other
"""

import argparse, os, re, csv, sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
from ultralytics import YOLO
import yaml

# ---------- helpers ----------
def image_to_label_path(img_path: str) -> str:
    s = img_path.replace("\\", "/")
    s = re.sub(r"[\\/]+images[\\/]+", "/labels/", s)
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", ".txt", s, flags=re.I)
    return s

def load_split_list(yaml_path: str, split: str) -> list[str]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    p = y.get(split)
    if p is None:
        raise ValueError(f"Split '{split}' not found in {yaml_path}")
    # If it's a file with a list of images, read it; if it's a folder, scan recursively.
    p = str(p)
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    # else treat as folder with 'images' subfolder
    imgs_dir = p
    if os.path.isdir(imgs_dir):
        exts = (".jpg",".jpeg",".png",".bmp")
        out = []
        for root, _, files in os.walk(imgs_dir):
            for fn in files:
                if fn.lower().endswith(exts):
                    out.append(os.path.join(root, fn))
        out.sort()
        return out
    raise FileNotFoundError(f"Could not resolve split path: {p}")

def read_gt_polygons(lbl_path: str) -> list[tuple[int, list[float]]]:
    out = []
    if not os.path.exists(lbl_path):
        return out
    with open(lbl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            parts = ln.split()
            try:
                cls = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                if len(coords) >= 6 and len(coords)%2==0:
                    out.append((cls, coords))
            except Exception:
                pass
    return out

def norm_to_pix(coords, W, H):
    pts = np.array(coords, dtype=np.float32).reshape(-1,2)
    pts[:,0] = np.clip(pts[:,0]*W, 0, W-1)
    pts[:,1] = np.clip(pts[:,1]*H, 0, H-1)
    return pts.astype(np.int32)

def rasterize_polys(polys, W, H, target_class: int) -> np.ndarray:
    """Union mask for a given class id from a list of (cls, coords_norm)."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for cls, coords in polys:
        if cls != target_class:
            continue
        pts = norm_to_pix(coords, W, H)
        if pts.shape[0] >= 3:
            cv2.fillPoly(mask, [pts], 1)
    return mask

def metrics_from_masks(gt: np.ndarray, pr: np.ndarray):
    """Return (IoU, Dice) given binary masks."""
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    inter = (gt & pr).sum()
    union = (gt | pr).sum()
    tp = inter
    fp = (~gt & pr).sum()
    fn = (gt & ~pr).sum()
    iou  = tp / union if union > 0 else 1.0  # if neither has positives, treat IoU=1
    dice = (2*tp) / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 1.0
    return iou, dice, tp, fp, fn

# ---------- main eval ----------
def evaluate(weights, data_yaml, split="test", imgsz=640, conf=0.25, iou_thr=0.5, device=None, out_csv="results_iou_dice.csv"):
    model = YOLO(weights)
    # names must match your order: [hard_coral, soft_coral]
    names = model.names
    # Map expected class ids
    HARD_ID = 0
    SOFT_ID = 1

    images = load_split_list(data_yaml, split)
    if len(images) == 0:
        raise RuntimeError(f"No images found for split '{split}'")

    # accumulators per class
    agg = {
        "hard": {"tp":0, "fp":0, "fn":0},
        "soft": {"tp":0, "fp":0, "fn":0},
        "other":{"tp":0, "fp":0, "fn":0}
    }

    for idx, im_path in enumerate(images, 1):
        im = cv2.imread(im_path)
        if im is None:
            continue
        H, W = im.shape[:2]

        # --- Ground truth union masks ---
        gt_polys = read_gt_polygons(image_to_label_path(im_path))
        gt_hard = rasterize_polys(gt_polys, W, H, HARD_ID)
        gt_soft = rasterize_polys(gt_polys, W, H, SOFT_ID)
        gt_other = (1 - np.clip(gt_hard + gt_soft, 0, 1)).astype(np.uint8)

        # --- Predictions ---
        # We get instance masks and union per class
        res = model.predict(source=im_path, imgsz=imgsz, conf=conf, iou=iou_thr, verbose=False, device=device)[0]

        pr_hard = np.zeros((H, W), dtype=np.uint8)
        pr_soft = np.zeros((H, W), dtype=np.uint8)

        if getattr(res, "masks", None) is not None and res.masks is not None and len(res.masks) > 0:
            # res.masks.data: (N, h, w) float; res.masks.xy: polygons; res.masks.orig_shape = (H, W)
            # We'll use the .data masks upsampled to the original size.
            md = res.masks.data.cpu().numpy()  # N x h x w in [0..1]
            # Upscale to original size
            for k, cls_id in enumerate(res.boxes.cls.cpu().numpy().astype(int)):
                m = md[k]
                m_up = (cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR) >= 0.5).astype(np.uint8)
                if cls_id == HARD_ID:
                    pr_hard = np.clip(pr_hard + m_up, 0, 1)
                elif cls_id == SOFT_ID:
                    pr_soft = np.clip(pr_soft + m_up, 0, 1)
        pr_other = (1 - np.clip(pr_hard + pr_soft, 0, 1)).astype(np.uint8)

        # --- Per-class IoU/Dice (accumulate TP/FP/FN so that we compute global scores) ---
        for cls_name, gt_mask, pr_mask in [
            ("hard", gt_hard, pr_hard),
            ("soft", gt_soft, pr_soft),
            ("other", gt_other, pr_other),
        ]:
            _, _, tp, fp, fn = metrics_from_masks(gt_mask, pr_mask)
            agg[cls_name]["tp"] += int(tp)
            agg[cls_name]["fp"] += int(fp)
            agg[cls_name]["fn"] += int(fn)

        if idx % 50 == 0:
            print(f"[{idx}/{len(images)}] processed")

    def finalize(cls):
        tp, fp, fn = agg[cls]["tp"], agg[cls]["fp"], agg[cls]["fn"]
        iou = tp / max(tp+fp+fn, 1)
        dice = (2*tp) / max(2*tp + fp + fn, 1)
        return iou, dice

    IoU_hard, Dice_hard = finalize("hard")
    IoU_soft, Dice_soft = finalize("soft")
    IoU_other, Dice_other = finalize("other")

    mIoU  = (IoU_hard + IoU_soft + IoU_other) / 3.0
    mDice = (Dice_hard + Dice_soft + Dice_other) / 3.0

    # write/append CSV
    run_name = Path(weights).parent.parent.name  # runs/<name>/weights/best.pt -> <name>
    header = ["run","mIoU","IoU_hard","IoU_soft","IoU_other","mDice","Dice_hard","Dice_soft","Dice_other","weights","split"]
    row = [run_name, mIoU, IoU_hard, IoU_soft, IoU_other, mDice, Dice_hard, Dice_soft, Dice_other, str(weights), split]

    out_path = Path(out_csv)
    write_header = (not out_path.exists())
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print("\n=== Final (global) metrics ===")
    print(f"run={run_name}")
    print(f"mIoU={mIoU:.4f} | IoU_hard={IoU_hard:.4f} | IoU_soft={IoU_soft:.4f} | IoU_other={IoU_other:.4f}")
    print(f"mDice={mDice:.4f} | Dice_hard={Dice_hard:.4f} | Dice_soft={Dice_soft:.4f} | Dice_other={Dice_other:.4f}")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to run weights, e.g. runs/baseline_s0/weights/best.pt")
    ap.add_argument("--data", required=True, help="data.yaml with 'test'/'val' split")
    ap.add_argument("--split", default="test", choices=["test","val"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou_thr", type=float, default=0.5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="iou_dice_results.csv")
    args = ap.parse_args()

    evaluate(args.weights, args.data, split=args.split, imgsz=args.imgsz,
             conf=args.conf, iou_thr=args.iou_thr, device=args.device, out_csv=args.out)
