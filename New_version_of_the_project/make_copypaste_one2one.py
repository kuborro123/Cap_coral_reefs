# make_copypaste_one2one.py
# ------------------------------------------------------------
# Build a new train list that achieves ~1:1 soft:hard by creating
# NEW images via SOFT-ONLY instance copy-paste.
#
# For each new image:
#   - pick ONE soft polygon instance from any source image,
#   - paste it onto a chosen target image (prefer low-hard targets),
#   - save the composited image + updated YOLOv8-seg label.
#
# Output:
#   - images_aug/<TAG>/...jpg
#   - labels_aug/<TAG>/...txt
#   - splits/train_copypaste_soft__<TAG>.txt   <-- feed this to training
#
# Usage (examples at the end of this file).
#
# Notes:
# - YOLOv8-seg format: each line = "<cls> x1 y1 x2 y2 ...", normalized [0..1]
# - We do NOT rotate polygons (keeps math simple & exact). We allow 0.8x–1.2x scaling.
# - If your dataset has very few hard-free images, we still bias toward fewer-hard targets.
# - This script may still add some hard instances (from chosen target), but far less than oversampling.
#
# Reproducibility: RNG seed fixed by --seed (default 0).

import argparse, os, re, math, random, shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

# ----------------------------- IO helpers -----------------------------

def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def norm_to_pix(poly_norm, W, H):
    # poly_norm: [x1,y1,x2,y2,...] in [0..1]
    pts = np.array(poly_norm, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= W
    pts[:, 1] *= H
    return pts

def pix_to_norm(pts_pix, W, H):
    pts = np.array(pts_pix, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0] / max(W,1), 0, 1)
    pts[:, 1] = np.clip(pts[:, 1] / max(H,1), 0, 1)
    return pts.reshape(-1)

def image_to_label_path(img_path):
    s = str(img_path).replace("\\", "/")
    s = re.sub(r"[\\/]+images[\\/]+", "/labels/", s)
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", ".txt", s, flags=re.I)
    return s

def label_to_image_path(lbl_path):
    s = str(lbl_path).replace("\\", "/")
    s = re.sub(r"[\\/]+labels[\\/]+", "/images/", s)
    s = re.sub(r"\.txt$", ".jpg", s)
    return s

def read_label(lbl_path):
    objs = []
    if not os.path.exists(lbl_path):
        return objs
    with open(lbl_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            try:
                cls = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    objs.append((cls, coords))
            except Exception:
                pass
    return objs

def write_label(lbl_path, objs):
    os.makedirs(os.path.dirname(lbl_path), exist_ok=True)
    with open(lbl_path, "w", encoding="utf-8") as f:
        for cls, coords in objs:
            coords_str = " ".join(f"{v:.6f}" for v in coords)
            f.write(f"{cls} {coords_str}\n")

# --------------------------- dataset indexing -------------------------

def build_index(train_list, soft_id, hard_id):
    """
    Returns:
      images: list of dict with keys: path, W,H, objs=[(cls, coords_norm)], cnt=Counter
      soft_instances: list of tuples (src_img_idx, coords_norm)   # per soft polygon
    """
    images = []
    soft_instances = []
    for img_path in train_list:
        img_path = Path(img_path)
        lbl_path = Path(image_to_label_path(img_path))
        objs = read_label(lbl_path)
        # lazy size read: open image to get W,H
        im = cv2.imread(str(img_path))
        if im is None:
            # skip missing/corrupt images
            continue
        H, W = im.shape[:2]
        cnt = Counter()
        for cls, coords in objs:
            cnt[cls] += 1
        images.append({
            "path": img_path, "W": W, "H": H,
            "objs": objs, "cnt": cnt
        })
        idx = len(images) - 1
        for cls, coords in objs:
            if cls == soft_id:
                soft_instances.append((idx, coords))
    return images, soft_instances

# ----------------------------- paste utils ----------------------------

def polygon_mask_from_pts(pts_pix, H, W):
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_pix.astype(np.int32)], 255)
    return mask

def crop_with_alpha(img, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None, None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop = img[y1:y2+1, x1:x2+1]
    alpha = mask[y1:y2+1, x1:x2+1]
    return crop, alpha

def paste_with_alpha(dst, crop, alpha, x, y):
    H, W = dst.shape[:2]
    h, w = crop.shape[:2]
    if x < 0 or y < 0 or x + w > W or y + h > H:
        return False
    roi = dst[y:y+h, x:x+w]
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    blended = (crop.astype(np.float32) * a + roi.astype(np.float32) * (1 - a)).astype(np.uint8)
    dst[y:y+h, x:x+w] = blended
    return True

# ---------------------------- main generator --------------------------

def generate_one2one(train_list, soft_id, hard_id, tag,
                     target_ratio=1.0, max_new=10000, max_per_target=2,
                     scale_min=0.8, scale_max=1.2, seed=0):
    rng = random.Random(seed)

    images, soft_instances = build_index(train_list, soft_id, hard_id)
    if not images:
        raise RuntimeError("No valid images found.")
    if not soft_instances:
        raise RuntimeError("No soft instances found. Check soft_id and labels.")

    # Current totals (instances across original train set)
    totals = Counter()
    for im in images:
        totals.update(im["cnt"])
    soft0, hard0 = totals[soft_id], totals[hard_id]
    if hard0 == 0:
        hard0 = 1
    print(f"Baseline totals: soft={soft0}, hard={hard0}, ratio={soft0/hard0:.3f}")

    # How many extra SOFT do we need (approx), ignoring new hard added by targets:
    need_soft = max(0, hard0 - soft0)  # to reach 1:1 ideally
    if need_soft == 0:
        print("[Info] Already ≥ 1:1; will still add some for robustness (capped by --max_new).")

    # Prepare output dirs and list
    out_list = []
    out_img_dir = Path("images_aug") / tag
    out_lbl_dir = Path("labels_aug") / tag
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # Target selection weights: prefer fewer-hard targets
    # weight = 1 / (1 + hard_count)  -> 1.0 for 0 hard, 0.5 for 1 hard, ~0.33 for 2, etc.
    idxs = list(range(len(images)))
    weights = []
    for i in idxs:
        h = images[i]["cnt"].get(hard_id, 0)
        weights.append(1.0 / (1 + h))
    wsum = sum(weights)
    weights = [w/wsum for w in weights]

    # Track how many augmented per target (avoid overloading single target)
    per_target_aug = Counter()

    added_soft = 0
    new_count = 0
    tries = 0
    max_tries = max_new * 50  # generous loop cap

    while new_count < max_new and (totals[soft_id] / max(totals[hard_id], 1)) < target_ratio and tries < max_tries:
        tries += 1

        # pick a soft instance
        src_idx, coords_norm = rng.choice(soft_instances)
        src = images[src_idx]
        src_img = cv2.imread(str(src["path"]))
        if src_img is None:
            continue

        # build soft crop + mask in pixel coords
        pts_src = norm_to_pix(coords_norm, src["W"], src["H"])
        mask_src = polygon_mask_from_pts(pts_src, src["H"], src["W"])
        crop, alpha = crop_with_alpha(src_img, mask_src)
        if crop is None:
            continue

        # random scale
        s = rng.uniform(scale_min, scale_max)
        new_w = max(2, int(crop.shape[1] * s))
        new_h = max(2, int(crop.shape[0] * s))
        crop_r = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        alpha_r = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        # scale polygon too
        pts_scaled = pts_src.copy()
        pts_scaled[:, 0] *= (new_w / crop.shape[1])
        pts_scaled[:, 1] *= (new_h / crop.shape[0])

        # choose a target image (biased to few-hard)
        tgt_idx = rng.choices(idxs, weights=weights, k=1)[0]
        if per_target_aug[tgt_idx] >= max_per_target:
            continue
        tgt = images[tgt_idx]
        tgt_img = cv2.imread(str(tgt["path"]))
        if tgt_img is None:
            continue
        Ht, Wt = tgt_img.shape[:2]

        # place top-left so the polygon fits: we translate so that the crop is fully inside
        max_x = Wt - new_w
        max_y = Ht - new_h
        if max_x < 1 or max_y < 1:
            continue
        x = rng.randint(0, max_x)
        y = rng.randint(0, max_y)

        # paste
        ok = paste_with_alpha(tgt_img, crop_r, alpha_r, x, y)
        if not ok:
            continue

        # transform polygon by translation (no rotation)
        pts_pasted = pts_scaled + np.array([x - int(pts_src[:,0].min()), y - int(pts_src[:,1].min())], dtype=np.float32)
        # BUT since we scaled relative to crop bbox, we need to rebuild pts relative to top-left corner of crop bbox.
        # Simpler: recompute from mask bounding box: pts_src bbox (x1,y1)
        xs, ys = np.where(mask_src > 0)
        x1_src, y1_src = xs.min(), ys.min()
        # After scaling, the polygon's origin is at (x, y) on target:
        # Map original pts_src -> normalize by original bbox, then scale, then translate
        # We'll approximate with the scaled pts we computed earlier, then translate by (x, y)
        # Correct translation: new_pts = (pts_src - [x1_src,y1_src]) * [sx,sy] + [x,y]
        sx = new_w / max(crop.shape[1], 1)
        sy = new_h / max(crop.shape[0], 1)
        pts_new = np.zeros_like(pts_src)
        pts_new[:, 0] = (pts_src[:, 0] - x1_src) * sx + x
        pts_new[:, 1] = (pts_src[:, 1] - y1_src) * sy + y

        # build new label = target objs + new soft polygon
        tgt_objs = list(tgt["objs"])  # copy
        tgt_objs.append((soft_id, pix_to_norm(pts_new, Wt, Ht).tolist()))

        # save new image + label
        base_name = f"{Path(tgt['path']).stem}__cpsoft_{new_count:06d}"
        out_img = out_img_dir / f"{base_name}.jpg"
        out_lbl = out_lbl_dir / f"{base_name}.txt"
        cv2.imwrite(str(out_img), tgt_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        write_label(str(out_lbl), tgt_objs)

        # update counters and bookkeeping
        out_list.append(str(out_img))
        per_target_aug[tgt_idx] += 1
        new_count += 1
        totals[soft_id] += 1                 # +1 soft instance pasted
        # Note: we also "added" all target's existing objs because it's a new image,
        # but those are already accounted via tgt_objs in the label we saved.
        # For the *ratio control* we only count the +1 soft explicitly; the target's
        # hard contribute to the dataset but bias was minimized via weights.

        # optional progress print
        if new_count % 500 == 0:
            r = totals[soft_id] / max(totals[hard_id], 1)
            print(f"[cp] new={new_count}  ratio≈{r:.3f}  soft={totals[soft_id]}  hard={totals[hard_id]}")

    # write list file
    list_path = Path("splits") / f"train_copypaste_soft__{tag}.txt"
    os.makedirs(list_path.parent, exist_ok=True)
    # final list includes ORIGINAL train images + AUGMENTED images
    final_list = list(train_list) + out_list
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_list))

    final_ratio = totals[soft_id] / max(totals[hard_id], 1)
    print("\n=== DONE ===")
    print(f"Created {new_count} augmented images in {out_img_dir}")
    print(f"Wrote train list: {list_path}  (lines: {len(final_list)})")
    print(f"Approx final soft:hard ≈ {final_ratio:.3f}")
    return str(list_path)

def main():
    ap = argparse.ArgumentParser(description="Soft copy-paste generator to reach ~1:1 soft:hard.")
    ap.add_argument("--train", required=True, help="Baseline train list (e.g., splits/train_realistic.txt)")
    ap.add_argument("--soft_id", type=int, required=True, help="Soft-coral class id (usually 1)")
    ap.add_argument("--hard_id", type=int, default=0, help="Hard-coral class id (usually 0)")
    ap.add_argument("--tag", type=str, required=True, help="Suffix for output files (e.g., one2one_realistic)")
    ap.add_argument("--target", type=float, default=1.0, help="Target soft:hard ratio (default 1.0)")
    ap.add_argument("--max_new", type=int, default=8000, help="Max number of NEW images to create")
    ap.add_argument("--max_per_target", type=int, default=2, help="Max augmented images per target image")
    ap.add_argument("--scale_min", type=float, default=0.8, help="Min scale for pasted soft instances")
    ap.add_argument("--scale_max", type=float, default=1.2, help="Max scale for pasted soft instances")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    args = ap.parse_args()

    train_list = read_lines(args.train)
    generate_one2one(
        train_list=train_list,
        soft_id=args.soft_id, hard_id=args.hard_id, tag=args.tag,
        target_ratio=args.target, max_new=args.max_new,
        max_per_target=args.max_per_target,
        scale_min=args.scale_min, scale_max=args.scale_max, seed=args.seed
    )

if __name__ == "__main__":
    main()
