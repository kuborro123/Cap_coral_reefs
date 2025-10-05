#!/usr/bin/env python3
import argparse, math, random, sys
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def log(*a): print(*a, flush=True)

def index_images(images_dir: Path):
    """Return list of all image paths under images/"""
    return [p.resolve() for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

def parse_yolo_labels(labels_dir: Path, image_paths):
    """Count instances per class from YOLO .txt labels"""
    per_image_classes = defaultdict(set)
    instance_counts = defaultdict(int)

    for img in image_paths:
        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists():
            continue
        with open(lbl, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cid = parts[0]
                per_image_classes[img].add(cid)
                instance_counts[cid] += 1
    return per_image_classes, instance_counts

def compute_rfs(instance_counts, T, cap):
    total_inst = max(1, sum(instance_counts.values()))
    freq = {c: max(1e-12, instance_counts[c]/total_inst) for c in instance_counts}
    r_class = {c: max(1.0, math.sqrt(T/f)) for c,f in freq.items()}
    r_class = {c: min(cap, r) for c,r in r_class.items()}
    return freq, r_class, total_inst

def write_list(path: Path, items):
    with open(path, "w", encoding="utf-8") as f:
        for p in items:
            f.write(str(p) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Merge subfolders into one dataset and create train/val/test + RFS.")
    ap.add_argument("--root", default= r'C:\Users\User\Downloads\website\website\benthic_datasets\mask_labels\reef_support', type=Path, help="Root containing dataset subfolders with images/labels/")
    ap.add_argument("--out", default=r'C:\Users\User\Downloads\website\website\benthic_datasets\mask_labels\reef_support\out_fold',  type=Path, help="Output folder to write split .txt files")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    # Collect images and labels from ALL subfolders
    all_images = []
    all_labels = []
    for ds in sorted([d for d in args.root.iterdir() if d.is_dir()]):
        images_dir = ds / "images"
        labels_dir = ds / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            log(f"[WARN] Skipping {ds} (no images/labels)")
            continue
        imgs = index_images(images_dir)
        all_images.extend(imgs)
        all_labels.append(labels_dir)

    if not all_images:
        log("[ERROR] No images found")
        sys.exit(1)

    random.shuffle(all_images)
    n_total = len(all_images)
    n_val = int(round(n_total * args.val_frac))
    n_test = int(round(n_total * args.test_frac))
    n_train = n_total - n_val - n_test

    train_imgs = all_images[:n_train]
    val_imgs   = all_images[n_train:n_train+n_val]
    test_imgs  = all_images[n_train+n_val:]

    args.out.mkdir(parents=True, exist_ok=True)
    write_list(args.out / "train_base.txt", train_imgs)
    write_list(args.out / "val.txt", val_imgs)
    write_list(args.out / "test.txt", test_imgs)

    log(f"Split: total={n_total} → train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

    # --- RFS ---
    # Note: need to try each labels/ folder to find matching txt
    per_image_classes = defaultdict(set)
    instance_counts = defaultdict(int)
    for lbl_dir in all_labels:
        p, c = parse_yolo_labels(lbl_dir, train_imgs)
        for k,v in p.items(): per_image_classes[k].update(v)
        for k,v in c.items(): instance_counts[k]+=v

    if not instance_counts:
        log("[WARN] No labels found for training set")
        return

    freq, r_class, total_inst = compute_rfs(instance_counts, args.threshold, args.cap)

    combined_records = []
    for img in train_imgs:
        cls_set = per_image_classes.get(img, set())
        if not cls_set:
            continue

        base = max(r_class.get(c, 1.0) for c in cls_set)
        r_img = int(round(min(args.cap, max(1.0, base))))
        combined_records.extend([str(img)] * r_img)

    with open(args.out / "train_rfs.txt", "w", encoding="utf-8") as f:
        for line in combined_records:
            f.write(line + "\n")

    log("\n===== RFS SUMMARY =====")
    log(f"Total train instances: {total_inst}")
    for c in sorted(freq):
        log(f"  class {c}: freq={freq[c]:.4f}, r(c)={r_class[c]:.3f}, instances={instance_counts[c]}")
    log(f"Final train_rfs.txt lines: {len(combined_records)}")

if __name__ == "__main__":
    main()
