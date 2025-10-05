#!/usr/bin/env python3
import argparse, math, sys
from pathlib import Path
from collections import defaultdict

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def log(*a): print(*a, flush=True)

def index_images_recursive(root: Path):
    """Build dict of image stem -> full path"""
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            idx[p.stem.lower()] = p.resolve()
    return idx

def parse_yolo_labels(label_dir: Path, allowed_keys=None):
    """
    Parse YOLO segmentation labels (*.txt).
    Returns:
      per_image_classes: dict[image_stem] -> set of class_ids
      instance_counts: dict[class_id] -> number of polygons
    """
    per_image_classes = defaultdict(set)
    instance_counts = defaultdict(int)

    for txt in label_dir.rglob("*.txt"):
        stem = txt.stem.lower()
        if allowed_keys and stem not in allowed_keys:
            continue

        with open(txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cid = parts[0]
                per_image_classes[stem].add(cid)
                instance_counts[cid] += 1
    return per_image_classes, instance_counts

def compute_rfs(instance_counts: dict[str,int], T: float, cap: int):
    total_inst = max(1, sum(instance_counts.values()))
    freq = {c: max(1e-12, instance_counts[c] / total_inst) for c in instance_counts}
    r_class = {c: max(1.0, math.sqrt(T / f)) for c, f in freq.items()}
    # Apply cap
    r_class = {c: min(cap, r) for c, r in r_class.items()}
    return freq, r_class, total_inst

def main():
    ap = argparse.ArgumentParser(description="Build RFS list from YOLO segmentation labels.")
    ap.add_argument("--root", type=Path, required=True, help="Root folder with images/ and labels/ subdirs")
    ap.add_argument("--out", type=Path, required=True, help="Output train_rfs.txt path")
    ap.add_argument("--train-base", type=Path, default=None,
                    help="Optional: path to train_base.txt (restricts to these images)")
    ap.add_argument("--threshold", type=float, default=0.3, help="LVIS T parameter (0.2–0.4 typical)")
    ap.add_argument("--cap", type=int, default=8, help="Max per-image repeat factor")
    args = ap.parse_args()

    root = args.root.resolve()
    labels_dir = root / "labels"
    images_dir = root / "images"
    if not labels_dir.exists():
        log(f"[ERROR] Labels dir not found: {labels_dir}"); sys.exit(1)
    if not images_dir.exists():
        log(f"[ERROR] Images dir not found: {images_dir}"); sys.exit(1)

    allowed_keys = None
    if args.train_base and args.train_base.exists():
        allowed_keys = set()
        with open(args.train_base, "r", encoding="utf-8") as f:
            for line in f:
                p = Path(line.strip())
                allowed_keys.add(p.stem.lower())
        log(f"[INFO] Loaded {len(allowed_keys)} train items from {args.train_base}")

    # Parse YOLO label files
    per_image_classes, instance_counts = parse_yolo_labels(labels_dir, allowed_keys)

    if not per_image_classes:
        log("[ERROR] No labeled training images found.")
        sys.exit(1)

    # Compute RFS
    freq, r_class, total_inst = compute_rfs(instance_counts, args.threshold, args.cap)

    # Index images
    idx = index_images_recursive(images_dir)

    combined_records = []
    missing = []
    seen = set()
    for stem, cls_set in per_image_classes.items():
        path = idx.get(stem)
        if not path:
            missing.append(stem)
            continue
        base = max(r_class.get(c, 1.0) for c in cls_set)
        r_img = int(round(min(args.cap, max(1.0, base))))
        if str(path) not in seen:
            seen.add(str(path))
            combined_records.append((str(path), r_img, tuple(sorted(cls_set))))

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, r, _ in combined_records:
            for _ in range(r):
                f.write(p + "\n")

    # Summary
    log("\n===== RFS SUMMARY =====")
    log(f"Training images (labels): {len(per_image_classes)}")
    log(f"Matched local images:     {len(combined_records)}")
    log(f"Missing images:           {len(missing)}")
    if missing:
        log(f"  e.g.: {missing[:5]}")
    log(f"Total instances:          {total_inst}")
    for c in sorted(freq):
        log(f"  class {c}: freq={freq[c]:.4f}, r(c)={r_class[c]:.3f}, instances={instance_counts[c]}")
    log(f"T (threshold):            {args.threshold}")
    log(f"Repeat cap:               {args.cap}")
    log(f"Combined output:          {args.out}  (lines={sum(r for _, r, _ in combined_records)})")

if __name__ == "__main__":
    main()
