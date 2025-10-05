#!/usr/bin/env python3
import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict

# Accept common image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------- helpers ----------
def log(*a): print(*a, flush=True)

def find_images_dir(ds_root: Path) -> Path | None:
    """
    Try common names; fallback to first dir that contains image files.
    """
    candidates = ["images", "Images", "imgs", "Imgs", "IMAGES"]
    for c in candidates:
        p = ds_root / c
        if p.exists() and p.is_dir():
            if any(next(p.rglob(f"*{ext}"), None) for ext in IMG_EXTS):
                return p
    # fallback: scan subdirs
    for p in ds_root.iterdir():
        if p.is_dir():
            if any(next(p.rglob(f"*{ext}"), None) for ext in IMG_EXTS):
                return p
    return None

def find_ndjsons(ds_root: Path) -> list[Path]:
    # Prefer *.ndjson directly under the dataset folder; if none, search deeper
    return list(ds_root.glob("*.ndjson")) or list(ds_root.glob("**/*.ndjson"))

def index_images_recursive(root: Path, loose: bool):
    """
    exact: key = file.name.lower()
    loose: key = file.stem.lower().replace(' ', '')
    """
    idx = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            key = (p.name.lower() if not loose else p.stem.lower().replace(" ", ""))
            idx.setdefault(key, p.resolve())
    return idx

def key_for_external_id(external_id: str, loose: bool):
    p = Path(external_id)
    return (p.name.lower() if not loose else p.stem.lower().replace(" ", ""))

def add_from_ndjson(ndjson_path: Path, per_image_classes, instance_counts, class_alias):
    """
    Parse Labelbox NDJSON and update:
      - per_image_classes[external_id] = set(class names)
      - instance_counts[class_name] += #objects
    """
    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            ext_id = rec.get("data_row", {}).get("external_id")
            if not ext_id:
                continue
            projects = rec.get("projects", {})
            for proj in projects.values():
                for lab in proj.get("labels", []):
                    anns = lab.get("annotations", {})
                    for obj in anns.get("objects", []):
                        cname = obj.get("name")
                        if not cname:
                            continue
                        cname = class_alias.get(cname, cname)
                        per_image_classes[ext_id].add(cname)
                        instance_counts[cname] += 1

def compute_rfs(instance_counts: dict[str,int], T: float):
    total_inst = max(1, sum(instance_counts.values()))
    freq = {c: max(1e-12, instance_counts[c] / total_inst) for c in instance_counts}
    r_class = {c: max(1.0, math.sqrt(T / f)) for c, f in freq.items()}
    return freq, r_class, total_inst

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(
        description="Auto-discover datasets under a root and build LVIS-style RFS list (Labelbox NDJSON)."
    )

    # Set YOUR defaults here so you can just run without flags
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\User\Desktop\YEAR 3\Q1\Capstone DC\Datasets\mask_labels\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\benthic_datasets\mask_labels\reef_support"),
        help="Root folder containing many dataset subfolders",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"C:\Users\User\Desktop\YEAR 3\Q1\Capstone DC\Datasets\mask_labels\content\gdrive\MyDrive\Data Challenge 3 - JBG060 AY2526\01_data\benthic_datasets\mask_labels\reef_support\train_rfs.txt"),
        help="Combined output train list (txt)",
    )
    ap.add_argument("--per-dataset-out", action="store_true", help="Also write <dataset>_train_rfs.txt per subfolder")
    ap.add_argument("--threshold", type=float, default=0.3, help="LVIS T parameter (try 0.2–0.4 for your skew)")
    ap.add_argument("--cap", type=int, default=8, help="Max per-image repeat factor")
    ap.add_argument("--match-mode", choices=["exact", "loose"], default="loose",
                    help="How to match NDJSON external_id to local filenames")
    ap.add_argument("--class-alias", action="append",
                    help="Optional alias: old=Soft_corals:new=Soft Coral (repeatable)")

    args = ap.parse_args()

    root = args.root.resolve()
    if not root.exists():
        log(f"[ERROR] Root not found: {root}"); sys.exit(1)

    # Build alias map
    class_alias = {}
    if args.class_alias:
        for ent in args.class_alias:
            try:
                old, new = ent.split(":")
                old = old.replace("old=","")
                new = new.replace("new=","")
                class_alias[old] = new
            except Exception:
                log(f"[WARN] Bad --class-alias entry: {ent}")

    # Discover datasets = immediate subfolders that contain both (images dir) and (an NDJSON)
    datasets = []
    for child in sorted([p for p in root.iterdir() if p.is_dir()]):
        ndjsons = find_ndjsons(child)
        if not ndjsons:
            continue
        images_dir = find_images_dir(child)
        if not images_dir:
            continue
        datasets.append((child.name, child, images_dir, ndjsons))

    if not datasets:
        log("[ERROR] No datasets found (need images/ and *.ndjson per subfolder).")
        sys.exit(1)

    log("=== Discovered datasets ===")
    for name, ds_root, im_dir, nds in datasets:
        log(f"- {name}: images={im_dir} ndjsons={len(nds)}")

    # Parse all NDJSONs and index images per dataset
    loose = (args.match_mode == "loose")
    global_per_image_classes = defaultdict(set)  # external_id -> classes (union over datasets)
    global_instance_counts = defaultdict(int)

    per_ds_indexes = {}  # name -> {match_key: path}
    for name, ds_root, images_dir, ndjsons in datasets:
        # index images in this dataset
        per_ds_indexes[name] = index_images_recursive(images_dir, loose)
        # gather labels
        for nd in ndjsons:
            add_from_ndjson(nd, global_per_image_classes, global_instance_counts, class_alias)

    if not global_per_image_classes:
        log("[ERROR] No labeled images found in any NDJSON."); sys.exit(1)

    # Compute repeat factors (instance-based)
    freq, r_class, total_inst = compute_rfs(global_instance_counts, args.threshold)

    # Resolve external_id -> path (first dataset that contains it wins), assign repeats
    combined_records = []
    missing = []
    seen_paths = set()

    dataset_order = [name for name, *_ in datasets]  # prefer earlier datasets on conflicts
    for ext_id, cls_set in global_per_image_classes.items():
        key = key_for_external_id(ext_id, loose)
        path = None
        for ds_name in dataset_order:
            path = per_ds_indexes[ds_name].get(key)
            if path:
                break
        if not path:
            missing.append(ext_id)
            continue

        base = max(r_class.get(c, 1.0) for c in cls_set) if cls_set else 1.0
        r_img = int(round(min(args.cap, max(1.0, base))))
        if str(path) not in seen_paths:
            seen_paths.add(str(path))
            combined_records.append((str(path), r_img, tuple(sorted(cls_set))))

    # Write combined list
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p, r, _ in combined_records:
            for _ in range(r):
                f.write(p + "\n")

    # Optionally write per-dataset lists
    if args.per_dataset_out:
        for name, ds_root, images_dir, ndjsons in datasets:
            out_b = args.out.with_name(f"{name}_train_rfs.txt")
            idx = per_ds_indexes[name]
            lines = 0
            with open(out_b, "w", encoding="utf-8") as f:
                for ext_id, cls_set in global_per_image_classes.items():
                    key = key_for_external_id(ext_id, loose)
                    p = idx.get(key)
                    if not p:
                        continue
                    base = max(r_class.get(c, 1.0) for c in cls_set) if cls_set else 1.0
                    r_img = int(round(min(args.cap, max(1.0, base))))
                    for _ in range(r_img):
                        f.write(str(p) + "\n")
                        lines += 1
            log(f"[INFO] Wrote {out_b} ({lines} lines)")

    # Summary
    log("\n===== RFS SUMMARY =====")
    log(f"Datasets found:          {len(datasets)}")
    for name, ds_root, images_dir, ndjsons in datasets:
        log(f"  - {name}: images={images_dir}, ndjsons={len(ndjsons)}")
    log(f"NDJSON-labeled images:   {len(global_per_image_classes)}")
    log(f"Matched local images:    {len(combined_records)}")
    log(f"Missing (no local file): {len(missing)}")
    if missing:
        log(f"  e.g.: {missing[:6]}")
    log(f"Total instances:         {total_inst}")
    for c in sorted(freq):
        log(f"  {c}: freq={freq[c]:.4f}  r(c)={r_class[c]:.3f}  instances={global_instance_counts[c]}")
    log(f"T (threshold):           {args.threshold}")
    log(f"Repeat cap:              {args.cap}")
    log(f"Combined output:         {args.out}  (lines={sum(r for _, r, _ in combined_records)})")

if __name__ == "__main__":
    main()
