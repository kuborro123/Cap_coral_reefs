# multimask_program.py
# Create multi-class masks (0=bg, 1=healthy, 2=bleached) from two-class PNGs,
# generate QC overlays, and save bleaching/coral-cover stats to CSV.

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import re

# ======================== CONFIG =========================
IMAGES_DIR   = Path("images")
BLEACHED_DIR = Path("masks_bleached")
HEALTHY_DIR  = Path("masks_non_bleached")

OUT_MULTI    = Path("masks_multi")   # training labels (grayscale 0/1/2)
OUT_OVERLAY  = Path("overlays")      # human QC (transparent + on-photo)
DOCS_DIR     = Path("docs")

RESIZE_MASKS_TO_PHOTO = True         # resize masks to match photo size
SAVE_TRANSPARENT_OVERLAY = True
SAVE_OVERLAY_ON_PHOTO    = True
OVERLAY_ALPHA = 120                  # 0..255
VERBOSE_FIRST_N = 10                 # print rich debug for first N images
QC_LIMIT = 0                         # 0 = process all; >0 = stop early after N images
# =========================================================


# -------- helper: file types --------
def is_img(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# -------- helpers: key normalization so names match --------
# We normalize any mask stem by stripping common tokens like:
# "_bleached", "_non_bleached", "_nonbleached", "_healthy", optional "mask"
BLEACHED_PAT = re.compile(r"(?:^|[_\-.])(mask_)?bleach(?:ed)?(?:[_\-.]|$)", re.I)
HEALTHY_PAT  = re.compile(r"(?:^|[_\-.])(mask_)?(?:non[_\-]?bleach(?:ed)?|healthy)(?:[_\-.]|$)", re.I)
MASK_PAT     = re.compile(r"(?:^|[_\-.])mask(?:[_\-.]|$)", re.I)

def normalize_mask_key(stem: str) -> str:
    s = stem.lower()
    s = MASK_PAT.sub("_", s)
    s = BLEACHED_PAT.sub("_", s)
    s = HEALTHY_PAT.sub("_", s)
    s = re.sub(r"[_\-.]+", "_", s).strip("_")
    return s

def img_key_from_path(p: Path) -> str:
    return p.stem.lower()

def index_images(folder: Path) -> dict:
    return {img_key_from_path(p): p for p in folder.rglob("*") if p.is_file() and is_img(p)}

def index_masks(folder: Path) -> dict:
    idx = {}
    for p in folder.rglob("*"):
        if p.is_file() and is_img(p):
            key = normalize_mask_key(p.stem)
            idx.setdefault(key, []).append(p)
    return idx


# -------- helpers: loading & binarizing masks robustly --------
def resize_if_needed(img: Image.Image, target_size) -> Image.Image:
    if RESIZE_MASKS_TO_PHOTO and img.size != target_size:
        return img.resize(target_size, Image.NEAREST)
    return img

def best_binarize(arr: np.ndarray):
    """
    Try multiple ways to get a clean binary mask from arbitrary PNGs.
    Returns (mask01, method_used, ones_count, ones_frac).
    Methods:
      A) arr > 0         (good for 0/1 masks)
      B) arr >= 128      (good for 0/255 masks)
      C) invert of A     (handles white-background, black-coral)
      D) invert of B
    We prefer not-all-zero and not-almost-all-ones (white bg) masks.
    """
    cand = []
    a0 = (arr > 0).astype(np.uint8)
    a1 = (arr >= 128).astype(np.uint8)
    a2 = 1 - a0
    a3 = 1 - a1
    for m, name in [(a0,"(>0)"), (a1,"(>=128)"), (a2,"invert(>0)"), (a3,"invert(>=128)")]:
        ones = int(m.sum())
        frac = ones / m.size
        cand.append((m, name, ones, frac))
    # scoring: prefer some positives, and a sparse-ish mask (target density ~ 0.15)
    best = None
    best_score = -1.0
    for m, name, ones, frac in cand:
        score = -1.0 if ones == 0 else 1.0 - abs(frac - 0.15)
        if score > best_score:
            best = (m, name, ones, frac)
            best_score = score
    return best  # (mask01, method_name, ones, frac)


# ========================= main ==========================
def main():
    # Ensure outputs exist
    OUT_MULTI.mkdir(exist_ok=True)
    OUT_OVERLAY.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)

    # Index inputs
    img_idx = index_images(IMAGES_DIR)
    ble_idx = index_masks(BLEACHED_DIR)
    hea_idx = index_masks(HEALTHY_DIR)

    rows = []
    n_created = 0
    n_skipped = 0
    unmatched_names = []

    for i, (img_key, img_path) in enumerate(sorted(img_idx.items())):
        # Load photo
        photo = Image.open(img_path).convert("RGB")
        W, H = photo.size

        # Find matching mask paths (exact key first)
        ble_paths = ble_idx.get(img_key, [])
        hea_paths = hea_idx.get(img_key, [])

        # Fuzzy fallback: allow partial contains either way
        if not ble_paths:
            for k, v in ble_idx.items():
                if img_key in k or k in img_key:
                    ble_paths = v
                    break
        if not hea_paths:
            for k, v in hea_idx.items():
                if img_key in k or k in img_key:
                    hea_paths = v
                    break

        ble_path = ble_paths[0] if ble_paths else None
        hea_path = hea_paths[0] if hea_paths else None

        if not ble_path and not hea_path:
            n_skipped += 1
            unmatched_names.append(img_path.name)
            rows.append({
                "image_id": img_key,
                "image_path": str(img_path),
                "mask_bleached_path": "",
                "mask_non_bleached_path": "",
                "mask_multi_path": "",
                "bleaching_pct": None,
                "coral_cover_pct": None,
                "notes": "no matching masks"
            })
            if QC_LIMIT and (n_created + n_skipped) >= QC_LIMIT:
                break
            continue

        # Load & binarize bleached
        if ble_path:
            ble_img = resize_if_needed(Image.open(ble_path).convert("L"), (W, H))
            ble_arr = np.array(ble_img)
            ble_bin, ble_meth, ble_ones, ble_frac = best_binarize(ble_arr)
        else:
            ble_bin = np.zeros((H, W), dtype=np.uint8); ble_meth="none"; ble_ones=0; ble_frac=0.0

        # Load & binarize healthy / non-bleached
        if hea_path:
            hea_img = resize_if_needed(Image.open(hea_path).convert("L"), (W, H))
            hea_arr = np.array(hea_img)
            hea_bin, hea_meth, hea_ones, hea_frac = best_binarize(hea_arr)
        else:
            hea_bin = np.zeros((H, W), dtype=np.uint8); hea_meth="none"; hea_ones=0; hea_frac=0.0

        # Debug for first few
        if i < VERBOSE_FIRST_N:
            print(f"[{img_key}]")
            print("  BLEACHED:", ble_path.name if ble_path else None,
                  f"method={ble_meth} ones={ble_ones} frac={ble_frac:.4f}")
            print("  HEALTHY :", hea_path.name if hea_path else None,
                  f"method={hea_meth} ones={hea_ones} frac={hea_frac:.4f}")

        # Fuse into multi-class: 0=bg, 1=healthy, 2=bleached (bleached wins)
        combined = np.zeros_like(ble_bin, dtype=np.uint8)
        combined[hea_bin == 1] = 1
        combined[ble_bin == 1] = 2

        # Save multi mask
        multi_path = OUT_MULTI / f"{img_key}.png"
        Image.fromarray(combined).save(multi_path)

        # Stats
        coral_pixels = int(np.count_nonzero((combined == 1) | (combined == 2)))
        bleached_pixels = int(np.count_nonzero(combined == 2))
        total_pixels = combined.size
        bleaching_pct = (bleached_pixels / max(coral_pixels, 1)) * 100.0
        coral_cover_pct = (coral_pixels / total_pixels) * 100.0

        if i < VERBOSE_FIRST_N:
            print("  COMBINED:",
                  f"healthy_sum={(combined==1).sum()}",
                  f"bleached_sum={(combined==2).sum()}",
                  f"coral_cover%={coral_cover_pct:.2f}")

        # Overlays (for human QC)
        if SAVE_TRANSPARENT_OVERLAY or SAVE_OVERLAY_ON_PHOTO:
            overlay = np.zeros((H, W, 4), dtype=np.uint8)
            overlay[combined == 1] = [0, 255, 0, OVERLAY_ALPHA]   # healthy = green
            overlay[combined == 2] = [255, 0, 0, OVERLAY_ALPHA]   # bleached = red

            if SAVE_TRANSPARENT_OVERLAY:
                Image.fromarray(overlay, "RGBA").save(OUT_OVERLAY / f"{img_key}_overlay.png")

            if SAVE_OVERLAY_ON_PHOTO:
                blended = Image.alpha_composite(photo.convert("RGBA"), Image.fromarray(overlay, "RGBA"))
                blended.save(OUT_OVERLAY / f"{img_key}_on_photo.png")

        # Row for CSV
        rows.append({
            "image_id": img_key,
            "image_path": str(img_path),
            "mask_bleached_path": str(ble_path) if ble_path else "",
            "mask_non_bleached_path": str(hea_path) if hea_path else "",
            "mask_multi_path": str(multi_path),
            "bleaching_pct": round(bleaching_pct, 4),
            "coral_cover_pct": round(coral_cover_pct, 4),
            "notes": ""
        })
        n_created += 1

        if QC_LIMIT and (n_created + n_skipped) >= QC_LIMIT:
            print(f"Stopping early after {QC_LIMIT} images for QC.")
            break

    # Write CSV
    df = pd.DataFrame(rows)
    DOCS_DIR.mkdir(exist_ok=True)
    out_csv = DOCS_DIR / "bleaching_stats.csv"
    df.to_csv(out_csv, index=False)

    # Summary
    print(f"\nDone. Combined masks created: {n_created}, images without masks skipped: {n_skipped}")
    if unmatched_names:
        print("\nImages with no matching masks (first 20):")
        for name in unmatched_names[:20]:
            print(" ", name)
    print(f"CSV saved to: {out_csv}")
    print(f"Outputs: {OUT_MULTI.resolve()}  |  {OUT_OVERLAY.resolve()}")


if __name__ == "__main__":
    # Ensure input dirs exist (nice error if not)
    for p in [IMAGES_DIR, BLEACHED_DIR, HEALTHY_DIR]:
        if not p.exists():
            print(f"[WARN] Missing folder: {p.resolve()} (script will skip if empty)")
    main()
