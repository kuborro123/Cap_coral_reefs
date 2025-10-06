# make_balanced_list.py
# Builds two new train lists under splits/:
#  - train_balanced_mild.txt
#  - train_balanced_strong.txt
# Works with YOLOv8-seg labels (polygon .txt) and class ids from your data.yaml (hard=0, soft=1).
import argparse, os, re, math, random
from collections import Counter

def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def image_to_label_path(img_path):
    # images/.../name.jpg -> labels/.../name.txt (handles Windows \ and /)
    s = img_path.replace("\\", "/")
    s = re.sub(r"[\\/]+images[\\/]+", "/labels/", s)
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", ".txt", s, flags=re.I)
    return s

def count_classes_in_label(lbl_path):
    c = Counter()
    if not os.path.exists(lbl_path):
        return c
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(float(parts[0]))
                c[cls] += 1
            except Exception:
                pass
    return c

def build_index(train_list):
    per_img = []
    totals  = Counter()
    for img in train_list:
        lbl = image_to_label_path(img)
        cnt = count_classes_in_label(lbl)
        per_img.append((img, cnt))
        totals.update(cnt)
    return per_img, totals

def make_oversampled(per_img, soft_id, hard_id, target_ratio=1.0, cap_repeat=6, light_undersample=True):
    """
    Repeat images that contain soft-class polygons so that the global soft:hard instance ratio
    moves toward 'target_ratio' (e.g., 1.0 ≈ balanced). 'cap_repeat' prevents extreme duplication.
    """
    tot = Counter()
    for _, c in per_img:
        tot.update(c)

    soft = tot[soft_id]
    hard = tot[hard_id]
    if soft == 0:
        raise RuntimeError("No soft (class_id) instances found—check class ids and labels.")

    current = soft / max(hard, 1)        # current soft:hard ratio
    gamma   = target_ratio / max(current, 1e-8)  # how much to boost soft images

    out, rng = [], random.Random(0)
    for img, c in per_img:
        s = c.get(soft_id, 0)
        h = c.get(hard_id, 0)
        repeats = 1

        # Oversample proportional to number of soft polygons
        if s > 0:
            extra = gamma * s                  # more soft polygons -> more repeats
            base  = min(cap_repeat-1, int(extra))
            frac  = min(1.0, extra - base)
            repeats += base + (1 if rng.random() < frac else 0)

        # Optional: lightly thin very hard-dominant images to keep epoch size reasonable
        if light_undersample and s == 0 and h >= 3:
            if rng.random() < 0.5:
                repeats = 0

        out.extend([img] * repeats)

    rng.shuffle(out)
    return out, dict(tot)

def write_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to baseline train.txt (e.g., splits/train.txt)")
    ap.add_argument("--soft_id", type=int, required=True, help="Soft coral class id (your yaml: 1)")
    ap.add_argument("--hard_id", type=int, default=0, help="Hard coral class id (your yaml: 0)")
    args = ap.parse_args()

    base = read_lines(args.train)
    per_img, totals = build_index(base)
    print("=== Baseline instance totals ===")
    for k in sorted(totals): print(f"class {k}: {totals[k]}")
    print(f"Images in baseline: {len(per_img)}")

    # Mild ≈ 1:1 target; Strong ≈ 1.5:1 target
    os_mild,  _ = make_oversampled(per_img, args.soft_id, args.hard_id, target_ratio=1.0, cap_repeat=6, light_undersample=True)
    os_strong, _ = make_oversampled(per_img, args.soft_id, args.hard_id, target_ratio=1.5, cap_repeat=8, light_undersample=True)

    p_mild   = os.path.join("splits", "train_balanced_mild.txt")
    p_strong = os.path.join("splits", "train_balanced_strong.txt")
    write_list(p_mild, os_mild)
    write_list(p_strong, os_strong)

    print("\nWrote:")
    print(f"  {p_mild}   (lines: {len(os_mild)})")
    print(f"  {p_strong} (lines: {len(os_strong)})")
    print("\nTip: use these with your train script via --names \"mild:splits/train_balanced_mild.txt\" etc.")

if __name__ == "__main__":
    main()
