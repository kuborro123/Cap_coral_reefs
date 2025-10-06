# make_rfs_list.py
# Build train lists using Repeat-Factor Sampling (instance-aware).
# Produces: splits/train_rfs_sqrt.txt  (sqrt scaling), and splits/train_rfs_linear.txt (stronger).
import argparse, os, re, random
from collections import Counter

def read_lines(p):
    with open(p, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def image_to_label_path(img_path):
    s = img_path.replace("\\", "/")
    s = re.sub(r"[\\/]+images[\\/]+", "/labels/", s)
    s = re.sub(r"\.(jpg|jpeg|png|bmp)$", ".txt", s, flags=re.I)
    return s

def count_classes_in_label(lbl_path):
    cnt = Counter()
    if not os.path.exists(lbl_path):
        return cnt
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                cls = int(float(line.split()[0]))
                cnt[cls] += 1
            except Exception:
                pass
    return cnt

def build_index(train_list):
    per_img = []
    inst_tot = Counter()
    for img in train_list:
        lbl = image_to_label_path(img)
        c = count_classes_in_label(lbl)   # instance counts per class in this image
        per_img.append((img, c))
        inst_tot.update(c)
    return per_img, inst_tot

def rfs_repeat_factors(inst_tot, alpha=0.5):
    """
    Compute per-class repeat factors r_c = max(1, (f_min / f_c) ** alpha),
    where f_c is instance frequency for class c, f_min is the minimum freq.
    alpha=0.5 gives sqrt scaling (LVIS-style); alpha=1.0 is linear/stronger.
    """
    tot_instances = sum(inst_tot.values())
    freq = {c: inst_tot[c] / max(tot_instances, 1) for c in inst_tot}
    f_min = min(freq.values()) if freq else 1.0
    r_class = {c: max(1.0, (f_min / max(freq[c], 1e-12)) ** alpha) for c in inst_tot}
    return r_class

def make_list(per_img, r_class, max_repeat=6):
    out = []
    rng = random.Random(0)
    for img, c in per_img:
        # image repeat factor = max over its classes of (class repeat * presence)
        # we multiply by sqrt(n_instances_in_image + 1e-6) to slightly prefer
        # images with more minority instances without exploding.
        r = 1.0
        for cls, n in c.items():
            if n > 0:
                r = max(r, r_class.get(cls,1.0) * (n ** 0.5))
        r = min(max_repeat, r)
        base = int(r)
        frac = r - base
        repeats = base + (1 if rng.random() < frac else 0)
        out.extend([img]*max(1, repeats))
    rng.shuffle(out)
    return out

def write_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="splits/train.txt")
    ap.add_argument("--soft_id", type=int, default=1)
    ap.add_argument("--hard_id", type=int, default=0)
    args = ap.parse_args()

    base = read_lines(args.train)
    per_img, inst_tot = build_index(base)
    print("Total instances:", dict(inst_tot))

    # sqrt scaling (gentle) and linear scaling (strong)
    r_class_sqrt  = rfs_repeat_factors(inst_tot, alpha=0.5)
    r_class_lin   = rfs_repeat_factors(inst_tot, alpha=1.0)

    lst_sqrt  = make_list(per_img, r_class_sqrt, max_repeat=6)
    lst_linear= make_list(per_img, r_class_lin,  max_repeat=8)

    p_sqrt   = os.path.join("splits", "train_rfs_sqrt.txt")
    p_linear = os.path.join("splits", "train_rfs_linear.txt")
    write_list(p_sqrt, lst_sqrt)
    write_list(p_linear, lst_linear)
    print(f"Wrote:\n  {p_sqrt}   (n={len(lst_sqrt)})\n  {p_linear} (n={len(lst_linear)})")
    print("Tip: compare against baseline and mild oversampling.")

if __name__ == "__main__":
    main()
