# make_rfs_list.py
# Build train lists using Repeat-Factor Sampling (instance-aware).
# Produces, for each split:
#   - splits/train_rfs_sqrt_<tag>.txt   (gentle RFS)
#   - splits/train_rfs_linear_<tag>.txt (strong RFS)

import argparse, os, re, random
from collections import Counter
from pathlib import Path

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
            line = line.strip()
            if not line:
                continue
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
        c = count_classes_in_label(lbl)
        per_img.append((img, c))
        inst_tot.update(c)
    return per_img, inst_tot

def rfs_repeat_factors(inst_tot, alpha=0.5):
    """
    r_c = max(1, (f_min / f_c)^alpha)
    alpha=0.5 -> sqrt scaling (gentle)
    alpha=1.0 -> linear scaling (stronger)
    """
    tot_instances = sum(inst_tot.values())
    if tot_instances == 0:
        return {c: 1.0 for c in inst_tot}
    freq = {c: inst_tot[c] / tot_instances for c in inst_tot}
    f_min = min(freq.values()) if freq else 1.0
    r_class = {c: max(1.0, (f_min / max(freq[c], 1e-12)) ** alpha) for c in inst_tot}
    return r_class

def make_list(per_img, r_class, max_repeat=6, seed=0):
    out = []
    rng = random.Random(seed)
    for img, c in per_img:
        r = 1.0
        for cls, n in c.items():
            if n > 0:
                r = max(r, r_class.get(cls, 1.0) * (n ** 0.5))
        r = min(max_repeat, r)
        base = int(r)
        frac = r - base
        repeats = base + (1 if rng.random() < frac else 0)
        out.extend([img] * max(1, repeats))
    rng.shuffle(out)
    return out

def write_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items))

def process_one(train_path, tag, args):
    base = read_lines(train_path)
    per_img, inst_tot = build_index(base)
    print(f"\n=== {tag} ===")
    print(f"Train list: {train_path}  (images: {len(base)})")
    print(f"Total instances: {dict(inst_tot)}")

    r_class_sqrt = rfs_repeat_factors(inst_tot, alpha=args.alpha_sqrt)
    r_class_lin  = rfs_repeat_factors(inst_tot, alpha=args.alpha_linear)

    lst_sqrt   = make_list(per_img, r_class_sqrt, max_repeat=args.sqrt_max_repeat,  seed=args.seed)
    lst_linear = make_list(per_img, r_class_lin,  max_repeat=args.linear_max_repeat, seed=args.seed)

    p_sqrt   = os.path.join("splits", f"train_rfs_sqrt_{tag}.txt")
    p_linear = os.path.join("splits", f"train_rfs_linear_{tag}.txt")
    write_list(p_sqrt, lst_sqrt)
    write_list(p_linear, lst_linear)

    print(f"Wrote:\n  {p_sqrt}   (n={len(lst_sqrt)})\n  {p_linear} (n={len(lst_linear)})")
    return p_sqrt, p_linear

def main():
    ap = argparse.ArgumentParser(description="Build RFS train lists for one or more train splits.")
    ap.add_argument("--trains", nargs="+", required=True, help="Paths to train list(s), e.g. splits/train_*.txt")
    ap.add_argument("--tags", nargs="*", help="Tags to append (auto from filename if omitted)")
    ap.add_argument("--alpha_sqrt", type=float, default=0.5, help="Alpha for sqrt/gentle scaling")
    ap.add_argument("--alpha_linear", type=float, default=1.0, help="Alpha for linear/strong scaling")
    ap.add_argument("--sqrt_max_repeat", type=int, default=6, help="Max repeats for sqrt version")
    ap.add_argument("--linear_max_repeat", type=int, default=8, help="Max repeats for linear version")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tags = args.tags or [Path(p).stem.replace("train_", "") for p in args.trains]
    if len(tags) != len(args.trains):
        raise SystemExit("[ERROR] Number of --tags must match number of --trains")

    for train_path, tag in zip(args.trains, tags):
        if not os.path.exists(train_path):
            print(f"[WARN] File not found: {train_path}")
            continue
        process_one(train_path, tag, args)

if __name__ == "__main__":
    main()
