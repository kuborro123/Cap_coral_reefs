# make_oversampling_list.py  (random oversampling, safe version)
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
    c = Counter()
    if not os.path.exists(lbl_path):
        return c
    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts = line.split()
            try:
                cls = int(float(parts[0])); c[cls] += 1
            except Exception:
                pass
    return c

def build_index(train_list):
    per_img, totals = [], Counter()
    for img in train_list:
        cnt = count_classes_in_label(image_to_label_path(img))
        per_img.append((img, cnt))
        totals.update(cnt)
    return per_img, totals

def compute_max_reachable_ratio(per_img, soft_id, hard_id, max_repeat_per_image):
    base = Counter()
    for _, cnt in per_img: base.update(cnt)
    S0, H0 = base[soft_id], (base[hard_id] if base[hard_id] > 0 else 1)
    S_add = H_add = 0
    for _, cnt in per_img:
        if cnt.get(soft_id, 0) > 0:
            S_add += cnt.get(soft_id, 0) * max_repeat_per_image
            H_add += cnt.get(hard_id, 0) * max_repeat_per_image
    S_max, H_max = S0 + S_add, H0 + H_add
    return S0, H0, S_max, H_max, (S_max / max(H_max, 1))

def random_oversample(per_img, soft_id, hard_id, target_ratio=1.0,
                      max_repeat_per_image=8, max_iterations=2_000_000,
                      progress_every=100_000):
    rng = random.Random(0)
    out = [img for img, _ in per_img]
    totals = Counter()
    soft_pool = []
    for img, cnt in per_img:
        totals.update(cnt)
        if cnt.get(soft_id, 0) > 0:
            soft_pool.append((img, cnt))
    if totals[soft_id] == 0:
        raise RuntimeError("No soft (minority) instances found—check class ids and labels.")
    if totals[hard_id] == 0:
        totals[hard_id] = 1
    def ratio(t): return t[soft_id] / max(t[hard_id], 1)

    # Reachability clamp
    _, _, _, _, max_reach = compute_max_reachable_ratio(per_img, soft_id, hard_id, max_repeat_per_image)
    if target_ratio > max_reach + 1e-9:
        print(f"[WARN] Target ratio {target_ratio:.3f} unreachable with max_repeat={max_repeat_per_image}. "
              f"Clamping to ≈ {max_reach:.3f}.")
        target_ratio = max_reach

    if ratio(totals) >= target_ratio:
        rng.shuffle(out); return out, dict(totals)

    repeats = Counter()
    iters = 0
    while ratio(totals) < target_ratio and iters < max_iterations:
        iters += 1
        img, cnt = rng.choice(soft_pool)
        if repeats[img] >= max_repeat_per_image:
            continue
        out.append(img); repeats[img] += 1; totals.update(cnt)
        if progress_every and (iters % progress_every) == 0:
            print(f"[oversample] iter={iters:,}  ratio={ratio(totals):.4f}  target={target_ratio:.4f}  "
                  f"capped_imgs={sum(1 for v in repeats.values() if v>=max_repeat_per_image)}")
    if iters >= max_iterations and ratio(totals) < target_ratio - 1e-6:
        print(f"[WARN] Stopped at {iters:,} iters; ratio={ratio(totals):.4f} < target={target_ratio:.4f}. "
              f"Try higher --max_repeat or lower --target.")
    rng.shuffle(out); return out, dict(totals)

def write_list(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(items))

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Random oversampling for class balance (soft coral).")
    ap.add_argument("--train", required=True, help="Path to baseline train list")
    ap.add_argument("--soft_id", type=int, required=True, help="Soft coral class id (usually 1)")
    ap.add_argument("--hard_id", type=int, default=0, help="Hard coral class id (usually 0)")
    ap.add_argument("--tag", type=str, default="", help="Suffix for output filename (e.g., realistic)")
    ap.add_argument("--target", type=float, default=1.0, help="Target soft:hard ratio")
    ap.add_argument("--max_repeat", type=int, default=8, help="Max repeats per image")
    ap.add_argument("--max_iter", type=int, default=2_000_000, help="Failsafe iteration cap")
    ap.add_argument("--progress_every", type=int, default=50_000, help="Progress print frequency (0 to disable)")
    args = ap.parse_args()

    base = read_lines(args.train)
    per_img, base_tot = build_index(base)
    print("=== Baseline instance totals ===")
    for k in sorted(base_tot): print(f"class {k}: {base_tot[k]}")
    print(f"Images in baseline: {len(per_img)}")
    S0,H0,_,_,max_reach = compute_max_reachable_ratio(per_img, args.soft_id, args.hard_id, args.max_repeat)
    curr = S0/max(H0,1)
    print(f"Current soft:hard ≈ {curr:.3f} | Max reachable with max_repeat={args.max_repeat}: ≈ {max_reach:.3f}")

    out_list, out_tot = random_oversample(
        per_img, args.soft_id, args.hard_id,
        target_ratio=args.target,
        max_repeat_per_image=args.max_repeat,
        max_iterations=args.max_iter,
        progress_every=args.progress_every
    )
    tag = f"__{args.tag}" if args.tag else ""
    out_path = os.path.join("splits", f"train_oversampling{tag}.txt")
    write_list(out_path, out_list)
    final_ratio = out_tot[args.soft_id] / max(out_tot[args.hard_id], 1)
    print(f"\nWrote:\n  {out_path} (lines: {len(out_list)})")
    print(f"Approx totals after oversampling: {out_tot}")
    print(f"Final soft:hard ≈ {final_ratio:.3f}")
    print(f'Use with: --names "{args.tag or "oversampling"}:{out_path}"')


if __name__ == "__main__":
    main()
