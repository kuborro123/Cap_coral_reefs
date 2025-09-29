import os, random
random.seed(0)

BASE = r"splits\train.txt"                 # your current train list
OUT  = r"splits\train_soft_1to10.txt"      # new list to write
MINORITY_ID, RATIO = 1, 10                   # 1=soft, make 1:10, here we can manipulate ratios

with open(BASE, "r", encoding="utf-8") as f:
    imgs = [l.strip().replace("\\","/") for l in f if l.strip()]

def lbl(p):
    q = p.replace("/images/","/labels/")
    return os.path.splitext(q)[0] + ".txt"

def has_cls(lbl_path, cls):
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            return any(line.strip().startswith(f"{cls} ") for line in f)
    except FileNotFoundError:
        return False

minor, major = [], []
for im in imgs:
    (minor if has_cls(lbl(im), MINORITY_ID) else major).append(im)

N = len(imgs)
n_minor = max(1, N // (RATIO + 1))    # 1 part minority
n_major = N - n_minor                 # RATIO parts majority

def sample(pool, k):
    return [random.choice(pool)]*k if len(pool)==1 else [random.choice(pool) for _ in range(k)]

new_list = sample(minor, n_minor) + sample(major, n_major)
random.shuffle(new_list)

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(new_list))
print(f"Wrote {OUT} ({len(new_list)} lines)")