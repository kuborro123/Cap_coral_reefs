import os
import numpy as np
from PIL import Image

project_root = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(project_root, "data")
bleached_dir = os.path.join(data_dir, "masks_bleached")
non_bleached_dir = os.path.join(data_dir, "masks_non_bleached")
combined_dir = os.path.join(data_dir, "masks_combined")

os.makedirs(combined_dir, exist_ok=True)

for fname in os.listdir(non_bleached_dir):
    if not fname.endswith(".png"):
        continue

    base = fname.replace("_non_bleached.png", "")
    bleached_file = base + "_bleached.png"
    non_bleached_file = base + "_non_bleached.png"

    path_non_bleached = os.path.join(non_bleached_dir, non_bleached_file)
    path_bleached = os.path.join(bleached_dir, bleached_file)

    if not (os.path.exists(path_non_bleached) and os.path.exists(path_bleached)):
        print(f"Missing masks for {base}")
        continue

    mnb = np.array(Image.open(path_non_bleached))
    mb = np.array(Image.open(path_bleached))

    if mnb.ndim == 3:
        mnb = mnb[:, :, 0]
    if mb.ndim == 3:
        mb = mb[:, :, 0]

    # Create combined mask (0,1,2)
    mask = np.zeros_like(mnb, dtype=np.uint8)
    mask[mnb > 0] = 1   # non-bleached
    mask[mb > 0] = 2    # bleached

    # Save combined mask
    out_path = os.path.join(combined_dir, base + "_combined.png")
    Image.fromarray(mask).save(out_path)

    print(f"Saved {out_path}")
