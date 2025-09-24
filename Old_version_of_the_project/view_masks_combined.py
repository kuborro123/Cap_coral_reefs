import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Choose file name to display
mask_file = "C1_BC_EP_T2_29nov24_CDaza_corr_combined.png"

project_root = os.path.dirname(__file__)
data_dir = os.path.join(project_root, "data")
combined_dir = os.path.join(data_dir, "masks_combined")
mask_path = os.path.join(combined_dir, mask_file)

if not os.path.exists(mask_path):
    raise FileNotFoundError(f"File not found: {mask_path}")

mask = np.array(Image.open(mask_path))

plt.imshow(mask, cmap="viridis")  # color map makes classes visible
plt.colorbar()
plt.title(f"Combined Mask: {mask_file}")
plt.show()
