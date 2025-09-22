from PIL import Image
import numpy as np


def get_pixel_stats(mask_path):
    """
    Calculate pixel counts for a coral reef segmentation mask.

    Labels:
        0 = background
        1 = healthy coral
        2 = bleached coral
        (3 = dead coral, optional)
    """
    # Load mask as numpy array
    mask = np.array(Image.open(mask_path))
    height, width = mask.shape
    total_pixels = height * width

    # Counts
    pix_bg = np.sum(mask == 0)
    pix_healthy = np.sum(mask == 1)
    pix_bleached = np.sum(mask == 2)
    pix_dead = np.sum(mask == 3) if np.any(mask == 3) else 0  # optional

    return {
        "file": mask_path,
        "width": width,
        "height": height,
        "total_pixels": total_pixels,
        "pix_bg": pix_bg,
        "pix_healthy": pix_healthy,
        "pix_bleached": pix_bleached,
        "pix_dead": pix_dead
    }


print(get_pixel_stats("masks_multi/c1_bc_em_t1_29nov24_cgomez_corr.png"))