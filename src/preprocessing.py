# ================================================================
# Preprocessing functions for BraTS 2021 MRI volumes
# ================================================================

import numpy as np


def normalize_image(image):
    """
    Apply z-score normalization over non-zero brain voxels.

    Background voxels remain zero.

    Parameters
    ----------
    image : np.ndarray
        2D MRI slice.

    Returns
    -------
    np.ndarray
        Normalized MRI slice.
    """

    image = image.astype(np.float32)
    brain_region = image > 0

    if np.sum(brain_region) > 0:
        mean = image[brain_region].mean()
        std = image[brain_region].std()
        image[brain_region] = (image[brain_region] - mean) / (std + 1e-8)

    image[~brain_region] = 0

    return image