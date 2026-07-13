"""
Speckle noise reduction with BM3D (implementation_plan.md, Setup > Preprocessing).
"""

import numpy as np
import bm3d # type: ignore


def denoise(bscan, sigma=10/255):
    """Denoise a single B-scan with BM3D.

    bscan: grayscale image as a uint8 or float array (any range).
    sigma: noise standard deviation in [0, 1] units.
    Returns a uint8 array of the same shape.
    """
    img_float = bscan.astype(np.float32) / 255.0

    denoised = bm3d.bm3d(
        z=img_float,
        sigma_psd=sigma,
        stage_arg=bm3d.BM3DStages.ALL_STAGES,
    )

    denoised = np.clip(denoised, 0, 1)
    return (denoised * 255).astype(np.uint8)


def estimate_sigma(bscan, background_rows=15):
    """Estimate BM3D's sigma_psd from a B-scan's noise-only background band.

    bscan: grayscale image as a uint8 or float array (any range).
    background_rows: number of rows from the top of the image assumed to be
        signal-free background (e.g. vitreous), per the flattening/cropping
        applied upstream in preprocessing.
    Returns sigma in [0, 1] units, ready to pass to denoise().
    """
    noise_patch = bscan[:background_rows, :].astype(np.float32)
    noise_patch = noise_patch[noise_patch > 0]  # exclude black padding from flattening
    return np.std(noise_patch) / 255.0