"""Fixed-canvas padding for the DL methods (implementation_plan.md, Setup >
Input size).

DUKE-DME B-scans are 224x768, HC-MS 128x1024. One model needs one input
shape, so both are zero-padded onto a 224x1024 canvas: padding goes bottom
and right only, so every (row, column) coordinate is unchanged and boundary
row labels need no offset. Predictions are cropped back to native shape
before they reach the eval harness.

Padding, never resizing: MAD is measured in axial pixels at the 3.87 um/px
scale shared by both datasets, so rescaling rows would make the two datasets'
errors non-comparable.
"""

import numpy as np

# max(height) over DUKE-DME (224) and HC-MS (128), max(width) over
# DUKE-DME (768) and HC-MS (1024). Both are divisible by 32, so a U-Net may
# downsample up to 5 times without a fractional feature map.
CANVAS_SHAPE = (224, 1024)

# Guards pad_to_canvas against being handed a (n_boundaries, width) position
# array, whose 6 rows would otherwise be padded to 224 as if they were image
# rows. No B-scan here is anywhere near this short; boundary arrays always
# are. Use pad_width_to_canvas for those.
MIN_IMAGE_HEIGHT = 32


def pad_to_canvas(array: np.ndarray, value: float = 0.0, shape: tuple[int, int] = CANVAS_SHAPE) -> np.ndarray:
    """Pad the trailing two axes of an image-shaped array up to the canvas.

    Args:
        array: (..., H, W) array. H, W must not exceed the canvas.
        value: fill for the padded region. Use 0.0 for images and
            training.IGNORE_INDEX for class targets.
        shape: (height, width) to pad to.
    Returns:
        (..., canvas_height, canvas_width) array, original content at the
        top-left.
    Raises:
        ValueError: if the array is larger than the canvas on either axis, or
            is too short to be an image (see MIN_IMAGE_HEIGHT).
    """
    height, width = array.shape[-2:]
    canvas_height, canvas_width = shape
    if height > canvas_height or width > canvas_width:
        raise ValueError(
            f"Array of {height}x{width} exceeds canvas {canvas_height}x{canvas_width}."
        )
    if height < MIN_IMAGE_HEIGHT:
        raise ValueError(
            f"Array is {height} rows tall, too short to be a B-scan. Boundary "
            f"position arrays should use pad_width_to_canvas."
        )
    pad_width = [(0, 0)] * (array.ndim - 2)
    pad_width += [(0, canvas_height - height), (0, canvas_width - width)]
    return np.pad(array, pad_width, constant_values=value)


def pad_width_to_canvas(array: np.ndarray, value: float = np.nan, width: int = CANVAS_SHAPE[1]) -> np.ndarray:
    """Pad only the last axis, for (n_boundaries, width) position arrays.

    The boundary axis is semantic, not spatial, so it is left alone. NaN is
    the default fill because a padded column is one no method annotated or
    predicted, which is what the coverage metric already reads NaN as.

    Args:
        array: (..., width) array of y-row positions.
        value: fill for the padded columns.
        width: canvas width to pad to.
    Returns:
        (..., width) array, original content at the left.
    Raises:
        ValueError: if the array is wider than the canvas.
    """
    if array.shape[-1] > width:
        raise ValueError(f"Array of width {array.shape[-1]} exceeds canvas width {width}.")
    pad_width = [(0, 0)] * (array.ndim - 1) + [(0, width - array.shape[-1])]
    return np.pad(array, pad_width, constant_values=value)


def crop_from_canvas(array, shape: tuple[int, int]):
    """Undo pad_to_canvas by taking the top-left corner back.

    Works on both numpy arrays and torch tensors.

    Args:
        array: (..., canvas_height, canvas_width) array or tensor.
        shape: (height, width) of the native B-scan to restore.
    Returns:
        The same type, trailing axes cropped to shape.
    """
    height, width = shape
    return array[..., :height, :width]
