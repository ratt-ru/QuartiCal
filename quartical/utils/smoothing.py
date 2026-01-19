from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d


def gaussian_filter1d_masked(
    input_array: np.ndarray, sigma: float, mask: Optional[np.ndarray] = None, **kwargs
) -> np.ndarray:
    """
    Applies a 1D Gaussian filter to an array using an explicit validity mask.

    Args:
        input_array: The input signal.
        sigma: Standard deviation for Gaussian kernel.
        mask: Boolean array of same shape as the input_array indivating valid
            values.
        kwargs: Keyword arguments passed into scipy.ndimage.gaussian_filter1d.
            See its documentation for a full list.

    Returns:
        The filtered array.
    """
    # Convert mask to float for convolution (0.0 vs 1.0)
    weights = mask.astype(float)

    # Set invalid vlaues to zero.
    input_filled = np.where(mask, input_array, 0.0)

    # Smooth the data.
    smoothed_data = gaussian_filter1d(input_filled, sigma, **kwargs)

    # Smooth the weights.
    smoothed_weights = gaussian_filter1d(weights, sigma, **kwargs)

    # Normalise the smoothed data.
    with np.errstate(invalid="ignore", divide="ignore"):
        output = smoothed_data / smoothed_weights

    # Retun a smoothed version of the data which is still zero at flagged locations.
    return np.where(mask, output, 0)
