import dask.array as da
import numpy as np


def initialize_weights(xds, data_col, corr_slice, opts):

    if opts._unity_weights:
        weight_col = da.ones_like(data_col[:, :1, :], dtype=np.float32)
    else:
        weight_col = xds[opts.input_ms_weight_column].data[..., corr_slice]

    # The following handles the fact that the chosen weight column might
    # not have a frequency axis.

    if weight_col.ndim == 2:
        weight_col = weight_col.map_blocks(
            lambda w: np.expand_dims(w, 1), new_axis=1)

    return weight_col
