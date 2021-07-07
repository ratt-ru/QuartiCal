import dask.array as da
import numpy as np
import uuid


def initialize_weights(xds, data_col, weight_col_name):
    """Given an input dataset, initializes the weights based on ms_opts.

    Initialises the weights. Data column is required in order to stat up unity
    weights.

    Inputs:
        xds: xarray.dataset on which the weight columns live.
        data_col: Chunked dask.array containing the data.
        weight_col_name: String containing name of input weight column.

    Outputs:
        weight_col: A chunked dask.array containing the weights.
    """

    if not weight_col_name:
        n_row, n_chan, n_corr = data_col.shape
        weight_col = da.ones((n_row, n_chan, n_corr),
                             chunks=data_col.chunks,
                             name="weights-" + uuid.uuid4().hex,
                             dtype=np.float32)

    else:
        weight_col = xds[weight_col_name].data

    # The following handles the fact that the chosen weight column might
    # not have a frequency axis.

    if weight_col.ndim == 2:
        weight_col = da.broadcast_to(weight_col[:, None, :],
                                     data_col.shape,
                                     chunks=data_col.chunks)

    return weight_col
