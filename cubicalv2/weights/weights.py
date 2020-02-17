import dask.array as da
import numpy as np
import uuid


def initialize_weights(xds, data_col, corr_slice, opts):
    """Given an input dataset, initializes the weights based on opts.

    Initialises the weights. Data column is required in order to stat up unity
    weights.

    Inputs:
        xds: xarray.dataset on which the weight columns live.
        data_col: Chunked dask.array containing the data.
        corr_slice: A tuple of Slice objects to select out releveant weights.
        opts: A Namespace of options.

    Outputs:
        weight_col: A chunked dask.array containing the weights.
    """

    if opts._unity_weights:
        n_row, n_chan, n_corr = data_col.shape
        row_chunks, chan_chunks, corr_chunks = data_col.chunks
        weight_col = da.ones((n_row, 1, n_corr), chunks=(row_chunks, (1,),
                             corr_chunks), name="weights-" + uuid.uuid4().hex,
                             dtype=np.float32)
    else:
        weight_col = xds[opts.input_ms_weight_column].data[..., corr_slice]

    # The following handles the fact that the chosen weight column might
    # not have a frequency axis.

    if weight_col.ndim == 2:
        weight_col = weight_col.map_blocks(
            lambda w: np.expand_dims(w, 1), new_axis=1)

    return weight_col
