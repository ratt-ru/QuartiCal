import dask.array as da
import numpy as np
import uuid


def initialize_weights(xds, data_col, corr_slice, opts):

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
