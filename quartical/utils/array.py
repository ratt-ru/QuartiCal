import numpy as np
import dask.array as da
from uuid import uuid4


def flat_ident_like(arr):
    """Given a dask.array, returns an equivalent "identity" array.

    An identity array is identitiy in its trailing dimension. This is a little
    unusual but comes up consistently when dealing with the measurement set.

    Args:
        arr: A dask.array with a given shape, dtype and chunking.

    Returns:
        A "identity" dask.array with identical shape, dtype and chunking.
    """

    if arr.shape[-1] not in (1, 2, 4):
        raise ValueError(
            "flat_ident_like only supports trailing dims of 1, 2 and 4."
        )

    def _flat_ident_like(chunk):
        out = np.zeros_like(chunk)

        last_dim = chunk.shape[-1]

        if last_dim == 4:
            out[..., (0, 3)] = 1
        elif last_dim in (1, 2):
            out[:] = 1
        else:
            raise ValueError(
                "flat_ident_like only supports trailing dims of 1, 2 and 4."
            )

        return out

    dim_str = [f"d{i}" for i in range(arr.ndim)]

    return da.blockwise(
        _flat_ident_like, dim_str,
        arr, dim_str,
        dtype=arr.dtype,
        name="flat_ident_like-" + uuid4().hex
    )
