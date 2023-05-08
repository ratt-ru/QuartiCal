import dask.array as da
import numpy as np
from itertools import cycle


class Converter(object):

    def __init__(self, conversion_functions, reversion_functions):
        self.conversion_functions = conversion_functions
        self.reversion_functions = reversion_functions

        def get_growth(cf, total=0):

            if isinstance(cf, list):
                if isinstance(cf[0], list):  # Nested!
                    nested = []
                    for ele in cf:
                        nested.append(get_growth(ele))
                    return sum(nested)
                else:
                    return True

    @property
    def conversion_ratio(self):

        input_fields = len(self.conversion_functions)

        output_fields = sum([len(x) for x in self.conversion_functions])

        return (input_fields, output_fields)

    def convert(self, arr, dtype):

        cr = self.conversion_ratio

        return da.blockwise(
            self._convert, 'tfadc',
            arr, 'tfadc',
            dtype, None,
            dtype=dtype,
            adjust_chunks={'c': arr.shape[-1] // cr[0] * cr[1]}
        )

    def _convert(self, arr, dtype):

        cr = self.conversion_ratio
        out_shape = arr.shape[:-1] + (arr.shape[-1] // cr[0] * cr[1],)

        out_arr = np.empty(out_shape, dtype=dtype)

        itr = zip(range(arr.shape[-1]), cycle(self.conversion_functions))

        offset = 0

        for i, cfsl in itr:
            for j, cfs in enumerate(cfsl):
                tmp = arr[..., i]
                for cf in cfs:
                    tmp = cf(tmp)
                out_arr[..., offset] = tmp[...]
                offset += 1

        return out_arr

    def revert(self, arr, dtype):

        cr = self.conversion_ratio

        return da.blockwise(
            self._revert, 'tfadc',
            arr, 'tfadc',
            dtype, None,
            dtype=dtype,
            adjust_chunks={'c': arr.shape[-1] // cr[1] * cr[0]}
        )

    def _revert(self, arr, dtype):

        cr = self.conversion_ratio
        out_shape = arr.shape[:-1] + (arr.shape[-1] // cr[1] * cr[0],)

        out_arr = np.empty(out_shape, dtype=dtype)

        j = 0
        for i, (n_req, rf) in enumerate(cycle(self.reversion_functions)):
            inputs = [arr[..., j + k] for k in range(n_req)]
            out_arr[..., i] = rf(*inputs)
            j += n_req
            if j >= arr.shape[-1]:
                break

        return out_arr


def noop(passthrough):
    return passthrough


def trig_to_phase(cos_arr, sin_arr):
    return np.arctan2(sin_arr, cos_arr)


def amp_trig_to_complex(amp_arr, cos_arr, sin_arr):
    return amp_arr * np.exp(1j * np.arctan2(sin_arr, cos_arr))
