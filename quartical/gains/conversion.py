import dask.array as da
import numpy as np
from quartical.gains.parameterized_gain import ParameterizedGain


class Converter(object):

    def __init__(self, gain_obj):
        self.parameterized = issubclass(gain_obj.__class__, ParameterizedGain)
        self.converted_dtype = gain_obj.converted_dtype
        self.native_dtype = gain_obj.native_dtype
        self.native_to_converted = gain_obj.native_to_converted
        self.converted_to_native = gain_obj.converted_to_native

    @property
    def conversion_ratio(self):

        input_fields = sum(cf[0] for cf in self.native_to_converted)

        output_fields = len(self.native_to_converted)

        return (input_fields, output_fields)

    def convert(self, arr):

        cr = self.conversion_ratio
        dtype = self.converted_dtype

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

        inp_ind = 0
        out_ind = 0

        while inp_ind < arr.shape[-1]:
            for (n_consumed, cfs) in self.native_to_converted:
                tmp = arr[..., inp_ind]
                for cf in cfs:
                    tmp = cf(tmp)
                out_arr[..., out_ind] = tmp[...]
                inp_ind += n_consumed
                out_ind += 1

        return out_arr

    def revert(self, arr):

        cr = self.conversion_ratio
        dtype = self.native_dtype

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

        inp_ind = 0
        out_ind = 0

        while inp_ind < arr.shape[-1]:
            for (n_consumed, rf) in self.converted_to_native:
                inputs = [arr[..., inp_ind + k] for k in range(n_consumed)]
                out_arr[..., out_ind] = rf(*inputs)
                inp_ind += n_consumed
                out_ind += 1

        return out_arr


def no_op(passthrough):
    return passthrough


def trig_to_angle(cos_arr, sin_arr):
    return np.arctan2(sin_arr, cos_arr)


def amp_to_complex(amp_arr):
    return amp_arr * np.exp(1j)


def amp_trig_to_complex(amp_arr, cos_arr, sin_arr):
    return amp_arr * np.exp(1j * np.arctan2(sin_arr, cos_arr))
