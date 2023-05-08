import numpy as np
import dask.array as da
import xarray
from quartical.interpolation.interpolants import (
    interpolate_missing,
    linear2d_interpolate_gains
)
from quartical.gains.gain import Gain
from quartical.gains.converter import (
    Converter,
    amp_trig_to_complex
)
from quartical.gains.complex.kernel import complex_solver, complex_args
from quartical.gains.complex.diag_kernel import diag_complex_solver


class Complex(Gain):

    solver = staticmethod(complex_solver)
    term_args = complex_args
    # Conversion functions required for interpolation NOTE: Non-parameterised
    # gains will always be reinterpreted and parameterised in amplitude and
    # phase for the sake of simplicity.
    conversion_functions = (
        (0, (np.abs,)),
        (0, (np.angle, np.cos)),
        (1, (np.angle, np.sin))
    )
    reversion_functions = (
        (3, amp_trig_to_complex),
    )

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def to_interpable(cls, xds):

        converter = Converter(
            cls.conversion_functions,
            cls.reversion_functions
        )

        params = converter.convert(xds.gains.data, xds.gains.data.real.dtype)
        param_flags = xds.gain_flags.data

        params = da.where(param_flags[..., None], np.nan, params)

        param_dims = xds.gains.dims[:-1] + ('parameter',)

        interpable_xds = xarray.Dataset(
            {
                "params": (param_dims, params),
                "param_flags": (param_dims[:-1], param_flags)
            },
            coords=xds.coords,
            attrs=xds.attrs
        )

        return interpable_xds

    @classmethod
    def interpolate(cls, source_xds, target_xds, term_opts):

        filled_params = interpolate_missing(source_xds.params)

        source_xds = source_xds.assign(
            {"params": (source_xds.params.dims, filled_params.data)}
        )

        interpolated_xds = linear2d_interpolate_gains(source_xds, target_xds)

        t_chunks = target_xds.GAIN_SPEC.tchunk
        f_chunks = target_xds.GAIN_SPEC.fchunk

        # We may be interpolating from one set of axes to another.
        t_t_axis, t_f_axis = target_xds.GAIN_AXES[:2]

        interpolated_xds = interpolated_xds.chunk(
            {
                t_t_axis: t_chunks,
                t_f_axis: f_chunks,
                "antenna": interpolated_xds.dims["antenna"]
            }
        )

        return interpolated_xds

    @classmethod
    def from_interpable(cls, xds):

        converter = Converter(
            cls.conversion_functions,
            cls.reversion_functions
        )

        gains = converter.revert(xds.params.data, np.complex128)

        gain_dims = xds.params.dims[:-1] + ('correlation',)

        native_xds = xarray.Dataset(
            {
                "gains": (gain_dims, gains),
            },
            coords=xds.coords,
            attrs=xds.attrs
        )

        return native_xds


class DiagComplex(Complex):

    solver = staticmethod(diag_complex_solver)

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)
