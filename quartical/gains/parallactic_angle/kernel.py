# -*- coding: utf-8 -*-
from numba import njit, literally
from numba.extending import overload
from quartical.utils.numba import JIT_OPTIONS
import quartical.gains.general.factories as factories


@njit(**JIT_OPTIONS)
def parallactic_angle_params_to_gains(
    params,
    gains,
    param_freq_map,
    feed_type,
    corr_mode
):
    return parallactic_angle_params_to_gains_impl(
        params,
        gains,
        param_freq_map,
        literally(feed_type),
        literally(corr_mode)
    )


def parallactic_angle_params_to_gains_impl(
    params,
    gains,
    param_freq_map,
    feed_type,
    corr_mode
):
    return NotImplementedError


@overload(
    parallactic_angle_params_to_gains_impl,
    jit_options=JIT_OPTIONS,
    prefer_literal=True
)
def nb_parallactic_angle_params_to_gains_impl(
    params,
    gains,
    param_freq_map,
    feed_type,
    corr_mode
):

    rotmat = factories.rotation_factory(corr_mode, feed_type)

    def impl(
        params,
        gains,
        param_freq_map,
        feed_type,
        corr_mode
    ):

        n_time, n_freq, n_ant, n_dir, n_corr = gains.shape

        for t in range(n_time):
            for f in range(n_freq):
                f_m = param_freq_map[f]
                for a in range(n_ant):
                    for d in range(n_dir):

                        g = gains[t, f, a, d]
                        p = params[t, f_m, a, d, 0]

                        rotmat(p, p, g)

    return impl