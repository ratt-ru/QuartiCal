import cubicalv2.kernels.complex as cmplx
import cubicalv2.kernels.phase as phase
from numba.extending import overload
from numba import jit, literally


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_jhj_jhr(model, gain_list, inverse_gain_list, residual, a1, a2,
                    weights, t_map_arr, f_map_arr, d_map_arr, active_term,
                    corr_mode, term_type):

    return _compute_jhj_jhr(model, gain_list, inverse_gain_list, residual, a1,
                            a2, weights, t_map_arr, f_map_arr, d_map_arr,
                            active_term, literally(corr_mode),
                            literally(term_type))


def _compute_jhj_jhr(model, gain_list, inverse_gain_list, residual, a1, a2,
                     weights, t_map_arr, f_map_arr, d_map_arr, active_term,
                     corr_mode, term_type):
    pass


@overload(_compute_jhj_jhr, inline="always")
def _compute_jhj_jhr_impl(model, gain_list, inverse_gain_list, residual, a1,
                          a2, weights, t_map_arr, f_map_arr, d_map_arr,
                          active_term, corr_mode, term_type):

    if corr_mode.literal_value == "diag":
        if term_type.literal_value == "cmplx":
            return cmplx.jhj_jhr_diag
        elif term_type.literal_value == "phase":
            return phase.jhj_jhr_diag
    else:
        if term_type.literal_value == "cmplx":
            return cmplx.jhj_jhr_full
        elif term_type.literal_value == "phase":
            raise NotImplementedError("Phase-only gain not yet supported in "
                                      "non-diagonal modes.")


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def compute_update(jhj, jhr, corr_mode, term_type):

    return _compute_update(jhj, jhr, literally(corr_mode),
                           literally(term_type))


def _compute_update(jhj, jhr, corr_mode, term_type):
    pass


@overload(_compute_update, inline="always")
def _compute_update_impl(jhj, jhr, corr_mode, term_type):

    if corr_mode.literal_value == "diag":
        if term_type.literal_value == "cmplx":
            return cmplx.update_diag
        elif term_type.literal_value == "phase":
            return phase.update_diag
    else:
        if term_type.literal_value == "cmplx":
            return cmplx.update_full
        elif term_type.literal_value == "phase":
            raise NotImplementedError("Phase-only gain not supported in "
                                      "non-diagonal modes.")


@jit(nopython=True, fastmath=True, parallel=False, cache=True, nogil=True)
def finalize_update(update, params, gain, i_num, dd_term, corr_mode,
                    term_type):

    return _finalize_update(update, params, gain, i_num, dd_term, corr_mode,
                            term_type)


def _finalize_update(update, params, gain, i_num, dd_term, corr_mode,
                     term_type):
    pass


@overload(_finalize_update, inline="always")
def _finalize_update_impl(update, params, gain, i_num, dd_term, corr_mode,
                          term_type):

    if corr_mode.literal_value == "diag":
        if term_type.literal_value == "cmplx":
            return cmplx.finalize_full  # Diagonal case is no different.
        elif term_type.literal_value == "phase":
            return phase.finalize_diag
    else:
        if term_type.literal_value == "cmplx":
            return cmplx.finalize_full
        elif term_type.literal_value == "phase":
            raise NotImplementedError("Phase-only gain not yet supported in "
                                      "non-diagonal modes.")
