# -*- coding: utf-8 -*-
from numba import prange, literally, generated_jit, types
from quartical.gains.general.convenience import (get_row,
                                                 get_chan_extents,
                                                 get_row_extents)
import quartical.gains.general.factories as factories
from collections import namedtuple


@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def compute_weights(model, gains, residual, icov, v,
                    weights, t_map_arr, f_map_arr, active_term, row_map, corr_mode):

    rb_weight_mult = factories.rb_weight_mult(corr_mode)
    rb_weight_upd = factories.rb_weight_upd(corr_mode)

    def impl(model, gains, residual, icov, v, 
            weights, t_map_arr, f_map_arr, active_term, row_map, corr_mode):
        _, n_chan, n_dir, n_corr = model.shape

        n_tint, n_fint, n_ant, n_gdir, _ = gains[active_term].shape
        n_int = n_tint*n_fint

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval. This could even be moved out for speed.
        row_starts, row_stops = get_row_extents(t_map_arr,
                                                active_term,
                                                n_tint)

        chan_starts, chan_stops = get_chan_extents(f_map_arr,
                                                   active_term,
                                                   n_fint,
                                                   n_chan)

        # Parallel over all solution intervals.
        for i in prange(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                
                for f in range(fs, fe):
                    r = residual[row, f]
                    denom = rb_weight_mult(r, icov)
                    rb_weight_upd(v, denom, weights[row, f])

        return
    return impl

@generated_jit(nopython=True, fastmath=True, parallel=False, cache=True,
               nogil=True)
def compute_cov(model, gains, residual, icov,
                    weights, t_map_arr, f_map_arr, active_term, row_map, corr_mode):

    rb_cov_mult = factories.rb_cov_mult(corr_mode)

    def impl(model, gains, residual, icov, 
            weights, t_map_arr, f_map_arr, active_term, row_map, corr_mode):
        _, n_chan, n_dir, n_corr = model.shape

        n_tint, n_fint, n_ant, n_gdir, _ = gains[active_term].shape
        n_int = n_tint*n_fint

        # Determine the starts and stops of the rows and channels associated
        # with each solution interval. This could even be moved out for speed.
        row_starts, row_stops = get_row_extents(t_map_arr,
                                                active_term,
                                                n_tint)

        chan_starts, chan_stops = get_chan_extents(f_map_arr,
                                                   active_term,
                                                   n_fint,
                                                   n_chan)

        # Parallel over all solution intervals. ... I change prange to range
        for i in range(n_int):

            ti = i//n_fint
            fi = i - ti*n_fint

            rs = row_starts[ti]
            re = row_stops[ti]
            fs = chan_starts[fi]
            fe = chan_stops[fi]

            for row_ind in range(rs, re):

                row = get_row(row_ind, row_map)
                
                for f in range(fs, fe):
                    r = residual[row, f]
                    w = weights[row, f, 0]
                    rb_cov_mult(r, icov, w)                    
        return
    return impl
