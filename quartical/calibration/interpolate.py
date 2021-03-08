# -*- coding: utf-8 -*-
from loguru import logger  # noqa
import dask.array as da
import numpy as np
import xarray
import glob
import re
from scipy.interpolate import pchip_interpolate


def sort_key(x):
    """Key for finding all gains"""
    # TODO: This needs to be more flexible/simpler - it is currently assuming
    # a bit too much  about the way the gain directory is structured.
    return float(re.findall("(\d+)", x)[-1])  # noqa


def load_and_interpolate_gains(gain_xds_list, opts):

    terms = opts.solver_gain_terms

    interp_xds_lol = []

    for term in terms:

        gain_path = getattr(opts, f"{term}_load_from", None)

        term_ind = terms.index(term)
        term_xds_list = [tlist[term_ind] for tlist in gain_xds_list]

        if gain_path is None:
            interp_xds_lol.append(term_xds_list)
            continue

        load_order = sorted(glob.glob(gain_path), key=sort_key)

        load_xds_list = [xarray.open_zarr(pth) for pth in load_order]

        interp_xds_list = []

        for load_xds in load_xds_list:
            # Convert the complex gain into amplitide and phase.
            interp_xds = load_xds.assign(
                {"phase": (("t_int", "f_int", "ant", "dir", "corr"),
                           da.angle(load_xds.gains.data)),
                 "amp": (("t_int", "f_int", "ant", "dir", "corr"),
                         da.absolute(load_xds.gains.data))})

            # Drop the unecessary data vars. TODO: Parametrised case.
            drop_vars = ("gains", "conv_perc", "conv_iter")
            interp_xds = interp_xds.drop_vars(drop_vars)

            interp_xds_list.append(interp_xds)

        time_lbounds = [xds.t_int.values[0] for xds in interp_xds_list]
        time_ubounds = [xds.t_int.values[-1] for xds in interp_xds_list]

        concat_requirements = []

        for txds in term_xds_list:

            glb = txds.t_int.data[0]
            gub = txds.t_int.data[-1]

            overlaps = ~((gub < time_lbounds) | (glb > time_ubounds))

            sel = np.where(overlaps)[0]
            slice_lb = sel[0]
            slice_ub = sel[-1] + 1  # Python indexing is not inclusive.

            # If the lower bound is zero, leave as is, else, include lb -1.
            slice_lb = slice_lb - 1 if slice_lb else slice_lb
            # Upper slice may fall off the end - this is safe.
            slice_ub = slice_ub + 1

            xds_slice = slice(slice_lb, slice_ub)

            concat_requirements.append(interp_xds_list[xds_slice])

        # Concatenate gains near the interpolation values.
        concat_xds_list = [xarray.concat(reqs, "t_int", join="exact")
                           for reqs in concat_requirements]

        # Remove the chunking from the concatenated datasets.
        concat_xds_list = [cxds.chunk({"t_int": -1, "f_int": -1})
                           for cxds in concat_xds_list]

        interp_xds_list = []

        # TODO: This is dodgy. Ideally we shouldn't be reasoning about the
        # values of the gains - we should just use the gain flags. This is
        # an interim solution.
        for txds, cxds in zip(term_xds_list, concat_xds_list):

            amp_sel = da.where(cxds.amp.data < 1e-6,
                               np.nan,
                               cxds.amp.data)

            phase_sel = da.where(cxds.amp.data < 1e-6,
                                 np.nan,
                                 cxds.phase.data)

            interp_xds = cxds.assign({"amp": (cxds.amp.dims, amp_sel),
                                      "phase": (cxds.phase.dims, phase_sel)})

            # TODO: This is INSANELY slow. Omitting until I come up with
            # a better solution.
            # interp_xds = interp_xds.interpolate_na("f_int",
            #                                        method="pchip")

            # This is a fast alternative to the above but it is definitely less
            # correct. Forward/back propagates values over missing entries.
            interp_xds = interp_xds.ffill("t_int")
            interp_xds = interp_xds.bfill("t_int")
            interp_xds = interp_xds.ffill("f_int")
            interp_xds = interp_xds.bfill("f_int")
            # If an entry is STILL missing, there is no data from which to
            # interp/propagate. Set to zero.
            interp_xds = interp_xds.fillna(0)

            interp_xds = interp_xds.interp(
                {"t_int": txds.t_int.data,
                 "f_int": txds.f_int.data},
                kwargs={"fill_value": "extrapolate"})

            gains = interp_xds.amp.data*da.exp(1j*interp_xds.phase.data)

            # TODO: This is an alternative to the 2D interpolation above.
            # gains = pchip_interpolate_gains(interp_xds, txds)

            interp_xds = txds.assign({"gains": (interp_xds.amp.dims, gains)})

            t_chunks = txds.GAIN_SPEC.tchunk
            f_chunks = txds.GAIN_SPEC.fchunk

            interp_xds = interp_xds.chunk({"t_int": t_chunks,
                                           "f_int": f_chunks})

            # TODO: This is only going to work for a single term.
            interp_xds_list.append(interp_xds)

        interp_xds_lol.append(interp_xds_list)

    return [list(xds_list) for xds_list in zip(*interp_xds_lol)]


def pchip_interpolate_gains(interp_xds, txds):

    iamp = da.blockwise(pchip_interpolate, "tfadc",
                        interp_xds.t_int.values, None,
                        interp_xds.amp.data, "tfadc",
                        txds.t_int.values, None,
                        dtype=np.float64,
                        adjust_chunks={"t": txds.dims["t_int"]})

    iamp = da.blockwise(pchip_interpolate, "tfadc",
                        interp_xds.f_int.values, None,
                        iamp, "tfadc",
                        txds.f_int.values, None,
                        axis=1,
                        dtype=np.float64,
                        adjust_chunks={"f": txds.dims["f_int"]})

    iphase = da.blockwise(pchip_interpolate, "tfadc",
                          interp_xds.t_int.values, None,
                          interp_xds.phase.data, "tfadc",
                          txds.t_int.values, None,
                          dtype=np.float64,
                          adjust_chunks={"t": txds.dims["t_int"]})

    iphase = da.blockwise(pchip_interpolate, "tfadc",
                          interp_xds.f_int.values, None,
                          iphase, "tfadc",
                          txds.f_int.values, None,
                          axis=1,
                          dtype=np.float64,
                          adjust_chunks={"f": txds.dims["f_int"]})

    gains = iamp*da.exp(1j*iphase)

    return gains
