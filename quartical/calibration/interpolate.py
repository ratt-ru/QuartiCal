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
    # a bit too much about the way the gain directory is structured.
    return float(re.findall("(\d+)", x)[-1])  # noqa


def load_and_interpolate_gains(gain_xds_list, opts):
    """Load and interpolate gains in accordance with opts.

    Given the gain datasets which are to be applied/solved for, determine
    whether any are to be loaded from disk. Interpolates on-disk datasets
    to be consistent with the solvable datasets.

    Args:
        gain_xds_list: List of xarray.Datasets containing gains.
        opts: A Namespace of globla options.

    Returns:
        A list like gain_xds_list with the relevant gains loaded from disk.
    """

    interp_xds_lol = []

    for term_ind, term in enumerate(opts.solver_gain_terms):

        gain_path = getattr(opts, f"{term}_load_from", None)

        # Pull out all the datasets for the current term into a flat list.
        term_xds_list = [tlist[term_ind] for tlist in gain_xds_list]

        # If the gain_path is None, this term doesn't require loading/interp.
        if gain_path is None:
            interp_xds_lol.append(term_xds_list)
            continue

        # TODO: We are assuming that the datasets are monotonically increasing
        # in time with file name. The multi-SPW case may not work.
        load_order = sorted(glob.glob(gain_path), key=sort_key)

        load_xds_list = [xarray.open_zarr(pth) for pth in load_order]

        # Convert to amp and phase. Drop unused data_vars.
        ampphase_xds_list = convert_and_drop(load_xds_list)

        # Figure out which datasets need to be concatenated.
        concat_xds_list = make_concat_xds_list(term_xds_list,
                                               ampphase_xds_list)

        interp_xds_list = make_interp_xds_list(term_xds_list,
                                               concat_xds_list)

        interp_xds_lol.append(interp_xds_list)

    # This return reverts the datasets to the expected ordering/structure.

    return [list(xds_list) for xds_list in zip(*interp_xds_lol)]


def convert_and_drop(load_xds_list):
    """Convert complex gain into amplitude and phase. Drop unused data_vars."""

    ampphase_xds_list = []

    for load_xds in load_xds_list:
        # Convert the complex gain into amplitide and phase.
        ampphase_xds = load_xds.assign(
            {"phase": (("t_int", "f_int", "ant", "dir", "corr"),
                       da.angle(load_xds.gains.data)),
             "amp": (("t_int", "f_int", "ant", "dir", "corr"),
                     da.absolute(load_xds.gains.data))})

        # Drop the unecessary data vars. TODO: Parametrised case.
        drop_vars = ("gains", "conv_perc", "conv_iter")
        interp_xds = ampphase_xds.drop_vars(drop_vars)

        ampphase_xds_list.append(interp_xds)

    return ampphase_xds_list


def make_concat_xds_list(term_xds_list, interp_xds_list):
    """Map datasets on disk to the dataset required for calibration."""

    # TODO: This is not yet adequate for the multi-SPW case. That will require
    # concatenation over two dimensions and this mapping will be more
    # complicated. Behaviour of solution ternsfer needs to be verified.

    # Figure out the upper and lower time bounds of each dataset.
    time_lbounds = [xds.t_int.values[0] for xds in interp_xds_list]
    time_ubounds = [xds.t_int.values[-1] for xds in interp_xds_list]

    concat_xds_list = []

    for txds in term_xds_list:

        glb = txds.t_int.data[0]
        gub = txds.t_int.data[-1]

        overlaps = ~((gub < time_lbounds) | (glb > time_ubounds))

        sel = np.where(overlaps)[0]
        slice_lb = sel[0]
        slice_ub = sel[-1] + 1  # Python indexing is not inclusive.

        # If the lower bound is zero, leave as is, else, include lb - 1.
        slice_lb = slice_lb - 1 if slice_lb else slice_lb
        # Upper slice may fall off the end - this is safe.
        slice_ub = slice_ub + 1

        concat_slice = slice(slice_lb, slice_ub)

        # Concatenate gains near the interpolation values.
        concat_xds = xarray.concat(interp_xds_list[concat_slice],
                                   "t_int",
                                   join="exact")

        # Remove the chunking from the concatenated datasets.
        concat_xds = concat_xds.chunk({"t_int": -1, "f_int": -1})

        concat_xds_list.append(concat_xds)

    return concat_xds_list


def make_interp_xds_list(term_xds_list, concat_xds_list):
    """Given the concatenated datasets, interp to the desired datasets."""

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

        interp_xds_list.append(interp_xds)

    return interp_xds_list


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
