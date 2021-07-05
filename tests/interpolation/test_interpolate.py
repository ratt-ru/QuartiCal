import pytest
import xarray
import dask
from itertools import product
from quartical.config.internal import gains_to_chain
from quartical.gains.gain import gain_spec_tup
from quartical.interpolation.interpolate import load_and_interpolate_gains
import numpy as np
from copy import deepcopy


@pytest.fixture(scope="module")
def opts(base_opts):

    # Don't overwrite base config - instead duplicate and update.

    _opts = deepcopy(base_opts)

    _opts.solver.terms = ["G"]
    _opts.G.load_from = ""
    _opts.G.interp_method = "2dlinear"
    _opts.G.interp_mode = "reim"
    _opts.B.load_from = ""
    _opts.B.interp_method = "2dlinear"
    _opts.B.interp_mode = "ampphase"

    return _opts


@pytest.fixture(scope="module")
def chain_opts(opts):
    return gains_to_chain(opts)


def mock_gain_xds_list(n_time, n_freq, t_offset, f_offset, n_xds_t, n_xds_f):

    n_ant = 7
    n_dir = 1
    n_corr = 4

    along_time = list(range(n_xds_t))
    along_freq = list(range(n_xds_f))

    time_ranges = [np.arange(n_time) + (n + 1)*t_offset for n in along_time]
    freq_ranges = [np.arange(n_freq) + (n + 1)*f_offset for n in along_freq]

    _gain_xds_list = []

    for t_ind, f_ind in product(along_time, along_freq):
        coords = {
            "gain_t": time_ranges[t_ind],
            "gain_f": freq_ranges[f_ind],
            "ant": np.arange(n_ant),
            "dir": np.arange(n_dir),
            "corr": np.arange(n_corr)
        }

        gains = np.zeros((n_time, n_freq, n_ant, n_dir, n_corr),
                         dtype=np.complex128)
        gains[..., (0, -1)] = 1

        data_vars = {
            "gains": (("gain_t", "gain_f", "ant", "dir", "corr"), gains)
        }

        attrs = {
            "GAIN_AXES": ("gain_t", "gain_f", "ant", "dir", "corr"),
            "GAIN_SPEC": gain_spec_tup((n_time,), (n_freq,), (n_ant,),
                                       (n_dir,), (n_corr,))
        }

        xds = xarray.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=attrs
        )

        _gain_xds_list.append(xds)

    return _gain_xds_list


@pytest.fixture(scope="module")
def gain_xds_list():
    return [[xds] for xds in mock_gain_xds_list(10, 2, 10, 4, 3, 3)]


def test_load_and_interpolate_gains(gain_xds_list,
                                    chain_opts,
                                    monkeypatch):

    monkeypatch.setattr("quartical.interpolation.interpolate.xds_from_zarr",
                        lambda store: mock_gain_xds_list(10, 2, 10, 4, 3, 3))

    interp_xds_list = load_and_interpolate_gains(gain_xds_list, chain_opts)

    foo = dask.compute(interp_xds_list)
    import pdb;pdb.set_trace()