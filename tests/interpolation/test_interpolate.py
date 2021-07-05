import pytest
import xarray
import dask
from itertools import product
from quartical.config.internal import gains_to_chain
from quartical.gains.gain import gain_spec_tup
from quartical.interpolation.interpolate import load_and_interpolate_gains
import numpy as np
from copy import deepcopy


def mock_gain_xds_list(start_time,
                       n_time,
                       gap_time,
                       n_xds_time,
                       start_freq,
                       n_freq,
                       gap_freq,
                       n_xds_freq):

    n_ant = 7
    n_dir = 1
    n_corr = 4

    _gain_xds_list = []

    for t_ind, f_ind in product(range(n_xds_time), range(n_xds_freq)):

        time_lb = start_time + t_ind*(n_time + gap_time)
        time_range = np.arange(time_lb, time_lb + n_time)

        freq_lb = start_freq + f_ind*(n_freq + gap_freq)
        freq_range = np.arange(freq_lb, freq_lb + n_freq)

        coords = {
            "gain_t": time_range,
            "gain_f": freq_range,
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


@pytest.fixture(scope="module")
def gain_xds_list():
    return [[xds] for xds in mock_gain_xds_list(10, 10, 10, 3, 4, 4, 2, 3)]


def test_load_and_interpolate_gains(gain_xds_list,
                                    chain_opts,
                                    monkeypatch):

    monkeypatch.setattr(
        "quartical.interpolation.interpolate.xds_from_zarr",
        lambda store: mock_gain_xds_list(10, 10, 10, 3, 4, 4, 2, 3)
    )

    interp_xds_list = load_and_interpolate_gains(gain_xds_list, chain_opts)

    foo = dask.compute(interp_xds_list)
    import pdb;pdb.set_trace()