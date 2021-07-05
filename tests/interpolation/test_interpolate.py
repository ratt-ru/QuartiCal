import pytest
import xarray
import dask
from itertools import product
from quartical.config.internal import gains_to_chain
from quartical.gains.gain import gain_spec_tup
from quartical.interpolation.interpolate import (load_and_interpolate_gains,
                                                 convert_and_drop,
                                                 domain_slice)
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
    return [[xds] for xds in mock_gain_xds_list(10, 10, 10, 3, 2, 4, 2, 3)]


@pytest.fixture(scope="module")
def load_xds_list():
    return mock_gain_xds_list(0, 10, 10, 4, 0, 2, 4, 4)


# ------------------------------convert_and_drop-------------------------------

def test_foo(load_xds_list):
    return
    # import pdb; pdb.set_trace()


expected_slicing = {
    (10, 19, (0, 20, 40), (9, 29, 49)): slice(0, 2),
    (10, 19, (0, 10, 20), (9, 19, 29)): slice(1, 2),
    ( 8, 22, (0, 10, 20), (9, 19, 29)): slice(0, 3),
    (12, 18, (0, 10, 20), (9, 19, 29)): slice(1, 2),
    (50, 59, (0, 20, 40, 60, 80), (9, 29, 49, 69, 89)): slice(2, 4),
    (30, 39, (0, 10, 20, 30, 40), (9, 19, 29, 39, 49)): slice(3, 4),
    (18, 42, (0, 10, 20, 30, 40), (9, 19, 29, 39, 49)): slice(1, 5),
    (22, 28, (0, 10, 20, 30, 40), (9, 19, 29, 39, 49)): slice(2, 3)
}


@pytest.mark.parametrize("input,expected", expected_slicing.items())
def test_slices(input, expected):
    lb, ub, lbounds, ubounds = input
    result = domain_slice(lb, ub, np.array(lbounds), np.array(ubounds))
    assert result == expected


def test_load_and_interpolate_gains(gain_xds_list,
                                    chain_opts,
                                    load_xds_list,
                                    monkeypatch):

    monkeypatch.setattr(
        "quartical.interpolation.interpolate.xds_from_zarr",
        lambda store: load_xds_list
    )

    # import pdb; pdb.set_trace()

    interp_xds_list = load_and_interpolate_gains(gain_xds_list, chain_opts)

    foo = dask.compute(interp_xds_list)
    import pdb;pdb.set_trace()