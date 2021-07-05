import pytest
import xarray
import dask
from itertools import product
from quartical.config.internal import gains_to_chain
from quartical.gains.gain import gain_spec_tup
from quartical.interpolation.interpolate import (load_and_interpolate_gains,
                                                 convert_and_drop,
                                                 make_concat_xds_list,
                                                 sort_datasets,
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
def term_xds_list(gain_xds_list):
    return [xds_list[0] for xds_list in gain_xds_list]


@pytest.fixture(scope="module")
def load_xds_list():
    return mock_gain_xds_list(0, 10, 10, 4, 0, 2, 4, 4)


# ------------------------------convert_and_drop-------------------------------

@pytest.fixture(scope="module", params=["reim", "ampphase"])
def interp_mode(request):
    return request.param


@pytest.fixture(scope="module")
def converted_xds_list(load_xds_list, interp_mode):
    return convert_and_drop(load_xds_list, interp_mode)


def test_data_vars(converted_xds_list, interp_mode):
    expected_keys = {"re", "im"} if interp_mode == "reim" else {"amp", "phase"}

    assert all([set(xds.keys()) ^ expected_keys == set()
                for xds in converted_xds_list])

# --------------------------------sort_datasets--------------------------------


@pytest.fixture(scope="module")
def sorted_xds_lol(converted_xds_list):
    return sort_datasets(converted_xds_list)


def test_time_grouping(sorted_xds_lol):
    assert all([len(set(xds.gain_t.values[0] for xds in xds_list)) == 1
                for xds_list in sorted_xds_lol])


def test_time_ordering(sorted_xds_lol):
    times = [xds_list[0].gain_t.values[0] for xds_list in sorted_xds_lol]
    assert sorted(times) == times


def test_freq_ordering(sorted_xds_lol):
    for xds_list in sorted_xds_lol:
        freqs = [xds.gain_f.values[0] for xds in xds_list]
        assert sorted(freqs) == freqs


def test_nogrid(converted_xds_list):
    with pytest.raises(ValueError):
        # Check that we fail when gains don't fall on a grid.
        sort_datasets(converted_xds_list[:-1])


# ---------------------------------domain_slice--------------------------------


expected_slicing = {
    (10, 19, (0, 20, 40), (9, 29, 49)): slice(0, 2),  # Between
    (10, 19, (0, 10, 20), (9, 19, 29)): slice(1, 2),  # Aligned
    ( 8, 22, (0, 10, 20), (9, 19, 29)): slice(0, 3),  # Overlap
    (12, 18, (0, 10, 20), (9, 19, 29)): slice(1, 2),  # Contain
    (30, 39, (0, 10, 20), (9, 19, 29)): slice(2, 3),  # Outside (upper)
    (0, 9, (10, 20, 30), (19, 29, 39)): slice(0, 1),  # Outside (lower)
    (18, 32, (0, 10, 20), (9, 19, 29)): slice(1, 3),  # Overlap + upper edge
    (8, 22, (10, 20, 30), (19, 29, 39)): slice(0, 2),  # Overlap + lower edge
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

# -----------------------------make_concat_xds_list----------------------------


@pytest.fixture(scope="module")
def concat_xds_list(term_xds_list, sorted_xds_lol):
    return make_concat_xds_list(term_xds_list, sorted_xds_lol)


def test_nxds(concat_xds_list, term_xds_list):
    assert len(concat_xds_list) == len(term_xds_list)


def test_time_lower_bounds(concat_xds_list, term_xds_list):

    concat_lb = [xds.gain_t.values[0] for xds in concat_xds_list]
    term_lb = [xds.gain_t.values[0] for xds in term_xds_list]

    assert [clb < tlb for clb, tlb in zip(concat_lb, term_lb)]


def test_time_upper_bounds(concat_xds_list, term_xds_list):

    concat_ub = [xds.gain_t.values[-1] for xds in concat_xds_list]
    term_ub = [xds.gain_t.values[-1] for xds in term_xds_list]

    assert [cub < tub for cub, tub in zip(concat_ub, term_ub)]


def test_freq_lower_bounds(concat_xds_list, term_xds_list):

    concat_lb = [xds.gain_f.values[0] for xds in concat_xds_list]
    term_lb = [xds.gain_f.values[0] for xds in term_xds_list]

    assert [clb < tlb for clb, tlb in zip(concat_lb, term_lb)]


def test_freq_upper_bounds(concat_xds_list, term_xds_list):

    concat_ub = [xds.gain_f.values[-1] for xds in concat_xds_list]
    term_ub = [xds.gain_f.values[-1] for xds in term_xds_list]

    assert [cub < tub for cub, tub in zip(concat_ub, term_ub)]


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
    # import pdb;pdb.set_trace()