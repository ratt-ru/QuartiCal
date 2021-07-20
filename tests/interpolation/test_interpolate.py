import pytest
import xarray
import dask.array as da
from itertools import product
from collections import namedtuple
from quartical.config.internal import gains_to_chain
from quartical.gains.gain import gain_spec_tup
from quartical.interpolation.interpolate import (load_and_interpolate_gains,
                                                 convert_and_drop,
                                                 sort_datasets,
                                                 domain_slice,
                                                 make_concat_xds_list,
                                                 make_interp_xds_list)
import numpy as np
from copy import deepcopy


# TODO: These deliberately have enough points to work with all interpolation
# methods. Add tests for the case when we don't.

GAIN_PROPERTIES = {
    "between": ((0, 2, 2, 2, 0, 2, 2, 2), (2, 2, 2, 1, 2, 2, 2, 1)),
    "aligned": ((0, 4, 4, 1, 0, 4, 4, 1), (0, 4, 4, 1, 0, 4, 4, 1)),
    "overlap": ((0, 2, 2, 2, 0, 2, 2, 2), (1, 4, 2, 1, 1, 4, 2, 1)),
    "contain": ((0, 4, 4, 1, 0, 4, 4, 1), (1, 2, 6, 1, 1, 2, 6, 1)),
    "outside": ((2, 4, 4, 1, 2, 4, 4, 1), (0, 2, 4, 2, 0, 2, 4, 2)),
}


BOUNDS = namedtuple("BOUNDS", "min_t max_t min_f max_f")


def mock_gain_xds_list(start_time,
                       n_time,
                       gap_time,
                       n_xds_time,
                       start_freq,
                       n_freq,
                       gap_freq,
                       n_xds_freq):

    n_ant = 3
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

        gains = da.zeros((n_time, n_freq, n_ant, n_dir, n_corr),
                         dtype=np.complex128)
        gains += da.array([1, 0, 0, 1])

        # Include a dummy data_var to check that it doesn't break anything.
        data_vars = {
            "gains": (("gain_t", "gain_f", "ant", "dir", "corr"), gains),
            "dummy": (("ant",), np.arange(n_ant))
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
def opts(base_opts, interp_mode, interp_method):

    # Don't overwrite base config - instead duplicate and update.

    _opts = deepcopy(base_opts)

    _opts.solver.terms = ["G", "B"]
    _opts.G.load_from = "ignored"  # Must not be None.
    _opts.G.interp_method = interp_method
    _opts.G.interp_mode = interp_mode

    return _opts


@pytest.fixture(scope="module")
def chain_opts(opts):
    return gains_to_chain(opts)


@pytest.fixture(scope="module", params=GAIN_PROPERTIES.values())
def _params(request):
    return request.param[0], request.param[1]


@pytest.fixture(scope="module")
def load_params(_params):
    return _params[0]


@pytest.fixture(scope="module")
def gain_params(_params):
    return _params[1]


@pytest.fixture(scope="module")
def gain_xds_lod(gain_params):
    return [{"G": xds, "B": xds} for xds in mock_gain_xds_list(*gain_params)]


@pytest.fixture(scope="module")
def term_xds_list(gain_xds_lod):
    return [xds_list["G"] for xds_list in gain_xds_lod]


@pytest.fixture(scope="module")
def load_xds_list(load_params):
    return mock_gain_xds_list(*load_params)


# ------------------------------convert_and_drop-------------------------------

@pytest.fixture(scope="module", params=["reim", "ampphase"])
def interp_mode(request):
    return request.param


@pytest.fixture(scope="module", params=["2dlinear",
                                        "2dspline",
                                        "smoothingspline"])
def interp_method(request):
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
    if len(converted_xds_list) > 1:  # A single dataset is always on a grid.
        with pytest.raises(ValueError):
            # Check that we fail when gains don't fall on a grid.
            sort_datasets(converted_xds_list[:-1])
    else:
        assert True


# ---------------------------------domain_slice--------------------------------


expected_slicing = {
    (10, 19, (0, 20, 40), (9, 29, 49)): slice(0, 2),  # Between
    (10, 19, (0, 10, 20), (9, 19, 29)): slice(1, 2),  # Aligned
    ( 8, 22, (0, 10, 20), (9, 19, 29)): slice(0, 3),  # Overlap  # noqa
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


@pytest.fixture(scope="module")
def bounds(load_params):
    start, step, skip, reps = load_params[:4]  # Time
    min_t = start
    max_t = start + (step + skip)*reps - skip - 1
    start, step, skip, reps = load_params[4:]  # Freq
    min_f = start
    max_f = start + (step + skip)*reps - skip - 1

    return BOUNDS(min_t, max_t, min_f, max_f)


def test_nxds(concat_xds_list, term_xds_list):
    assert len(concat_xds_list) == len(term_xds_list)


def test_time_lower_bounds(concat_xds_list, term_xds_list, bounds):

    concat_lb = [xds.gain_t.values[0] for xds in concat_xds_list]
    term_lb = [xds.gain_t.values[0] for xds in term_xds_list]
    req_extrapolation = [tlb < bounds.min_t for tlb in term_lb]

    zipper = zip(concat_lb, term_lb, req_extrapolation)

    assert all([(clb <= tlb) or re for clb, tlb, re in zipper])


def test_time_upper_bounds(concat_xds_list, term_xds_list, bounds):

    concat_ub = [xds.gain_t.values[-1] for xds in concat_xds_list]
    term_ub = [xds.gain_t.values[-1] for xds in term_xds_list]
    req_extrapolation = [tub > bounds.max_f for tub in term_ub]

    zipper = zip(concat_ub, term_ub, req_extrapolation)

    assert all([(cub >= tub) or re for cub, tub, re in zipper])


def test_freq_lower_bounds(concat_xds_list, term_xds_list, bounds):

    concat_lb = [xds.gain_f.values[0] for xds in concat_xds_list]
    term_lb = [xds.gain_f.values[0] for xds in term_xds_list]
    req_extrapolation = [tlb < bounds.min_f for tlb in term_lb]

    zipper = zip(concat_lb, term_lb, req_extrapolation)

    assert all([(clb <= tlb) or re for clb, tlb, re in zipper])


def test_freq_upper_bounds(concat_xds_list, term_xds_list, bounds):

    concat_ub = [xds.gain_f.values[-1] for xds in concat_xds_list]
    term_ub = [xds.gain_f.values[-1] for xds in term_xds_list]
    req_extrapolation = [tub > bounds.max_f for tub in term_ub]

    zipper = zip(concat_ub, term_ub, req_extrapolation)

    assert all([(cub >= tub) or re for cub, tub, re in zipper])


# -----------------------------make_interp_xds_list----------------------------


@pytest.fixture(scope="module")
def interp_xds_list(term_xds_list,
                    concat_xds_list,
                    interp_mode,
                    interp_method):
    return make_interp_xds_list(term_xds_list,
                                concat_xds_list,
                                interp_mode,
                                interp_method)


def test_has_gains(interp_xds_list):
    assert all(hasattr(xds, "gains") for xds in interp_xds_list)


def test_chunking(interp_xds_list, term_xds_list):
    # TODO: Chunking behaviour is tested but not adequately probed yet.
    assert all(ixds.chunks == txds.chunks
               for ixds, txds in zip(interp_xds_list, term_xds_list))

# -----------------------------load_and_interpolate----------------------------


@pytest.fixture(scope="function")
def interp_xds_lol(gain_xds_lod, chain_opts, load_xds_list, monkeypatch):

    monkeypatch.setattr(
        "quartical.interpolation.interpolate.xds_from_zarr",
        lambda store: load_xds_list
    )

    return load_and_interpolate_gains(gain_xds_lod, chain_opts)


@pytest.fixture(scope="function")
def compute_interp_xds_lol(interp_xds_lol):
    return da.compute(interp_xds_lol)[0]


def test_cixl_has_gains(compute_interp_xds_lol):
    assert all([hasattr(xds, "gains")
               for xds_dict in compute_interp_xds_lol
               for xds in xds_dict.values()])


def test_cixl_gains_ident(compute_interp_xds_lol):
    # NOTE: Splines will not be exactly identity due to numerical precision.
    assert all(np.allclose(xds.gains.values, np.array([1, 0, 0, 1]))
               for xds_dict in compute_interp_xds_lol
               for xds in xds_dict.values())

# -----------------------------------------------------------------------------
