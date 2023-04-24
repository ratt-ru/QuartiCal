import pytest
import xarray
import dask.array as da
from itertools import product
from collections import namedtuple
from quartical.gains.gain import gain_spec_tup
from quartical.interpolation.interpolate import (
    load_and_interpolate_gains,
    make_interpolated_xds_list
)
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
            "gain_time": time_range,
            "gain_freq": freq_range,
            "antenna": np.arange(n_ant),
            "direction": np.arange(n_dir),
            "correlation": np.arange(n_corr)
        }

        gains = da.zeros((n_time, n_freq, n_ant, n_dir, n_corr),
                         dtype=np.complex128)
        gains += da.array([1, 0, 0, 1])

        flags = da.zeros((n_time, n_freq, n_ant, n_dir),
                         dtype=np.int8)

        gain_axes = (
            "gain_time",
            "gain_freq",
            "antenna",
            "direction",
            "correlation"
        )

        # Include a dummy data_var to check that it doesn't break anything.
        data_vars = {
            "gains": (gain_axes, gains),
            "gain_flags": (gain_axes[:-1], flags),
            "dummy": (("antenna",), np.arange(n_ant))
        }

        attrs = {
            "TYPE": 'complex',
            "GAIN_AXES": gain_axes,
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
                                        "2dspline"])
def interp_method(request):
    return request.param

# -----------------------------make_interp_xds_list----------------------------


@pytest.fixture(scope="module")
def merged_xds(load_xds_list):
    xds = xarray.combine_by_coords(
        [xds[['gains', 'gain_flags']] for xds in load_xds_list],
        combine_attrs='drop_conflicts'
    )
    return xds.chunk({**xds.dims, "antenna": 1})


@pytest.fixture(scope="module")
def interpolated_xds_list(
    term_xds_list,
    merged_xds,
    interp_mode,
    interp_method
):

    return make_interpolated_xds_list(
        term_xds_list,
        merged_xds,
        interp_mode,
        interp_method
    )


def test_has_gains(interpolated_xds_list):
    assert all(hasattr(xds, "gains") for xds in interpolated_xds_list)


def test_chunking(interpolated_xds_list, term_xds_list):
    # TODO: Chunking behaviour is tested but not adequately probed yet.
    assert all(ixds.chunks == txds.chunks
               for ixds, txds in zip(interpolated_xds_list, term_xds_list))

# -----------------------------load_and_interpolate----------------------------


@pytest.fixture(scope="function")
def interp_xds_lol(gain_xds_lod, chain, load_xds_list, monkeypatch):

    monkeypatch.setattr(
        "quartical.interpolation.interpolate.xds_from_zarr",
        lambda store: load_xds_list
    )

    return load_and_interpolate_gains(gain_xds_lod, chain)


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
