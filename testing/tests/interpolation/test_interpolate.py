import pytest
import xarray
import dask.array as da
from itertools import product
from collections import namedtuple
from daskms.experimental.zarr import xds_to_zarr
from quartical.config.internal import gains_to_chain
from quartical.gains import TERM_TYPES
from quartical.gains.gain import gain_spec_tup, param_spec_tup
from quartical.interpolation.interpolate import (
    load_and_interpolate_gains
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
                       n_xds_freq,
                       antennas=None,
                       amplitudes=None,
                       flagged_antennas=()):

    antennas = np.arange(3) if antennas is None else np.asarray(antennas)
    n_ant = antennas.size
    n_dir = 1
    n_corr = 4

    # Scaling the identity per antenna makes the antenna axis identifiable.
    amplitudes = np.ones(n_ant) if amplitudes is None else amplitudes

    _gain_xds_list = []

    for t_ind, f_ind in product(range(n_xds_time), range(n_xds_freq)):

        time_lb = start_time + t_ind*(n_time + gap_time)
        time_range = np.arange(time_lb, time_lb + n_time)

        freq_lb = start_freq + f_ind*(n_freq + gap_freq)
        freq_range = np.arange(freq_lb, freq_lb + n_freq)

        coords = {
            "gain_time": time_range,
            "gain_freq": freq_range,
            "antenna": antennas,
            "direction": np.arange(n_dir),
            "correlation": np.arange(n_corr)
        }

        gains = da.zeros((n_time, n_freq, n_ant, n_dir, n_corr),
                         dtype=np.complex128)
        gains += da.array([1, 0, 0, 1])
        gains *= da.array(amplitudes)[None, None, :, None, None]

        flags = np.zeros((n_time, n_freq, n_ant, n_dir), dtype=np.int8)
        flags[:, :, np.isin(antennas, flagged_antennas)] = 1
        flags = da.from_array(flags)

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
            "NAME": "G",
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


@pytest.fixture(scope="function")
def opts(base_opts, interp_mode, interp_method, tmp_path_factory):

    # Don't overwrite base config - instead duplicate and update.

    _opts = deepcopy(base_opts)

    _opts.solver.terms = ["G", "B"]
    _opts.output.gain_directory = str(tmp_path_factory.mktemp("writes.qc"))
    _opts.G.load_from = str(tmp_path_factory.mktemp("loads.qc")) + "/G"
    _opts.G.interp_method = interp_method
    _opts.G.interp_mode = interp_mode

    return _opts


@pytest.fixture(scope="function")
def chain(opts):
    return gains_to_chain(opts)


@pytest.fixture(scope="function", params=GAIN_PROPERTIES.values())
def _params(request):
    return request.param[0], request.param[1]


@pytest.fixture(scope="function")
def load_params(_params):
    return _params[0]


@pytest.fixture(scope="function")
def gain_params(_params):
    return _params[1]


@pytest.fixture(scope="function")
def gain_xds_lod(gain_params):
    return [{"G": xds, "B": xds} for xds in mock_gain_xds_list(*gain_params)]


@pytest.fixture(scope="function")
def term_xds_list(gain_xds_lod):
    return [xds_list["G"] for xds_list in gain_xds_lod]


@pytest.fixture(scope="function")
def load_xds_list(load_params, opts):

    mock_loads = mock_gain_xds_list(*load_params)

    path = '::'.join(opts.G.load_from.rsplit('/', maxsplit=1))

    da.compute(xds_to_zarr(mock_loads, path))

    return mock_loads


@pytest.fixture(scope="function", params=["reim", "ampphase"])
def interp_mode(request):
    return request.param


@pytest.fixture(scope="function", params=["2dlinear", "2dspline"])
def interp_method(request):
    return request.param

# -----------------------------make_interp_xds_list----------------------------


@pytest.fixture(scope="function")
def interpolated_xds_lod(
    gain_xds_lod,
    chain,
    opts,
    load_xds_list
):

    return load_and_interpolate_gains(
        gain_xds_lod,
        chain,
        opts.output.gain_directory
    )


def test_has_gains(interpolated_xds_lod):
    assert all(
        all(hasattr(xds, "gains") for xds in d.values())
        for d in interpolated_xds_lod
    )


def test_chunking(interpolated_xds_lod, term_xds_list):
    # TODO: Chunking behaviour is tested but not adequately probed yet.
    interpolated_xds_list = [ixds['G'] for ixds in interpolated_xds_lod]

    assert all(ixds.chunks == txds.chunks
               for ixds, txds in zip(interpolated_xds_list, term_xds_list))

# -----------------------------load_and_interpolate----------------------------


@pytest.fixture(scope="function")
def compute_interpolated_xds_lod(interpolated_xds_lod):
    return da.compute(interpolated_xds_lod)[0]


def test_cixl_has_gains(compute_interpolated_xds_lod):
    assert all([hasattr(xds, "gains")
               for xds_dict in compute_interpolated_xds_lod
               for xds in xds_dict.values()])


def test_cixl_gains_ident(compute_interpolated_xds_lod):
    # NOTE: Splines will not be exactly identity due to numerical precision.
    assert all(np.allclose(xds.gains.values, np.array([1, 0, 0, 1]))
               for xds_dict in compute_interpolated_xds_lod
               for xds in xds_dict.values())

# -----------------------------------------------------------------------------

# -------------------------------antenna alignment-----------------------------

# Solutions may be transferred between observations which do not span the same
# antennas. Each case gives the antennas of the loaded solutions and of the
# datasets they are interpolated onto.
ALIGNMENT_CASES = {
    "identical": ([0, 1, 2], [0, 1, 2]),
    "dropped": ([0, 1, 2], [0, 2]),
    "added": ([0, 2], [0, 1, 2]),
    "reordered": ([2, 1, 0], [0, 1, 2]),
}


@pytest.fixture(scope="function")
def alignment_opts(base_opts, tmp_path_factory):

    _opts = deepcopy(base_opts)

    _opts.solver.terms = ["G"]
    _opts.output.gain_directory = str(tmp_path_factory.mktemp("writes.qc"))
    _opts.G.load_from = str(tmp_path_factory.mktemp("loads.qc")) + "/G"
    _opts.G.interp_method = "2dlinear"
    _opts.G.interp_mode = "reim"

    return _opts


@pytest.fixture(
    scope="function",
    params=ALIGNMENT_CASES.values(),
    ids=ALIGNMENT_CASES.keys()
)
def alignment_case(request):
    return request.param


@pytest.fixture(scope="function")
def alignment_xds(alignment_case, alignment_opts):

    load_antennas, target_antennas = alignment_case
    load_params, gain_params = GAIN_PROPERTIES["aligned"]

    # The gains of the nth loaded antenna are n + 1 times the identity, which
    # makes the interpolated antenna axis verifiable.
    load_xds_list = mock_gain_xds_list(
        *load_params,
        antennas=load_antennas,
        amplitudes=np.arange(len(load_antennas)) + 1
    )

    path = '::'.join(alignment_opts.G.load_from.rsplit('/', maxsplit=1))
    da.compute(xds_to_zarr(load_xds_list, path))

    gain_xds_lod = [
        {"G": xds} for xds in
        mock_gain_xds_list(*gain_params, antennas=target_antennas)
    ]

    interpolated_xds_lod = load_and_interpolate_gains(
        gain_xds_lod,
        gains_to_chain(alignment_opts),
        alignment_opts.output.gain_directory
    )

    return da.compute(interpolated_xds_lod)[0][0]["G"]


def test_alignment_antennas(alignment_xds, alignment_case):
    """The interpolated antenna axis is that of the target datasets."""

    _, target_antennas = alignment_case

    assert list(alignment_xds.antenna.values) == target_antennas


def test_alignment_gains(alignment_xds, alignment_case):
    """Antennas common to both datasets retain their loaded gains."""

    load_antennas, target_antennas = alignment_case

    common = [(i, a) for i, a in enumerate(target_antennas)
              if a in load_antennas]

    assert all(
        np.allclose(
            alignment_xds.gains.values[:, :, i],
            (load_antennas.index(a) + 1) * np.array([1, 0, 0, 1])
        )
        for i, a in common
    )


def test_alignment_flags(alignment_xds, alignment_case):
    """Antennas absent from the loaded solutions are fully flagged."""

    load_antennas, target_antennas = alignment_case

    flags = alignment_xds.gain_flags.values

    assert all(
        flags[:, :, i].all() == (a not in load_antennas)
        for i, a in enumerate(target_antennas)
    )


# ------------------------------unsolved antennas------------------------------

# An antenna whose loaded solutions are flagged at every time and frequency has
# nothing from which to interpolate, exactly as if it were missing altogether.
FLAGGED_ANTENNA = 1


@pytest.fixture(scope="function")
def fully_flagged_xds(alignment_opts):

    load_params, gain_params = GAIN_PROPERTIES["between"]

    load_xds_list = mock_gain_xds_list(
        *load_params, flagged_antennas=[FLAGGED_ANTENNA]
    )

    path = '::'.join(alignment_opts.G.load_from.rsplit('/', maxsplit=1))
    da.compute(xds_to_zarr(load_xds_list, path))

    gain_xds_lod = [{"G": xds} for xds in mock_gain_xds_list(*gain_params)]

    interpolated_xds_lod = load_and_interpolate_gains(
        gain_xds_lod,
        gains_to_chain(alignment_opts),
        alignment_opts.output.gain_directory
    )

    return da.compute(interpolated_xds_lod)[0][0]["G"]


def test_fully_flagged_antenna_flagged(fully_flagged_xds):
    """An antenna flagged throughout the loaded solutions is flagged."""

    assert fully_flagged_xds.gain_flags.values[:, :, FLAGGED_ANTENNA].all()


def test_fully_flagged_antenna_leaves_others_unflagged(fully_flagged_xds):
    """Antennas with loaded solutions somewhere remain unflagged."""

    flags = fully_flagged_xds.gain_flags.values

    assert not np.delete(flags, FLAGGED_ANTENNA, axis=2).any()


# ---------------------------parameterised unsolved----------------------------

# A parameterised term interpolates its parameters, so an antenna with nothing
# to interpolate from must be flagged on the parameter grid and on the gain grid
# which init_term merges its gain flags from.
CORRELATIONS = ["XX", "XY", "YX", "YY"]

UNSOLVED_CASES = {
    "missing": dict(load_antennas=[0, 2], flagged_antennas=[]),
    "fully_flagged": dict(load_antennas=[0, 1, 2], flagged_antennas=[1]),
}


def mock_delay_xds(antennas, flagged_antennas=(), scaffold=False):
    """A delay dataset solved on a 4x1 parameter grid and a 4x4 gain grid.

    A loaded dataset carries parameters of one and the flags given by
    flagged_antennas. A scaffold carries identity gains, identity parameters
    and no raised flags on both grids, as make_gain_xds_lod produces them.
    """

    antennas = np.asarray(antennas)
    param_names = TERM_TYPES["delay"].make_param_names(CORRELATIONS)

    n_time, n_freq, n_pfreq = 4, 4, 1
    n_ant, n_dir, n_corr, n_param = antennas.size, 1, 4, len(param_names)

    gain_axes = ("gain_time", "gain_freq", "antenna", "direction",
                 "correlation")
    param_axes = ("param_time", "param_freq", "antenna", "direction",
                  "param_name")

    coords = {
        "gain_time": np.arange(n_time, dtype=np.float64),
        "gain_freq": np.arange(n_freq, dtype=np.float64),
        "param_time": np.arange(n_time, dtype=np.float64),
        "param_freq": np.array([1.5]),
        "antenna": antennas,
        "direction": np.arange(n_dir),
        "correlation": np.array(CORRELATIONS),
        "param_name": np.array(param_names),
    }

    param_flags = np.zeros((n_time, n_pfreq, n_ant, n_dir), dtype=np.int8)

    if scaffold:
        gains = np.zeros((n_time, n_freq, n_ant, n_dir, n_corr),
                         dtype=np.complex128)
        gains[..., (0, 3)] = 1
        data_vars = {
            "gains": (gain_axes, da.from_array(gains)),
            "gain_flags": (gain_axes[:-1], da.zeros(gains.shape[:-1],
                                                    dtype=np.int8)),
            "params": (param_axes, da.zeros(
                (n_time, n_pfreq, n_ant, n_dir, n_param))),
            "param_flags": (param_axes[:-1], da.from_array(param_flags)),
        }
    else:
        param_flags[:, :, np.isin(antennas, flagged_antennas)] = 1
        data_vars = {
            "params": (param_axes, da.ones(
                (n_time, n_pfreq, n_ant, n_dir, n_param))),
            "param_flags": (param_axes[:-1], da.from_array(param_flags)),
        }

    attrs = {
        "NAME": "G",
        "TYPE": "delay",
        "GAIN_AXES": gain_axes,
        "GAIN_SPEC": gain_spec_tup((n_time,), (n_freq,), (n_ant,),
                                   (n_dir,), (n_corr,)),
        "PARAM_AXES": param_axes,
        "PARAM_SPEC": param_spec_tup((n_time,), (n_pfreq,), (n_ant,),
                                     (n_dir,), (n_param,)),
    }

    return xarray.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


@pytest.fixture(
    scope="function",
    params=UNSOLVED_CASES.values(),
    ids=UNSOLVED_CASES.keys()
)
def unsolved_delay_xds(request, alignment_opts):

    alignment_opts.G.type = "delay"

    load_xds = mock_delay_xds(
        request.param["load_antennas"],
        request.param["flagged_antennas"]
    )

    path = '::'.join(alignment_opts.G.load_from.rsplit('/', maxsplit=1))
    da.compute(xds_to_zarr([load_xds], path))

    gain_xds_lod = [{"G": mock_delay_xds([0, 1, 2], scaffold=True)}]

    interpolated_xds_lod = load_and_interpolate_gains(
        gain_xds_lod,
        gains_to_chain(alignment_opts),
        alignment_opts.output.gain_directory
    )

    return da.compute(interpolated_xds_lod)[0][0]["G"]


def test_unsolved_antenna_param_flags(unsolved_delay_xds):
    """Only the antenna with nothing to interpolate from is param flagged."""

    flags = unsolved_delay_xds.param_flags.values

    assert flags[:, :, FLAGGED_ANTENNA].all()
    assert not np.delete(flags, FLAGGED_ANTENNA, axis=2).any()


def test_unsolved_antenna_gain_flags(unsolved_delay_xds):
    """Only the antenna with nothing to interpolate from is gain flagged."""

    flags = unsolved_delay_xds.gain_flags.values

    assert flags[:, :, FLAGGED_ANTENNA].all()
    assert not np.delete(flags, FLAGGED_ANTENNA, axis=2).any()
