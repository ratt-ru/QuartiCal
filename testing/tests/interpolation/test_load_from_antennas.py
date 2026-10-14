# -*- coding: utf-8 -*-
"""End-to-end check of load_from when the antennas of two observations differ.

A gain store is built from the target's own gain datasets and then altered to
reproduce each realistic mismatch before being loaded back through the full
calibration graph. An antenna without usable solutions in the store must end
up fully flagged: in its gains and, through flag propagation, in the data.
"""
from copy import deepcopy

import dask
import numpy as np
import pytest
import xarray
from daskms.experimental.zarr import xds_to_zarr

from quartical.calibration.calibrate import add_calibration_graph
from quartical.config.internal import gains_to_chain
from quartical.gains.datasets import make_gain_xds_lod

# Index of the antenna which each case perturbs in the gain store.
PERTURBED = 1

# Case -> whether the perturbed antenna is left without usable solutions.
CASES = {
    "missing": True,        # Absent from the store.
    "fully_flagged": True,  # Present in the store but flagged everywhere.
    "extra": False,         # The store holds an antenna the target lacks.
    "reordered": False,     # The store lists the antennas in reverse.
}


@pytest.fixture(scope="module")
def opts(base_opts):

    _opts = deepcopy(base_opts)
    _opts.solver.terms = ["G"]

    return _opts


@pytest.fixture(scope="module", params=CASES.keys())
def case(request):
    return request.param


@pytest.fixture(scope="module", params=["complex", "delay"])
def term_type(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 5], ids=lambda i: f"iters{i}")
def iters(request):
    return request.param


def with_solutions(xds):
    """Give a gain dataset identity solutions and no raised flags."""

    gain_shape = tuple(xds.sizes[ax] for ax in xds.GAIN_AXES)
    gains = np.zeros(gain_shape, dtype=np.complex128)
    gains[..., (0, -1)] = 1  # The identity for one, two or four correlations.

    solutions = {
        "gains": (xds.GAIN_AXES, gains),
        "gain_flags": (
            xds.GAIN_AXES[:-1], np.zeros(gain_shape[:-1], dtype=np.int8)
        ),
    }

    if hasattr(xds, "PARAM_SPEC"):
        param_shape = tuple(xds.sizes[ax] for ax in xds.PARAM_AXES)
        solutions["params"] = (xds.PARAM_AXES, np.zeros(param_shape))
        solutions["param_flags"] = (
            xds.PARAM_AXES[:-1], np.zeros(param_shape[:-1], dtype=np.int8)
        )

    return xds.assign(solutions)


def perturb(xds, case, flag_field):
    """Reproduce an antenna mismatch in a single gain dataset."""

    n_ant = xds.sizes["antenna"]

    if case == "missing":
        keep = [a for a in range(n_ant) if a != PERTURBED]
        return xds.isel(antenna=keep)
    elif case == "fully_flagged":
        flags = xds[flag_field].values.copy()
        flags[:, :, PERTURBED] = 1
        return xds.assign({flag_field: (xds[flag_field].dims, flags)})
    elif case == "extra":
        extra = xds.isel(antenna=[PERTURBED])
        extra = extra.assign_coords(antenna=["NOT_IN_TARGET"])
        return xarray.concat(
            [xds, extra], dim="antenna", data_vars="minimal", coords="minimal"
        )
    elif case == "reordered":
        return xds.isel(antenna=slice(None, None, -1))

    raise ValueError(f"Unknown case {case}.")


@pytest.fixture(scope="module")
def loaded_outputs(
    predicted_xds_list,
    stats_xds_list,
    opts,
    case,
    term_type,
    iters,
    tmp_path_factory
):
    """Load a perturbed store through the calibration graph and compute."""

    _opts = deepcopy(opts)
    _opts.G.type = term_type
    _opts.solver.iter_recipe = [iters]
    _opts.output.gain_directory = str(tmp_path_factory.mktemp("writes.qc"))
    load_dir = tmp_path_factory.mktemp("loads.qc")
    _opts.G.load_from = f"{load_dir}/G"

    chain = gains_to_chain(_opts)
    flag_field = chain[0].interpolation_targets[1]

    store_xds_list = [
        perturb(with_solutions(xds_dict["G"]), case, flag_field)
        for xds_dict in make_gain_xds_lod(predicted_xds_list, chain)
    ]
    dask.compute(xds_to_zarr(store_xds_list, f"{load_dir}::G"))

    gain_xds_lod, _, data_xds_list, _, _ = add_calibration_graph(
        predicted_xds_list,
        stats_xds_list,
        _opts.solver,
        chain,
        _opts.output
    )

    return dask.compute(
        [xds_dict["G"].gain_flags for xds_dict in gain_xds_lod],
        [xds.FLAG for xds in data_xds_list],
        [xds.FLAG for xds in predicted_xds_list],
        [xds.ANTENNA1 for xds in predicted_xds_list],
        [xds.ANTENNA2 for xds in predicted_xds_list],
    )


def perturbed_rows(ant1, ant2):
    return (ant1.values == PERTURBED) | (ant2.values == PERTURBED)


def test_perturbed_antenna_has_data(loaded_outputs):
    """The target must hold unflagged data for the perturbed antenna."""

    _, _, input_flags, ant1s, ant2s = loaded_outputs

    assert any(
        not flags.values[perturbed_rows(ant1, ant2)].all()
        for flags, ant1, ant2 in zip(input_flags, ant1s, ant2s)
    )


def test_unsolved_antenna_gains_flagged(loaded_outputs, case):
    """Gains are fully flagged exactly when there were no solutions."""

    gain_flags, *_ = loaded_outputs

    fully_flagged = all(
        flags.values[:, :, PERTURBED].all() for flags in gain_flags
    )

    assert fully_flagged == CASES[case]


def test_unsolved_antenna_data_flagged(loaded_outputs, case):
    """Data is fully flagged for an antenna without loaded solutions."""

    if not CASES[case]:
        pytest.skip("The perturbed antenna has loaded solutions.")

    _, output_flags, _, ant1s, ant2s = loaded_outputs

    assert all(
        flags.values[perturbed_rows(ant1, ant2)].all()
        for flags, ant1, ant2 in zip(output_flags, ant1s, ant2s)
    )


def test_other_data_flags_untouched(loaded_outputs, iters):
    """Applying loaded gains leaves the flags of other antennas unchanged."""

    if iters:
        pytest.skip("Solving may legitimately raise further flags.")

    _, output_flags, input_flags, ant1s, ant2s = loaded_outputs

    for out_flags, in_flags, ant1, ant2 in zip(
        output_flags, input_flags, ant1s, ant2s
    ):
        others = ~perturbed_rows(ant1, ant2)
        np.testing.assert_array_equal(
            out_flags.values[others], in_flags.values[others]
        )
