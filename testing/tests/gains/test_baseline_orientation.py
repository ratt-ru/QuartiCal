"""A baseline's orientation must not change any antenna's JHJ.

Each row contributes to both of its antennas: to ANTENNA1 through V_pq and to
ANTENNA2 through V_qp, the conjugate transpose of V_pq. Writing every baseline
the other way round - swapping its antennas, conjugate transposing its data and
model and transposing its weights - describes the same measurements, so each
antenna's JHJ has to come out the same. V_qp's XY correlation is V_pq's YX, so
it carries the YX weight.

The weights differ between XY and YX here, as robust reweighting makes them.
Equal cross-hand weights would hide a qp side which reads them untransposed.
"""
import numpy as np
import pytest

from quartical.calibration.solver import meta_args_nt
from quartical.gains import TERM_TYPES

N_ANT = 4
N_TIME = 3
N_CHAN = 4
CORRELATIONS = ["XX", "XY", "YX", "YY"]

CHAN_FREQ = np.linspace(8.5e8, 1.7e9, N_CHAN)
CHAN_WIDTH = np.full(N_CHAN, CHAN_FREQ[1] - CHAN_FREQ[0])

# Indices which transpose a row-major (XX, XY, YX, YY) correlation axis.
TRANSPOSE = [0, 2, 1, 3]

SOLVABLE_TERMS = [
    name for name, term in TERM_TYPES.items() if term.solver is not None
]


def make_ms_inputs(term_class, flip):
    """Return random ms_inputs, with every baseline flipped if requested."""

    rng = np.random.default_rng(452)

    ant1, ant2 = np.triu_indices(N_ANT, k=1)
    n_bl = ant1.size
    n_row = N_TIME * n_bl

    antenna1 = np.tile(ant1, N_TIME).astype(np.int32)
    antenna2 = np.tile(ant2, N_TIME).astype(np.int32)

    vis_shape = (n_row, N_CHAN, len(CORRELATIONS))
    model = rng.normal(size=vis_shape) + 1j * rng.normal(size=vis_shape)
    data = rng.normal(size=vis_shape) + 1j * rng.normal(size=vis_shape)
    weight = rng.uniform(0.1, 2.0, size=vis_shape)

    if flip:
        antenna1, antenna2 = antenna2, antenna1
        model = model[..., TRANSPOSE].conj()
        data = data[..., TRANSPOSE].conj()
        weight = weight[..., TRANSPOSE]

    freq_fields = {
        "CHAN_FREQ": CHAN_FREQ,
        "MIN_FREQ": CHAN_FREQ.min(),
        "MAX_FREQ": CHAN_FREQ.max()
    }

    return term_class.ms_inputs(
        MODEL_DATA=model[:, :, None, :],
        DATA=data,
        ANTENNA1=antenna1,
        ANTENNA2=antenna2,
        WEIGHT=weight,
        FLAG=np.zeros(vis_shape[:2], dtype=np.int8),
        ROW_MAP=None,
        ROW_WEIGHTS=None,
        TIME=np.repeat(np.arange(N_TIME, dtype=np.float64), n_bl),
        **{
            field: freq_fields[field]
            for field in term_class.ms_inputs._fields if field in freq_fields
        }
    )


def compute_jhj(term_class, flip):
    """Return the term's JHJ at identity gains over one solution interval."""

    ms_inputs = make_ms_inputs(term_class, flip)
    n_row = ms_inputs.ANTENNA1.size

    time_bins = np.zeros(N_TIME, dtype=np.int32)
    time_map = np.zeros(n_row, dtype=np.int32)
    freq_map = term_class._make_freq_map(CHAN_FREQ, CHAN_WIDTH, 0)
    dir_map = np.zeros(1, dtype=np.int32)

    gains = np.zeros(
        (1, freq_map.max() + 1, N_ANT, 1, len(CORRELATIONS)),
        dtype=np.complex128
    )
    gains[..., (0, 3)] = 1
    gain_flags = np.zeros(gains.shape[:-1], dtype=np.int8)

    mapping_fields = {
        "time_bins": (time_bins,),
        "time_maps": (time_map,),
        "freq_maps": (freq_map,),
        "dir_maps": (dir_map,)
    }
    chain_fields = {"gains": (gains,), "gain_flags": (gain_flags,)}

    if term_class.is_parameterized:
        param_freq_map = term_class._make_param_freq_map(
            CHAN_FREQ, CHAN_WIDTH, 0
        )
        n_param = len(term_class.make_param_names(CORRELATIONS))
        params = np.full(
            (1, param_freq_map.max() + 1, N_ANT, 1, n_param),
            term_class.param_identity_fill
        )
        mapping_fields.update(
            param_time_bins=(time_bins,),
            param_time_maps=(time_map,),
            param_freq_maps=(param_freq_map,)
        )
        chain_fields.update(
            params=(params,),
            param_flags=(np.zeros(params.shape[:-1], dtype=np.int8),)
        )

    # A solve of zero iterations returns the JHJ at the starting point.
    meta_inputs = meta_args_nt(
        iters=0,
        active_term=0,
        stop_frac=1.0,
        stop_crit=1e-6,
        threads=1,
        robust=False,
        reference_antenna=0,
        scalar=False,
        dd_term=False,
        pinned_directions=(0,),
        solve_per="antenna",
        referenced=False
    )

    jhj, _, _ = term_class.solver(
        ms_inputs,
        term_class.mapping_inputs(**mapping_fields),
        term_class.chain_inputs(**chain_fields),
        meta_inputs,
        len(CORRELATIONS)
    )

    return jhj


@pytest.mark.parametrize("term_type", SOLVABLE_TERMS)
def test_jhj_independent_of_baseline_orientation(term_type):

    term_class = TERM_TYPES[term_type]

    jhj = compute_jhj(term_class, flip=False)
    flipped_jhj = compute_jhj(term_class, flip=True)

    assert np.any(jhj), "JHJ is zero!"
    np.testing.assert_allclose(
        flipped_jhj, jhj, rtol=1e-10, atol=1e-10 * np.abs(jhj).max()
    )
