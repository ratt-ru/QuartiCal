import pytest
import numpy as np
from copy import deepcopy


@pytest.fixture(scope="module")
def opts(base_opts, time_chunk, freq_chunk, time_int, freq_int):

    # Don't overwrite base config - instead duplicate and update.

    _opts = deepcopy(base_opts)

    _opts.input_ms.time_chunk = time_chunk
    _opts.input_ms.freq_chunk = freq_chunk
    _opts.G.time_interval = time_int
    _opts.B.time_interval = 2*time_int
    _opts.G.freq_interval = freq_int
    _opts.B.freq_interval = 2*freq_int
    _opts.mad_flags.enable = True

    return _opts


@pytest.fixture(scope="module")
def expected_t_ints(single_xds, chain):

    n_row = single_xds.dims["row"]
    t_ints = [term.time_interval or n_row for term in chain]

    expected_t_ints = []

    da_ivl_col = single_xds.INTERVAL.data
    da_time_col = single_xds.TIME.data

    for t_int in t_ints:
        if isinstance(t_int, float):
            term_expected_t_ints = []
            for ind in range(da_ivl_col.npartitions):
                ivl_col = da_ivl_col.blocks[ind].compute()
                time_col = da_time_col.blocks[ind].compute()
                _, uinds = np.unique(time_col, return_index=True)
                utime_ivl = ivl_col[uinds]
                num_int = 0
                sol_width = 0
                for ivl in utime_ivl:
                    sol_width += ivl
                    if sol_width >= t_int:
                        num_int += 1
                        sol_width = 0
                if sol_width:
                    num_int += 1
                    sol_width = 0
                term_expected_t_ints.append(num_int)
            expected_t_ints.append(term_expected_t_ints)
        else:
            expected_t_ints.append([np.ceil(tc/t_int)
                                    for tc in single_xds.UTIME_CHUNKS])

    return expected_t_ints


@pytest.fixture(scope="module")
def expected_f_ints(single_xds, chain):

    n_chan = single_xds.dims["chan"]
    f_ints = [term.freq_interval or n_chan for term in chain]

    expected_f_ints = []

    da_chan_freq_col = single_xds.CHAN_FREQ.data
    da_chan_width_col = single_xds.CHAN_WIDTH.data

    for f_int in f_ints:
        if isinstance(f_int, float):
            term_expected_f_ints = []
            for ind in range(da_chan_freq_col.npartitions):
                chan_width_col = da_chan_width_col.blocks[ind].compute()
                num_int = 0
                sol_width = 0
                for cw in chan_width_col:
                    sol_width += cw
                    if sol_width >= f_int:
                        num_int += 1
                        sol_width = 0
                if sol_width:
                    num_int += 1
                    sol_width = 0
                term_expected_f_ints.append(num_int)
            expected_f_ints.append(term_expected_f_ints)
        else:
            expected_f_ints.append([np.ceil(fc/f_int)
                                    for fc in da_chan_freq_col.chunks[0]])

    return expected_f_ints


# ----------------------------make_gain_xds_list-------------------------------

@pytest.mark.calibrate
def test_nterm(gain_xds_lod, solver_opts):
    """Each gain term should produce a gain xds."""
    assert len(solver_opts.terms) == len(gain_xds_lod[0])


@pytest.mark.calibrate
def test_data_coords(single_xds, term_xds_dict):
    """Check that dimensions shared between the gains and data are the same."""

    data_coords = ["ant", "dir", "corr"]
    gain_coords = ["antenna", "direction", "correlation"]

    assert all(single_xds.dims[dc] == gxds.dims[gc]
               for gxds in term_xds_dict.values()
               for dc, gc in zip(data_coords, gain_coords))


@pytest.mark.calibrate
def test_t_chunking(single_xds, term_xds_dict):
    """Check that time chunking of the gain xds list is correct."""

    assert all(len(single_xds.UTIME_CHUNKS) == gxds.dims["time_chunk"]
               for gxds in term_xds_dict.values())


@pytest.mark.calibrate
def test_f_chunking(single_xds, term_xds_dict):
    """Check that frequency chunking of the gain xds list is correct."""

    assert all(len(single_xds.chunks["chan"]) == gxds.dims["freq_chunk"]
               for gxds in term_xds_dict.values())


@pytest.mark.calibrate
def test_t_ints(term_xds_dict, expected_t_ints):
    """Check that the time intervals are correct."""

    assert all(int(sum(eti)) == gxds.dims["gain_time"]
               for eti, gxds in zip(expected_t_ints, term_xds_dict.values()))


@pytest.mark.calibrate
def test_f_ints(term_xds_dict, expected_f_ints):
    """Check that the frequency intervals are correct."""

    assert all(int(sum(efi)) == gxds.dims["gain_freq"]
               for efi, gxds in zip(expected_f_ints, term_xds_dict.values()))


@pytest.mark.calibrate
def test_attributes(single_xds, term_xds_dict):
    """Check that the attributes of the gains are the same as the data."""

    data_attributes = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

    assert all(single_xds.attrs[a] == gxds.attrs[a]
               for gxds in term_xds_dict.values()
               for a in data_attributes)


@pytest.mark.calibrate
def test_chunk_spec(single_xds, term_xds_dict, expected_t_ints,
                    expected_f_ints):
    """Check that the chunking specs are correct."""

    n_row, n_chan, n_ant, n_dir, n_corr = \
        [single_xds.dims[d] for d in ["row", "chan", "ant", "dir", "corr"]]

    specs = [tuple([tuple(tic), tuple(fic), (n_ant,), (n_dir,), (n_corr,)])
             for tic, fic in zip(expected_t_ints, expected_f_ints)]

    assert all(spec == gxds.attrs["GAIN_SPEC"]
               for spec, gxds in zip(specs, term_xds_dict.values()))

# ---------------------------add_calibration_graph-----------------------------


@pytest.mark.calibrate
def test_ngains(gain_xds_lod, solver_opts):
    """Check that calibration produces one xds per gain per input xds."""

    assert all([len(term_xds_dict) == len(solver_opts.terms)
                for term_xds_dict in gain_xds_lod])


@pytest.mark.calibrate
def test_has_gain_field(gain_xds_lod):
    """Check that calibration assigns the gains to the relevant xds."""

    assert all([hasattr(term_xds, "gains")
                for term_xds_dict in gain_xds_lod
                for term_xds in term_xds_dict.values()])

# -----------------------------------------------------------------------------
