import numpy as np
from quartical.calibration.solver import solver_wrapper
from quartical.utils.dask import Blocker
from collections import namedtuple


term_spec_tup = namedtuple("term_spec_tup", "name type shape pshape")


def construct_solver(data_xds_list,
                     gain_xds_list,
                     t_map_list,
                     f_map_list,
                     d_map_list,
                     opts):
    """Constructs the dask graph for the solver layer.

    This constructs a custom dask graph for the solver layer given the slew
    of solver inputs. This is arguably the most important function in V2 and
    should not be tampered with without a certain level of expertise with dask.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        gain_xds_list: A list of lists containing xarray.Dataset objects
            describing the gain terms.
        t_map_list: List of dask.Array objects containing time mappings.
        f_map_list: List of dask.Array objects containing frequency mappings.
        d_map_list: List of dask.Array objects containing direction mappings.
        opts: A Namespace object containing global options.

    Returns:
        A list of lists containing xarray.Datasets describing the solved gains.
    """

    solved_gain_xds_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        model_col = data_xds.MODEL_DATA.data
        data_col = data_xds.DATA.data
        ant1_col = data_xds.ANTENNA1.data
        ant2_col = data_xds.ANTENNA2.data
        weight_col = data_xds.WEIGHT.data
        chan_freqs = data_xds.CHAN_FREQ.data
        t_map_arr = t_map_list[xds_ind]
        f_map_arr = f_map_list[xds_ind]
        d_map_arr = d_map_list[xds_ind]
        gain_terms = gain_xds_list[xds_ind]

        # Grab the number of input chunks - doing this on the data should be
        # safe.
        n_t_chunks, n_f_chunks, _ = data_col.numblocks

        # Take the compact chunking info on the gain xdss and expand it.
        spec_list = expand_specs(gain_terms)

        # Create a blocker object.
        blocker = Blocker(solver_wrapper, "rf")

        # Add relevant inputs to the blocker object.
        blocker.add_input("model", model_col, "rfdc")
        blocker.add_input("data", data_col, "rfc")
        blocker.add_input("a1", ant1_col, "r")
        blocker.add_input("a2", ant2_col, "r")
        blocker.add_input("weights", weight_col, "rfc")
        blocker.add_input("t_map_arr", t_map_arr, "rj")
        blocker.add_input("f_map_arr", f_map_arr, "fj")
        blocker.add_input("d_map_arr", d_map_arr)
        blocker.add_input("corr_mode", opts.input_ms_correlation_mode)
        blocker.add_input("term_spec_list", spec_list, "rf")
        blocker.add_input("chan_freqs", chan_freqs, "f")

        if opts.input_ms_is_bda:
            blocker.add_input("row_map", data_xds.ROW_MAP.data, "r")
            blocker.add_input("row_weights", data_xds.ROW_WEIGHTS.data, "r")
        else:
            blocker.add_input("row_map", None)
            blocker.add_input("row_weights", None)

        # Add relevant outputs to blocker object.
        for gi, gn in enumerate(opts.solver_gain_terms):

            chunks = gain_terms[gi].GAIN_SPEC
            blocker.add_output(f"{gn}-gain", "rfadc", chunks, np.complex128)

            # If there is a PARAM_SPEC on the gain xds, it is also an output.
            if hasattr(gain_terms[gi], "PARAM_SPEC"):
                chunks = gain_terms[gi].PARAM_SPEC
                blocker.add_output(f"{gn}-param", "rfadpc", chunks,
                                   np.complex128)

            chunks = ((1,)*n_t_chunks, (1,)*n_f_chunks)
            blocker.add_output(f"{gn}-conviter", "rf", chunks, np.int64)
            blocker.add_output(f"{gn}-convperc", "rf", chunks, np.float64)

        # Apply function to inputs to produce dask array outputs (as dict).
        output_dict = blocker.get_dask_outputs()

        # Assign results to the relevant gain xarray.Dataset object.
        solved_gain_terms = []

        for gi, gain_xds in enumerate(gain_terms):

            result_vars = {}

            gain = output_dict[f"{gain_xds.NAME}-gain"]
            result_vars["gains"] = (gain_xds.GAIN_AXES, gain)

            convperc = output_dict[f"{gain_xds.NAME}-convperc"]
            result_vars["conv_perc"] = (("t_chunk", "f_chunk"), convperc)

            conviter = output_dict[f"{gain_xds.NAME}-conviter"]
            result_vars["conv_iter"] = (("t_chunk", "f_chunk"), conviter)

            if hasattr(gain_xds, "PARAM_SPEC"):
                params = output_dict[f"{gain_xds.NAME}-param"]
                result_vars["params"] = (gain_xds.PARAM_AXES, params)

            solved_xds = gain_xds.assign(result_vars)

            solved_gain_terms.append(solved_xds)

        solved_gain_xds_list.append(solved_gain_terms)

    return solved_gain_xds_list


def expand_specs(gain_terms):
    """Convert compact spec to a per-term list per-chunk."""

    # TODO: This was rejiggered to work with the updated Blocker. Could stand
    # to be made a little neater/smarter, but works for now. Assembles nested
    # list where the outer list represnts time chunks, the middle list
    # represents frequency chunks and the inner-most list contains the
    # specs per term.

    n_t_chunks = gain_terms[0].dims["t_chunk"]
    n_f_chunks = gain_terms[0].dims["f_chunk"]

    tc_list = []
    for tc_ind in range(n_t_chunks):
        fc_list = []
        for fc_ind in range(n_f_chunks):
            term_list = []
            for gxds in gain_terms:

                term_name = gxds.NAME
                term_type = gxds.TYPE
                gain_chunk_spec = gxds.GAIN_SPEC

                tc = gain_chunk_spec.tchunk[tc_ind]
                fc = gain_chunk_spec.fchunk[fc_ind]

                ac = gain_chunk_spec.achunk[0]  # No chunking.
                dc = gain_chunk_spec.dchunk[0]  # No chunking.
                cc = gain_chunk_spec.cchunk[0]  # No chunking.

                term_shape = (tc, fc, ac, dc, cc)

                # Check if we have a spec for the parameters.
                parm_chunk_spec = getattr(gxds, "PARAM_SPEC", ())
                if parm_chunk_spec:
                    tc_p = parm_chunk_spec.tchunk[tc_ind]
                    fc_p = parm_chunk_spec.fchunk[fc_ind]
                    pc = parm_chunk_spec.pchunk[0]

                    parm_shape = (tc_p, fc_p, ac, dc, pc, cc)
                else:
                    parm_shape = ()

                term_list.append(term_spec_tup(term_name,
                                               term_type,
                                               term_shape,
                                               parm_shape))

            fc_list.append(term_list)
        tc_list.append(fc_list)

    return tc_list
