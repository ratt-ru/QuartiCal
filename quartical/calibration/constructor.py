import numpy as np
from quartical.calibration.solver import solver_wrapper
from quartical.utils.dask import Blocker, get_block_id_arr
from collections import namedtuple


term_spec_tup = namedtuple("term_spec_tup", "name type shape pshape")
aux_info_fields = ("SCAN_NUMBER", "FIELD_ID", "DATA_DESC_ID")


def construct_solver(data_xds_list,
                     gain_xds_lod,
                     t_bin_list,
                     t_map_list,
                     f_map_list,
                     d_map_list,
                     solver_opts,
                     chain_opts):
    """Constructs the dask graph for the solver layer.

    This constructs a custom dask graph for the solver layer given the slew
    of solver inputs. This is arguably the most important function in V2 and
    should not be tampered with without a certain level of expertise with dask.

    Args:
        data_xds_list: A list of xarray.Dataset objects containing MS data.
        gain_xds_lod: A list of dicts containing xarray.Dataset objects
            describing the gain terms.
        t_map_list: List of dask.Array objects containing time mappings.
        f_map_list: List of dask.Array objects containing frequency mappings.
        d_map_list: List of dask.Array objects containing direction mappings.
        solver_opts: A Solver config object.
        chain_opts: A Chain config object.

    Returns:
        solved_gain_xds_lod: A list of dicts containing xarray.Datasets
            describing the solved gains.
    """

    solved_gain_xds_lod = []
    output_data_xds_list = []

    for xds_ind, data_xds in enumerate(data_xds_list):

        model_col = data_xds.MODEL_DATA.data
        data_col = data_xds.DATA.data
        ant1_col = data_xds.ANTENNA1.data
        ant2_col = data_xds.ANTENNA2.data
        weight_col = data_xds.WEIGHT.data
        flag_col = data_xds.FLAG.data
        chan_freqs = data_xds.CHAN_FREQ.data
        t_bin_arr = t_bin_list[xds_ind]
        t_map_arr = t_map_list[xds_ind]
        f_map_arr = f_map_list[xds_ind]
        d_map_arr = d_map_list[xds_ind]
        gain_terms = gain_xds_lod[xds_ind]
        corr_mode = data_xds.dims["corr"]

        block_id_arr = get_block_id_arr(data_col)
        aux_block_info = {
            k: data_xds.attrs.get(k, "?") for k in aux_info_fields
        }

        # Grab the number of input chunks - doing this on the data should be
        # safe.
        n_t_chunks, n_f_chunks, _ = data_col.numblocks

        # Take the compact chunking info on the gain xdss and expand it.
        spec_list = expand_specs(gain_terms)

        # Create a blocker object.
        blocker = Blocker(solver_wrapper, "rf")

        # Add relevant inputs to the blocker object. TODO: Only pass in values
        # required by the specific terms in use.
        blocker.add_input("model", model_col, "rfdc")
        blocker.add_input("data", data_col, "rfc")
        blocker.add_input("a1", ant1_col, "r")
        blocker.add_input("a2", ant2_col, "r")
        blocker.add_input("weights", weight_col, "rfc")
        blocker.add_input("flags", flag_col, "rf")
        blocker.add_input("t_bin_arr", t_bin_arr, "prj")  # Not always needed.
        blocker.add_input("t_map_arr", t_map_arr, "prj")
        blocker.add_input("f_map_arr", f_map_arr, "pfj")
        blocker.add_input("d_map_arr", d_map_arr)
        blocker.add_input("corr_mode", corr_mode)
        blocker.add_input("term_spec_list", spec_list, "rf")
        blocker.add_input("chan_freqs", chan_freqs, "f")  # Not always needed.
        blocker.add_input("block_id_arr", block_id_arr, "rfc")
        blocker.add_input("aux_block_info", aux_block_info)
        blocker.add_input("solver_opts", solver_opts)
        blocker.add_input("chain_opts", chain_opts)

        # If the gain dataset already has a gain variable, we want to pass
        # it in to initialize the solver.
        for term_name, term_xds in gain_terms.items():
            if "gains" in term_xds.data_vars:
                blocker.add_input(f"{term_name}_initial_gain",
                                  term_xds.gains.data,
                                  "rfadc")

        if hasattr(data_xds, "ROW_MAP"):  # We are dealing with BDA.
            blocker.add_input("row_map", data_xds.ROW_MAP.data, "r")
            blocker.add_input("row_weights", data_xds.ROW_WEIGHTS.data, "r")
        else:
            blocker.add_input("row_map", None)
            blocker.add_input("row_weights", None)

        # Add relevant outputs to blocker object.
        blocker.add_output("weights",
                           "rfc",
                           weight_col.chunks,
                           weight_col.dtype)

        blocker.add_output("flags",
                           "rf",
                           flag_col.chunks,
                           flag_col.dtype)

        for term_name, term_xds in gain_terms.items():

            blocker.add_output(f"{term_name}-gain",
                               "rfadc",
                               term_xds.GAIN_SPEC,
                               np.complex128)

            blocker.add_output(f"{term_name}-gain_flags",
                               "rfad",
                               term_xds.GAIN_SPEC[:-1],
                               np.int8)

            # If there is a PARAM_SPEC on the gain xds, it is also an output.
            if hasattr(term_xds, "PARAM_SPEC"):
                blocker.add_output(f"{term_name}-param",
                                   "rfadp",
                                   term_xds.PARAM_SPEC,
                                   np.float64)

                blocker.add_output(f"{term_name}-param_flags",
                                   "rfad",
                                   term_xds.PARAM_SPEC[:-1],
                                   np.int8)

            else:  # Only non-parameterised gains return a jhj (for now).
                blocker.add_output(f"{term_name}-jhj",
                                   "rfadc",
                                   term_xds.GAIN_SPEC,
                                   np.complex128)

            chunks = ((1,)*n_t_chunks, (1,)*n_f_chunks)
            blocker.add_output(f"{term_name}-conviter",
                               "rf",
                               chunks,
                               np.int64)
            blocker.add_output(f"{term_name}-convperc",
                               "rf",
                               chunks,
                               np.float64)

        # Apply function to inputs to produce dask array outputs (as dict).
        output_dict = blocker.get_dask_outputs()

        # Assign column results to the relevant data xarray.Dataset object.
        # NOTE: Only update FLAG if we are honouring solver flags.
        flag_field = "FLAG" if solver_opts.propagate_flags else "_FLAG"

        output_data_xds = data_xds.assign(
            {"_WEIGHT": (data_xds.WEIGHT.dims, output_dict["weights"]),
             flag_field: (data_xds.FLAG.dims, output_dict["flags"])}
        )
        output_data_xds_list.append(output_data_xds)

        # Assign results to the relevant gain xarray.Dataset object.
        solved_gain_dict = {}

        for term_name, term_xds in gain_terms.items():

            result_vars = {}

            gain = output_dict[f"{term_name}-gain"]
            result_vars["gains"] = (term_xds.GAIN_AXES, gain)

            flags = output_dict[f"{term_name}-gain_flags"]
            result_vars["gain_flags"] = (term_xds.GAIN_AXES[:-1], flags)

            convperc = output_dict[f"{term_name}-convperc"]
            result_vars["conv_perc"] = (("t_chunk", "f_chunk"), convperc)

            conviter = output_dict[f"{term_name}-conviter"]
            result_vars["conv_iter"] = (("t_chunk", "f_chunk"), conviter)

            if hasattr(term_xds, "PARAM_SPEC"):
                params = output_dict[f"{term_name}-param"]
                result_vars["params"] = (term_xds.PARAM_AXES, params)

                param_flags = output_dict[f"{term_name}-param_flags"]
                result_vars["param_flags"] = \
                    (term_xds.PARAM_AXES[:-1], param_flags)
            else:
                jhj = output_dict[f"{term_name}-jhj"]
                result_vars["jhj"] = (term_xds.GAIN_AXES, jhj)

            solved_xds = term_xds.assign(result_vars)

            solved_gain_dict[term_name] = solved_xds

        solved_gain_xds_lod.append(solved_gain_dict)

    return solved_gain_xds_lod, output_data_xds_list


def expand_specs(gain_terms):
    """Convert compact spec to a per-term list per-chunk."""

    # TODO: This was rejiggered to work with the updated Blocker. Could stand
    # to be made a little neater/smarter, but works for now. Assembles nested
    # list where the outer list represnts time chunks, the middle list
    # represents frequency chunks and the inner-most list contains the
    # specs per term. Might be possible to do this with arrays instead.

    n_t_chunks = set(xds.dims["t_chunk"] for xds in gain_terms.values())
    n_f_chunks = set(xds.dims["f_chunk"] for xds in gain_terms.values())

    assert len(n_t_chunks) == 1, "Chunking in time is inconsistent."
    assert len(n_f_chunks) == 1, "Chunking in freq is inconsistent."

    n_t_chunks = n_t_chunks.pop()
    n_f_chunks = n_f_chunks.pop()

    tc_list = []
    for tc_ind in range(n_t_chunks):
        fc_list = []
        for fc_ind in range(n_f_chunks):
            term_list = []
            for xds in gain_terms.values():

                term_name = xds.NAME
                term_type = xds.TYPE
                gain_chunk_spec = xds.GAIN_SPEC

                tc = gain_chunk_spec.tchunk[tc_ind]
                fc = gain_chunk_spec.fchunk[fc_ind]

                ac = gain_chunk_spec.achunk[0]  # No chunking.
                dc = gain_chunk_spec.dchunk[0]  # No chunking.
                cc = gain_chunk_spec.cchunk[0]  # No chunking.

                term_shape = (tc, fc, ac, dc, cc)

                # Check if we have a spec for the parameters.
                parm_chunk_spec = getattr(xds, "PARAM_SPEC", ())
                if parm_chunk_spec:
                    tc_p = parm_chunk_spec.tchunk[tc_ind]
                    fc_p = parm_chunk_spec.fchunk[fc_ind]
                    pc = parm_chunk_spec.pchunk[0]

                    parm_shape = (tc_p, fc_p, ac, dc, pc)
                else:
                    parm_shape = (0,) * 5  # Used for creating a dummy array.

                term_list.append(term_spec_tup(term_name,
                                               term_type,
                                               term_shape,
                                               parm_shape))

            fc_list.append(term_list)
        tc_list.append(fc_list)

    return tc_list
