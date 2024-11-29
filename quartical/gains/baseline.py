import xarray
import numpy as np
import dask.array as da
from daskms.experimental.zarr import xds_to_zarr
from numba import njit
from numba.extending import overload
from quartical.utils.numba import coerce_literal, JIT_OPTIONS
import quartical.gains.general.factories as factories
from quartical.gains.general.convenience import get_dims, get_row


def write_baseline_datasets(bl_corr_xds_list, output_opts):

    if bl_corr_xds_list:
        return xds_to_zarr(
            bl_corr_xds_list,
            f"{output_opts.gain_directory}::BLCORR"
        )
    else:
        return None


def compute_baseline_corrections(
    data_xds_list,
    solved_gain_xds_lod,
    mapping_xds_list
):

    bl_corr_xdsl = []

    itr = zip(data_xds_list, mapping_xds_list, solved_gain_xds_lod)

    for (data_xds, mapping_xds, gain_dict) in itr:
        data_col = data_xds.DATA.data
        model_col = data_xds.MODEL_DATA.data
        flag_col = data_xds.FLAG.data
        weight_col = data_xds._WEIGHT.data  # The weights exiting the solver.
        ant1_col = data_xds.ANTENNA1.data
        ant2_col = data_xds.ANTENNA2.data
        corr_mode = data_xds.sizes["corr"]

        time_maps = tuple(
            [mapping_xds.get(f"{k}_time_map").data for k in gain_dict.keys()]
        )
        freq_maps = tuple(
            [mapping_xds.get(f"{k}_freq_map").data for k in gain_dict.keys()]
        )
        dir_maps = tuple(
            [mapping_xds.get(f"{k}_dir_map").data for k in gain_dict.keys()]
        )

        is_bda = hasattr(data_xds, "ROW_MAP")  # We are dealing with BDA.
        row_map = data_xds.ROW_MAP.data if is_bda else None
        row_weights = data_xds.ROW_WEIGHTS.data if is_bda else None

        req_itr = zip(gain_dict.values(), time_maps, freq_maps, dir_maps)

        req_args = []

        for g, tb, fm, dm in req_itr:
            req_args.extend([g.gains.data, g.gains.dims])
            req_args.extend([tb, ("gain_time",)])
            req_args.extend([fm, ("gain_freq",)])
            req_args.extend([dm, ("direction",)])

        n_ant = data_xds.sizes["ant"]
        n_bla = int((n_ant*(n_ant - 1))/2 + n_ant)

        bl_corr = da.blockwise(
            dask_compute_baseline_corrections,
            ("rowlike", "baseline", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            model_col, ("rowlike", "chan", "dir", "corr"),
            weight_col, ("rowlike", "chan", "corr"),
            flag_col, ("rowlike", "chan"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            *((row_map, ("rowlike",)) if is_bda else (None, None)),
            *((row_weights, ("rowlike",)) if is_bda else (None, None)),
            corr_mode, None,
            *req_args,
            dtype=data_col.dtype,
            align_arrays=False,
            concatenate=True,
            new_axes={"baseline": n_bla},
            adjust_chunks={
                "rowlike": ((1,)*len(data_xds.chunks['row'])),
                "chan": data_col.chunks[1]
            }
        )

        a1_inds = [x for x in range(n_ant) for _ in range(x, n_ant)]
        a2_inds = [y for x in range(n_ant) for y in range(x, n_ant)]

        bl_corr_xds = xarray.Dataset(
            {"bl_correction": (("time", "bl_id", "chan", "corr"), bl_corr)},
            coords={
                "time": (("time",), np.arange(len(data_xds.chunks['row']))),
                "bl_id": (("bl_id",), np.arange(n_bla)),
                "chan": (("chan",), data_xds.chan.values),
                "corr": (("corr",), data_xds.corr.values),
                "antenna1": (("bl_id", a1_inds)),
                "antenna2": (("bl_id", a2_inds))
            }
        )

        bl_corr_xdsl.append(bl_corr_xds)

    return bl_corr_xdsl


def dask_compute_baseline_corrections(
    data,
    model,
    weight,
    flags,
    a1,
    a2,
    row_map,
    row_weights,
    corr_mode,
    *args
):

    gains = tuple(args[::4])
    time_maps = tuple(args[1::4])
    freq_maps = tuple(args[2::4])
    dir_maps = tuple(args[3::4])

    return _compute_baseline_corrections(
        data,
        model,
        weight,
        flags,
        gains,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode,
    )


@njit(**JIT_OPTIONS)
def _compute_baseline_corrections(
    data,
    model,
    weight,
    flags,
    gains,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):
    return _compute_baseline_corrections_impl(
        data,
        model,
        weight,
        flags,
        gains,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    )


def _compute_baseline_corrections_impl(
    data,
    model,
    weight,
    flags,
    gains,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):
    return NotImplementedError


@overload(_compute_baseline_corrections_impl, jit_options=JIT_OPTIONS)
def nb_compute_baseline_corrections_impl(
    data,
    model,
    weight,
    flags,
    gains,
    a1,
    a2,
    time_maps,
    freq_maps,
    dir_maps,
    row_map,
    row_weights,
    corr_mode
):

    coerce_literal(nb_compute_baseline_corrections_impl, ["corr_mode"])

    imul_rweight = factories.imul_rweight_factory(corr_mode, row_weights)
    v1_imul_v2 = factories.v1_imul_v2_factory(corr_mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(corr_mode)
    iadd = factories.iadd_factory(corr_mode)
    iunpack = factories.iunpack_factory(corr_mode)
    valloc = factories.valloc_factory(corr_mode)

    def impl(
        data,
        model,
        weight,
        flags,
        gains,
        a1,
        a2,
        time_maps,
        freq_maps,
        dir_maps,
        row_map,
        row_weights,
        corr_mode
    ):

        n_rows, n_chan, n_dir, n_corr = get_dims(model, row_map)

        n_ant = int(max(np.max(a1), np.max(a2))) + 1

        n_bla = int((n_ant*(n_ant - 1))/2 + n_ant)  # bls plus autos

        jhj = np.zeros((n_bla, n_chan, n_corr), dtype=np.complex64)
        jhr = np.zeros((n_bla, n_chan, n_corr), dtype=np.complex64)

        bl_ids = (n_bla - ((n_ant - a1 + 1)*(n_ant - a1))//2 + a2 - a1)

        n_gains = len(gains)

        dir_loop = np.arange(n_dir)

        for row_ind in range(n_rows):

            row = get_row(row_ind, row_map)
            a1_m, a2_m = a1[row], a2[row]
            bl_m = bl_ids[row]
            v = valloc(np.complex128)  # Hold GMGH.

            for f in range(n_chan):

                if flags[row, f]:
                    continue

                m = model[row, f]
                w = weight[row, f]
                r = data[row, f]

                for d in dir_loop:

                    iunpack(v, m[d])

                    for g in range(n_gains - 1, -1, -1):

                        t_m = time_maps[g][row_ind]
                        f_m = freq_maps[g][f]
                        d_m = dir_maps[g][d]  # Broadcast dir.

                        gain = gains[g][t_m, f_m]
                        gain_p = gain[a1_m, d_m]
                        gain_q = gain[a2_m, d_m]

                        v1_imul_v2(gain_p, v, v)
                        v1_imul_v2ct(v, gain_q, v)

                    imul_rweight(v, v, row_weights, row_ind)
                    iadd(jhj[bl_m, f], v.conjugate() * w * v)
                    iadd(jhr[bl_m, f], v.conjugate() * w * r)

        bl_corrections = np.ones_like(jhr).ravel()

        sel = np.where(jhj.ravel() != 0)

        bl_corrections[sel] = jhr.ravel()[sel]/jhj.ravel()[sel]

        return bl_corrections.reshape((1, n_bla, n_chan, n_corr))

    return impl


def apply_baseline_corrections(data_xds_list, bl_xds_list):

    bl_corr_xds = []

    for xds, blxds in zip(data_xds_list, bl_xds_list):

        data_col = xds._CORRECTED_DATA.data
        bl_corrections = blxds.bl_correction.data
        ant1_col = xds.ANTENNA1.data
        ant2_col = xds.ANTENNA2.data

        corres = da.blockwise(
            dask_apply_baseline_corrections, ("rowlike", "chan", "corr"),
            data_col, ("rowlike", "chan", "corr"),
            bl_corrections, ("rowlike", "baseline", "chan", "corr"),
            ant1_col, ("rowlike",),
            ant2_col, ("rowlike",),
            dtype=data_col.dtype,
            align_arrays=False,
            concatenate=True,
            adjust_chunks={"rowlike": data_col.chunks[0],
                           "chan": data_col.chunks[1]})

        new_xds = xds.assign(
            {
                "_CORRECTED_DATA": ((xds._CORRECTED_DATA.dims), corres)
            }
        )

        bl_corr_xds.append(new_xds)

    return bl_corr_xds


def dask_apply_baseline_corrections(
    data,
    bl_corrections,
    a1,
    a2,
):

    return _apply_baseline_corrections(
        data,
        bl_corrections,
        a1,
        a2,
    )


@njit(**JIT_OPTIONS)
def _apply_baseline_corrections(data, bl_corrections, a1, a2):

    data = data.copy()

    n_rows, n_chan, n_corr = data.shape

    n_ant = int(max(np.max(a1), np.max(a2))) + 1

    n_bla = int((n_ant*(n_ant - 1))/2 + n_ant)  # bls plus autos

    bl_ids = (n_bla - ((n_ant - a1 + 1)*(n_ant - a1))//2 + a2 - a1)

    for row in range(n_rows):

        bl_m = bl_ids[row]

        for f in range(n_chan):

            v = data[row, f]

            for c in range(n_corr):

                if bl_corrections[0, bl_m, f, c]:
                    blg = 1/bl_corrections[0, bl_m, f, c]
                else:
                    blg = 1

                data[row, f, c] = blg * v[c]

    return data
