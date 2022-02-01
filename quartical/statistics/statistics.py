import dask.array as da


def compute_presolve_chisq(data_xds_list):

    chisq_per_xds = []

    for xds in data_xds_list:

        data = xds.DATA.data
        model = xds.MODEL_DATA.data
        weights = xds.WEIGHT.data
        inv_flags = da.where(xds.FLAG.data == 0, 1, 0)[:, :, None]

        residual = data - model.sum(axis=2)

        chisq = da.map_blocks(
            compute_chisq,
            residual,
            weights,
            inv_flags,
            chunks=(1, 1),
            drop_axis=-1,
        )

        chisq_per_xds.append(chisq)

    return chisq_per_xds


def compute_postsolve_chisq(data_xds_list):

    chisq_per_xds = []

    for xds in data_xds_list:

        weights = xds._WEIGHT.data
        residual = xds._RESIDUAL.data
        inv_flags = da.where(xds.FLAG.data == 0, 1, 0)[:, :, None]

        chisq = da.map_blocks(
            compute_chisq,
            residual,
            weights,
            inv_flags,
            chunks=(1, 1),
            drop_axis=-1,
        )

        chisq_per_xds.append(chisq)

    return chisq_per_xds


def compute_chisq(residual, weights, inv_flags):

    eff_weights = weights * inv_flags

    chisq = (residual * eff_weights * residual.conj()).real
    chisq = chisq.sum(keepdims=True)

    counts = inv_flags.sum(keepdims=True) * residual.shape[-1]

    return (chisq/counts)[..., -1]
