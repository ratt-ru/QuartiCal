from loguru import logger
import dask.array as da
import dask
import Tigger
from daskms import xds_from_table

from cubicalv2.utils.dask import blockwise_unique
from cubicalv2.utils.collections import freeze_default_dict

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import (phase_delay as compute_phase_delay,
                                 predict_vis,
                                 parallactic_angles as
                                 compute_parallactic_angles,
                                 feed_rotation as compute_feed_rotation)
from africanus.model.coherency.dask import convert
from africanus.model.spectral.dask import spectral_model
from africanus.model.shape.dask import gaussian as gaussian_shape

from collections import namedtuple, defaultdict

_einsum_corr_indices = 'ijkl'

_empty_spectrum = object()


def _brightness_schema(corrs, index):
    if corrs == 4:
        return "sf" + _einsum_corr_indices[index:index + 2], index + 1
    else:
        return "sfi", index


def _phase_delay_schema(corrs, index):
    return "srf", index


def _spi_schema(corrs, index):
    return "s", index


def _gauss_shape_schema(corrs, index):
    return "srf", index


def _bl_jones_output_schema(corrs, index):
    if corrs == 4:
        return "->srfi" + _einsum_corr_indices[index]
    else:
        return "->srfi"


_rime_term_map = {
    'brightness': _brightness_schema,
    'phase_delay': _phase_delay_schema,
    'spi': _spi_schema,
    'gauss_shape': _gauss_shape_schema,
}


def parse_sky_models(opts):
    """Parses a Tigger sky model.

    Args:
        opts: A Namespace object containing options.
    Returns:
        sky_model_dict: A dictionary of source data.
    """

    sky_model_dict = {}

    for sky_model_tuple in opts._sky_models:

        sky_model_name, sky_model_tags = sky_model_tuple

        sky_model = Tigger.load(sky_model_name, verbose=False)

        sources = sky_model.sources

        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for source in sources:

            tagged = any([source.getTag(tag) for tag in sky_model_tags])

            parent_group = source.getTag("cluster") if tagged else "DIE"

            ra = source.pos.ra
            dec = source.pos.dec
            typecode = source.typecode.lower()
            flux = [getattr(source.flux, sto, 0) for sto in "IQUV"]
            spectrum = getattr(source, "spectrum", _empty_spectrum)

            # Attempts to grab the source reference frequency. Failing that,
            # the skymodel reference frequency is used. If that isn't set,
            # defaults to 1e9. TODO: Error out? Not sure this is sane.
            ref_freq = getattr(spectrum, "freq0", sky_model.freq0) or 1e9

            # Extract SPI for I, defaulting to -0.7. TODO: Default to 0?
            spi = [[getattr(spectrum, "spi", 0)]*4]

            if typecode == "gau":
                emaj = source.shape.ex
                emin = source.shape.ey
                pa = source.shape.pa

                gauss_params = groups[parent_group]["gauss"]

                gauss_params["radec"].append([ra, dec])
                gauss_params["stokes"].append(flux)
                gauss_params["spi"].append(spi)
                gauss_params["ref_freq"].append(ref_freq)
                gauss_params["shape"].append([emaj, emin, pa])

            elif typecode == "pnt":

                point_params = groups[parent_group]["point"]

                point_params["radec"].append([ra, dec])
                point_params["stokes"].append(flux)
                point_params["spi"].append(spi)
                point_params["ref_freq"].append(ref_freq)

            else:
                raise ValueError("Unknown typecode - {}".format(typecode))

        # Recursively freeze the default dict so that accessing non-existent
        # keys will fail as expected hereafter.
        sky_model_dict[sky_model_tuple] = freeze_default_dict(groups)

        msg = "".join(
            "\n  {:<8}: {} point source/s, {} Gaussian source/s".format(
                key,
                len(value["point"]["stokes"]) if "point" in value else 0,
                len(value["gauss"]["stokes"]) if "gauss" in value else 0)
            for key, value in sky_model_dict[sky_model_tuple].items())

        logger.info("Source groups/clusters for {}:{}", sky_model_name, msg)

    return sky_model_dict


def daskify_sky_model_dict(sky_model_dict, opts):
    """Converts source parameter dictionary into a dictionary of dask arrays.

    Args:
        sky_model_dict: Dictionary of sources.
        opts: Namespace object containing options.

    Returns:
        dask_sky_model_dict: A dicitonary of dask arrays.
    """

    dask_sky_model_dict = sky_model_dict.copy()  # Avoid mutating input.

    # This single line can have a large impact on memory/performance. Sets
    # the source chunking strategy for the predict.

    chunks = opts.input_model_source_chunks

    Point = namedtuple("Point", ["radec", "stokes", "spi", "ref_freq"])
    Gauss = namedtuple("Gauss", ["radec", "stokes", "spi", "ref_freq",
                                 "shape"])

    for model_name, model_group in sky_model_dict.items():
        for group_name, group_sources in model_group.items():

            if "point" in group_sources:

                point_params = group_sources["point"]

                dask_sky_model_dict[model_name][group_name]['point'] = \
                    Point(
                        da.from_array(
                            point_params["radec"], chunks=(chunks, -1)),
                        da.from_array(
                            point_params["stokes"], chunks=(chunks, -1)),
                        da.from_array(
                            point_params["spi"], chunks=(chunks, 1, -1)),
                        da.from_array(
                            point_params["ref_freq"], chunks=chunks))

            if "gauss" in group_sources:

                gauss_params = group_sources["gauss"]

                dask_sky_model_dict[model_name][group_name]['gauss'] = \
                    Gauss(
                        da.from_array(
                            gauss_params["radec"], chunks=(chunks, -1)),
                        da.from_array(
                            gauss_params["stokes"], chunks=(chunks, -1)),
                        da.from_array(
                            gauss_params["spi"], chunks=(chunks, 1, -1)),
                        da.from_array(
                            gauss_params["ref_freq"], chunks=chunks),
                        da.from_array(
                            gauss_params["shape"], chunks=(chunks, -1)))

    return dask_sky_model_dict


def get_support_tables(opts):
    """Get the support tables necessary for the predict.

    Args:
        opts: A Namespace object of global options.

    Returns:
        lazy_tables: Dictionary of support table datasets.
    """

    n = {k: '::'.join((opts.input_ms_name, k)) for k
         in ("ANTENNA", "DATA_DESCRIPTION", "FIELD",
             "SPECTRAL_WINDOW", "POLARIZATION", "FEED")}

    # All rows at once
    lazy_tables = {"ANTENNA": xds_from_table(n["ANTENNA"]),
                   "FEED": xds_from_table(n["FEED"])}

    compute_tables = {
        # Fixed shape rows
        "DATA_DESCRIPTION": xds_from_table(n["DATA_DESCRIPTION"]),
        # Variably shaped, need a dataset per row
        "FIELD":
            xds_from_table(n["FIELD"], group_cols="__row__"),
        "SPECTRAL_WINDOW":
            xds_from_table(n["SPECTRAL_WINDOW"], group_cols="__row__"),
        "POLARIZATION":
            xds_from_table(n["POLARIZATION"], group_cols="__row__"),
    }

    lazy_tables.update(dask.compute(compute_tables)[0])

    return lazy_tables


def corr_schema(pol):
    """
    Parameters
    ----------
    pol : Dataset
    Returns
    -------
    corr_schema : list of list
        correlation schema from the POLARIZATION table,
        `[[9, 10], [11, 12]]` for example
    """

    corrs = pol.NUM_CORR.values
    corr_types = da.squeeze(pol.CORR_TYPE.values)

    if corrs == 4:
        return [[corr_types[0], corr_types[1]],
                [corr_types[2], corr_types[3]]]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]    # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]                   # (1, ) shape
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


def baseline_jones_multiply(corrs, *args):
    names = args[::2]
    arrays = args[1::2]

    input_einsum_schemas = []
    corr_index = 0

    for name, array in zip(names, arrays):
        try:
            # Obtain function for prescribing the input einsum schema
            schema_fn = _rime_term_map[name]
        except KeyError:
            raise ValueError("Unknown RIME term '%s'" % name)
        else:
            # Extract it and the next corr index
            einsum_schema, corr_index = schema_fn(corrs, corr_index)
            input_einsum_schemas.append(einsum_schema)

            if not len(einsum_schema) == array.ndim:
                raise ValueError("%s len(%s) == %d != %s.ndim"
                                 % (name, einsum_schema,
                                    len(einsum_schema), array.shape))

    output_schema = _bl_jones_output_schema(corrs, corr_index)
    schema = ",".join(input_einsum_schemas) + output_schema

    return da.einsum(schema, *arrays, optimize=True)


def compute_p_jones(parallactic_angles, feed_xds, opts):
    """Compte the P-Jones (parallactic angle + receptor angle) matrix.

    Args:
        parallactic_angles: Dask array of parallactic angles.
        feed_xds: xarray datset containing feed information.
        opts: A Namepspace of global options.

    Returns:
        A dask array of feed rotations per antenna per time.
    """
    # This is a DI term when there are no DD terms, otherwise it needs to be
    # applied before the beam. This currently makes assumes identical
    # receptor angles. TODO: Remove assumption when Codex gains functionality.

    receptor_angles = feed_xds.RECEPTOR_ANGLE.data

    if not da.all(receptor_angles[:, 0] == receptor_angles[:, 1]):
        logger.warning("RECEPTOR_ANGLE indicates non-orthoganal "
                       "receptors. Currently, P-Jones cannot account "
                       "for non-uniform offsets. Using 0.")
        return compute_feed_rotation(parallactic_angles, opts._feed_type)
    else:
        return compute_feed_rotation(
            parallactic_angles + receptor_angles[None, :, 0], opts._feed_type)


def die_factory(utime_val, frequency, ant_xds, feed_xds, phase_dir, opts):
    """Produces a net direction-independent matrix per time, channe, antenna.

    Args:
        utime_val: Dask array of unique time values.
        frequency: Dask array of frequency values.
        ant_xds: xarray dataset containing antenna information.
        feed_xds: xarray dataset containing feed information.
        phase_dir: The phase direction in radians.
        opts: A Namepspace of global options.

    Returns:
        die_jones: A dask array representing the net direction-independent
            effects.
    """

    die_jones = None

    # If the beam is enabled, P-Jones has to be applied before the beam.
    if opts.input_model_apply_p_jones and not opts.input_model_beam:

        parallactic_angles = compute_parallactic_angles(utime_val,
                                                        ant_xds["POSITION"],
                                                        phase_dir)

        p_jones = compute_p_jones(parallactic_angles, feed_xds, opts)

        # Broadcasts the P-Jones matrix to the expeceted DIE dimensions.
        # TODO: This is only appropriate whilst there are no other DIE terms.
        n_t, n_a, _, _ = p_jones.shape
        n_c = frequency.size

        chunks = (utime_val.chunks[0], n_a, frequency.chunks[0], 2, 2)

        die_jones = da.broadcast_to(p_jones[:, :, None, :, :],
                                    (n_t, n_a, n_c, 2, 2),
                                    chunks=chunks)

    return die_jones


def vis_factory(opts, source_type, sky_model, ms, ant, field, spw, pol, feed):
    """Generates a graph describing the predict for an xds, model and type.

    Args:
        opts: A Namepspace of global options.
        source_type: A string - either "point" or "gauss".
        sky_model: The daskified sky model containing dask arrays of params.
        ms: An xarray.dataset containing a piece of the MS.
        ant: An xarray.dataset corresponding to the antenna subtable.
        field: An xarray.dataset corresponding to the antenna subtable.
        spw: An xarray.dataset corresponding to the spectral window subtable.
        pol: An xarray.dataset corresponding the polarization subtable.
        feed: An xarray.dataset corresponding the feed subtable.

    Returns:
        The result of predict_vis - a graph describing the predict.
    """

    # Array containing source parameters.
    sources = sky_model[source_type]

    # Select single dataset rows
    corrs = pol.NUM_CORR.data[0]
    # Necessary to chunk the predict in frequency. TODO: Make this less hacky
    # when this is improved in dask-ms.
    frequency = da.from_array(spw.CHAN_FREQ.data[0], chunks=ms.chunks['chan'])
    phase_dir = field.PHASE_DIR.data[0][0]  # row, poly

    lm = radec_to_lm(sources.radec, phase_dir)
    # This likely shouldn't be exposed. TODO: Disable this switch?
    uvw = -ms.UVW.data if opts.input_model_invert_uvw else ms.UVW.data

    # Generate per-source K-Jones (source, row, frequency).
    phase = compute_phase_delay(lm, uvw, frequency)

    # Apply spectral model to stokes parameters (source, frequency, corr).
    stokes = spectral_model(sources.stokes,
                            sources.spi,
                            sources.ref_freq,
                            frequency,
                            base=0)

    # Convery from stokes parameters to brightness matrix.
    brightness = convert(stokes, ["I", "Q", "U", "V"], corr_schema(pol))

    utime_val, utime_ind = blockwise_unique(ms.TIME.data,
                                            chunks=(ms.UTIME_CHUNKS,),
                                            return_inverse=True)

    bl_jones_args = ["phase_delay", phase]

    # Add any visibility amplitude terms
    if source_type == "gauss":
        bl_jones_args.append("gauss_shape")
        bl_jones_args.append(gaussian_shape(uvw, frequency, sources.shape))

    bl_jones_args.extend(["brightness", brightness])

    jones = baseline_jones_multiply(corrs, *bl_jones_args)
    # TODO: Add DI and DD factories. DI will include P-jones when there is no
    # DE term. Otherwise P must be applied before E.
    die = die_factory(utime_val, frequency, ant, feed, phase_dir, opts)
    # dde = dde_factory(opts, ms, ant, field, pol, lm, utime, frequency)

    return predict_vis(utime_ind, ms.ANTENNA1.data, ms.ANTENNA2.data,
                       None, jones, None, die, None, die)


def predict(data_xds_list, opts):
    """Produces graphs describing predict operations.

    Args:
        data_xds_list: A list of xarray datasets corresponing to the input
            measurement set data.
        opts: A Namepspace of global options.

    Returns:
        predict_list: A list of dictionaries containing dask graphs describing
            the predict.
    """

    # Read in a Tigger .lsm.html and produce a dictionary of sources per
    # unique sky model and tag combination. Tags determine clustering.

    sky_model_dict = parse_sky_models(opts)

    # Convert sky model dictionary into a dictionary of per-model dask arrays.

    dask_sky_model_dict = daskify_sky_model_dict(sky_model_dict, opts)

    # Get the support tables (as lists), and give them sensible names.
    tables = get_support_tables(opts)

    ant_xds_list = tables["ANTENNA"]
    field_xds_list = tables["FIELD"]
    ddid_xds_list = tables["DATA_DESCRIPTION"]
    spw_xds_list = tables["SPECTRAL_WINDOW"]
    pol_xds_list = tables["POLARIZATION"]
    feed_xds_list = tables["FEED"]

    # List of predict operations
    predict_list = []

    for data_xds in data_xds_list:

        # Perform subtable joins.
        ant_xds = ant_xds_list[0]
        feed_xds = feed_xds_list[0]
        field_xds = field_xds_list[data_xds.attrs['FIELD_ID']]
        ddid_xds = ddid_xds_list[data_xds.attrs['DATA_DESC_ID']]
        spw_xds = spw_xds_list[ddid_xds.SPECTRAL_WINDOW_ID.data[0]]
        pol_xds = pol_xds_list[ddid_xds.POLARIZATION_ID.data[0]]

        corrs = opts._ms_ncorr

        model_vis = defaultdict(list)

        # Generate visibility expressions per model, per direction for each
        # source type.
        for model_name, model_group in dask_sky_model_dict.items():
            for group_name, group_sources in model_group.items():

                # Generate visibilities per source type.
                source_vis = [vis_factory(opts, stype, group_sources,
                                          data_xds, ant_xds, field_xds,
                                          spw_xds, pol_xds, feed_xds)
                              for stype in group_sources.keys()]

                # Sum the per-source-type visibilitites together.
                vis = sum(source_vis)

                # Reshape (2, 2) correlation to shape (4,)
                if corrs == 4:
                    vis = vis.reshape(vis.shape[:-2] + (4,))

                # Append group_vis to the appropriate list.
                model_vis[model_name].append(vis)

        predict_list.append(freeze_default_dict(model_vis))

    return predict_list
