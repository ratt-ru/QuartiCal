from loguru import logger
import dask.array as da
import Tigger
from daskms import xds_from_table

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
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


def parse_sky_model(opts):
    """Parses a Tigger sky model.

    Args:
        opts: A Namespace object containing options.
    Returns:
        source_data: A dictionary of source data.
    """

    # This is naive. TODO: Add support for arbitrary recipes.

    def dict_factory():

        return dict(point=dict(radec=[],
                               stokes=[],
                               spi=[],
                               ref_freq=[],
                               n_src=0),
                    gauss=dict(radec=[],
                               stokes=[],
                               spi=[],
                               ref_freq=[],
                               shape=[],
                               n_src=0))

    # Currently looping over multiple sky models will not work as expected.
    # TODO: Fix this behaviour.

    for sky_model_name, sky_model_tag in opts._sky_models.items():

        sky_model = Tigger.load(sky_model_name, verbose=False)

        sources = sky_model.sources

        groups = defaultdict(dict_factory)

        for source in sources:

            parent_group = source.getTag("cluster") \
                if source.getTag(sky_model_tag) else "DIE"

            gauss_params = groups[parent_group]["gauss"]
            point_params = groups[parent_group]["point"]

            ra = source.pos.ra
            dec = source.pos.dec
            typecode = source.typecode.lower()

            I = source.flux.I  # noqa
            Q = source.flux.Q
            U = source.flux.U
            V = source.flux.V

            spectrum = (getattr(source, "spectrum", _empty_spectrum)
                        or _empty_spectrum)

            # Attempts to grab the source reference frequency. Failing that,
            # the skymodel reference frequency is used. If that isn't set,
            # defaults to 1e9.
            ref_freq = getattr(spectrum, "freq0", sky_model.freq0) or 1e9

            try:
                # Extract SPI for I.
                # Zero Q, U and V to get 1 on the exponential
                spi = [[spectrum.spi, 0, 0, 0]]
            except AttributeError:
                # Default I SPI to -0.7
                spi = [[-0.7, 0, 0, 0]]

            if typecode == "gau":
                emaj = source.shape.ex
                emin = source.shape.ey
                pa = source.shape.pa

                gauss_params["radec"].append([ra, dec])
                gauss_params["stokes"].append([I, Q, U, V])
                gauss_params["spi"].append(spi)
                gauss_params["ref_freq"].append(ref_freq)
                gauss_params["shape"].append([emaj, emin, pa])
                gauss_params["n_src"] += 1

            elif typecode == "pnt":
                point_params["radec"].append([ra, dec])
                point_params["stokes"].append([I, Q, U, V])
                point_params["spi"].append(spi)
                point_params["ref_freq"].append(ref_freq)
                point_params["n_src"] += 1
            else:
                raise ValueError("Unknown source morphology %s" % typecode)

        logger.info("Source groups/clusters for {}:{}",
                    sky_model_name,
                    "".join("\n  {:<8}: {} point source/s, "
                            "{} Gaussian source/s".format(
                                key,
                                value["point"]["n_src"],
                                value["gauss"]["n_src"])
                            for key, value in groups.items()))

        # # Currently, we hard code the model chunks to include 10 sources.
        # # TODO: Make this dynamic if necessary.

        chunks = 10

        # # The code is currently only suitable for a direction independent
        # # predict. TODO: Add clustering behaviour.

        Point = namedtuple("Point", ["radec", "stokes", "spi", "ref_freq"])
        Gauss = namedtuple("Gauss", ["radec", "stokes", "spi", "ref_freq",
                                     "shape"])

        source_data = {key: {} for key in groups.keys()}

        for name, params in groups.items():

            gauss_params = params["gauss"]
            point_params = params["point"]

            if point_params["n_src"] > 0:
                source_data[name]['point'] = \
                    Point(da.from_array(point_params["radec"],
                                        chunks=(chunks, -1)),
                          da.from_array(point_params["stokes"],
                                        chunks=(chunks, -1)),
                          da.from_array(point_params["spi"],
                                        chunks=(chunks, 1, -1)),
                          da.from_array(point_params["ref_freq"],
                                        chunks=chunks))
            if gauss_params["n_src"] > 0:
                source_data[name]['gauss'] = \
                    Gauss(da.from_array(gauss_params["radec"],
                                        chunks=(chunks, -1)),
                          da.from_array(gauss_params["stokes"],
                                        chunks=(chunks, -1)),
                          da.from_array(gauss_params["spi"],
                                        chunks=(chunks, 1, -1)),
                          da.from_array(point_params["ref_freq"],
                                        chunks=chunks),
                          da.from_array(gauss_params["shape"],
                                        chunks=(chunks, -1)))

    return source_data


def support_tables(opts, tables):
    """
    Parameters
    ----------
    args : object
        Script argument objects
    tables : list of str
        List of support tables to open
    Returns
    -------
    table_map : dict of Dataset
        {name: dataset}
    """
    return {t: [ds.compute() for ds in
                xds_from_table("::".join((opts.input_ms_name, t)),
                               group_cols="__row__")]
            for t in tables}


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


def vis_factory(opts, source_type, sky_model, time_index,
                ms, field, spw, pol):
    try:
        source = sky_model[source_type]
    except KeyError:
        raise ValueError("Source type '%s' unsupported" % source_type)

    corrs = pol.NUM_CORR.values

    # Added the squeeze - no idea if this is correct. TODO: Ask Simon.
    # The phase direction has dimensions (row, poly, 2). Currently we assume
    # that both row and poly are size one and can be squeezed out. Similarly,
    # we squeeze the row dimension out of the spectral window.

    if field.NUM_POLY.data != 0:
        raise ValueError("Polynomials in FIELD table currently unsupported.")

    lm = radec_to_lm(source.radec, field.PHASE_DIR.data[0][0])
    uvw = -ms.UVW.data if opts.input_model_invert_uvw else ms.UVW.data
    frequency = spw.CHAN_FREQ.data[0]

    # (source, row, frequency)
    phase = phase_delay(lm, uvw, frequency)

    # (source, spi, corrs)
    # Apply spectral mode to stokes parameters
    stokes = spectral_model(source.stokes,
                            source.spi,
                            source.ref_freq,
                            frequency,
                            base=[1, 0, 0, 0])

    brightness = convert(stokes, ["I", "Q", "U", "V"],
                         corr_schema(pol))

    args = ["phase_delay", phase]

    # Add any visibility amplitude terms
    if source_type == "gauss":
        args.append("gauss_shape")
        args.append(gaussian_shape(uvw, frequency, source.shape))

    args.extend(["brightness", brightness])

    jones = baseline_jones_multiply(corrs, *args)

    return predict_vis(time_index, ms.ANTENNA1.data, ms.ANTENNA2.data,
                       None, jones, None, None, None, None)


def predict(data_xds, opts):
    # Convert source data into dask arrays
    sky_model = parse_sky_model(opts)

    # Get the support tables
    tables = support_tables(opts, ["FIELD", "DATA_DESCRIPTION",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])

    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    # List of predict operations
    predict_xds = []

    for xds in data_xds:

        # Extract frequencies from the spectral window associated
        # with this data descriptor id
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]

        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.values[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.values[0]]

        corrs = opts._ms_ncorr

        _, time_index = da.unique(xds.TIME.data, return_inverse=True)

        vis = []

        # Generate visibility expressions for each source type
        for name, sources in sky_model.items():

            source_vis = [vis_factory(opts, stype, sources, time_index,
                                      xds, field, spw, pol)
                          for stype in sources.keys()]

            # Sum visibilities together
            vis.append(sum(source_vis))

        vis = da.stack(vis, axis=2).rechunk({2: len(sky_model)})

        # Reshape (2, 2) correlation to shape (4,)
        if corrs == 4:
            vis = vis.reshape(vis.shape[:3] + (4,))

        # Assign visibilities to MODEL_DATA array on the dataset
        xds = xds.assign({"MODEL_DATA": (("row", "chan", "dir", "corr"), vis)})

        # Add to the list of predicts.
        predict_xds.append(xds)

    return predict_xds
