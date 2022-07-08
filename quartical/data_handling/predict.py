from collections import defaultdict
from functools import lru_cache
import weakref

from astropy.io import fits
import dask.array as da
import dask
from xarray import DataArray, Dataset
from dask.graph_manipulation import clone
from daskms import xds_from_storage_table
from loguru import logger
import numpy as np
import Tigger

from africanus.util.casa_types import STOKES_ID_MAP
from africanus.util.beams import beam_filenames, beam_grids

from africanus.experimental.rime.fused import RimeSpecification
from africanus.experimental.rime.fused.dask import rime

from quartical.utils.collections import freeze_default_dict

_empty_spectrum = object()


def parse_sky_models(sky_models):
    """Parses a Tigger sky model.

    Args:
        sky_models: A list of SkyModel objects.
    Returns:
        sky_model_dict: A dictionary of source data.
    """

    sky_model_dict = {}

    for sky_model_tuple in sky_models:

        sky_model_name, sky_model_tags = sky_model_tuple

        sky_model = Tigger.load(sky_model_name, verbose=False)

        sources = sky_model.sources

        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        if sky_model.freq0 is None:
            s0 = sources[0]
            s0_spectrum = getattr(s0, "spectrum", _empty_spectrum)
            fallback_freq0 = getattr(s0_spectrum, "freq0", None)
            logger.info(f"No reference frequency found for {sky_model_name}. "
                        f"Reference frequency for first source in the model "
                        f"is {fallback_freq0}. If this is None, "
                        f"all sources will be treated as flat spectrum.")
        else:
            fallback_freq0 = sky_model.freq0
            logger.info(f"Setting the default reference frequency for "
                        f"{sky_model_name} to {fallback_freq0}.")

        for source in sources:

            tagged = any([source.getTag(tag) for tag in sky_model_tags])

            parent_group = source.getTag("cluster") if tagged else "DIE"

            ra = source.pos.ra
            dec = source.pos.dec
            typecode = source.typecode.lower()
            flux = [getattr(source.flux, sto, 0) for sto in "IQUV"]
            spectrum = getattr(source, "spectrum", _empty_spectrum)

            # Attempts to grab the source reference frequency. Failing that,
            # the skymodel reference frequency is used. If that still fails,
            # will error out. However, if the first source has a reference
            # frequency set, we will instead default to that.

            if (fallback_freq0 is None) or (fallback_freq0 == 0):
                ref_freq = 1e9  # Non-zero default.
                spi = [[0]*4]  # Flat spectrum.
            else:
                ref_freq = getattr(spectrum, "freq0", fallback_freq0)
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


def daskify_sky_model_dict(sky_model_dict, chunk_size):
    """Converts source parameter dictionary into a dictionary of dask arrays.

    Args:
        sky_model_dict: Dictionary of sources.
        chunk_size: Interger number of sources in a chunk.

    Returns:
        dask_sky_model_dict: A dictionary of dask arrays.
    """

    dask_sky_model_dict = sky_model_dict.copy()  # Avoid mutating input.

    # This single line can have a large impact on memory/performance. Sets
    # the source chunking strategy for the predict.

    for model_name, model_group in sky_model_dict.items():
        for group_name, group_sources in model_group.items():
            if "point" in group_sources:
                arrays = group_sources["point"]

                def darray(name, chunks):
                    return da.from_array(arrays[name], chunks=chunks)

                dask_sky_model_dict[model_name][group_name]['point'] = \
                    Dataset({
                        "radec": (("source", "radec_comp"),
                                  darray("radec", (chunk_size, -1))),
                        "stokes": (("source", "corr"),
                                   darray("stokes", (chunk_size, -1))),
                        "spi": (("source", "spec_idx", "corr"),
                                darray("spi", (chunk_size, 1, -1))),
                        "ref_freq": (("source",),
                                     darray("ref_freq", chunk_size))})
            if "gauss" in group_sources:
                arrays = group_sources["gauss"]

                def darray(name, chunks):
                    return da.from_array(arrays[name], chunks=chunks)

                dask_sky_model_dict[model_name][group_name]['gauss'] = \
                    Dataset({
                        "radec": (("source", "radec_comp"),
                                  darray("radec", (chunk_size, -1))),
                        "stokes": (("source", "corr"),
                                   darray("stokes", (chunk_size, -1))),
                        "spi": (("source", "spec_idx", "corr"),
                                darray("spi", (chunk_size, 1, -1))),
                        "ref_freq": (("source",),
                                     darray("ref_freq", chunk_size)),
                        "gauss_shape": (("source", "gauss_comp"),
                                        darray("shape", (chunk_size, -1)))})

    return dask_sky_model_dict


@lru_cache(maxsize=16)
def load_beams(beam_file_schema, corr_types, beam_l_axis, beam_m_axis):
    """Loads in the beams spcified by the beam schema.

    Adapted from https://github.com/ska-sa/codex-africanus.

    Args:
        beam_file_schema: String conatining MeqTrees like schema.
        corr_types: Tuple of correlation types obtained from the MS.
        beam_l_axis: String identifying beam l axis.
        beam_m_axis: String identifying beam m axis.

    Returns:
        beam: (npix, npix, nchan, ncorr) dask array of beam values.
        beam_lm_ext: (2, 2) dask array of the beam's extent in the lm plane.
        beam_freq_grid: (nchan) dask array of frequency values at which we
            have beam samples.
    """

    class FITSFile(object):
        """ Exists so that fits file is closed when last ref is gc'd """

        def __init__(self, filename):
            self.filename = filename
            self.hdul = hdul = fits.open(filename)
            assert len(hdul) == 1
            self.__del_ref = weakref.ref(self, lambda r: hdul.close())

        def __reduce__(self):
            return (FITSFile, (self.filename,))

    # Open files and get headers
    beam_files = []
    headers = []

    for corr, (re, im) in beam_filenames(beam_file_schema, corr_types).items():
        re_f = FITSFile(re)
        im_f = FITSFile(im)
        beam_files.append((corr, (re_f, im_f)))
        headers.append((corr, (re_f.hdul[0].header, im_f.hdul[0].header)))

    # All FITS headers should agree (apart from DATE)
    flat_headers = []

    for corr, (re_header, im_header) in headers:
        if "DATE" in re_header:
            del re_header["DATE"]
        if "DATE" in im_header:
            del im_header["DATE"]
        flat_headers.append(re_header)
        flat_headers.append(im_header)

    if not all(flat_headers[0] == h for h in flat_headers[1:]):
        raise ValueError("BEAM FITS Header Files differ")

    #  Map FITS header type to NumPy type
    BITPIX_MAP = {8: np.dtype('uint8').type,
                  16: np.dtype('int16').type,
                  32: np.dtype('int32').type,
                  -32: np.dtype('float32').type,
                  -64: np.dtype('float64').type}

    header = flat_headers[0]
    bitpix = header['BITPIX']

    try:
        dtype = BITPIX_MAP[bitpix]
    except KeyError:
        raise ValueError("No mapping from BITPIX %s to a numpy type" % bitpix)
    else:
        dtype = np.result_type(dtype, np.complex64)

    if not header['NAXIS'] == 3:
        raise ValueError("FITS must have exactly three axes. "
                         "L or X, M or Y and FREQ. NAXIS != 3")

    (l_ax, l_grid), (m_ax, m_grid), (nu_ax, nu_grid) = \
        beam_grids(header,
                   beam_l_axis.replace("~", "-"),
                   beam_m_axis.replace("~", "-"))

    # Shape of each correlation
    shape = (l_grid.shape[0], m_grid.shape[0], nu_grid.shape[0])

    # Axis tranpose, FITS is FORTRAN ordered
    ax = (nu_ax - 1, m_ax - 1, l_ax - 1)

    def _load_correlation(re, im, ax):
        # Read real and imaginary for each correlation
        return (re.hdul[0].data.transpose(ax) +
                im.hdul[0].data.transpose(ax)*1j)

    # Create delayed loads of the beam
    beam_loader = dask.delayed(_load_correlation)

    beam_corrs = [beam_loader(re, im, ax)
                  for c, (corr, (re, im)) in enumerate(beam_files)]
    beam_corrs = [da.from_delayed(bc, shape=shape, dtype=dtype)
                  for bc in beam_corrs]

    # Stack correlations and rechunk to one great big block
    beam = da.stack(beam_corrs, axis=3)
    beam = beam.rechunk(shape + (len(corr_types),))

    # Dask arrays for the beam extents and beam frequency grid
    beam_lm_ext = np.array([[l_grid[0], l_grid[-1]], [m_grid[0], m_grid[-1]]])
    beam_lm_ext = da.from_array(beam_lm_ext, chunks=beam_lm_ext.shape)
    beam_freq_grid = da.from_array(nu_grid, chunks=nu_grid.shape)

    return beam, beam_lm_ext, beam_freq_grid


def get_support_tables(ms_path):
    """Get the support tables necessary for the predict.

    Adapted from https://github.com/ska-sa/codex-africanus.

    Args:
        ms_path: Path to the input MS.

    Returns:
        lazy_tables: Dictionary of support table datasets.
    """

    n = {k: '::'.join((ms_path, k)) for k
         in ("ANTENNA", "DATA_DESCRIPTION", "FIELD",
             "SPECTRAL_WINDOW", "POLARIZATION", "FEED")}

    # All rows at once
    lazy_tables = {"ANTENNA": xds_from_storage_table(n["ANTENNA"]),
                   "FEED": xds_from_storage_table(n["FEED"])}

    compute_tables = {
        # NOTE: Even though this has a fixed shape, I have ammended it to
        # also group by row. This just makes life fractionally easier.
        "DATA_DESCRIPTION": xds_from_storage_table(n["DATA_DESCRIPTION"],
                                                   group_cols="__row__"),
        # Variably shaped, need a dataset per row
        "FIELD":
            xds_from_storage_table(n["FIELD"], group_cols="__row__"),
        "SPECTRAL_WINDOW":
            xds_from_storage_table(n["SPECTRAL_WINDOW"], group_cols="__row__"),
        "POLARIZATION":
            xds_from_storage_table(n["POLARIZATION"], group_cols="__row__"),
    }

    lazy_tables.update(dask.compute(compute_tables)[0])

    return lazy_tables


def build_rime_spec(stokes, corrs, source_type, model_opts):
    left, middle, right = [], ["Kpq", "Bpq"], []
    terms = {}

    if source_type == "point":
        pass
    elif source_type == "gauss":
        middle.insert(0, "Cpq")
        terms["C"] = "Gaussian"
    else:
        raise ValueError(f"Unhandled source_type {source_type}")

    if model_opts.apply_p_jones:
        left.insert(0, "Lp")
        right.append("Lq")

    if model_opts.beam:
        left.insert(0, "Ep")
        right.append("Eq")

    onion = ",".join(left + middle + right)
    bits = ["(", onion, "): ",
            "[", ",".join(stokes), "] -> [", ",".join(corrs), "]"]
    return RimeSpecification("".join(bits), terms=terms)


def predict(data_xds_list, model_vis_recipe, ms_path, model_opts):
    """Produces graphs describing predict operations.

    Adapted from https://github.com/ska-sa/codex-africanus.

    Args:
        data_xds_list: A list of xarray datasets corresponing to the input
            measurement set data.
        model_vis_recipe: A Recipe object.
        ms_path: Path to input MS.
        model_opts: A ModelInputs configuration object.

    Returns:
        predict_list: A list of dictionaries containing dask graphs describing
            the predict.
    """

    # Read in a Tigger .lsm.html and produce a dictionary of sources per
    # unique sky model and tag combination. Tags determine clustering.

    sky_model_dict = parse_sky_models(model_vis_recipe.ingredients.sky_models)

    # Convert sky model dictionary into a dictionary of per-model dask arrays.

    dask_sky_model_dict = daskify_sky_model_dict(sky_model_dict,
                                                 model_opts.source_chunks)

    # Get the support tables (as lists), and give them sensible names.
    tables = get_support_tables(ms_path)

    ant_xds_list = tables["ANTENNA"]
    field_xds_list = tables["FIELD"]
    ddid_xds_list = tables["DATA_DESCRIPTION"]
    spw_xds_list = tables["SPECTRAL_WINDOW"]
    pol_xds_list = tables["POLARIZATION"]
    feed_xds_list = tables["FEED"]

    assert len(ant_xds_list) == 1
    assert len(feed_xds_list) == 1

    # List of predict operations
    predict_list = []
    messages = set()

    for data_xds in data_xds_list:
        # Perform subtable joins.
        ant_xds = ant_xds_list[0]
        feed_xds = feed_xds_list[0]
        field_xds = field_xds_list[data_xds.attrs['FIELD_ID']]
        ddid_xds = ddid_xds_list[data_xds.attrs['DATA_DESC_ID']]
        spw_xds = spw_xds_list[ddid_xds.SPECTRAL_WINDOW_ID.data[0]]
        pol_xds = pol_xds_list[ddid_xds.POLARIZATION_ID.data[0]]
        corr_type = tuple(pol_xds.CORR_TYPE.data[0])
        corr_schema = [STOKES_ID_MAP[ct] for ct in corr_type]
        stokes_schema = ["I", "Q", "U", "V"]
        chan_freq = da.from_array(spw_xds.CHAN_FREQ.data[0],
                                  chunks=data_xds.chunks['chan'])
        phase_dir = da.from_array(field_xds.PHASE_DIR.data[0][0])  # row, poly
        extras = {
            "phase_dir": clone(phase_dir),
            "chan_freq": clone(chan_freq),
            "antenna_position": clone(ant_xds.POSITION.data),
            "receptor_angle": clone(feed_xds.RECEPTOR_ANGLE.data),
            "convention": "casa" if model_opts.invert_uvw else "fourier"
        }

        if model_opts.beam:
            beam, lm_ext, freq_map = load_beams(model_opts.beam,
                                                corr_type,
                                                model_opts.beam_l_axis,
                                                model_opts.beam_m_axis)

            extras["beam"] = beam
            extras["beam_lm_extents"] = lm_ext
            extras["beam_freq_map"] = freq_map

        model_vis = defaultdict(list)

        # Generate visibility expressions per model, per direction for each
        # source type.
        for model_name, model_group in dask_sky_model_dict.items():
            for group_name, group_sources in model_group.items():
                source_vis = []

                # Generate visibilities per source type.
                for source_type, sky_model in group_sources.items():
                    spec = build_rime_spec(stokes_schema, corr_schema,
                                           source_type, model_opts)

                    messages.add(
                        f"Predicting {source_type} sources using {spec}."
                    )

                    source_vis.append(
                        rime(spec, data_xds, clone(sky_model), extras)
                    )

                vis = DataArray(da.stack(source_vis).sum(axis=0),
                                dims=["row", "chan", "corr"],
                                coords={"corr": corr_schema})

                vis = vis.sel(corr=data_xds.corr.values)

                # Append group_vis to the appropriate list.
                model_vis[model_name].append(vis.data)

        predict_list.append(freeze_default_dict(model_vis))

    for message in messages:
        logger.info(message)

    return predict_list
