from loguru import logger
import dask.array as da
import dask
import Tigger
import weakref
import multiprocessing
from daskms import xds_from_table
import numpy as np

from quartical.utils.dask import blockwise_unique
from quartical.utils.collections import freeze_default_dict

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import (phase_delay as compute_phase_delay,
                                 predict_vis,
                                 parallactic_angles as
                                 compute_parallactic_angles,
                                 feed_rotation as compute_feed_rotation,
                                 beam_cube_dde)
from africanus.model.coherency.dask import convert
from africanus.model.spectral.dask import spectral_model
from africanus.model.shape.dask import gaussian as gaussian_shape
from africanus.util.beams import beam_filenames, beam_grids
from africanus.linalg.geometry import (BoundingBox,
                                       BoundingBoxFactory)
from africanus.gridding.perleypolyhedron import kernels
from africanus.gridding.perleypolyhedron.degridder import (
    degridder as np_degridder)
from africanus.gridding.perleypolyhedron.degridder import (
    degridder_serial as np_degridder_serial)
from africanus.gridding.perleypolyhedron.policies import (
    stokes_conversion_policies)

from collections import namedtuple, defaultdict
from functools import lru_cache
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
import math
import os

import re
import pickle
import pyfftw

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

        fallback_freq0 = sky_model.freq0

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
            ref_freq = getattr(spectrum, "freq0", fallback_freq0)

            if ref_freq is None and hasattr(spectrum, "spi"):
                raise ValueError("Reference frequency not found for source {} "
                                 "in {}. Please set reference frequency for "
                                 "either this source or the entire file."
                                 "".format(source.name, sky_model_name))
            else:
                fallback_freq0 = fallback_freq0 or ref_freq

            if ref_freq is not None:
                # Extract SPI for I, defaulting to 0 - flat spectrum.
                spi = [[getattr(spectrum, "spi", 0)]*4]
            else: # no reference frequency set and no spi set (error above)
                spi = [[0]*4] # flat spectrum source assumed
                ref_freq = 1.0e-10

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

    chunks = opts.dft_predict_source_chunks \
        if opts.input_model_predict_mode == "dft" else -1

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
            self.hdul = hdul = fits.open(filename)
            assert len(hdul) == 1
            self.__del_ref = weakref.ref(self, lambda r: hdul.close())

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


def get_support_tables(opts):
    """Get the support tables necessary for the predict.

    Adapted from https://github.com/ska-sa/codex-africanus.

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
        # NOTE: Even though this has a fixed shape, I have ammended it to
        # also group by row. This just makes life fractionally easier.
        "DATA_DESCRIPTION": xds_from_table(n["DATA_DESCRIPTION"],
                                           group_cols="__row__"),
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
    """Define correlation schema.

    Adapted from https://github.com/ska-sa/codex-africanus.

    Args:
        pol: xarray dataset containing POLARIZATION subtable information.

    Returns:
        corr_schema: A list of lists containing the correlation schema from
        the POLARIZATION table, e.g. `[[9, 10], [11, 12]]` for example.
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
    """Multiplies per-baseline Jones terms together.

    Args:
        corrs: Integer number of corellations.
        *args: Variable length list of alternating term names and assosciated
            dask arrays.

    Returns:
        Dask array contatining the result of multiplying the input Jones terms.
    """

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

    NOTE: This lacks test coverage.

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


def _zero_pes(parangles, frequency, dtype_):
    """ Create zeroed pointing errors """
    ntime, na = parangles.shape
    nchan = frequency.shape[0]
    return np.zeros((ntime, na, nchan, 2), dtype=dtype_)


def _unity_ant_scales(parangles, frequency, dtype_):
    """ Create zeroed antenna scalings """
    _, na = parangles[0].shape
    nchan = frequency.shape[0]
    return np.ones((na, nchan, 2), dtype=dtype_)


def dde_factory(ms, utime, frequency, ant, feed, field, pol, lm, opts):
    """Multiplies per-antenna direction-dependent Jones terms together.

    Adapted from https://github.com/ska-sa/codex-africanus.

    Args:
        ms: xarray dataset containing measurement set data.
        utime: A dask array of unique TIME column vales.
        frequency: A dask array of per-channel frequency values.
        ant: xarray dataset containing ANTENNA subtable data.
        feed: xarray dataset containing FEED subtable data.
        field: xarray dataset containing FIELD subtable data.
        pol: xarray dataset containing POLARIZATION subtable data.
        lm: dask array of per-source lm values.
        opts: A Namespace object containing global options.

    Returns:
        Dask array containing the result of multiplying the
            direction-dependent Jones terms together.
    """
    if opts.input_model_beam is None:
        return None

    # Beam is requested
    corr_type = tuple(pol.CORR_TYPE.data[0])

    if not len(corr_type) == 4:
        raise ValueError("Need four correlations for DDEs")

    parangles = compute_parallactic_angles(utime, ant.POSITION.data,
                                           field.PHASE_DIR.data[0][0])

    # Construct feed rotation
    p_jones = compute_p_jones(parangles, feed, opts)

    dtype = np.result_type(parangles, frequency)

    # Create zeroed pointing errors
    zpe = da.blockwise(_zero_pes, ("time", "ant", "chan", "comp"),
                       parangles, ("time", "ant"),
                       frequency, ("chan",),
                       dtype, None,
                       new_axes={"comp": 2},
                       dtype=dtype)

    # Created zeroed antenna scaling factors
    zas = da.blockwise(_unity_ant_scales, ("ant", "chan", "comp"),
                       parangles, ("time", "ant"),
                       frequency, ("chan",),
                       dtype, None,
                       new_axes={"comp": 2},
                       dtype=dtype)

    # Load the beam information
    beam, lm_ext, freq_map = load_beams(opts.input_model_beam,
                                        corr_type,
                                        opts.input_model_beam_l_axis,
                                        opts.input_model_beam_m_axis)

    # Introduce the correlation axis
    beam = beam.reshape(beam.shape[:3] + (2, 2))

    beam_dde = beam_cube_dde(beam, lm_ext, freq_map, lm, parangles,
                             zpe, zas,
                             frequency)

    # Multiply the beam by the feed rotation to form the DDE term
    return da.einsum("stafij,tajk->stafik", beam_dde, p_jones)

def degridder(uvw,
              gridstack,
              lambdas,
              chanmap,
              cell,
              image_centres,
              phase_centre,
              convolution_kernel,
              convolution_kernel_width,
              convolution_kernel_oversampling,
              baseline_transform_policy,
              phase_transform_policy,
              stokes_conversion_policy,
              convolution_policy,
              vis_dtype=np.complex128,
              rowparallel=False):
    """
    2D Convolutional degridder, discrete to contiguous

    Adapted from https://github.com/ska-sa/codex-africanus.

    @uvw: value coordinates, (nrow, 3)
    @gridstack: complex gridded data, (nband, npix, npix)
    @lambdas: wavelengths of data channels
    @chanmap: MFS band mapping per channel
    @cell: cell_size in degrees
    @image_centre: new phase centre of image (radians, ra, dec)
    @phase_centre: original phase centre of data (radians, ra, dec)
    @convolution_kernel: packed kernel as generated by kernels package
    @convolution_kernel_width: number of taps in kernel
    @convolution_kernel_oversampling: number of oversampled points in kernel
    @baseline_transform_policy: any accepted policy in
                                .policies.baseline_transform_policies,
                                can be used to tilt image planes for
                                polyhedron faceting
    @phase_transform_policy: any accepted policy in
                             .policies.phase_transform_policies,
                             can be used to facet at provided
                             facet @image_centre
    @stokes_conversion_policy: any accepted correlation to stokes
                               conversion policy in
                               .policies.stokes_conversion_policies
    @convolution_policy: any accepted convolution policy in
                         .policies.convolution_policies
    @vis_dtype: accumulation vis dtype (default complex 128)
    @rowparallel: adds additional threading per row per chunk. This may be
                  necessary for cases where there are few facets and few chunks
                  to get optimal performance. Requires TBB to be installed
                  from your distribution package management system. See numba
                  documentation
                  http://numba.pydata.org/numba-doc/0.46.0/user/threading-layer.html
                  Must set:
                  from numba import config # have to be openmp to support
                  nested parallelism config.THREADING_LAYER = 'threadsafe'
                  before calling this function
    """
    def __degrid(uvw,
             gridstack,
             lambdas,
             chanmap,
             image_centres,
             phase_centre,
             cell=None,
             convolution_kernel=None,
             convolution_kernel_width=None,
             convolution_kernel_oversampling=None,
             baseline_transform_policy=None,
             phase_transform_policy=None,
             stokes_conversion_policy=None,
             convolution_policy=None,
             vis_dtype=np.complex128,
             rowparallel=False):
        image_centres = image_centres[0]
        if image_centres.ndim != 2:
            raise ValueError(
                "Image centres for DASK wrapper expects list of image centres, "
                "one per facet in radec radians"
            )
        if image_centres.shape[1] != 2:
            raise ValueError("Image centre must be a list of tuples")
        uvw = uvw[0]
        if uvw.ndim != 2 or uvw.shape[1] != 3:
            raise ValueError("UVW array must be nrow x 3")
        gridstack = gridstack[0][0][0]
        if gridstack.ndim != 4:
            raise ValueError("Gridstack must be nfacet x nband x ny x nx")
        lambdas = lambdas
        chanmap = chanmap
        if chanmap.size != lambdas.size:
            raise ValueError(
                "Chanmap and corresponding lambdas must match in shape")
        nchan = lambdas.size
        nrow = uvw.shape[0]
        nfacet, _ = image_centres.shape
        ncorr = stokes_conversion_policies.ncorr_outpy(
            policy_type=stokes_conversion_policy)()
        vis = np.zeros((nfacet, nrow, nchan, ncorr), dtype=vis_dtype)
        degridcall = np_degridder_serial if not rowparallel else np_degridder
        for fi, f in enumerate(image_centres):
            # add contributions from all facets
            vis[fi, :, :, :] = \
                degridcall(uvw,
                           gridstack[fi, :, :, :],
                           lambdas,
                           chanmap,
                           cell,
                           f,
                           phase_centre,
                           convolution_kernel,
                           convolution_kernel_width,
                           convolution_kernel_oversampling,
                           baseline_transform_policy,
                           phase_transform_policy,
                           stokes_conversion_policy,
                           convolution_policy,
                           vis_dtype=vis_dtype)
        return vis

    if image_centres.ndim != 2:
        raise ValueError(
            "Image centres for DASK wrapper expects list of "
            "image centres, one per facet in radec radians"
        )
    if image_centres.shape[1] != 2:
        raise ValueError("Image centre must be a list of tuples")
    if gridstack.ndim != 4 or gridstack.shape[0] != image_centres.shape[0]:
        raise ValueError(
            "Grid stack must be nfacet x nband x yy x xx and match number "
            "of image centres"
        )
    vis = da.blockwise(
        __degrid, ("nfacet", "row", "chan", "corr"),
        uvw, ("row", "uvw"),
        gridstack, ("nfacet", "nband", "y", "x"),
        lambdas, ("chan", ),
        chanmap, ("chan", ),
        image_centres, ("nfacet", "coord"),
        convolution_kernel=convolution_kernel,
        convolution_kernel_width=convolution_kernel_width,
        convolution_kernel_oversampling=convolution_kernel_oversampling,
        baseline_transform_policy=baseline_transform_policy,
        phase_transform_policy=phase_transform_policy,
        stokes_conversion_policy=stokes_conversion_policy,
        convolution_policy=convolution_policy,
        cell=cell,
        phase_centre=phase_centre,
        vis_dtype=vis_dtype,
        new_axes={
            "corr":
            stokes_conversion_policies.ncorr_outpy(
                policy_type=stokes_conversion_policy)()
        },
        dtype=vis_dtype,
        meta=np.empty(
            (0, 0, 0),
            dtype=vis_dtype)  # row, chan, correlation product as per MSv2 spec
    )
    return vis

def render_gaussians(stokes, gaussian_extent, gaussian_shape, sradec,
                    frequency, phase_centre, cellsize, npix):
    """Renders sources onto a cube to create a model for degridding purposes
    Args:
        stokes: da array of stokes parameters per band  (nsrc x nband)
        gaussian_shape: da array of (emaj, emin, pos) tripples (nsrc x 3),
                        emaj and emin given at fwhm, pos in radians
        gaussian_extent: da array of gaussian support size in number of pixels
        sradec: da array of source ra, dec radians (nsrc x 2)
        frequency: da array of frequencies of band centres (nband)
        cellsize: model resolution in radians
        npix: size of the (square) grid to render model to
        phase_centre: phase centre of model in radians
    Returns:
        da array of shape (nband x npix x npix) containing rendered model
    """
    def __render(stokes, gaussian_extent, gaussian_shape,
                 sradec, frequency, phase_centre, cellsize, npix):
        ndegridband = frequency.size
        fwhm_inv = 1.0 / (2*np.sqrt(2*np.log(2)))
        if stokes.shape[1] != ndegridband:
            raise ValueError("Source stokes vector must have channel "
                             "dimension equal to #degridbands")
        model = np.zeros((1, ndegridband, npix, npix), dtype=np.float32)
        for si, (s, sup, g, (ra, dec)) in enumerate(zip(stokes,
                                                        gaussian_extent,
                                                        gaussian_shape,
                                                        sradec)):
            emaj, emin, pa = g
            xx = (np.arange(sup) - sup//2)
            x, y = np.meshgrid(xx, xx)
            xr = np.cos(pa) * x + np.sin(pa) * y
            yr = -np.sin(pa) * x + np.cos(pa) * y
            # unnormalized gaussian rendered at FWHM
            g = np.exp(-0.5 * ((xr/(emaj/cellsize*fwhm_inv))**2 + 
                               (yr/(emin/cellsize*fwhm_inv))**2))
            # render overlapping portion of g onto model
            hdr = {"NAXIS": 2, 
                   "NAXIS1":npix, 
                   "NAXIS2":npix, 
                   "CDELT1":-np.rad2deg(cellsize), # RA 
                   "CDELT2":np.rad2deg(cellsize),  # DEC
                   "CRPIX1":npix*0.5 + 1, # for FORTRAN indexing
                   "CRPIX2":npix*0.5 + 1, # for FORTRAN indexing
                   "CRVAL1": np.rad2deg(phase_centre[0]), 
                   "CRVAL2": np.rad2deg(phase_centre[1]), 
                   "CTYPE1": "deg", 
                   "CTYPE2": "deg"}
            w = WCS(hdr)
            sx, sy = w.all_world2pix([[np.rad2deg(ra), 
                                       np.rad2deg(dec)]], 
                                     1)[0] - 1 # FORTRAN ordering, starting at 1
            sx = int(np.round(sx))
            sy = int(np.round(sy))
            grb = max(0, sup - max(0, sx + sup // 2 + 1 - npix))
            glb = min(sup, max(0, 0 - (sx - sup // 2)))
            gub = max(0,sup - max(0, sx + sup // 2 + 1 - npix))
            gbb = min(sup, max(0, 0 - (sx - sup // 2)))
            gsel = g[gbb:gub,glb:grb]
            mlb = max(sx - sup // 2, 0)
            mrb = max(min(sx + sup // 2 + 1, npix), 0)
            mbb = max(sy - sup // 2, 0)
            mub = max(min(sy + sup // 2 + 1, npix), 0)
            modsel = model[0, :, mbb:mub,mlb:mrb].view()
            if np.prod(modsel.shape[1:3]) != 0:
                assert modsel.shape[1:3] == gsel.shape
                modsel += gsel[None, :, :] * s[:, None, None]
        return model
    assert len(stokes.chunks) == len(stokes.shape)
    assert len(gaussian_extent.chunks) == len(gaussian_extent.shape)
    assert len(gaussian_shape.chunks) == len(gaussian_shape.shape)
    assert len(sradec.chunks) == len(sradec.shape)
    assert len(frequency.chunks) == len(frequency.shape)
    assert stokes.shape[0] == gaussian_extent.shape[0]
    assert stokes.shape[0] == gaussian_shape.shape[0]
    assert stokes.shape[0] == sradec.shape[0]
    assert stokes.shape[1] == frequency.shape[0]
    assert stokes.chunks[0] == gaussian_extent.chunks[0]
    assert stokes.chunks[0] == gaussian_shape.chunks[0]
    assert stokes.chunks[0] == sradec.chunks[0]
    assert stokes.chunks[1] == frequency.chunks[0]
    assert gaussian_extent.ndim == 1
    assert gaussian_shape.ndim == 2
    assert gaussian_shape.shape[1] == 3
    assert sradec.shape[1] == 2
    assert stokes.ndim == 3
    # render chunks of rows in parallel, reduce over first dimension
    # for now only support stokes I
    return da.blockwise(__render, ("nsrc", "nband", "y", "x"),
                        stokes[:, :, 0], ("nsrc", "nband"),
                        gaussian_extent, ("nsrc",),
                        gaussian_shape, ("nsrc", "gaussshape"),
                        sradec, ("nsrc", "radec"),
                        frequency, ("nband",),
                        phase_centre=phase_centre, 
                        cellsize=cellsize, 
                        npix=npix,
                        dtype=np.float32,
                        meta=np.empty((0,0,0,0), dtype=np.float32),
                        concatenate=True,
                        new_axes={"x": npix,
                                  "y": npix},
                        # one model cube per source row chunk
                        adjust_chunks={"nsrc": 1}
                       ).sum(axis=0)

def gaussian_extents(gaussian_shape, cellsize, 
                     truncate_sigma=10.0):
    """Calculates source extents in number of pixels
    Args:
        gaussian_shape: da array of (emaj, emin, pos) tripples (nsrc x 3),
                        emaj and emin given at fwhm, pos in radians
        cellsize: model resolution in radians
        truncate_sigma: truncate rendered gaussians at specified sigma
    Returns:
        da array of source extents in number of pixels
    """
    def __extents(gaussian_shape, cellsize, 
                  truncate_sigma=10.0):
        fwhm_inv = 1.0 / (2*np.sqrt(2*np.log(2)))
        sext = np.zeros(gaussian_shape.shape[0], dtype=np.int64)
        for si, g in enumerate(gaussian_shape):
            emaj, emin, pa = g
            sigpx = emaj / cellsize * fwhm_inv
            nx = max(int(np.ceil(sigpx * truncate_sigma)), 1)
            nx = nx + 1 if nx % 2 == 0 else nx
            sext[si] = nx
        return sext
    return da.blockwise(__extents, ("nsrc",),
                        gaussian_shape, ("nsrc", "gaussshape"),
                        cellsize=cellsize,
                        truncate_sigma=truncate_sigma,
                        dtype=np.int64,
                        meta=np.empty((0,), dtype=np.int64),
                        concatenate=True
                       )

def bandmapping(vis_freqs, band_frequencies):
    """ 
        Gives the frequency mapping for visibility to degrid band 
        Assume linear regular spacing, so we may end up
        with a completely empty band somewhere in the middle
        if we have spws that are separated in frequency
        Args:
            vis_freqs: channel centre frequencies for visibility data
            band_frequencies: should be a linear spaced array of band
                              frequencies
        Returns:
            da array of band mapping per channel
    """
    def __map(vis_freqs, band_frequencies):
        def find_nearest(array, value):
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
                return idx - 1
            else:
                return idx
        freq_mapping = [find_nearest(band_frequencies, v) for v in vis_freqs]
        return np.array(freq_mapping, dtype=np.int32)
    return da.blockwise(__map, ("chan", ),
                        vis_freqs, ("chan", ),
                        band_frequencies, ("band", ),
                        dtype=np.int32,
                        meta=np.empty((0,), dtype=np.int64),
                        concatenate=True)

def extract_facet_models(model, facet_bbs):
    """Extracts facet subcubes from model cube
        Arguments:
            model: must be nband x npix x npix model cube
            facet_bbs: ndarray of nfacet x square bounding boxes
                        of shape nfpix x nfpix
        Returns:
            da array of shape nfacet x nfpix x nfpix
    """
    if not all(map(lambda f: f == facet_bbs[0].box_npx,
                   map(lambda f: f.box_npx, facet_bbs))):
        raise ValueError("Facets must all be of same size")
    if not facet_bbs[0].box_npx[0] == facet_bbs[0].box_npx[1]:
        raise ValueError("Facets must all be square")
    if model.ndim != 3:
        raise ValueError("Expect model cube to be 3 dimensional")
    ndegridband = model.shape[0]
    nfpix = facet_bbs[0].box_npx[0]
    def __extract(model, facet_bbs, ndegridband, nfpix):
        facet_models = np.zeros((len(facet_bbs), 
                                ndegridband, 
                                nfpix, 
                                nfpix),
                                dtype=np.complex64)
        for bbi, bb in enumerate(facet_bbs):
            subcube = BoundingBox.regional_data(bb, 
                                                model,
                                                axes=(1, 2),
                                                oob_value=0)[0]
            facet_models[bbi, :, :, :] = subcube
        return facet_models

    return da.blockwise(__extract, ("nfacet", "nband", "y", "x"),
                        model, ("nband", "my", "mx"),
                        da.from_array(facet_bbs, chunks=(len(facet_bbs))), ("nfacet",),
                        ndegridband=ndegridband,
                        nfpix=nfpix,
                        dtype=np.complex64,
                        meta=np.empty((0, 0, 0, 0), dtype=np.complex64),
                        new_axes={"nband": ndegridband,
                                  "y": nfpix,
                                  "x": nfpix},
                        concatenate=True)

def dask_fft_cached_wisdom(opts, model, wisdom_file):
    """FFTs a 4d cube of nfacet x nband x npix x npix
       Arguments:
            opts: global options dict
            model: (da array) 4d cube of nfacet x nband x npix x npix
       Postcondition:
            Caches FFTW wisdom for this shape and reuse if possible
       Returns:
            FFT (with DC component at npix//2, npix//2) of model
    """
    # PyFFTW documenation incorrect. Implementation is missing necessary kwargs
    # define some replacements
    def __rfft4x2(a, **kwargs):
        return da.blockwise(pyfftw.interfaces.numpy_fft.rfftn, ("nfacet", "nband", "y", "x"),
                            a, ("nfacet", "nband", "y", "x"),
                            dtype=a.dtype,
                            concatenate=True,
                            meta=np.empty((0,0,0,0), dtype=a.dtype),
                            **kwargs)
    def __irfft4x2(a, **kwargs):
        return da.blockwise(pyfftw.interfaces.numpy_fft.irfftn, ("nfacet", "nband", "y", "x"),
                            a, ("nfacet", "nband", "y", "x"),
                            dtype=a.dtype,
                            concatenate=True,
                            meta=np.empty((0,0,0,0), dtype=a.dtype),
                            **kwargs)
    def __fft4x2(a, **kwargs):
        return da.blockwise(pyfftw.interfaces.numpy_fft.fftn, ("nfacet", "nband", "y", "x"),
                            a, ("nfacet", "nband", "y", "x"),
                            dtype=a.dtype,
                            concatenate=True,
                            meta=np.empty((0,0,0,0), dtype=a.dtype),
                            **kwargs)
    def __ifft4x2(a, **kwargs):
        return da.blockwise(pyfftw.interfaces.numpy_fft.ifftn, ("nfacet", "nband", "y", "x"),
                            a, ("nfacet", "nband", "y", "x"),
                            dtype=a.dtype,
                            concatenate=True,
                            meta=np.empty((0,0,0,0), dtype=a.dtype),
                            **kwargs)
    if model.ndim != 4:
        raise ValueError("Expect model cube to be 4 dimensional")
    nfacet, ndegridband, _, npix = model.shape
    fft_nthread = multiprocessing.cpu_count() if opts.fft_predict_fft_nthreads == -1 else \
            opts.fft_predict_fft_nthreads

    def __learn_wisdom(nfacet, nstack, npix, dtype=np.complex64):
        """Learns FFTW wisdom for 2D FFT of npix x npix stack of images"""
        logger.info("Computing fftw wisdom FFTs for shape [{} x {} x {} x {}] and dtype {}".format(
            nfacet, nstack, npix, npix, dtype.name))
        test = da.zeros((nfacet, nstack, npix, npix), dtype=dtype, chunks=(nfacet, nstack, npix, npix))
        if "float" in dtype.name:
            a = __rfft4x2(test, 
                          overwrite_input=True,
                          threads=fft_nthread,
                          axes=(2, 3),
                          planner_effort="FFTW_MEASURE")
            b = __irfft4x2(a, 
                           overwrite_input=True,
                           threads=fft_nthread,
                           axes=(2, 3),
                           planner_effort="FFTW_MEASURE")
        elif "complex" in dtype.name:
            a = __fft4x2(test,
                         overwrite_input=True,
                         threads=fft_nthread,
                         axes=(2, 3),
                         planner_effort="FFTW_MEASURE")
            b = __ifft4x2(a, 
                          overwrite_input=True,
                          threads=fft_nthread,
                          axes=(2, 3),
                          planner_effort="FFTW_MEASURE")
        b.compute()
    
    fft_plannereffort = "FFTW_ESTIMATE"
    if os.path.exists(wisdom_file) and os.path.isfile(wisdom_file):
        try:
            with open(wisdom_file, "rb") as wf:
                logger.info("Loading cached FFTW wisdom from {}".format(wisdom_file))
                dicowisdom = pickle.load(wf)
            pyfftw.import_wisdom(dicowisdom["Wisdom"])

            if (model.dtype.name, nfacet, ndegridband, npix, npix) not in dicowisdom["CachedShapes"]:
                dicowisdom["CachedShapes"].append((model.dtype.name, nfacet, ndegridband, npix, npix))
                __learn_wisdom(nfacet, ndegridband, npix, dtype=model.dtype)
                with open(wisdom_file, "wb") as wf:
                    logger.info("Caching FFTW wisdom to {}".format(wisdom_file))
                    pickle.dump(dicowisdom, wf)
        except:
            logger.error("Failed to load FFTW wisdom. Cache file is corrupt. Will recompute.")
            
            __learn_wisdom(nfacet, ndegridband, npix, dtype=model.dtype)
            dicowisdom = {
                "Wisdom": pyfftw.export_wisdom(),
                "CachedShapes": [(model.dtype.name, nfacet, ndegridband, npix, npix)]
            }
            with open(wisdom_file, "wb") as wf:
                logger.info("Caching FFTW wisdom to {}".format(wisdom_file))
                pickle.dump(dicowisdom, wf)
    else:
        __learn_wisdom(nfacet, ndegridband, npix, dtype=model.dtype)
        dicowisdom = {
            "Wisdom": pyfftw.export_wisdom(),
            "CachedShapes": [(model.dtype.name, nfacet, ndegridband, npix, npix)]
        }
        with open(wisdom_file, "wb") as wf:
            logger.info("Caching FFTW wisdom to {}".format(wisdom_file))
            pickle.dump(dicowisdom, wf)

    # finally fft
    return da.fft.fftshift(__ifft4x2(da.fft.ifftshift(model, 
                                                      axes=(2, 3)),
                                     axes=(2, 3),
                                     overwrite_input=True,
                                     threads=fft_nthread,
                                     planner_effort=fft_plannereffort
                                    ),
                           axes=(2, 3))

def vis_factory(opts, source_type, sky_model, ms, ant, field, spw, pol, feed,
                model_name, group_name):
    """Generates a graph describing the predict for an xds, model and type.

    Adapted from https://github.com/ska-sa/codex-africanus.

    Args:
        opts: A Namepspace of global options.
        source_type: A string - either "point" or "gauss".
        sky_model: The daskified sky model containing dask arrays of params.
        ms: An xarray.dataset containing a piece of the MS.
        ant: An xarray.dataset corresponding to the ANTENNA subtable.
        field: An xarray.dataset corresponding to the FIELD subtable.
        spw: An xarray.dataset corresponding to the SPECTRAL_WINDOW subtable.
        pol: An xarray.dataset corresponding the POLARIZATION subtable.
        feed: An xarray.dataset corresponding the FEED subtable.
        model_name: Name of model file
        group_name: Name of group / region within model                  
    Returns:
        The result of predict_vis - a graph describing the predict.
    """
    forward_transform = opts.input_model_predict_mode
    # Array containing source parameters.
    sources = sky_model[source_type]

    # Select single dataset rows
    corrs = pol.NUM_CORR.data[0]
    # Necessary to chunk the predict in frequency. TODO: Make this less hacky
    # when this is improved in dask-ms.
    frequency = da.from_array(spw.CHAN_FREQ.data[0], chunks=ms.chunks['chan'])
    phase_dir = field.PHASE_DIR.data[0][0]  # row, poly

    utime_val, utime_ind = blockwise_unique(ms.TIME.data,
                                            chunks=(ms.UTIME_CHUNKS,),
                                            return_inverse=True)
    if forward_transform == "dft":
        # Necessary to chunk the predict in frequency. TODO: Make this less hacky
        # when this is improved in dask-ms.
        frequency = da.from_array(spw.CHAN_FREQ.data[0], chunks=ms.chunks['chan'])
        lm = radec_to_lm(sources.radec, phase_dir)
        # This likely shouldn't be exposed. TODO: Disable this switch?
        uvw = -ms.UVW.data if opts.dft_predict_invert_uvw else ms.UVW.data

        # Apply spectral model to stokes parameters (source, frequency, corr).
        stokes = spectral_model(sources.stokes,
                                sources.spi,
                                sources.ref_freq,
                                frequency,
                                base=0)

        # Convery from stokes parameters to brightness matrix.
        brightness = convert(stokes, ["I", "Q", "U", "V"], corr_schema(pol))

        # Generate per-source K-Jones (source, row, frequency).
        phase = compute_phase_delay(lm, uvw, frequency)

        bl_jones_args = ["phase_delay", phase]

        # Add any visibility amplitude terms
        if source_type == "gauss":
            bl_jones_args.append("gauss_shape")
            bl_jones_args.append(gaussian_shape(uvw, frequency, sources.shape))

        bl_jones_args.extend(["brightness", brightness])

        jones = baseline_jones_multiply(corrs, *bl_jones_args)
    elif forward_transform == "fft":
        if group_name == "DIE":
            direction_phase_dir = phase_dir
        else:
            direction_phase_dir = da.mean(sources.radec, axis=0).compute() # barycentre of source cluster
        ndegridband = opts.fft_predict_no_degrid_band
        frequency = da.from_array(spw.CHAN_FREQ.data[0], chunks=ms.chunks['chan'])
        bandfreq = da.from_array(np.linspace(spw.CHAN_FREQ.data[0][0],
                                             spw.CHAN_FREQ.data[0][-1],
                                             ndegridband + 2)[1:-1], chunks=ndegridband)
        bandmap = bandmapping(frequency, bandfreq)
        logger.info("Degridding band frequencies: {}".format(
            ", ".join(map(lambda nu: "{0:.2f} MHz".format(nu), bandfreq.compute()*1e-6))))
        logger.info("Degridding band mapping: {}".format(
            ", ".join(map(str, bandmap.compute()))))
        nsrc = sources.stokes.shape[0]
        nsrc_chunks = sources.stokes.chunks[0]
        if da.max(sources.stokes[:,1:-1]).compute() > 1e-15:
            logger.warning("Model is polarized. FFT-based prediction currently does not "
                           "support polarized models. Stokes U, Q and V will be ignored!")
        stokes = spectral_model(sources.stokes,
                                sources.spi,
                                sources.ref_freq,
                                bandfreq,
                                base=0)
        cellsize = np.deg2rad(opts.fft_predict_model_resolution / 3600.0)
        sext = gaussian_extents(getattr(sources, "shape", da.zeros((nsrc, 2), 
                                                                   dtype=np.float32,
                                                                   chunks=(nsrc_chunks, 2))), 
                                cellsize)
        # work out how big the encompassing sky should be if automatic. It will be at least one facet
        # around the direction centre
        lm = radec_to_lm(sources.radec, direction_phase_dir)
        npix = min(opts.fft_predict_minimum_facet_size_px if opts.fft_predict_model_npix_max == -1 else 
                   opts.fft_predict_model_npix_max,
                   int(2 * np.ceil(da.max(np.pi*0.5*da.sqrt(da.sum(lm**2, axis=1))).compute() / cellsize)))
        npix = npix + 1 if npix % 2 == 1 else npix
        model = render_gaussians(stokes,
                                 sext,
                                 sources.shape,
                                 sources.radec,
                                 bandfreq,
                                 direction_phase_dir,
                                 cellsize,
                                 npix)
        nfacetxx = max(1, 
                       int(np.ceil(npix * np.rad2deg(cellsize) / 
                           opts.fft_predict_maximum_facet_size)))
        facetnpixpad = int(np.ceil(max(opts.fft_predict_maximum_facet_size / np.rad2deg(cellsize) * 
                                       opts.fft_predict_facet_padding_factor,
                                       opts.fft_predict_minimum_facet_size_px)))
        facetnpixpad = facetnpixpad + 1 if facetnpixpad % 2 == 1 else facetnpixpad
        logger.info("Building facet tesselation scheme for '{}' direction '{}'. "
                    "This may take some time.".format(model_name.name, group_name))
        model_bb = BoundingBoxFactory.AxisAlignedBoundingBox(BoundingBox(0, npix-1, 0, npix-1, 
                                                                         name="{}.{}".format(model_name[0], 
                                                                                             group_name)), 
                                                             square=True, enforce_odd=False)
        facet_bbs = list(map(lambda fb: BoundingBoxFactory.AxisAlignedBoundingBox(fb, 
                                                                                  square=True, 
                                                                                  enforce_odd=False),
                             map(lambda fb: BoundingBoxFactory.PadBox(fb, facetnpixpad, facetnpixpad),
                                 BoundingBoxFactory.SplitBox(model_bb, nsubboxes=nfacetxx))))
        
        # facet centres off the phase centre of the telescope:
        # facets are constructed relative to the bary centre of the direction
        # so we have to compute their offsets from the phase centre
        # for use in phase shifting visibilities to newly computed centres
        facet_centres = ((da.from_array(np.array(list(map(lambda f: f.centre, facet_bbs))), 
                                       chunks=(len(facet_bbs), 2)) -
                          da.from_array(np.array([[npix//2, npix//2]]),
                                        chunks=(1,2))
                         ) * da.from_array(np.array([[-cellsize, cellsize]]),
                                           chunks=(1,2)) +
                         (da.from_array(direction_phase_dir.reshape((1, 2)), chunks=(1,2)) - 
                          da.from_array(phase_dir.reshape((1, 2)), chunks=(1,2))) +
                         da.from_array(phase_dir.reshape((1, 2)), chunks=(1,2))
                        ) # in radians

        # facet centre lm cosines for use in E-Jones application
        # w.r.t to the pointing centre of the telescope
        # TODO: this should read from the pointing centre of the telescope
        # not the phase centre!
        lm = radec_to_lm(facet_centres, phase_dir)
        facet_models = extract_facet_models(model, facet_bbs)
        logger.info("Model '{0:s}' direction '{1:s}' rendered to {2:d} band {3:d} {4:d}x{4:d} px facetted "
                    "map at resolution of {5:0.1f}\"".format(model_name[0],
                                                             group_name,
                                                             ndegridband,
                                                             len(facet_bbs),
                                                             facetnpixpad,
                                                             np.rad2deg(cellsize)*3600.0))
        import os
        wisdom_file = os.path.join(opts.fft_predict_fftw_wisdom_cache_dir, 
                                   "{0:s}.{1:s}.wisdom".format(
            "tesselation",
            "pyfftw{}".format(pyfftw.__version__)))

        ft_models = dask_fft_cached_wisdom(opts, facet_models, wisdom_file)

        sup = opts.fft_predict_kernel_support
        OS = opts.fft_predict_kernel_oversampling
        aafilter = kernels.pack_kernel(kernels.kbsinc(sup, oversample=OS), 
                                       sup, oversample=OS)
        detaper = kernels.compute_detaper_dft_seperable(facet_bbs[0].box_npx[0], 
                                                        kernels.unpack_kernel(aafilter,
                                                                              sup,
                                                                              oversample=OS),
                                                        sup,
                                                        oversample=OS)
        aafilter = da.from_array(aafilter, chunks=(aafilter.size,))
        detaper = da.from_array(detaper, chunks=(facet_bbs[0].box_npx[0], 
                                                 facet_bbs[0].box_npx[0],))
        # detaper model stack (equal sized images)
        ft_models /= detaper[None, None, :, :]

        # generate 2x2 coherencies per row x channel for each of the facets
        # in this direction. The DDE matricies will then be computed towards
        # each of the directions
        if opts.fft_predict_degrid_rowparallel:
            from numba import config
            config.THREADING_LAYER = 'threadsafe'
        if list(np.array(corr_schema(pol)).flatten()) == [9, 10, 11, 12] or \
           list(np.array(corr_schema(pol)).flatten()) == [9, 12]:
            stokes_conversion = "XXXYYXYY_FROM_I"
        elif list(np.array(corr_schema(pol)).flatten()) == [5, 6, 7, 8] or \
             list(np.array(corr_schema(pol)).flatten()) == [5, 8]:
            stokes_conversion = "RRRLLRLL_FROM_I"
        else:
            raise RuntimeError("Degridding only supports [RR, RL, LR, LL] or [XX, XY, YX, YY] or "
                               "[RR, LL] or [XX, YY] correlation datasets at present.")
        jones = degridder(uvw=ms.UVW.data,
                          gridstack=ft_models,
                          lambdas=frequency,
                          chanmap=bandmap,
                          cell=np.rad2deg(cellsize),
                          image_centres=facet_centres,
                          phase_centre=phase_dir,
                          convolution_kernel=aafilter,
                          convolution_kernel_width=sup,
                          convolution_kernel_oversampling=OS,
                          baseline_transform_policy="rotate",
                          phase_transform_policy="phase_rotate",
                          stokes_conversion_policy=stokes_conversion,
                          convolution_policy="conv_1d_axisymmetric_packed_gather",
                          vis_dtype=ft_models.dtype,
                          rowparallel=opts.fft_predict_degrid_rowparallel).reshape(facet_centres.shape[0],
                                                                                   ms.UVW.shape[0], 
                                                                                   frequency.shape[0], 
                                                                                   2, 2)
    else:
        raise ValueError("Unknown mode {} for forward transform".format(forward_transform))
    
    # DI will include P-jones when there is no other DE term. Otherwise P must
    # be applied before other DD terms.
    die = die_factory(utime_val, frequency, ant, feed, phase_dir, opts)
    dde = dde_factory(ms, utime_val, frequency, ant, feed, field, pol, lm,
                      opts)

    dft = predict_vis(utime_ind, ms.ANTENNA1.data, ms.ANTENNA2.data,
                      dde, jones, dde, die, None, die)
    return dft


def predict(data_xds_list, opts):
    """Produces graphs describing predict operations.

    Adapted from https://github.com/ska-sa/codex-africanus.

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

        model_vis = defaultdict(list)

        # Generate visibility expressions per model, per direction for each
        # source type.
        for model_name, model_group in dask_sky_model_dict.items():
            for group_name, group_sources in model_group.items():

                # Generate visibilities per source type.
                source_vis = [vis_factory(opts, stype, group_sources,
                                          data_xds, ant_xds, field_xds,
                                          spw_xds, pol_xds, feed_xds,
                                          model_name, group_name)
                              for stype in group_sources.keys()]

                # Sum the per-source-type visibilitites together.
                vis = sum(source_vis)

                # Reshape (2, 2) correlation to shape (4,)
                if vis.ndim == 4:
                    vis = vis.reshape(vis.shape[:-2] + (4,))

                # Append group_vis to the appropriate list.
                model_vis[model_name].append(vis)

        predict_list.append(freeze_default_dict(model_vis))

    return predict_list
