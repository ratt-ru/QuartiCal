import os
# to avoid numpy parallellism
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
import argparse
from daskms import xds_from_ms, xds_from_table
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
from pathlib import Path
import time
import dask
import dask.array as da
import numpy as np
from quartical.interpolation.interpolants import gpr_interpolate_gains
import xarray as xr
from collections import namedtuple
from quartical.data_handling import CORR_TYPES


def smoothcal():
    parser = argparse.ArgumentParser(
        description='Smooth gain solutions using Gaussian process regression.'
    )

    parser.add_argument(
        '--ms-path',
        type=Path,
        help='Path to input measurement set, e.g. path/to/dir/foo.MS.'
    )
    parser.add_argument(
        '--gain-dir',
        type=Path,
        help='Path to gain solutions produced by QuartiCal.'
    )
    parser.add_argument(
        '--gain-term',
        type=str,
        default='G',
        help='Gain term ro interpolate.'
    )
    parser.add_argument(
        '--select-corr',
        type=float,
        nargs='+',
        default=[0,-1],
        help='Target correlations to smooth. '
        'Must be consistent with gain dataset.'
    )
    parser.add_argument(
        '--time-length-scales',
        type=float,
        nargs=2,
        help='Length scales for time dimension for ampltudes '
        'and phases respectively.'
    )
    parser.add_argument(
        '--freq-length-scales',
        type=float,
        nargs=2,
        help='Length scales for frequency dimension for '
        'amplitudes and phases respectively.'
    )
    parser.add_argument(
        '--noise-inflation',
        type=float,
        help='Noise inflation factor'
    )

    opts = parser.parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # create empty output datasets corresponding to MS
    ms_path = opts.ms_path.resolve()
    ms_name = str(ms_path)
    group_cols = ("FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER")
    xds = xds_from_ms(ms_name,
                      chunks={"row":-1},
                      columns=['TIME','ANTENNA1','ANTENNA2'],
                      index_cols=("TIME",),
                      group_cols=group_cols)
    f = xds_from_table(f'{ms_name}::SPECTRAL_WINDOW')[0].CHAN_FREQ.values
    f = f.squeeze()
    nchan = f.size
    fname = xds_from_table(f'{ms_name}::FIELD')[0].NAME.values[0]
    ant_names = xds_from_table(f'{ms_name}::ANTENNA')[0].NAME.values
    corrs = xds_from_table(f'{ms_name}::POLARIZATION')[0].CORR_TYPE
    corrs = [corrs.values[0][i] for i in opts.select_corr]
    corrs = np.array([CORR_TYPES[i] for i in corrs], dtype=object)
    ncorr = corrs.size
    output_xds = []
    nant = None
    for ds in xds:
        t = np.unique(ds.TIME.values)
        ntime = t.size

        ant1 = ds.ANTENNA1.values
        ant2 = ds.ANTENNA2.values

        if nant is None:
            nant = np.maximum(ant1.max(), ant2.max()) + 1
        else:
            assert (np.maximum(ant1.max(), ant2.max()) + 1) == nant

        fid = ds.FIELD_ID
        ddid = ds.DATA_DESC_ID
        sid = ds.SCAN_NUMBER

        gain_spec_tup = namedtuple('gains_spec_tup',
                                   'tchunk fchunk achunk dchunk cchunk')
        attrs = {
            'DATA_DESC_ID': int(ddid),
            'FIELD_ID': int(fid),
            'FIELD_NAME': fname,
            'GAIN_AXES': ('gain_t', 'gain_f', 'ant', 'dir', 'corr'),
            'GAIN_SPEC': gain_spec_tup(tchunk=(int(ntime),),
                                       fchunk=(int(nchan),),
                                       achunk=(int(nant),),
                                       dchunk=(int(1),),
                                       cchunk=(int(ncorr),)),
            'NAME': opts.gain_term,
            'SCAN_NUMBER': int(sid),
            'TYPE': 'complex'
        }

        coords = {
            'gain_f': (('gain_f',), f),
            'gain_t': (('gain_t',), t),
            'ant': (('ant'), ant_names),
            'corr': (('corr'), corrs),
            'dir': (('dir'), np.array([0], dtype=np.int32)),
            'f_chunk': (('f_chunk'), np.array([0], dtype=np.int32)),
            't_chunk': (('t_chunk'), np.array([0], dtype=np.int32))
        }

        gain = da.zeros((ntime, nchan, nant, 1, ncorr), dtype=np.complex128)
        data_vars = {
        'gains':(('gain_t', 'gain_f', 'ant', 'dir', 'corr'), gain)
        }
        output_xds.append(xr.Dataset(data_vars, coords=coords, attrs=attrs))


    gain_dir = opts.gain_dir.resolve()
    gain_name = f'{str(gain_dir)}/gains.qc::{opts.gain_term}'
    input_xds = xds_from_zarr(gain_name)

    # concatenate scans
    input_xds = xr.concat(input_xds, dim='gain_t')

    gpr_params = (opts.time_length_scales,
                  opts.freq_length_scales,
                  opts.noise_inflation)

    interp_xds = gpr_interpolate_gains(input_xds, output_xds, gpr_params)

    rechunked_xds = []
    for ds in interp_xds:
        dso = ds.chunk({'gain_t': 'auto', 'gain_f': 'auto'})
        rechunked_xds.append(dso)

    writes = xds_to_zarr(rechunked_xds,
                         f'{str(gain_dir)}/smoothed.qc::{opts.gain_term}',
                         columns='ALL')
    dask.compute(writes)
