import os
# to avoid numpy parallellism
os.environ["OMP_NUM_THREADS"] = '1'
os.environ["OPENBLAS_NUM_THREADS"] = '1'
os.environ["MKL_NUM_THREADS"] = '1'
os.environ["VECLIB_MAXIMUM_THREADS"] = '1'
import sys
import argparse
from daskms import xds_from_ms, xds_from_table
from daskms.experimental.zarr import xds_to_zarr, xds_from_zarr
from pathlib import Path
import shutil
import time
import dask
import dask.array as da
import numpy as np
from quartical.interpolation.interpolants import gpr_interpolate_gains
import xarray as xr
from collections import namedtuple
from quartical.data_handling import CORR_TYPES
from quartical.logging import InterceptHandler
import logging
from loguru import logger
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.interpolate import bisplrep, bisplev


def rspline_interpolate_gains(input_xds, output_xds, s, k, mode):
    """
    Interpolatres from input_xds to output_xds using a RectBivariateSpline.
    Sets flaged data to one

    """
    t = input_xds.gain_t.values
    f = input_xds.gain_f.values
    jhj = input_xds.jhj.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1, 4:-1})
    gain = input_xds.gains.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1, 4:-1})
    flag = input_xds.gain_flags.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1})
    gref = gain[:,:,-1]
    _, _, nant, _, _ = gain.shape
    p = da.arange(nant, chunks=1)

    interpo = da.blockwise(rspline_solve, 'adcx',
                           gain, 'tfadc',
                           jhj, 'tfadc',
                           flag, 'tfad',
                           p, 'a',
                           t, None,
                           f, None,
                           gref, None,
                           s, None,
                           k, None,
                           mode, None,
                           new_axes={'x':2},
                           meta=np.empty((1,1,1,1), dtype=object))

    out_ds = []
    for ds in output_xds:
        tp = ds.gain_t.values
        fp = ds.gain_f.values
        gain = da.blockwise(rspline_interp, 'tfadc',
                            interpo, 'adcx',
                            tp, None,
                            fp, None,
                            mode, None,
                            new_axes={'t': tp.size, 'f': fp.size},
                            dtype=np.complex128)

        dso = ds.assign(**{'gains': (ds.GAIN_AXES, gain.rechunk({2:-1}))})
        out_ds.append(dso)
    return out_ds


def rspline_solve(gain, jhj, flag, p, t, f, gref, s, k, mode):
    return _rspline_solve(gain[0][0],
                         jhj[0][0],
                         flag[0][0],
                         p,
                         t, f, gref, s, k, mode)

def _rspline_solve(gain, jhj, flag, p, t, f, gref, s, k, mode):
    ntime, nchan, nant, ndir, ncorr = gain.shape
    sol = np.zeros((nant, ndir, ncorr, 2), dtype=object)
    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                # mask where flagged or jhj is zero
                inval = np.logical_or(flag[:, :, p, d],
                                      jhj[:, :, p, d, c]==0)
                It, If = np.where(inval)
                # replace flagged data with ones
                g = gain[:, :, p, d, c]
                g[It, If] = 1.0 + 0j
                gr = gref[:, :, d, c]
                gr[It, If] = 1.0 + 0j

                if mode == 'reim':
                    gamp = np.real(g)
                    gphase = np.imag(g)
                elif mode == 'ampphase':
                    gamp = np.log(np.abs(g))
                    gphase = np.angle(g) # * np.conj(gr))
                    # gphase = np.unwrap(np.unwrap(gphase, axis=0), axis=1)

                ampo = rbs(t, f, gamp, kx=k, ky=k, s=s)
                sol[p, d, c, 0] = ampo
                phaseo = rbs(t, f, gphase, kx=k, ky=k, s=s)
                sol[p, d, c, 1] = phaseo

    return sol


def rspline_interp(interpo, tp, fp, mode):
    return _rspline_interp(interpo[0], tp, fp, mode)


def _rspline_interp(interpo, tp, fp, mode):
    nant, ndir, ncorr, _ = interpo.shape
    ntime = tp.size
    nchan = fp.size

    sol = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.complex128)

    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                logampo = interpo[p, d, c, 0]
                logamp = logampo(tp, fp)

                phaseo = interpo[p, d, c, 1]
                phase = phaseo(tp, fp)

                if mode=="reim":
                    sol[:, :, p, d, c] = logamp + 1.0j*phase
                elif mode=="ampphase":
                    sol[:, :, p, d, c] = np.exp(logamp + 1.0j*phase)

    return sol


def spline_interpolate_gains(input_xds, output_xds, s, k, mode):
    '''
    Interpolates from input_xds to output_xds using interp2d.
    Slow but
    '''
    t = input_xds.gain_t.values
    f = input_xds.gain_f.values
    jhj = input_xds.jhj.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1, 4:-1})
    gain = input_xds.gains.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1, 4:-1})
    flag = input_xds.gain_flags.data.rechunk({0:-1, 1:-1, 2: 1, 3:-1})
    gref = gain[:,:,-1]
    _, _, nant, _, _ = gain.shape
    p = da.arange(nant, chunks=1)

    interpo = da.blockwise(spline_solve, 'adcx',
                           gain, 'tfadc',
                           jhj, 'tfadc',
                           flag, 'tfad',
                           p, 'a',
                           t, None,
                           f, None,
                           gref, None,
                           s, None,
                           k, None,
                           mode, None,
                           new_axes={'x':2},
                           meta=np.empty((1,1,1,1), dtype=object))

    out_ds = []
    for ds in output_xds:
        tp = ds.gain_t.values
        fp = ds.gain_f.values
        gain = da.blockwise(spline_interp, 'tfadc',
                            interpo, 'adcx',
                            tp, None,
                            fp, None,
                            mode, None,
                            new_axes={'t': tp.size, 'f': fp.size},
                            dtype=np.complex128)

        dso = ds.assign(**{'gains': (ds.GAIN_AXES, gain.rechunk({2:-1}))})
        out_ds.append(dso)
    return out_ds


def spline_solve(gain, jhj, flag, p, t, f, gref, s, k, mode):
    return _spline_solve(gain[0][0],
                         jhj[0][0],
                         flag[0][0],
                         p,
                         t, f, gref, s, k, mode)

def _spline_solve(gain, jhj, flag, p, t, f, gref, s, k, mode):
    tt, ff = np.meshgrid(t, f, indexing='ij')
    ntime, nchan, nant, ndir, ncorr = gain.shape
    sol = np.zeros((nant, ndir, ncorr, 2), dtype=object)
    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                # select valid data
                ival = ~np.logical_or(flag[:, :, p, d],
                                      jhj[:, :, p, d, c]==0)
                It, If = np.where(ival)
                g = gain[:, :, p, d, c]
                tp = tt[It, If]
                fp = ff[It, If]
                gp = g[It, If]

                # first logamp
                gamp = np.log(np.abs(gp))
                ampo = bisplrep(tp, fp, gamp, kx=k, ky=k, s=s)
                # ampo = res[0]  #, chisq, ier, msg

                sol[p, d, c, 0] = ampo

                # unwrapped phase
                gphase = np.angle(gp) # * np.conj(gr))
                # gphase = np.unwrap(np.unwrap(gphase, axis=0), axis=1)
                phaseo = bisplrep(tp, fp, gamp, kx=k, ky=k, s=s)
                # phaseo = res[0]  #, chisq, ier, msg

                sol[p, d, c, 1] = phaseo

    return sol


def spline_interp(interpo, tp, fp, mode):
    return _spline_interp(interpo[0], tp, fp, mode)


def _spline_interp(interpo, tp, fp, mode):
    nant, ndir, ncorr, _ = interpo.shape
    ntime = tp.size
    nchan = fp.size

    tt, ff = np.meshgrid(tp, fp, indexing='ij')
    t = tt.flatten()
    f = ff.flatten()

    sol = np.zeros((ntime, nchan, nant, ndir, ncorr), dtype=np.complex128)

    for p in range(nant):
        for d in range(ndir):
            for c in range(ncorr):
                logampo = interpo[p, d, c, 0]
                logamp = bisplev(tp, fp, logampo)

                phaseo = interpo[p, d, c, 1]
                phase = bisplev(tp, fp, phaseo)

                logamp = logamp.reshape(ntime, nchan)
                phase = phase.reshape(ntime, nchan)

                sol[:, :, p, d, c] = np.exp(logamp + 1.0j*phase)

    return sol


def configure_loguru(output_dir):
    logging.basicConfig(handlers=[InterceptHandler()], level="WARNING")

    # Put together a formatting string for the logger. Split into pieces in
    # order to improve legibility.

    tim_fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green>"
    lvl_fmt = "<level>{level}</level>"
    src_fmt = "<cyan>{module}</cyan>:<cyan>{function}</cyan>"
    msg_fmt = "<level>\n{message}</level>"

    fmt = " | ".join([tim_fmt, lvl_fmt, src_fmt, msg_fmt])

    output_path = Path(output_dir)
    output_name = Path("{time:YYYYMMDD_HHmmss}.smoothcal.qc")

    config = {
        "handlers": [
            {"sink": sys.stderr,
             "level": "INFO",
             "format": fmt},
            {"sink": str(output_path / output_name),
             "level": "INFO",
             "format": fmt}
        ],
    }

    logger.configure(**config)


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
        '--output-dir',
        type=Path,
        help='Directory to write smoothed gain solutions to. '
        'Solutions will be stored as output_dir/smoothed.qc/gain_term. '
        'If None solutions are stored in parent of gain-dir. '
    )
    parser.add_argument(
        '--overwrite',
        action='store_true'
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
    parser.add_argument(
        '--method',
        type=str,
        default='spline',
        help='Type of smothig to do. Options are "gpr" or "spline"'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='spline',
        help='reim or ampphase'
    )
    parser.add_argument(
        '--s',
        type=float,
        default=0.0,
        help='Smoothing factor for spline. '
        'Default is zero which is interpolation. '
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='Spline order. Default of one means linear. '
    )
    opts = parser.parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    gain_dir = opts.gain_dir.resolve()

    configure_loguru(gain_dir.parent)

    if opts.output_dir is not None:
        output_dir = (opts.output_dir / 'smoothed.qc') / opts.gain_term
    else:
        output_dir = (opts.gain_dir.parent / 'smoothed.qc') / opts.gain_term
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if opts.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"output-dir {str(output_dir)} exists. "
                             "Use --overwrite flag if contents should be "
                             "ovrwritten. ")
    output_dir.mkdir(parents=True, exist_ok=True)
    opts.output_dir = str(output_dir)

    # create empty output datasets corresponding to MS
    ms_path = opts.ms_path.resolve()
    ms_name = str(ms_path)

    GD = vars(opts)
    msg = ''
    for key in GD.keys():
        msg += '     %25s = %s \n' % (key, GD[key])

    logger.info(msg)

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


    gain_name = f'{str(gain_dir)}::{opts.gain_term}'
    input_xds = xds_from_zarr(gain_name)

    # concatenate scans
    input_xds = xr.concat(input_xds, dim='gain_t')

    if opts.method.lower() == 'gpr':
        gpr_params = (opts.time_length_scales,
                      opts.freq_length_scales,
                      opts.noise_inflation)
        interp_xds = gpr_interpolate_gains(input_xds, output_xds,
                                           gpr_params)
    elif opts.method.lower() == 'spline':
        interp_xds = spline_interpolate_gains(input_xds, output_xds,
                                              opts.s, opts.k, opts.mode)
    elif opts.method.lower() == 'rspline':
        interp_xds = rspline_interpolate_gains(input_xds, output_xds,
                                               opts.s, opts.k, opts.mode)

    rechunked_xds = []
    for ds in interp_xds:
        dso = ds.chunk({'gain_t': 'auto', 'gain_f': 'auto'})
        rechunked_xds.append(dso)

    writes = xds_to_zarr(rechunked_xds,
                         str(output_dir),
                         columns='ALL')
    dask.compute(writes)

    logger.info('Success!')
