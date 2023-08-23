import argparse
import math
import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from itertools import product, chain
from daskms.experimental.zarr import xds_from_zarr
from daskms.fsspec_store import DaskMSStore
from concurrent.futures import ProcessPoolExecutor, as_completed


TRANSFORMS = {
    "raw": np.array,
    "amplitude": np.abs,
    "phase": np.angle,
    "real": np.real,
    "imag": np.imag
}


class CustomFormatter(ticker.ScalarFormatter):

    def __init__(self, *args, precision=None, **kwargs):

        super().__init__(*args, *kwargs)

        self.precision = precision

    def _set_format(self):
        # set the format string to format all the ticklabels
        if len(self.locs) < 2:
            # Temporarily augment the locations with the axis end points.
            _locs = [*self.locs, *self.axis.get_view_interval()]
        else:
            _locs = self.locs
        locs = (np.asarray(_locs) - self.offset) / 10. ** self.orderOfMagnitude
        loc_range = np.ptp(locs)
        # Curvilinear coordinates can yield two identical points.
        if loc_range == 0:
            loc_range = np.max(np.abs(locs))
        # Both points might be zero.
        if loc_range == 0:
            loc_range = 1
        if len(self.locs) < 2:
            # We needed the end points only for the loc_range calculation.
            locs = locs[:-2]
        loc_range_oom = int(math.floor(math.log10(loc_range)))
        # first estimate:
        sigfigs = max(0, 3 - loc_range_oom)
        # refined estimate:
        thresh = 1e-3 * 10 ** loc_range_oom
        while sigfigs >= 0:
            if np.abs(locs - np.round(locs, decimals=sigfigs)).max() < thresh:
                sigfigs -= 1
            else:
                break
        sigfigs += 1
        sigfigs = self.precision or sigfigs
        self.format = f'%{sigfigs + 3}.{sigfigs}f'
        if self._usetex or self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def cli():

    parser = argparse.ArgumentParser(
        description="Rudimentary plotter for QuartiCal gain solutions."
    )

    parser.add_argument(
        "input_path",
        type=DaskMSStore,
        help="Path to input gains, e.g. path/to/dir/G. Accepts valid s3 urls."
    )
    parser.add_argument(
        "output_path",
        type=DaskMSStore,
        help="Path to desired output location."
    )
    parser.add_argument(
        "--plot-var",
        type=str,
        default="gains",
        help="Name of data variable to plot."
    )
    parser.add_argument(
        "--flag-var",
        type=str,
        default="gain_flags",
        help="Name of data variable to use as flags."
    )
    parser.add_argument(
        "--xaxis",
        type=str,
        default="gain_time",
        choices=("gain_time", "gain_freq", "param_time", "param_freq"),
        help="Name of coordinate to use for x-axis."
    )
    parser.add_argument(
        "--transform",
        type=str,
        default="raw",
        choices=list(TRANSFORMS.keys()),
        help="Transform to apply to data before plotting."
    )
    parser.add_argument(
        "--iter-axes",
        type=str,
        nargs="+",
        default=["antenna", "direction", "correlation"],
        help="Axes over which to iterate when generating plots."
    )
    parser.add_argument(
        "--agg-axis",
        type=str,
        default="gain_freq",
        help="Axis over which take the average when plotting."
    )
    parser.add_argument(
        "--time-range",
        type=float,
        nargs=2,
        default=[None],
        help="Time range to plot."
    )
    parser.add_argument(
        "--freq-range",
        type=float,
        nargs=2,
        default=[None],
        help="Frequency range to plot."
    )
    parser.add_argument(
        "--merge-scans",
        action="store_true",
        help="Controls whether or not scans are merged before plotting."
    )
    parser.add_argument(
        "--merge-spws",
        action="store_true",
        help="Controls whether or not scans are merged before plotting."
    )
    parser.add_argument(
        "--nworker",
        type=int,
        default=1,
        help="Number of processes to use while plotting."
    )

    return parser.parse_args()


def to_plot_dict(xdsl, merge_scans=False, merge_spws=False):

    if merge_scans and merge_spws:
        merged_xds = xarray.combine_by_coords(
            xdsl, combine_attrs="drop_conflicts"
        )
        return {("SCAN-ALL", "SPW-ALL"): merged_xds}
    elif merge_scans:
        ddids = {xds.attrs.get("DATA_DESC_ID", "ALL") for xds in xdsl}

        merge_dict = {
            ("SCAN-ALL", f"SPW-{ddid}"): [
                xds for xds in xdsl
                if xds.attrs.get("DATA_DESC_ID", "ALL") == ddid
            ]
            for ddid in ddids
        }

        merge_dict = {
            k: xarray.combine_by_coords(v, combine_attrs="drop_conflicts")
            for k, v in merge_dict.items()
        }

        return merge_dict

    elif merge_spws:
        sids = {xds.attrs.get("SCAN_NUMBER", "ALL") for xds in xdsl}

        merge_dict = {
            (f"SCAN-{sid}", "SPW-ALL"): [
                xds for xds in xdsl
                if xds.attrs.get("SCAN_NUMBER", "ALL") == sid
            ]
            for sid in sids
        }

        merge_dict = {
            k: xarray.combine_by_coords(v, combine_attrs="drop_conflicts")
            for k, v in merge_dict.items()
        }

        return merge_dict

    else:
        return {
            (f"SCAN-{xds.SCAN_NUMBER}", f"SPW-{xds.DATA_DESC_ID}"): xds
            for xds in xdsl
        }


def _plot(k, xds, args):

    xds = xds.compute(scheduler="single-threaded")

    if args.freq_range or args.time_range:
        time_ax, freq_ax = xds[args.plot_var].dims[:2]
        xds = xds.sel(
            {
                time_ax: slice(*args.time_range),
                freq_ax: slice(*args.freq_range)
            }
        )

    dims = xds[args.plot_var].dims  # Dimensions of plot quantity.
    assert all(map(lambda x: x in dims, args.iter_axes)), (
        f"Some or all of {args.iter_axes} are not present on "
        f"{args.plot_var}."
    )

    # Grab the required transform from the dict.
    transform = TRANSFORMS[args.transform]

    # NOTE: This mututates the data variables in place.
    data = xds[args.plot_var].values
    flags = xds[args.flag_var].values
    data[np.where(flags)] = np.nan  # Set flagged values to nan (not plotted).
    xds = xds.drop_vars(args.flag_var)  # No more use for flags.

    # Construct list of lists containing axes over which we iterate i.e.
    # produce a plot per combination of these values.
    iter_axes = [xds[x].values.tolist() for x in args.iter_axes]

    fig, ax = plt.subplots(figsize=(4, 3))

    for ia in product(*iter_axes):

        sel = {ax: val for ax, val in zip(args.iter_axes, ia)}

        xda = xds.sel(sel)[args.plot_var]

        ax.clear()

        for i in range(xda.sizes[args.agg_axis]):
            pxda = xda.isel({args.agg_axis: i})

            ax.plot(
                pxda[args.xaxis].values,
                transform(pxda.values),
                "k",
                linewidth=0.1
            )

        mxda = xda.mean(args.agg_axis)

        ax.plot(
            mxda[args.xaxis].values,
            transform(mxda.values),
            "r",
            label="mean"
        )

        ax.title.set_text("\n".join([f"{k}: {v}" for k, v in sel.items()]))
        ax.title.set_fontsize("medium")
        ax.set_xlabel(f"{args.xaxis}")
        ax.set_ylabel(f"{args.transform}({ia[-1]})")

        formatter = CustomFormatter(precision=2)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=5))
        ax.legend()

        fig_name = "-".join(map(str, chain.from_iterable(sel.items())))

        root_subdir = f"{xds.NAME}-{args.plot_var}-{args.transform}"
        leaf_subdir = "-".join(k)
        subdir_path = f"{root_subdir}/{leaf_subdir}"

        args.output_path.makedirs(subdir_path, exist_ok=True)

        fig.savefig(
            f"{args.output_path.full_path}/{subdir_path}/{fig_name}.png",
            bbox_inches="tight"  # SLOW, but slightly unavoidable.
        )

    plt.close()


def plot():

    args = cli()

    # Path to gain location.
    gain_path = DaskMSStore("::".join(args.input_path.url.rsplit("/", 1)))

    xdsl = xds_from_zarr(gain_path)

    # Select only the necessary fields for plotting on each dataset.
    xdsl = [xds[[args.plot_var, args.flag_var]] for xds in xdsl]

    # Partitioned dictionary of xarray.Datasets.
    xdsd = to_plot_dict(xdsl, args.merge_scans, args.merge_spws)

    with ProcessPoolExecutor(max_workers=args.nworker) as ppe:
        futures = [ppe.submit(_plot, k, xds, args) for k, xds in xdsd.items()]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Exception raised in process pool: {exc}")
