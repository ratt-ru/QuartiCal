import argparse
import math
import xarray
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm
from itertools import product, chain
from daskms.experimental.zarr import xds_from_zarr
from daskms.fsspec_store import DaskMSStore
from concurrent.futures import ProcessPoolExecutor, as_completed
from quartical.utils.collections import flatten
from quartical.utils.datasets import recursive_group_by_attr


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
        "--iter-attrs",
        type=str,
        nargs="+",
        default=["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"],
        help=(
            "Attributes (datasets) over which to iterate. Omission will "
            "result in concatenation in the omitted axis i.e. omit "
            "SCAN_NUMBER to to include all scans in a single plot."
        )
    )
    parser.add_argument(
        "--iter-axes",
        type=str,
        nargs="+",
        default=["antenna", "direction", "correlation"],
        help=(
            "Axes over which to iterate when generating plots i.e. produce a "
            "plot per unique combination of the specified axes."
        )
    )
    parser.add_argument(
        "--mean-axis",
        type=str,
        default=None,
        help=(
            "If set, will plot a heavier line to indicate the mean of the "
            "plotted quantity along this axis."
        )
    )
    parser.add_argument(
        "--colourize-axis",
        type=str,
        default=None,
        help="Axis to colour by."
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
        "--nworker",
        type=int,
        default=1,
        help="Number of processes to use while plotting."
    )
    parser.add_argument(
        "--colourmap",
        type=str,
        default="plasma",
        help=(
            "Colourmap to use with --colourize-axis. Supports all matplotlib "
            "colourmaps."
        )
    )
    parser.add_argument(
        "--fig-size",
        type=float,
        nargs=2,
        default=[5, 5],
        help="Figure size in inches. Expects two values, width and height."
    )
    return parser.parse_args()


def to_plot_dict(xdsl, iter_attrs):

    grouped = recursive_group_by_attr(xdsl, iter_attrs)

    return {
        k: xarray.combine_by_coords(v, combine_attrs="drop_conflicts")
        for k, v in flatten(grouped).items()
    }


def _plot(group, xds, args):
    # get rid of question marks
    qstrip = lambda x: x.replace('?', 'N/A')
    group = tuple(map(qstrip, group))

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
    iter_axes_itr = [xds[x].values.tolist() for x in args.iter_axes]

    # Figure out axes included in a single plot.
    excluded_dims = {*args.iter_axes, args.xaxis}
    agg_axes = [d for d in dims if d not in excluded_dims]
    agg_axes_itr = [range(xds.sizes[x]) for x in agg_axes]

    # Figure out axes included in a single plot after taking the mean.
    excluded_dims = {*args.iter_axes, args.xaxis, args.mean_axis}
    mean_agg_axes = [d for d in dims if d not in excluded_dims]
    mean_agg_axes_itr = [range(xds.sizes[x]) for x in mean_agg_axes]

    if args.colourize_axis:
        n_colour = xds.sizes[args.colourize_axis]
        colourmap = cm.get_cmap(args.colourmap)
        colours = [colourmap(i / n_colour) for i in range(n_colour)]
    else:
        n_colour = 2
        colours = ["k", "r"]

    fig, ax = plt.subplots(figsize=args.fig_size)

    for ia in product(*iter_axes_itr):

        sel = {ax: val for ax, val in zip(args.iter_axes, ia)}

        xda = xds.sel(sel)[args.plot_var]

        ax.clear()

        for aa in product(*agg_axes_itr):

            subsel = {ax: val for ax, val in zip(agg_axes, aa)}
            pxda = xda.isel(subsel)

            ax.plot(
                pxda[args.xaxis].values,
                transform(pxda.values),
                color=colours[subsel.get(args.colourize_axis, 0)],
                linewidth=0.1
            )

        if args.mean_axis:

            mxda = xda.mean(args.mean_axis)

            for ma in product(*mean_agg_axes_itr):

                subsel = {ax: val for ax, val in zip(mean_agg_axes, ma)}
                pxda = mxda.isel(subsel)

                ax.plot(
                    pxda[args.xaxis].values,
                    transform(pxda.values),
                    color=colours[subsel.get(args.colourize_axis, 1)]
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

        fig_name = "-".join(map(str, chain.from_iterable(sel.items())))

        root_subdir = f"{xds.NAME}-{args.plot_var}-{args.transform}"
        leaf_subdir = "-".join(group)
        subdir_path = f"{root_subdir}/{leaf_subdir}"

        args.output_path.makedirs(subdir_path, exist_ok=True)

        fig.savefig(
            f"{args.output_path.full_path}/{subdir_path}/{fig_name}.png",
            bbox_inches="tight"  # SLOW, but slightly unavoidable.
        )

    plt.close()


def plot():

    args = cli()

    non_colourizable_axes = {*args.iter_axes, args.mean_axis, args.xaxis}
    if args.colourize_axis and args.colourize_axis in non_colourizable_axes:
        raise ValueError(f"Cannot colourize using axis {args.colourize_axis}.")

    # Path to gain location.
    gain_path = DaskMSStore("::".join(args.input_path.url.rsplit("/", 1)))

    xdsl = xds_from_zarr(gain_path)

    # Select only the necessary fields for plotting on each dataset.
    xdsl = [xds[[args.plot_var, args.flag_var]] for xds in xdsl]

    # Partitioned dictionary of xarray.Datasets.
    xdsd = to_plot_dict(xdsl, args.iter_attrs)

    with ProcessPoolExecutor(max_workers=args.nworker) as ppe:
        futures = [ppe.submit(_plot, k, xds, args) for k, xds in xdsd.items()]

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"Exception raised in process pool: {exc}")
