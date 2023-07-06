import argparse
import xarray
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, chain
from daskms.experimental.zarr import xds_from_zarr
from daskms.fsspec_store import DaskMSStore


TRANSFORMS = {
    "amplitude": np.abs,
    "phase": np.angle,
    "real": np.real,
    "imag": np.imag
}


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
        default=None,
        help="Name of coordinate to use for y-axis."
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

    return parser.parse_args()


def plot():

    args = cli()

    # Path to gain location.
    gain_path = "::".join(args.input_path.full_path.rsplit("/", 1))
    # Name of gain to be plotted.
    gain_name = args.input_path.full_path.rsplit("/", 1)[1]

    xdsl = xds_from_zarr(gain_path)

    # Select only the necessary fields for plotting on each dataset.
    xdsl = [xds[[args.plot_var, args.flag_var]] for xds in xdsl]

    # Combine all the datasets into a single dataset for simplicity.
    # TODO: This currently precludes plotting per scan/spw and doesn't
    # work for overlapping spws.
    xds = xarray.combine_by_coords(xdsl, combine_attrs="drop").compute()

    dims = xds[args.plot_var].dims  # Dimensions of plot quantity.
    assert all(map(lambda x: x in dims, args.iter_axes)), (
        f"Some or all of {args.iter_axes} are not present on {args.plot_var}."
    )

    # If the user has requested a transform, grab it from the dict.
    if args.transform and args.transform in TRANSFORMS:
        transform = TRANSFORMS[args.transform]
    else:
        def no_op(x):
            return x
        transform = no_op

    # NOTE: This mututates the data variables in place.
    data = xds[args.plot_var].values
    flags = xds[args.flag_var].values
    data[np.where(flags)] = np.nan  # Set flagged values to nan (not plotted).
    xds = xds.drop_vars(args.flag_var)  # No more use for flags.

    # Construct list of lists containing axes over which we iterate i.e.
    # produce a plot per combination of these values.
    iter_axes = [xds[x].values.tolist() for x in args.iter_axes]

    for ia in product(*iter_axes):

        sel = {ax: val for ax, val in zip(args.iter_axes, ia)}

        xda = xds.sel(sel)[args.plot_var]

        fig, ax = plt.subplots(figsize=(7, 10))

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

        ax.title.set_text(f"{sel}")
        ax.set_xlabel(f"{args.xaxis}")
        if args.transform:
            ax.set_ylabel(f"{args.transform}({ia[-1]})")
        else:
            ax.set_ylabel(f"{ia[-1]}")
        ax.legend()

        fig_name = "-".join(map(str, chain.from_iterable(sel.items())))

        subdir_name = f"{gain_name}-{args.transform}"
        args.output_path.makedirs(subdir_name, exist_ok=True)

        fig.savefig(
            f"{args.output_path.full_path}/{subdir_name}/{fig_name}.png",
            bbox_inches='tight'
        )
        plt.close()
