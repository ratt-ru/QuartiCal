import numpy as np
from numpy.typing import NDArray
from quartical.utils.smoothing import gaussian_filter1d_masked


def estimate_delay_and_tec(
    data: NDArray[np.complex128],
    flags: NDArray[np.int8],
    antenna1: NDArray[np.int32],
    antenna2: NDArray[np.int32],
    time_map: NDArray[np.int32],
    freq_map: NDArray[np.int32],
    chan_freq: NDArray[np.float64],
    gain_shape: tuple[int, int, int, int, int],
    ref_ant: int,
    estimate_resolution: float = 0.01,
    scalar: bool = False
) -> NDArray[np.float64]:
    """Estimate delay and dTEC from visibility data.

    This function estimates delay and dTEC parameters from visibility data by
    performing windowed FFTs along the frequency axis. The locations of the
    peaks in the resulting power spectra are used as the input to a
    least-squares fit to determine the delay and TEC values for each antenna.

    Args:
        data: Complex visibility data array.
        flags: Flag array indicating invalid measurements.
        antenna1: First antenna indices for each baseline.
        antenna2: Second antenna indices for each baseline.
        time_map: Time interval indices for each visibility.
        freq_map: Frequency interval indices for each visibility.
        chan_freq: Channel frequencies (possibly scaled).
        gain_shape: Shape of the gain solutions array
            (n_tint, n_f_int, n_ant, n_dir, n_corr).
        ref_ant: Reference antenna index.
        estimate_resolution: Used to determine the padding required in the
            FFT to ensure that peaks can be resolved to this level.
        scalar: If True, collapse correlation axis for scalar solutions.

    Returns:
        Estimated delay and dTEC values.

    Raises:
        ValueError: If the number of correlations is not supported.
    """

    a1, a2 = antenna1, antenna2  # Alias for brevity.
    _, n_chan, n_ant, _, n_corr = gain_shape

    # We only need the baselines which include the ref_ant.
    sel = np.where((a1 == ref_ant) | (a2 == ref_ant))
    a1 = a1[sel]
    a2 = a2[sel]
    time_map = time_map[sel]
    data = data[sel]
    flags = flags[sel]

    data[flags == 1] = 0  # Ignore UV-cut, otherwise there may be no est.

    utint = np.unique(time_map)
    ufint = np.unique(freq_map)
    n_tint = utint.size
    n_fint = ufint.size

    # NOTE: This determines the number of subintervals which are used to
    # estimate the delay and tec values. More subintervals will typically
    # yield better estimates at the cost of SNR.
    n_subint = max(int(np.ceil(n_chan / 1024)), 2)

    if n_corr == 1:
        n_paramk = 1 # Total number of delay parameters.
        corr_slice = slice(None)
    elif n_corr == 2:
        n_paramk = 2
        corr_slice = slice(None)
    elif n_corr == 4:
        n_paramk = 2
        corr_slice = slice(0, 4, 3)
    else:
        raise ValueError("Unsupported number of correlations.")

    # Loop over all antennas except the reference antenna.
    loop_ants = list(range(n_ant))
    loop_ants.pop(ref_ant)

    subint_delays = np.zeros((n_tint, n_fint, n_ant, n_paramk, n_subint))
    gradients = np.zeros((n_tint, n_fint, n_ant, n_subint))

    # We loop over baseline (to the reference antenna) at the outermost level.
    # This allows us to do things like smooth the data on the baseline prior
    # to performing the estimates.

    for a in loop_ants:

        bl_sel = np.where(
            (a1 != a2) & ((a1 == ref_ant) & (a2 == a)) | ((a2 == ref_ant) & (a1 == a))
        )

        baseline_data = data[bl_sel]
        baseline_flags = flags[bl_sel]
        baseline_time_map = time_map[bl_sel]

        # NOTE: Collapse correlation axis when term is scalar.
        if scalar:
            baseline_data[..., :] = baseline_data.sum(axis=-1, keepdims=True)

        # Apply simple 1D smoothing in time to the input data. This helps
        # improve the SNR on particularly bad baselines. Currently, this
        # cannot be configured from the CLI.
        baseline_data = gaussian_filter1d_masked(
            baseline_data, 1, mask=(baseline_flags==0)[...,None], axis=0
        )

        for ut in utint:
            t_sel = np.where(baseline_time_map == ut)

            t_sel_data = baseline_data[t_sel].sum(axis=0)
            t_sel_counts = (baseline_flags[t_sel] == 0).sum(axis=0)
            np.divide(
                t_sel_data,
                t_sel_counts[..., None],
                where=t_sel_counts[..., None] != 0,
                out=t_sel_data
            )
            t_sel_mask = t_sel_counts.astype(bool)

            for uf in ufint:

                f_sel = np.where(freq_map == uf)[0]
                f_sel_chan = chan_freq[f_sel]

                f_sel_data = t_sel_data[f_sel]
                f_sel_mask = t_sel_mask[f_sel]

                if not f_sel_mask.any():  # No valid data in the interval.
                    continue

                # NOTE: Previous attempts to set the resolution per
                # subinterval and antenna caused problems. Consequently, we
                # set the resolution per solution interval. This has not
                # been tested comprehensively in the case where we have
                # multiple solution intervals in frequency.
                dfreq = np.abs(f_sel_chan[-1] - f_sel_chan[-2])
                max_n_wrap = (f_sel_chan[-1] - f_sel_chan[0]) / (2 * dfreq)
                nbins = int((2 * max_n_wrap) / estimate_resolution)

                mask_indices = np.flatnonzero(f_sel_mask)
                n_valid = mask_indices.size
                subint_stride = int(np.ceil(n_valid / n_subint))

                for i, si in enumerate(range(0, n_valid, subint_stride)):

                    subint_indices = mask_indices[si: si + subint_stride]
                    subint_start = subint_indices[0]
                    subint_end = subint_indices[-1]

                    subint_data = f_sel_data[subint_start:subint_end]

                    # Fit straight line to 1/nu, where nu are the unflagged
                    # channel frequencies in this solution interval.
                    subint_freq = f_sel_chan[subint_indices]
                    subint_ifreq = 1/subint_freq
                    gradients[ut, uf, a, i], _ = np.polyfit(
                        subint_freq, subint_ifreq, deg=1
                    )

                    # Perform the FFT on the subinterval data and find
                    # the location of the resulting peak in the power
                    # spectrum.
                    fft = np.fft.fft(
                        subint_data[:, corr_slice].copy(),
                        axis=0,
                        n=nbins
                    )
                    fft_freq = np.fft.fftfreq(nbins, dfreq)

                    subint_delays[ut, uf, a, :, i] = \
                        fft_freq[np.argmax(np.abs(fft), axis=0)]

    # We can combine the estimates of the gradeint of a straight line fit
    # to 1/nu in each interval and the location of the peak in the
    # associated power spectrum to estimate the clock and dTEC using the
    # normal equations.
    A = np.ones((n_tint, n_fint, n_ant, n_subint, 2), dtype=np.float64)

    A[..., 1] = gradients

    ATA = A.transpose(0,1,2,4,3) @ A

    ATA00 = ATA[..., 0, 0]
    ATA01 = ATA[..., 0, 1]
    ATA10 = ATA[..., 1, 0]
    ATA11 = ATA[..., 1, 1]

    # Determine the determinant for each 2x2 element of ATA. Re-add
    # axes to ensure compatibility with ATAinv array.
    ATA_det  = (ATA00 * ATA11 - ATA01 * ATA10)[..., None, None]

    ATAinv = np.zeros_like(ATA)

    ATAinv[..., 0, 0] = ATA11
    ATAinv[..., 0, 1] = -ATA01
    ATAinv[..., 1, 0] = -ATA10
    ATAinv[..., 1, 1] = ATA00

    np.divide(ATAinv, ATA_det, where=ATA_det!=0, out=ATAinv)
    ATAinvAT = ATAinv @ A.transpose(0,1,2,4,3)

    b = subint_delays
    # NOTE: Matmul doesn't work in the particular case so we explicitly
    # einsum instead. Trailing dim (the matrix column) gets trimmed.
    x = np.einsum("abcij,abcdjk->abcdik", ATAinvAT, b[..., None])[..., 0]

    # Flip the sign on antennas > reference as they correspond to G^H.
    x[:, :, ref_ant:] = -x[:, :, ref_ant:]

    return x
