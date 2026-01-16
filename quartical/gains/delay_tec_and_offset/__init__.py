import numpy as np
from collections import namedtuple
from quartical.utils.smoothing import gaussian_filter1d_masked
from quartical.gains.conversion import no_op, trig_to_angle
from quartical.gains.parameterized_gain import ParameterizedGain
from quartical.gains.delay_tec_and_offset.kernel import (
    delay_tec_and_offset_solver,
    delay_tec_and_offset_params_to_gains
)
from quartical.gains.general.flagging import (
    apply_gain_flags_to_gains,
    apply_param_flags_to_params
)

# Overload the default measurement set inputs to include the frequencies.
ms_inputs = namedtuple(
    'ms_inputs', ParameterizedGain.ms_inputs._fields + (
        'CHAN_FREQ',
        'MIN_FREQ',
        'MAX_FREQ'
    )
)


class DelayTecAndOffset(ParameterizedGain):

    solver = staticmethod(delay_tec_and_offset_solver)
    ms_inputs = ms_inputs

    native_to_converted = (
        (0, (np.cos,)),
        (1, (np.sin,)),
        (1, (no_op,)),
        (1, (no_op,))
    )
    converted_to_native = (
        (2, trig_to_angle),
        (1, no_op),
        (1, no_op)
    )
    converted_dtype = np.float64
    native_dtype = np.float64

    def __init__(self, term_name, term_opts):

        super().__init__(term_name, term_opts)

    @classmethod
    def _make_freq_map(cls, chan_freqs, chan_widths, freq_interval):
        # Overload gain mapping construction - we evaluate it in every channel.
        return np.arange(chan_freqs.size, dtype=np.int32)

    @classmethod
    def make_param_names(cls, correlations):

        # TODO: This is not dasky, unlike the other functions. Delayed?
        parameterisable = ["XX", "YY", "RR", "LL"]

        param_corr = [c for c in correlations if c in parameterisable]

        template = ("offset_{}", "tec_{}", "delay_{}")

        return [n.format(c) for c in param_corr for n in template]

    def init_term(self, term_spec, ref_ant, ms_kwargs, term_kwargs, meta=None):
        """Initialise the gains (and parameters)."""

        gains, gain_flags, params, param_flags = super().init_term(
            term_spec, ref_ant, ms_kwargs, term_kwargs
        )

        # Convert the parameters into gains.
        delay_tec_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            ms_kwargs["MIN_FREQ"],
            ms_kwargs["MAX_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        if self.load_from or not self.initial_estimate:

            apply_param_flags_to_params(param_flags, params, 0)
            apply_gain_flags_to_gains(gain_flags, gains)

            return gains, gain_flags, params, param_flags

        data = ms_kwargs["DATA"]  # (row, chan, corr)
        flags = ms_kwargs["FLAG"]  # (row, chan)
        a1 = ms_kwargs["ANTENNA1"]
        a2 = ms_kwargs["ANTENNA2"]
        chan_freq = ms_kwargs["CHAN_FREQ"]
        t_map = term_kwargs[f"{term_spec.name}_time_map"]
        f_map = term_kwargs[f"{term_spec.name}_param_freq_map"]
        _, n_chan, n_ant, n_dir, n_corr = gains.shape

        # Rescale the channel frequencies.
        scale_factor = (chan_freq.min() + chan_freq.max()) / 2
        scaled_chan_freq = chan_freq / scale_factor

        est_resolution = 0.01  # NOTE: Make this lower.

        # We only need the baselines which include the ref_ant.
        sel = np.where((a1 == ref_ant) | (a2 == ref_ant))
        a1 = a1[sel]
        a2 = a2[sel]
        t_map = t_map[sel]
        data = data[sel]
        flags = flags[sel]

        data[flags == 1] = 0  # Ignore UV-cut, otherwise there may be no est.

        utint = np.unique(t_map)
        ufint = np.unique(f_map)
        n_tint = utint.size
        n_fint = ufint.size
        # NOTE: This determines the number of subintervals which are used to
        # estimate the delay and tec values. More subintervals will typically
        # yield better estimates at the cost of SNR. TODO: Thus doesn't factor
        # in the flagging behaviour which may make some estimates worse than
        # others. Should we instead only consider unflagged regions or weight
        # the mean calaculation?
        n_subint = max(int(np.ceil(n_chan / 1024)), 2)

        if n_corr == 1:
            n_paramt = 1 #number of parameters in TEC
            n_paramk = 1 #number of parameters in delay
            corr_slice = slice(None)
        elif n_corr == 2:
            n_paramt = 2
            n_paramk = 2
            corr_slice = slice(None)
        elif n_corr == 4:
            n_paramt = 2
            n_paramk = 2
            corr_slice = slice(0, 4, 3)
        else:
            raise ValueError("Unsupported number of correlations.")

        # Loop over all antennas except the reference antenna.
        loop_ants = list(range(n_ant))
        loop_ants.pop(ref_ant)

        subint_delays = np.zeros((n_tint, n_fint, n_ant, n_paramk, n_subint))
        gradients = np.zeros((n_tint, n_fint, n_ant, n_subint))

        # TODO: We should make baseline the outermost axis for the initial
        # estimates. This allows us to do things like smooth the data on
        # the baseline prior to performing the estimates.

        for a in loop_ants:

            bl_sel = np.where(
                (a1 != a2) & ((a1 == ref_ant) & (a2 == a)) | ((a2 == ref_ant) & (a1 == a))
            )

            baseline_data = data[bl_sel]
            baseline_flags = flags[bl_sel]
            baseline_t_map = t_map[bl_sel]

            # NOTE: Collapse correlation axis when term is scalar.
            if self.scalar:
                baseline_data[..., :] = baseline_data.sum(axis=-1, keepdims=True)

            # Apply simple 1D smoothing in time to the input data. This helps
            # improve the SNR on particularly bad baselines. Currently, this
            # cannot be configured from the CLI.
            baseline_data = gaussian_filter1d_masked(
                baseline_data, 1, mask=(baseline_flags==0)[...,None], axis=0
            )

            for ut in utint:
                t_sel = np.where(baseline_t_map == ut)

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

                    f_sel = np.where(f_map == uf)[0]
                    f_sel_chan = scaled_chan_freq[f_sel]

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
                    nbins = int((2 * max_n_wrap) / est_resolution)

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

        params[:, :, :, 0, 1::3] = x[..., 1] * scale_factor
        params[:, :, :, 0, 2::3] = x[..., 0] / scale_factor

        # Flip the sign on antennas > reference as they correspond to G^H.
        params[:, :, ref_ant:] = -params[:, :, ref_ant:]

        delay_tec_and_offset_params_to_gains(
            params,
            gains,
            ms_kwargs["CHAN_FREQ"],
            ms_kwargs["MIN_FREQ"],
            ms_kwargs["MAX_FREQ"],
            term_kwargs[f"{self.name}_param_freq_map"],
        )

        apply_param_flags_to_params(param_flags, params, 0)
        apply_gain_flags_to_gains(gain_flags, gains)

        return gains, gain_flags, params, param_flags

