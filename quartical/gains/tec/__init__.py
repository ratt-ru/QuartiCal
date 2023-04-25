from quartical.gains.gain import ParameterizedGain
from quartical.gains.tec.kernel import tec_solver, tec_args
import numpy as np


class TEC(ParameterizedGain):

    solver = staticmethod(tec_solver)
    term_args = tec_args

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

        template = ("phase_offset_{}", "TEC_{}")

        return [n.format(c) for c in param_corr for n in template]
