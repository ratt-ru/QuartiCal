import numpy as np
from numba import generated_jit
import quartical.gains.general.factories as factories
from quartical.utils.numba import coerce_literal


@generated_jit(nopython=True, nogil=True, cache=True)
def apply_gains(model, gains, ant1, ant2, row_ind, mode):

    coerce_literal(apply_gains, ["mode"])

    v1_imul_v2 = factories.v1_imul_v2_factory(mode)
    v1_imul_v2ct = factories.v1_imul_v2ct_factory(mode)
    iunpack = factories.iunpack_factory(mode)

    def impl(model, gains, ant1, ant2, row_ind, mode):

        n_row, n_chan, n_dir, n_corr = model.shape

        data = np.zeros((n_row, n_chan, n_corr), dtype=np.complex64)

        for row in range(n_row):
            r = row_ind[row]
            a1 = ant1[row]
            a2 = ant2[row]

            tmp = np.zeros(n_corr, dtype=model.dtype)

            for f in range(n_chan):

                for d in range(n_dir):

                    gp = gains[r, f, a1, d]
                    gq = gains[r, f, a2, d]
                    iunpack(tmp, model[row, f, d])

                    v1_imul_v2(gp, tmp, tmp)
                    v1_imul_v2ct(tmp, gq, tmp)

                    data[row, f] += tmp

        return data
    return impl
