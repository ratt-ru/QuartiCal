# -*- coding: utf-8 -*-
"""Referencing in gain space, shared by the non-parameterised terms.

A per-antenna solution is determined by the data only up to a right
multiplication by a constant matrix X drawn from the stabiliser of the model:
V_pq = G_p M_pq G_q^H is unchanged by G_p -> G_p X exactly when
X M_pq X^H = M_pq for every baseline. Referencing fixes that freedom by
building X from the reference antenna, which makes the solution reproducible
between runs instead of landing wherever the iteration happened to stop.

The parameter-space counterpart is
``general/parameters.py:reference_params_factory``.
"""
import numpy as np
from numba import njit
from numba.extending import overload
from quartical.utils.numba import coerce_literal, JIT_OPTIONS
from quartical.gains.general.flagging import apply_gain_flags_to_gains
import quartical.gains.general.factories as factories


@njit(**JIT_OPTIONS)
def reference_gains(chain_inputs, meta_inputs, mode):
    return reference_gains_impl(chain_inputs, meta_inputs, mode)


def reference_gains_impl(chain_inputs, meta_inputs, mode):
    raise NotImplementedError


@overload(reference_gains_impl, jit_options=JIT_OPTIONS)
def nb_reference_gains_impl(chain_inputs, meta_inputs, mode):
    """Pin the reference antenna's diagonal phases to zero.

    X is the unit-modulus diagonal of the reference antenna's gain,
    conjugated, so the right multiplication leaves every antenna's gain moduli
    untouched and drives the reference antenna's diagonal real and positive.
    Amplitude is never gauge - the model fixes it - hence the normalisation,
    and the reference gain's off-diagonal elements are discarded because X has
    to stay unitary and diagonal. Using the reference gain's inverse instead
    would not: G^-1 = H^-1 U^H by polar decomposition, and the Hermitian
    factor is not a symmetry of any model.

    Fixing both diagonal phases forces the cross-hand phase out of the term,
    which is deliberate: that phase belongs to a dedicated crosshand_phase
    term. It is free to discard only when the model has no cross-hand
    coherency, the case for every Stokes I model. Against a model with
    cross-hand power the data constrain it, and the per-term ``referenced``
    option is what leaves it in place.
    """

    coerce_literal(nb_reference_gains_impl, ["mode"])
    v1_imul_v2 = factories.v1_imul_v2_factory(mode)

    def impl(chain_inputs, meta_inputs, mode):

        active_term = meta_inputs.active_term
        ref_ant = meta_inputs.reference_antenna

        gains = chain_inputs.gains[active_term]
        gain_flags = chain_inputs.gain_flags[active_term]

        n_ti, n_fi, n_ant, n_dir, n_corr = gains.shape

        ref_gains = gains[:, :, ref_ant: ref_ant + 1, :, :].copy()

        for t in range(n_ti):
            for f in range(n_fi):
                for d in range(n_dir):

                    if gain_flags[t, f, ref_ant, d]:  # TODO: Flagged refant?
                        continue
                    elif n_corr in (1, 2):
                        rg = ref_gains[t, f, 0, d]
                        rg[...] = rg.conjugate()/np.abs(rg)
                    else:
                        rg = ref_gains[t, f, 0, d]
                        rg[1:3] = 0
                        rg[::3] = rg[::3].conjugate()/np.abs(rg[::3])

        for t in range(n_ti):
            for f in range(n_fi):
                for a in range(n_ant):
                    for d in range(n_dir):

                        g = gains[t, f, a, d]
                        rg = ref_gains[t, f, 0, d]

                        v1_imul_v2(g, rg, g)

        apply_gain_flags_to_gains(gain_flags, gains)

    return impl
