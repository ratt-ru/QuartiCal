# Domain Primer

> **Purpose:** The radio-interferometry and optimisation background QuartiCal's code assumes —
> RIME, Jones chains, gain solving as complex NLLS.
> **Last verified:** 2b75ebd, 2026-07-07

This page states the physics and optimisation background as standard, textbook-level material and
maps it onto QuartiCal's symbols, config, and files. Physics/maths statements are kept
conservative; the authoritative reference for the calibration *formulation* used here is the
QuartiCal paper, Kenyon et al. 2025 (Astronomy and Computing 52, 100962; arXiv:2412.10072), with
Smirnov & Tasse 2015 (complex/Wirtinger calibration) and Kenyon et al. 2018 (CubiCal) as the
lineage it refines. For the
implementation of everything sketched here, see [solver-architecture.md](solver-architecture.md);
this page is the "why", that page is the "how".

## The RIME in one page

Radio interferometry measures **visibilities**: for a pair of antennas (a baseline) `p`–`q`, the
correlator produces the correlations of the two antennas' feeds. With dual-polarisation feeds the
per-baseline measurement is a 2×2 **coherency matrix** `V_pq` (four correlations, e.g. `XX XY YX
YY` for linear feeds or `RR RL LR LL` for circular feeds).

The Radio Interferometer Measurement Equation (RIME) expresses the observed coherency as the true
sky coherency corrupted by per-antenna, 2×2 **Jones matrices** that describe the signal path
(receiver gain, bandpass, ionosphere, feed rotation, leakage, ...):

```
V_pq = J_p B_pq J_q^H
```

Here `B_pq` is the model (true) coherency for baseline `p`–`q`, `J_p` is antenna `p`'s Jones
matrix, and `^H` is the conjugate transpose (Hermitian adjoint). The `J_q^H` on the right (rather
than `J_q`) is what makes this a coherency/correlation relation rather than a plain linear map.
The physical derivation and the general (direction-dependent, integral-over-sky) form are in the
references above; QuartiCal works with this per-baseline, per-direction discrete form.

**What QuartiCal takes as given vs solves for.** The model coherencies `B_pq` are an *input* —
supplied as an MS column and/or predicted from a sky model (`data_handling/model_handler.py`;
carried into the kernels as `ms_inputs.MODEL_DATA`, and the measured `V_pq` as `ms_inputs.DATA`).
QuartiCal *solves for* the Jones terms `J_p`. In the code the model can carry a direction axis
(`model[row, f, d]` in the kernels), so `B_pq` is really a sum over model directions `d`; each
gain term is either shared across directions or solved per direction (see below).

## Jones chains

A single physical `J_p` is usually a product of several distinct effects. QuartiCal represents
this explicitly: the user names an ordered chain via `solver.terms=['G', 'B', ...]`, and the
per-antenna Jones matrix is the matrix product of the terms:

```
J_p = G_p B_p ...
```

so the measurement model becomes `V_pq = G_p B_p ... B_pq ... B_q^H G_q^H`.

**Order matters because 2×2 matrices do not commute.** The chain is an *ordered* product, and each
term sits at a definite position relative to the model. In the residual kernel
(`quartical/gains/general/generics.py`, `compute_residual_impl`) the model is built by looping the
gain tuple from the last index inward — `for g in range(n_gains-1, -1, -1): v = gain_p @ v @
gain_q^H` — so the **first** term in `solver.terms` (index 0) is the **outermost / leftmost**
factor and the last term is closest to the model. The order you request is therefore physically
meaningful, not cosmetic.

**Sky-side vs antenna-side placement.** Terms nearer the model in the product act "closer to the
sky" (e.g. direction-dependent effects, leakage, parallactic-angle/feed rotation), while terms at
the outer end act "closer to the receiver" (e.g. bandpass `B`, complex gain `G`). This ordering is
purely a consequence of where a term appears in the chain — QuartiCal has no separate sky/antenna
flag; you encode it by chain position. During the alternating solve for one term, all other terms
are held fixed, and the kernels split the chain around the active term into "left of active" and
"right of active" operators (`gt_active` / `lt_active` loops in
`quartical/gains/delay/kernel.py:nb_compute_jhj_jhr`).

**Direction-dependent (DD) vs direction-independent (DI).** Each term carries a
`direction_dependent` flag (config option in `quartical/config/gain_schema.yaml`; attribute on
`Gain`). DI terms (the default) are solved once and broadcast across all model directions — the
`dir_map` is all-zeros so every model direction indexes gain direction 0
(`quartical/gains/gain.py:_make_dir_map`). DD terms are solved per direction: `dir_map =
arange(n_dir)` and the stored gain has a real direction axis. Computationally, a DD term multiplies
the solved-for degrees of freedom by the number of directions and changes flag propagation (DI gain
flags can be pushed back into the MS `FLAG` column via `apply_gain_flags_to_flag_col`; the kernels
guard this with `if not dd_term`). `pinned_directions` lets you freeze chosen directions of a DD
term during updates.

## Gain solving as complex NLLS

Calibration is posed as **non-linear least squares**: choose the Jones terms that minimise the
weighted residual between data and the gain-corrupted model,

```
minimise   sum over (p,q,t,nu)  w_pq  || V_pq  -  J_p B_pq J_q^H ||^2
```

where `w_pq` are per-visibility weights (`ms_inputs.WEIGHT`), flagged points are skipped, and the
sum runs over baselines, times, and channels. The residual `r_pq = V_pq - model` is exactly what
`compute_residual` forms and what the solver drives toward zero.

**Why complex optimisation (Wirtinger calculus).** The unknowns are complex gains, and the
objective depends on both a gain and its conjugate (through the `J_q^H`), so it is not holomorphic
and ordinary complex differentiation does not apply. QuartiCal instead uses **Wirtinger
calculus**, treating a variable and its conjugate as independent and forming complex Gauss–Newton
updates in that basis. Two documented simplifications make this tractable at scale (Kenyon et al.
2025, Section 2): the *AllJones* diagonal approximation — all off-diagonal blocks of `J^H W J` are
discarded (equivalent to assuming no covariance between parameters), which avoids large matrix
products and guarantees the approximate `J^H W J` is invertible — and **one-term-at-a-time
updates** within a chain (updating all terms simultaneously is computationally impractical and can
be degenerate; term-at-a-time has no convergence guarantee but is observed to work empirically).
The update equations also admit the per-correlation weights present in real MS data, relaxing an
assumption of Kenyon et al. 2018. Consult Kenyon et al. 2025 (and Smirnov & Tasse 2015 for the
underlying complex calculus) for the derivation of the complex Jacobian and the update equations;
this page does not reproduce the algebra.

**Iteration structure.** Solving is per data **chunk** and per term, alternating:

- The calibration graph fires one solver task per `(row, chan)` chunk
  (`quartical/calibration/constructor.py`); each task sees a whole antenna/direction/correlation
  slice for its block.
- Within a chunk the terms are solved in an **alternating** (block-coordinate) fashion — one term
  is "active" at a time, all others held fixed; the chain rotates through the terms.
- Each term's numba kernel iterates: accumulate the (Wirtinger) `JHJ` and `JHr` over solution
  intervals (`nb_compute_jhj_jhr`), solve the normal equations for a parameter/gain update
  (`compute_update`, via `inversion.invert_factory`), apply it (`finalize_update`), update flags,
  and stop when the converged fraction reaches `stop_frac`. See the "Numba kernel conventions"
  section of [solver-architecture.md](solver-architecture.md).

`JHJ` (the Gauss–Newton approximation to the Hessian) is retained per term and written out — it is
a natural per-solution error estimate.

**Solution intervals: the resolution/SNR trade-off.** A term need not have one solution per
timeslot/channel. `time_interval` and `freq_interval` (config, per term) define **solution
intervals** — blocks of time/frequency sharing a single solved value. A short interval tracks fast
variation but averages fewer visibilities, so each solution is noisier; a long interval averages
more data for higher SNR but cannot follow variation within the interval. Choosing intervals is the
classic bias/variance (time-frequency resolution vs signal-to-noise) trade-off. Mechanically, the
interval mappings (`time_map`, `freq_map`, ...) translate each data row/channel to its solution
cell; see the "Interval mappings" section of [solver-architecture.md](solver-architecture.md).
Because the update equations simply sum over the samples inside an interval, every solution
interval can be solved entirely independently of its neighbours — the property that makes
calibration embarrassingly parallel and underpins the chunked dask design (Kenyon et al. 2025,
Section 2.3; [dask-machinery.md](dask-machinery.md)).

## Parameterised terms

Some effects are physically constrained to a low-dimensional, smooth form rather than a free 2×2
complex matrix per time/channel. QuartiCal solves those in **parameter space**: it optimises a few
physical parameters and then *converts* them to complex gains. Base class:
`ParameterizedGain` (`quartical/gains/parameterized_gain.py`), subclassed by `delay`, `phase`,
`amplitude`, `tec_and_offset`, `rotation`, `rotation_measure`, `delay_and_tec`, and the offset
variants (full list in [solver-architecture.md](solver-architecture.md)).

**Why.** Fewer degrees of freedom and built-in smoothness. A delay is one real number per antenna
per parameterisable diagonal correlation (XX/YY or RR/LL — see `make_param_names`,
`quartical/gains/delay/__init__.py`; off-diagonal correlations are not parameterised), versus a
complex gain per channel; and it enforces the physical constraint that a delay produces a phase
that is **linear in frequency**. Concretely, a delay `d` gives a diagonal
gain `g(nu) = exp(i * 2*pi * d * (nu - nu_c))` (see `docs/source/gain_types.rst` and the conversion
code): one scalar determines the phase across the whole band, which both reduces the unknowns and
regularises the solution against per-channel noise. Analogous physically-motivated forms apply to
phase (constant phase), amplitude (real positive),
TEC (dispersive: the leading physical dependence is `~ ν⁻¹`, but the implemented coefficient in
`tec_and_offset` is `2π(bandwidth/ν + log(ν_min/ν_max))` — a 1/ν dispersive part plus a constant
log offset that enforces a zero-mean correction over the band; see
`quartical/gains/tec_and_offset/kernel.py` ~line 472), and rotation (a rotation angle mixing
correlations).

**How params convert to gains.** The kernel accumulates `JHJ`/`JHr` in the *parameter* basis, so
the Jacobian carries the extra chain-rule factor from the parameterisation — formally, the
parameterised Jacobian is the complex Jacobian multiplied by `∂g/∂u`, the derivative of the gains
with respect to the (real-valued) parameters (Kenyon et al. 2025, Section 2.4). In the delay
kernel this factor appears as the
`coeff = 2*pi*(chan_freq/cf_mid - 1)` term in `nb_compute_jhj_jhr` and `nb_finalize_update` in
`quartical/gains/delay/kernel.py`. After each update the params are mapped to complex gains by a
`param_to_gain` factory — for delay, `gain = exp(i * coeff * param)` per correlation
(`param_to_gain_factory` / `delay_params_to_gains`). The delay solver also works internally in a
rescaled parameter `D' = D*(nu_min+nu_max)/2` for numerical conditioning, undoing the scaling
before returning. To connect the physical and kernel pictures: the physical model is
`exp(i·2π·D·(ν − ν_c))`, but the kernel computes `exp(i·coeff·D')` where
`coeff = 2π(ν/ν_mid − 1)` and `D' = D·ν_mid`; because `ν/ν_mid − 1 = (ν − ν_mid)/ν_mid`,
this is exactly `exp(i·2π·D·(ν − ν_mid))` — so `ν_c = ν_mid = (ν_min+ν_max)/2`, not zero, and
`D'` carries the units of `D·Hz` rather than plain delay.

**Consequences for interpolation.** Because parameters are the natural, smooth quantity, terms
declare how to move between native gains and an interpolation space via
`native_to_converted`/`converted_to_native`/`native_dtype`/`converted_dtype`. A `Complex` term
converts to amplitude+trig components (so interpolation respects phase wrapping); a `Delay` uses
`no_op` conversions and a `float64 native_dtype` because the delay parameter is already the smooth
quantity to interpolate. Parameterised terms also solve gains per channel (`Delay._make_freq_map`
returns `arange(n_chan)`) while the *parameter* grid may be coarser. See
[interpolation.md](interpolation.md) for how prior solutions are loaded and regridded.

## Vocabulary map

The five columns below trace one term from user config to on-disk artefact. Grounded in
`quartical/gains/__init__.py` (`TERM_TYPES`), `quartical/config/gain_schema.yaml`
(`gain.type.choices`), and `quartical/gains/datasets.py` (zarr writing). "Class" is the
`Gain`/`ParameterizedGain` subclass; the config section name is whatever the user put in
`solver.terms`, and its `type` field selects the registry key.

| Config term name | Config section | `TERM_TYPES` key | `Gain` subclass | On-disk (zarr) |
| --- | --- | --- | --- | --- |
| `G` (user choice) | section `G`, `G.type=` | value of `G.type` | class for key | `<output>::G` |
| e.g. `G`, `type=complex` | `G` | `complex` | `Complex` | `<output>::G` |
| e.g. `B`, `type=diag_complex` | `B` | `diag_complex` | `DiagComplex` | `<output>::B` |
| e.g. `K`, `type=delay` | `K` | `delay` | `Delay` | `<output>::K` |

The registry (`TERM_TYPES`) has 16 keys: `complex`, `diag_complex`, `amplitude`, `phase`, `delay`,
`delay_and_offset`, `tec_and_offset`, `rotation`, `rotation_measure`, `crosshand_phase`,
`crosshand_phase_null_v`, `leakage`, `delay_and_tec`, `parallactic_angle`, `feed_flip`,
`delay_tec_and_offset`. The 16-entry `gain.type.choices` list in
`quartical/config/gain_schema.yaml` matches the registry one-to-one. The on-disk group is written
by `write_gain_datasets` at `f"{directory}::{term_name}"`, one zarr group per term (plus optional
`<terms>-net` effective-gain groups); a solved term dataset holds `gains`, `gain_flags`, `jhj`,
`conv_perc`, `conv_iter`, and — for parameterised terms — `params`, `param_flags`. These data
variables are assigned in `quartical/calibration/constructor.py` (~lines 237–260);
`scaffold_from_data_xds` (`quartical/gains/datasets.py`) builds only the dataset coords and
attrs.

**Glossary.**

- **chunk** — a `(row, chan)` block of the MS as seen by dask; the unit of a single solver task
  (contraction axes must be single-chunk). See [dask-machinery.md](dask-machinery.md).
- **solution interval** — a block of timeslots/channels sharing one solved value for a term,
  set by `time_interval`/`freq_interval`; controls the resolution vs SNR trade-off.
- **mapping** — an integer array translating a data coordinate to a solution-interval index
  (`time_map`, `freq_map`, `dir_map`, plus `param_*` variants), built in
  `quartical/calibration/mapping.py`.
- **direction** — a model-sky direction index `d`; a term is direction-independent (broadcast one
  direction) or direction-dependent (solved per direction).
- **xds** — an `xarray.Dataset`; QuartiCal represents each MS partition and each solved gain term
  as an xds backed by dask arrays.

## Pointers

- User-facing gain-type reference with the exact 2×2 form of each term:
  `docs/source/gain_types.rst`. Related user docs: `docs/source/interpolation.rst`,
  `docs/source/gain_files.rst`, `docs/source/options.rst`.
- Implementation of everything above: [solver-architecture.md](solver-architecture.md) (term
  abstraction, `Gain`/`ParameterizedGain`, graph, mappings, kernel conventions) and
  [dask-machinery.md](dask-machinery.md) (chunking, single-compute, scheduling).
- Authoritative reference: Kenyon, J.S., Perkins, S.J., Bester, H.L., Smirnov, O.M.,
  Russeeawon, C. & Hugo, B.V., 2025, "Africanus II. QuartiCal: Calibrating radio interferometer
  data at scale using Numba and Dask", Astronomy and Computing 52, 100962
  (doi:10.1016/j.ascom.2025.100962; arXiv:2412.10072; open access). The updated formalism for
  arbitrary-length chains of parameterised terms, the AllJones approximation, and the dask/numba
  implementation rationale.
- Lineage references: Smirnov & Tasse 2015 (complex/Wirtinger calibration); Kenyon et al. 2018
  ("CubiCal", MNRAS). Consult the papers for the RIME derivation, the complex Jacobian, and the
  update equations, which this page deliberately does not reproduce.
