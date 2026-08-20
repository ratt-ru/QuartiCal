---
type: architecture
title: Solver Architecture
description: "How gain terms, mappings, and the calibration graph fit together — read before touching quartical/gains/ or quartical/calibration/."
timestamp: 2026-08-20
last_verified_commit: 5b39871
---

# Solver Architecture

The published description of this machinery is Kenyon et al. 2025, "Africanus II. QuartiCal"
(Astronomy and Computing 52, 100962; arXiv:2412.10072) — Section 2 for the update equations the
kernels implement, Section 4.3 for the solver design. [domain-primer.md](domain-primer.md)
summarises the maths.

## The term abstraction

A "term" is one Jones factor in the solved gain chain. The user names the chain via
`solver.terms` (e.g. `['G', 'B']`); each name becomes a config section whose `type` field
selects a class from the `TERM_TYPES` registry in `quartical/gains/__init__.py`. That dict maps
16 type strings to classes: `complex`→`Complex`, `diag_complex`→`DiagComplex`, `amplitude`,
`phase`, `delay`, `delay_and_offset`, `tec_and_offset`, `rotation`, `rotation_measure`,
`crosshand_phase`, `crosshand_phase_null_v`, `leakage`, `delay_and_tec`, `parallactic_angle`,
`feed_flip`, `delay_tec_and_offset`. `quartical/config/internal.py:gains_to_chain` instantiates
these into the `chain` (a list of `Gain` objects) using the per-term options; the same choices are
enumerated under `gain.type.choices` in `quartical/config/gain_schema.yaml`.

A term package under `quartical/gains/<type>/` provides a contract consumed by the graph and the
solver wrapper:

- **A class** (in `<type>/__init__.py`) inheriting `Gain` or `ParameterizedGain`, registered in
  `TERM_TYPES`.
- **`solver = staticmethod(...)`** — the numba-jitted per-chunk solve function, imported from the
  package's `kernel.py`. Example: `Complex.solver = staticmethod(complex_solver)`
  (`quartical/gains/complex/__init__.py`); `Delay.solver = staticmethod(delay_solver)`
  (`quartical/gains/delay/__init__.py`).
- **Class attributes** describing behaviour: `native_to_converted` / `converted_to_native` /
  `converted_dtype` / `native_dtype` drive interpolation-space conversion. `Complex` converts to
  amplitude+trig (`(np.abs,)`, `(np.angle, np.cos)`, `(np.angle, np.sin)`) and back via
  `amp_trig_to_complex`; `Delay` uses `no_op` conversions with `native_dtype = np.float64`.
- **Optional mapping overrides.** `Delay._make_freq_map` overrides the base to solve in every
  channel (`return np.arange(chan_freqs.size, dtype=np.int32)`).
- **`ms_inputs` override** when the kernel needs extra MS-derived quantities. `Delay` extends the
  base `ms_inputs` namedtuple with `CHAN_FREQ`, `MIN_FREQ`, `MAX_FREQ`
  (`quartical/gains/delay/__init__.py`).
- **Parameterised terms only:** `make_param_names(correlations)` (returns the `param_name` coord
  labels, e.g. `Delay` returns `delay_XX`/`delay_YY`/…) and an `init_term` that seeds parameters and
  converts them to gains.

Two worked examples span the space: `Complex` is a plain (non-parameterised) full-Jones gain whose
solver optimises the complex gain entries directly; `Delay` is parameterised — it solves for a
scalar delay per correlation and maps that to a phase-slope gain.

## Gain vs ParameterizedGain

`Gain` (`quartical/gains/gain.py`) is the base class. It holds per-term config on `__init__`
(`name`, `type`, `solve_per`, `scalar`, `direction_dependent`, `pinned_directions`,
`time_interval`, `freq_interval`, `respect_scan_boundaries`, `initial_estimate`, `load_from`,
`interp_mode`, `interp_method`) and provides the classmethods that build the time/freq/direction
mappings (`make_time_bins`, `make_time_map`, `make_freq_map`, `make_dir_map`, and the `_make_*`
numpy internals plus `make_*_chunks`/`make_*_coords`). It defines `gain_axes = ("gain_time",
"gain_freq", "antenna", "direction", "correlation")`, `is_parameterized = False`, and the
`ms_inputs`/`mapping_inputs`/`chain_inputs` namedtuple classes used to marshal kernel arguments.
`Gain.init_term` allocates a complex128 gain array (2×2 identity when `n_corr == 4`) and initial
gain flags via `init_flags`, returning `(gains, gain_flags)`.

`ParameterizedGain` (`quartical/gains/parameterized_gain.py`) subclasses `Gain`, sets
`is_parameterized = True`, adds `param_axes = ("param_time", "param_freq", "antenna", "direction",
"param_name")`, and extends the three namedtuples: `mapping_inputs` gains `param_time_bins`,
`param_time_maps`, `param_freq_maps`; `chain_inputs` gains `params` and `param_flags`. It adds the
parameter mapping builders (`make_param_time_bins`, `make_param_time_map`, `make_param_freq_map`,
etc., which delegate to the base `_make_*` implementations) and declares `make_param_names` as
`NotImplementedError` (each term must supply it). Its `init_term` returns
`(gains, gain_flags, params, param_flags)`: it allocates a float64 `params` array (shape
`param_shape`), inits `param_flags` and `gain_flags` from data coverage, and expects the subclass to
convert params→gains. `Delay.init_term` calls `super().init_term` then `delay_params_to_gains`, and
(unless `load_from`/`not initial_estimate`) runs an FFT-based initial delay estimate against the
reference antenna.

So parameterisation means: **solve in parameter space, then convert parameters to complex gains.**
The kernel accumulates JHJ/JHr in the parameter basis and its `finalize_update` writes gains via a
`param_to_gain` factory (see `quartical/gains/delay/kernel.py:param_to_gain_factory` and
`nb_finalize_update`). Terms subclassing `ParameterizedGain` (verified in each `__init__.py`):
`amplitude`, `phase`, `delay`, `delay_and_offset`, `tec_and_offset`, `rotation`, `rotation_measure`,
`crosshand_phase`, `crosshand_phase_null_v`, `parallactic_angle`, `delay_and_tec`,
`delay_tec_and_offset`. Terms subclassing `Gain` directly: `complex`, `diag_complex` (via
`Complex`), `leakage`, `feed_flip`. Note that `amplitude`/`phase` are parameterised even though they
are diagonal complex effects — they solve in amplitude/phase space respectively.

## Graph construction path

`quartical/executor.py` calls `add_calibration_graph` once, lazily; the concrete compute happens
later inside `utils/dask.py:compute_context`. The hops:

1. **`add_calibration_graph(data_xds_list, stats_xds_list, solver_opts, chain, output_opts)`**
   (`quartical/calibration/calibrate.py`). Builds the gain-dataset scaffolds
   (`make_gain_xds_lod`), the mapping datasets (`make_mapping_datasets`), optionally loads/
   interpolates prior gains, then calls `construct_solver`. Returns
   `(gain_xds_lod, net_xds_lod, data_xds_list, stats_xds_list, bl_corr_xds_list)`.

2. **`make_mapping_datasets(data_xds_list, chain)`** (`quartical/calibration/mapping.py`). Per data
   xds, per term, builds `<name>_time_bins`, `<name>_time_map`, `<name>_freq_map`, `<name>_dir_map`
   (and the `param_*` variants for parameterised terms) as dask arrays, returning one
   `xarray.Dataset` per data xds.

3. **`construct_solver(data_xds_list, mapping_xds_list, stats_xds_list, gain_xds_lod, solver_opts,
   chain)`** (`quartical/calibration/constructor.py`). Per data xds it creates a
   `Blocker(solver_wrapper, ("row", "chan"))`, adds the required MS columns (only those in any
   term's `ms_inputs._fields`), the mapping arrays (time-dimensioned inputs relabelled to `row`),
   the compact per-chunk `term_spec_list` (from `expand_specs`), and scalars (`corr_mode`,
   `solver_opts`, `chain`, `data_xds_meta`, `block_id_arr`). It declares outputs (`weights`,
   `flags`, `presolve_chisq`, `postsolve_chisq`, and per term `<name>_gains`, `<name>_gain_flags`,
   `<name>_jhj`, `<name>_conviter`, `<name>_convperc`, plus `<name>_params`/`<name>_param_flags` for
   parameterised terms), calls `blocker.get_dask_outputs()`, and assigns results back onto the
   gain / data / stats datasets. Returns
   `(solved_gain_xds_lod, output_data_xds_list, output_stats_xds_list)`.

4. **`Blocker.get_dask_outputs`** (`quartical/utils/dask.py`) wires one `solver_wrapper` call per
   `(row, chan)` chunk into a custom `HighLevelGraph`. Inputs are passed as a kwargs dict; each
   declared output is extracted per chunk with `getitem(chunk_result, name)`.

5. **`solver_wrapper(term_spec_list, solver_opts, chain, block_id_arr, data_xds_meta, corr_mode,
   **kwargs)`** (`quartical/calibration/solver.py`) — the per-chunk Python driver. It runs on one
   data chunk and dispatches into the numba `solver` kernels (Section: Numba kernel conventions).

Invariant: the contraction axes handed to `Blocker` (everything except `row` and `chan`) must be
single-chunk — `Blocker._check_axis` raises if a contraction axis has multiple chunks. Thus one
solver task sees a whole antenna/direction/correlation slice for its `(row, chan)` block.

## Interval mappings

Mappings translate data coordinates (rows/channels) into solution-interval indices, so a solve can
be coarser than the data. All are built in `quartical/calibration/mapping.py:make_mapping_datasets`
by calling classmethods on the `Gain` objects (defined in `quartical/gains/gain.py`):

- **`time_bins`** (`make_time_bins`/`_make_time_bins`): a per-unique-time int array mapping each
  unique timeslot to a solution-interval bin number. `time_interval` as an `int` counts timeslots;
  as a `float` it accumulates `INTERVAL` seconds. `respect_scan_boundaries` forces a new bin at
  scan edges. Shape: `(n_utime,)`.
- **`time_map`** (`make_time_map`/`_make_time_map`): per-row (per data element along the rowlike
  axis) index into the solution-time axis, computed as `time_bins[utime_inverse]`. Shape:
  `(n_row,)`.
- **`freq_map`** (`make_freq_map`/`_make_freq_map`): per-channel index into the solution-freq axis;
  `freq_interval` int → `arange(n_chan)//freq_interval`, float → accumulate `CHAN_WIDTH`. Shape:
  `(n_chan,)`.
- **`dir_map`** (`make_dir_map`/`_make_dir_map`): per-model-direction index into the gain's
  direction axis. `arange(n_dir)` when `direction_dependent`, else all-zeros (broadcast one
  direction). Shape: `(n_dir,)`.
- **Parameterised terms** additionally get `param_time_bins`, `param_time_map`, `param_freq_map`
  (same construction, allowing a distinct parameter solution grid).

These are consumed inside the numba kernels via the `mapping_inputs` namedtuple. In the shared
JHJ/JHr accumulation loop (`quartical/gains/general/solver_components.py:build_jhj_jhr_impl`, bound
by every kernel's `nb_compute_jhj_jhr`), each data point looks up its gain slice as
`gains[gi][time_maps[gi][row_ind], freq_maps[gi][f]][antenna, dir_maps[gi][d]]`. The kernels also
call `convenience.get_extents` on the (upsampled) time map and freq map to precompute the row/
channel start–stop spans of each solution interval, and iterate `prange` over `n_tint*n_fint`
intervals.

Invariants: gain-array shape along time/freq equals `time_bins.max()+1` / `freq_map.max()+1`
(`make_time_chunks`/`make_freq_chunks` compute exactly this); `time_map` has one entry per rowlike
element and `freq_map` one per channel; `dir_map` length equals model `n_dir` and its max+1 equals
the gain's stored direction count. For parameterised terms `Delay` forces
`freq_map = arange(n_chan)` (gains evaluated per channel) while `param_freq_map` may bin coarsely.

## Numba kernel conventions

QuartiCal supports 1, 2, and 4 correlations, with markedly different maths per case. To avoid
per-element runtime branching on correlation count, kernels use the **factory pattern** in
`quartical/gains/general/factories.py`. A factory is a plain Python function taking `mode` (a numba
literal carrying `mode.literal_value`); it selects the correct closure body at compile time and
returns it wrapped by `qcjit = njit(**JIT_OPTIONS, inline="always")`. Because `corr_mode` is
coerced to a literal (`quartical/utils/numba.py:coerce_literal`), each correlation count compiles
to its own specialised, branch-free machine code. The **typical** pattern is a full three-way
dispatch on `mode.literal_value` ∈ {1, 2, 4} (e.g. `v1_imul_v2_factory`, `iunpack_factory`,
`set_identity_factory`, `valloc_factory`), but the granularity varies per factory. Two concrete
exceptions: `loop_var_factory` (~line 564) branches two ways only — `== 4` vs `else` — because the
loop-variable construction for scalar (1-corr) and diagonal (2-corr) gains is identical.
`a_kron_bt_factory` (~line 774) takes `corr_mode` but has a single unconditional implementation
body; `corr_mode` flows only into an internal `unpack_factory` call, so the factory itself
performs no correlation-count branching.

Factories come in two styles. The **tuple style** (`tuple_unpack_factory`,
`tuple_v1_mul_v2_factory`, and friends — see the "Tuple-based helpers" block in `factories.py`)
operates on and returns tuples, which are immutable SSA values inside jitted code: per-visibility
intermediaries stay in registers instead of round-tripping through memory. It is THE pattern for
kernel accumulation: the complex kernel was rewritten on it in 2026-07 for a measured 2.1-2.8x
single-thread speedup, and the rewrite was then propagated to every other solvable kernel via
the shared accumulation loop (below) for measured 1.4-2.8x speedups (see design-decisions.md,
"Tuple-based kernel maths" and "Shared hook-parameterised accumulation loop"). The original
**array-buffer style** (`iunpack_factory`, `v1_imul_v2_factory`, `valloc_factory`, ...) writes
results into small array buffers; it is legacy for accumulation loops — no kernel's
`compute_jhj_jhr` uses it any more — but survives outside them (residual computation in
`general/generics.py`, inversion buffers, `finalize_update` bodies, and the flagging kernels).
The tuple style always returns tuples even in the
1-correlation case (unlike `unpack_factory`, which returns a bare scalar) so results can be fed
back into other tuple helpers. One hard-won constraint: **never return nested tuples from an
inlined (`qcjit`) helper called inside a `prange` body** — numba's parfor array analysis misreads
tuple-of-tuples returns as array shapes and dies with `AssertionError: Dimension mismatch`.
Return flat tuples and slice by literal index instead (this is why the jhj/jhr accumulator in
the shared accumulation loop is one flat tuple).

Kernel structure (see `quartical/gains/complex/kernel.py` and `.../delay/kernel.py`), all following
the same skeleton:

- A thin `@njit(**JIT_OPTIONS)` entry point (`complex_solver` / `delay_solver`) forwarding to an
  `_impl` that is `@overload`-ed. The overload calls `coerce_literal(..., ["corr_mode"])`, binds
  one of the two shared solver-loop builders (see "The shared solver loop" below) with its
  module-local hooks, and returns a module-local trampoline that inlines the built loop.
- The shared solve body sets up flagging/solving intermediaries (`native_intermediaries`,
  `upsampled_itermediaries`, `flag_intermediaries`), then loops `for loop_idx in
  range(max_iter or 1)`: `compute_jhj_jhr` → optional `downsample_jhj_jhr` /
  `per_array_jhj_jhr` / the scalar collapse (`collapse_to_scalar_jhj_jhr` for a
  non-parameterised term, the generic `scalar_jhj_jhr` for a parameterised one) →
  `compute_update` (matrix inversion via
  `inversion.invert_factory`) → `finalize_update` → `update_gain_flags` (which also returns the
  converged percentage; parameterised terms then propagate flags via `update_param_flags`) →
  break at `conv_perc >= meta_inputs.stop_frac`.
- `compute_jhj_jhr` is `@overload`-ed with `PARALLEL_JIT_OPTIONS` and binds the shared
  accumulation loop (next subsection), which does the `prange` over solution intervals,
  accumulating per-antenna JHJ and JHr from the residual in flat register-resident tuples.
  4-corr JHJ elements use algebraically expanded Kronecker forms in the per-term accumulate hooks
  (the old explicit `a_kron_bt` array temp survives only in comments and in
  `general/generics.py`). The expanded entries are named `jh_ij` after their position in
  `a_kron_bt_factory`'s row-major product — remember it unpacks `rop` transposed — and both
  contractions are written in terms of them: `jh_i = jh_i0 + jh_i3` is the per-correlation J^H
  element (the Kronecker row contracted over the diagonal-correlation columns, used to normalise
  the residual) and `r_i = sum_k jh_ik * nres_k` is the JHr contraction, with `nres_*` the
  normalised weighted residual. Gain-basis kernels accumulate only the upper triangle and mirror
  the lower triangle once per solution interval via the mirror hook. The loop also has a fast path
  for the single-direction case (`single_dir`) which accumulates each row's JHJ/JHr
  contributions in registers and flushes to memory once per row — valid because the antenna
  pair, and hence the accumulation target, is fixed along a row.
- `compute_update` (the invert-over-intervals loop) exists exactly once, in
  `quartical/gains/general/solver_components.py`. The solver-loop builders in `solver_loop.py`
  resolve it in their own module scope, so kernel modules do not import it at all — the twelve
  dead `# noqa` re-export imports were removed 2026-07-28. The one exception is
  `crosshand_phase/null_v_kernel.py`, which keeps a hand-written solver loop and so calls
  `compute_update` directly.
- Every solve returns `(native_imdry.jhj, loop_idx + 1, conv_perc)`.

**The shared accumulation loop.** The tuple-based `compute_jhj_jhr` body lives once in
`quartical/gains/general/solver_components.py` as `build_jhj_jhr_impl(...)`. This is THE pattern:
every solvable kernel binds it (complex, diag_complex, phase, amplitude, delay,
delay_and_offset, tec_and_offset, delay_and_tec, delay_tec_and_offset, crosshand_phase,
crosshand_phase_null_v, rotation, rotation_measure; leakage imports complex's
`compute_jhj_jhr` wholesale) — there are no array-buffer holdouts. The loop itself
(the `prange` over solution intervals, the chain-product construction of the
`lop_pq/rop_pq/lop_qp/rop_qp` operators, the single-direction fast path, and the general
multi-direction path) is term-independent; all per-term maths arrives through hook factories:

All arguments are keyword-only (the leading `*` in the signature) so call sites read as a
labelled hook table rather than a run of positional factories.

- `accumulate_jhj_jhr_factory(corr_mode) -> accumulate_jhj_jhr(lop, rop, w, gain, channel_coeffs, wres, jhj_jhr) -> jhj_jhr` —
  accumulates one weighted JHJ/JHr element into the flat register-resident accumulator tuple,
  which packs the JHJ block first and the JHr block after it. `gain` is the active-term gain
  tuple for the relevant antenna (parameterised terms need it for the chain rule; the complex
  hook ignores it and LLVM eliminates the fetch). `channel_coeffs` is the per-channel coefficient
  tuple (empty for terms with no channel-coefficient hook).
- `zero_jhj_jhr_factory(corr_mode) -> zero_jhj_jhr(ref_elem)` — the zero accumulator tuple (JHJ
  block first, then JHr).
- `flush_jhj_jhr_factory(corr_mode) -> flush_jhj_jhr(jhj_el, jhr_el, jhj_jhr)` — adds an
  accumulator into the JHJ/JHr array slices (once per row on the fast path, once per direction
  otherwise).
- `compute_residual_factory(corr_mode) -> compute_residual(r, v)` — returns the per-correlation
  residual tuple. Three implementations in `general/residuals.py` cover every term:
  `standard_residual_factory` (`r - v`: complex, diag_complex, rotation, rotation_measure,
  crosshand_phase_null_v), `phase_only_residual_factory` (rescales `r` to `|v|`, so only phase
  reaches the normal equations: phase, crosshand_phase and the delay/TEC families) and
  `amplitude_only_residual_factory` (rescales `v` to `|r|`: amplitude).
- `compute_channel_coeffs_factory(corr_mode) -> compute_channel_coeffs(ms_inputs, meta_inputs, f)`
  (optional) — the per-channel coefficient tuple for terms with a frequency-dependent parameter
  (delay/TEC families, rotation_measure); its output IS the `channel_coeffs` tuple passed to the
  accumulate hook. `None` yields an empty tuple.
- `mirror_jhj_factory(corr_mode) -> mirror(jhj_tifi)` (optional) — fills the lower triangle of the
  per-interval JHJ elements; `None` yields a no-op.

"Optional" above means the hook may be `None`, not that the argument may be omitted. All three
builders (`build_jhj_jhr_impl`, `build_gain_solver_impl`, `build_param_solver_impl`) are
keyword-only with no defaults, so every kernel spells out its whole hook contract and adding a hook
forces every kernel to be visited rather than silently defaulting.

Every parameterised kernel declares one module-level constant, `PARAMS_PER_CORR` — the number of
parameters the term solves per diagonal correlation, or `None` for a term whose single parameter set
acts on the full 2x2. That one constant feeds all three consumers in the kernel:
`accumulator.py:triangular_accumulator_factories(PARAMS_PER_CORR)` at module scope (bound as
`accumulator` and passed as `accumulator.zero` / `.flush` / `.mirror`),
`parameters.py:get_identity_params(corr_mode, PARAMS_PER_CORR)`, and `params_per_corr=` on
`build_param_solver_impl`. It is also the stride `generics.py:scalar_jhj_jhr` uses when collapsing
to a scalar solve.

`parameters.py:get_n_param(corr_mode, params_per_corr)` turns it into `n_param`, the flat
parameter-vector length: `2 * params_per_corr` at 2 or 4 correlations, `params_per_corr` at 1, and 1
when `params_per_corr` is `None` (four correlations only). `None` covers exactly crosshand_phase,
crosshand_phase_null_v, rotation and rotation_measure, and it is the same `None` those kernels pass
to `build_param_solver_impl` — collapsing JHJ/JHr to a scalar solve is possible if and only if there
is a parameter set per correlation to collapse, so the two uses cannot diverge. Their `(1, 1)` JHJ
has no triangle, so they also pass `mirror_jhj_factory=None`.

`n_param` is both the accumulator's JHJ dimension and the identity vector's length, so neither can
drift from the other, and `testing/tests/gains/test_parameters.py` pins it against each gain class's
own `make_param_names` for every term and correlation mode. JHJ is a real symmetric
`(n_param, n_param)` matrix whose upper triangle is packed row-major ahead of the `n_param` JHr
entries. `complex` and `diag_complex` are not covered: their JHJ is a correlation-space block with a
mix of real and complex slots and a Hermitian mirror, so they keep hand-written hooks.

`zero` and `flush` are built as numba `@intrinsic`s so that the layout can be walked by a plain
python loop at compile time while every emitted access still names its accumulator slot by a literal
index — a computed index there costs the loop its register-resident accumulator, which is measured
in [design-decisions.md](design-decisions.md). A useful side effect: `flush`'s typing phase rejects
an accumulator of the wrong length or dtype, so an `accumulate_jhj_jhr` hook whose tuple disagrees
with the term's `n_param` fails to compile rather than writing the wrong entries.

Worked example — delay's staged-coeff hook (`quartical/gains/delay/kernel.py`). Delay is the
first consumer of the `channel_coeffs` (stage) hook. Its stage returns the single-element flat
tuple `(coeff,)` with `coeff = 2*pi*(chan_freq[f]/cf_mid - 1)` and `cf_mid = (MIN_FREQ + MAX_FREQ)/2`
(the same band-midpoint rescaling the solver applies to the parameters). That tuple is the entire
`channel_coeffs` passed to the accumulate hook, so `coeff` is at `channel_coeffs[0]` in every corr
mode. For its residual it passes the shared `phase_only_residual_factory` — `r*normf - v` with
`normf = |v|/|r|`. Delay's accumulate hook is phase's with the
chain-rule coefficient folded in — it scales JHr by `coeff` and JHJ by `coeff**2` (from
differentiating the frequency-dependent exponent) and, like phase, recomputes its own
operator-based normalisation. This shows that a flat coefficient tuple from the channel-coefficient
hook is a numba-safe way to thread per-channel data into the accumulate hook (a nested tuple return
would trip the parfor array analysis; see the NOTE in `solver_components.py`).

All hook factories are plain-Python compile-time compositions returning `qcjit`
(`inline="always"`) closures, so the indirection is free after inlining — extracting the loop
from the complex kernel measured as parity (min/min speedups 0.97-1.00 across corr modes and
a 3-direction run, checksums bitwise-identical). Each kernel keeps its own ~15-line
`compute_jhj_jhr` + `@overload` boilerplate binding its hooks, so each kernel still owns a
distinct overload symbol (`quartical/gains/phase/kernel.py` is a representative consumer;
`leakage` imports complex's `compute_jhj_jhr` wholesale).

**The shared solver loop.** One level up from the accumulation loop, the outer solver
iteration (each kernel's `*_solver_impl` body) also lives once, in
`quartical/gains/general/solver_loop.py`, as two hook-parameterised builders (extracted
2026-07-20 as pure code motion: checksums bitwise-identical per term and corr mode vs the
pre-extraction tree, timing at parity). Both builders take their hooks as **keyword-only
arguments** (a leading bare `*` in each signature); every kernel call site passes them by
name, so the opaque positional `build_param_solver_impl(None, ..., 1e9, ..., None)`
form is a `TypeError`:

- `build_gain_solver_impl(*, get_jhj_dims, compute_jhj_jhr, collapse_to_scalar_jhj_jhr,
  scalar_error_message, finalize_update, reference_gains)` — non-parameterised terms
  (complex, diag_complex, leakage). The body is complex's historic impl. diag_complex
  differs only via the builder inputs: `identity_dims` (its jhj is gain-shaped rather than
  `get_jhj_dims_factory`'s block shape), its own one-arg `collapse_to_scalar_jhj_jhr` (scalar
  mode supported; `None` means unsupported and raises `scalar_error_message`, which the builder
  requires to be non-`None` in that case), and a
  `reference_gains(chain_inputs, meta_inputs, corr_mode)` stage after `finalize_gain_flags`.
  That collapse hook is deliberately separate from the generic two-arg
  `generics.scalar_jhj_jhr`: a gain-shaped jhj element is a flat correlation vector, so
  collapsing it is a sum along the correlation axis, whereas the generic routine indexes the
  `(n_param, n_param)` block a parameterised term carries and folds its halves together using
  `values_per_correlation` as the stride.
- `build_param_solver_impl(*, pre_solve, compute_jhj_jhr,
  params_per_corr, scalar_error_message, finalize_update, numbness, identity_params,
  reference_params, post_solve)` — the ten parameterised terms. The body is delay's
  historic impl. Extents always come from `param_freq_maps`: jhj/jhr/update are allocated
  on the parameter shape, so the parameter grid is the only consistent source. The gain
  grid (`freq_maps`) is either bit-identical to it — for terms that don't override
  `_make_freq_map` (phase, amplitude, crosshand_phase, rotation), since
  `ParameterizedGain._make_param_freq_map` delegates to `Gain._make_freq_map` with the same
  args — or deliberately finer (delay/tec families and rotation_measure solve in every
  channel), which would be inconsistent with the parameter shape. A former
  `solve_on_param_grid` build flag that could select `freq_maps` was removed 2026-07-20 as
  dead: no term needed the gain-grid path (the three that set it never differed from the
  param grid, and rotation already solved on the param grid despite matching grids).
  `params_per_corr` is the width passed to the generic `scalar_jhj_jhr` collapse (`None`
  means scalar unsupported, raise — and a term which says so without supplying
  `scalar_error_message` fails to build); `numbness` forwards to `update_gain_flags` (1e9
  everywhere except amplitude's default 1e-6). That 1e9 does more than switch off trend
  flagging: the nine accumulate hooks which consume the `gain` (delay/tec families, phase,
  crosshand_phase, rotation, rotation_measure) linearise about the gain, and `set_identity`
  forces a hard-flagged gain element to the identity while `update_param_flags` only resets
  a parameter interval whose contributing gain intervals are ALL flagged — so for the six
  with a per-channel gain grid, one hard-flagged channel would contribute a derivative
  linearised at the identity against a non-zero parameter. 1e9 is what keeps mid-solve hard
  flagging (and hence that state) unreachable; full argument in the linearisation-point note
  in `solver_components.py`.
  `identity_params` forwards to `update_param_flags` and comes from
  `general/parameters.py:get_identity_params(corr_mode, PARAMS_PER_CORR, fill=)` — sized by
  `get_n_param`, so `params_per_corr=None` gives the single parameter of the four-correlation-only
  whole-2x2 terms, and `fill=1.0` is amplitude's multiplicative identity.
  `reference_params(ms_inputs, mapping_inputs, chain_inputs, meta_inputs)` runs after
  `finalize_gain_flags` where present (phase, delay/tec families). The five delay/tec members
  come from `general/parameters.py:reference_params_factory(params_to_gains=)`, which works
  because their `*_params_to_gains` share one signature; phase states its own, as
  `phase_params_to_gains` takes no frequency arguments.
  `pre_solve(ms_inputs, chain_inputs, meta_inputs)` and `post_solve(ms_inputs,
  chain_inputs, meta_inputs, native_imdry)` are opaque jitted closures owned by each
  kernel module — deliberately NOT a declarative rescaling abstraction — used to enter and
  leave a scaled solver basis: the delay/tec families' mid_freq/bandwidth strided rescales.
  They change units only; the band-referenced coefficients that decorrelate a delay or a TEC
  from an offset belong to each term's model and are carried by `params_to_gains` in both of
  its `rescaled` modes, so the two modes describe identical gains
  (`testing/tests/gains/test_solver_basis.py`).
  A term solving in the basis `p' = Sp` has `jhj = S jhj' S`. Every `post_solve` unscales
  the diagonal blocks of its rescaled parameters (`jhj[..., i::ppc, i::ppc]`) and leaves
  the blocks coupling those to unrescaled parameters in the solver basis; `delay` scales
  the whole array because every one of its parameters carries the same factor. Only the
  jhj diagonal survives `calibration/solver.py`, so the untouched blocks never reach a
  caller — see the ledger entry on the jhj unscaling convention.

`None` hooks resolve at build time (an empty `qcjit` closure or the raising scalar variant
is substituted when the builder runs), so the compiled body carries no runtime branch for an
absent hook. `finalize_update` is called with a single standardised signature per family:
5-arg `(chain_inputs, meta_inputs, native_imdry, loop_idx, corr_mode)` for gain-basis terms,
7-arg `(ms_inputs, mapping_inputs, ...)` for parameterised terms (rotation_measure
recomputes `lambda_sq` from `ms_inputs.CHAN_FREQ` inside its finalize impl).
`crosshand_phase_null_v` is the one deliberate holdout: its loop builds a typed List of
inverse gains before iterating, passes it as an extra leading argument to its own
`compute_jhj_jhr`, and refreshes the active inverse each iteration — not expressible as
verbatim code motion through these hooks — so it keeps a private copy of the loop in
`null_v_kernel.py`.

**The module-local trampoline is mandatory.** `build_jhj_jhr_impl` and both solver-loop
builders return their loops wrapped in `qcjit` (`inline="always"`), and each kernel's
`nb_compute_jhj_jhr` / `nb_<term>_solver_impl` must NOT hand the built closure back to
numba directly: it returns a module-local `impl` that simply calls (and therefore inlines)
the shared loop. This exists for on-disk cache correctness, not style. Numba keys disk-cache
entries by source location plus argument-type signature, discriminated only by a
nondeterministic cloudpickle hash of the closure cells; a directly-returned shared closure is
lowered as ONE cache unit for all kernels with identical signatures, so a stale
multi-session cache could silently load the wrong kernel's machine code (this happened —
corrupted solves, no error; see design-decisions.md, "Per-kernel numba disk-cache
namespaces"). The trampoline gives each kernel a private cache namespace; `prange` survives
the inlining. The constraint is documented as the CACHE CORRECTNESS CONSTRAINT in
`solver_components.py` (canonical) and restated at the top of `solver_loop.py` — any new consumer
of either shared loop must copy the trampoline shape.

Flagging hooks live in `quartical/gains/general/flagging.py` and are called by the kernels:
`update_gain_flags` (trend-based "trendy flagging": soft/hard flags diverging solutions, resets
hard-flagged gains to identity, returns `conv_perc`), `finalize_gain_flags` (clears surviving soft
flags, hard-flags points with a bad trend), `apply_gain_flags_to_flag_col` (propagates DI gain flags
back into the MS `FLAG` column), and for parameterised terms `update_param_flags` (propagates gain
flags to parameter flags via the bin mappings), plus `init_flags` (used at init to flag intervals
with no data). `apply_gain_flags_to_gains` / `apply_param_flags_to_params` reset flagged entries to
identity.

## Outputs

Each solved term produces one `xarray.Dataset` per data xds (a list-of-dicts `gain_xds_lod`, keyed
by term name), assembled in `construct_solver`. Data variables on a solved term xds:
`gains` (dims `GAIN_AXES`, complex128), `gain_flags` (int8), `conv_perc` and `conv_iter`
(per `(time_chunk, freq_chunk)`), `jhj`, and — for parameterised terms — `params`, `param_flags`
(dims `PARAM_AXES`). The scaffold (coords/attrs incl. `NAME`, `TYPE`, `GAIN_SPEC`, `GAIN_AXES`, and
for parameterised terms `PARAM_SPEC`, `PARAM_AXES`, `param_name`) is built by
`quartical/gains/datasets.py:scaffold_from_data_xds`.

`jhj` is declared per term in `construct_solver`, and the declaration differs by family: a
parameterised term gets `("row", "chan", "ant", "dir", "param")` chunked by its own `PARAM_SPEC`
with dtype `float64`, everything else `("row", "chan", "ant", "dir", "corr")` chunked by its
`GAIN_SPEC` with dtype `complex128`. Both facts matter to `solver_wrapper`, because a term with no
solver (`parallactic_angle`, `feed_flip`) still has to produce an array: it allocates zeros from
*its own* `term_spec` (`pshape` or `shape`) in the declared dtype, since the declaration is
per term and a chunk holds one spec per term rather than one shared shape.

`quartical/gains/datasets.py:write_gain_datasets` rechunks each term's xds list to sensible
(<2 GB, regular) chunks and writes to zarr via `daskms.experimental.zarr.xds_to_zarr` at
`f"{directory}::{term_name}"` (one zarr group per term, plus any requested `*-net` effective-gain
groups from `make_net_xds_lod`/`populate_net_xds_list`).

Per-visibility outputs are produced by `quartical/calibration/calibrate.py:make_visibility_output`,
which uses `da.blockwise` over the `dask_residual` / `dask_corrected_residual` /
`dask_corrected_weights` wrappers (calling into `quartical/gains/general/generics.py`) and assigns
underscore-prefixed data vars onto the data xds: `_RESIDUAL`, `_CORRECTED_RESIDUAL`,
`_CORRECTED_DATA`, `_CORRECTED_WEIGHT`. `construct_solver` also assigns `_WEIGHT` and either `FLAG`
(if `solver_opts.propagate_flags`) or `_FLAG`. These MS-side columns are later written by
`data_handling.ms_handler:write_xds_list`. See [dask-machinery.md](dask-machinery.md) for the
single-compute design and the `Blocker`.

## Adding a gain type

1. **New package** `quartical/gains/<type>/` with `__init__.py` (the class) and `kernel.py`
   (the numba solver). Inherit `Gain` for a directly-solved complex gain or `ParameterizedGain` for
   a parameter-space term. Set `solver = staticmethod(<your_solver>)`, the conversion attributes
   (`native_to_converted`, `converted_to_native`, `converted_dtype`, `native_dtype`), and override
   `ms_inputs`/`_make_freq_map` only if the kernel needs it. Parameterised terms must implement
   `make_param_names` and an `init_term` that converts params→gains.
2. **Register** the type string → class in `TERM_TYPES` (`quartical/gains/__init__.py`) and import
   the class at the top of that file.
3. **Expose options** by adding the type string to `gain.type.choices` in
   `quartical/config/gain_schema.yaml` (and any term-specific option fields there). See
   [config-system.md](config-system.md) (stub) for how the schema materialises into per-term
   dataclasses.
4. **Write the kernel** following the factory + `@overload` skeleton above; reuse
   `quartical/gains/general/factories.py`, `.../residuals.py`, `.../parameters.py`,
   `.../accumulator.py`,
   `.../flagging.py`, `.../inversion.py`, `.../convenience.py`, and `.../generics.py` rather than
   reimplementing correlation dispatch.
5. **Add a per-type test** `testing/tests/gains/test_<type>.py`, mirroring an existing one (e.g.
   `test_delay.py` for parameterised, `test_complex.py` for plain). These parametrise over
   `select_corr`, set `solver.terms=['G']` with `G.type=<type>`, run `add_calibration_graph`, and
   check the recovered gains against a known truth.
