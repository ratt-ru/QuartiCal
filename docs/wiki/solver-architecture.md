# Solver Architecture

> **Purpose:** How gain terms, mappings, and the calibration graph fit together — read before
> touching quartical/gains/ or quartical/calibration/.
> **Last verified:** 50207c9, 2026-07-07

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

These are consumed inside the numba kernels via the `mapping_inputs` namedtuple. In the JHJ/JHr
accumulation loops (`quartical/gains/complex/kernel.py:nb_compute_jhj_jhr`,
`quartical/gains/delay/kernel.py:nb_compute_jhj_jhr`), each data point looks up its gain slice as
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

Kernel structure (see `quartical/gains/complex/kernel.py` and `.../delay/kernel.py`), all following
the same skeleton:

- A thin `@njit(**JIT_OPTIONS)` entry point (`complex_solver` / `delay_solver`) forwarding to an
  `_impl` that is `@overload`-ed. The overload calls `coerce_literal(..., ["corr_mode"])`, builds
  its factory helpers once, then returns the `impl` closure.
- The solve `impl` sets up flagging/solving intermediaries (`native_intermediaries`,
  `upsampled_itermediaries`, `flag_intermediaries`), then loops `for loop_idx in
  range(max_iter or 1)`: `compute_jhj_jhr` → optional `downsample_jhj_jhr` /
  `per_array_jhj_jhr` / `scalar_jhj_jhr` → `compute_update` (matrix inversion via
  `inversion.invert_factory`) → `finalize_update` → `update_gain_flags` (which also returns the
  converged percentage) → break at `conv_perc >= meta_inputs.stop_frac`.
- `compute_jhj_jhr` is `@overload`-ed with `PARALLEL_JIT_OPTIONS` and does the `prange` over
  solution intervals, accumulating per-antenna JHJ (a Kronecker/`a_kron_bt` form for 4-corr) and
  JHr from the residual.
- Every solve returns `(native_imdry.jhj, loop_idx + 1, conv_perc)`.

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
   `quartical/gains/general/factories.py`, `.../flagging.py`, `.../inversion.py`,
   `.../convenience.py`, and `.../generics.py` rather than reimplementing correlation dispatch.
5. **Add a per-type test** `testing/tests/gains/test_<type>.py`, mirroring an existing one (e.g.
   `test_delay.py` for parameterised, `test_complex.py` for plain). These parametrise over
   `select_corr`, set `solver.terms=['G']` with `G.type=<type>`, run `add_calibration_graph`, and
   check the recovered gains against a known truth.
