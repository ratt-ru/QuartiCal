---
type: architecture
title: Gain Interpolation
description: "Loading previously solved gains and interpolating them onto the current solution grid — the load path, grid mismatch handling, and how loaded values and flags reach the solver."
timestamp: 2026-09-30
last_verified_commit: ff732aa
---

# Gain Interpolation

`quartical/interpolation/` implements transfer calibration: reading a previously written
gain store from disk and putting its contents onto the solution grid of the current run.
It runs once per pipeline invocation, from
`calibration/calibrate.py:add_calibration_graph`, between `make_gain_xds_lod` (which
builds the empty per-term scaffolds) and `construct_solver`. Loaded solutions are always
an *initial estimate* — whether they are then refined depends on `solver.iter_recipe`; a
term given zero iterations is applied as-is (see "Zero-iteration terms" below).

## Entry point

`interpolate.py:load_and_interpolate_gains(gain_xds_lod, chain, output_directory)` walks
the chain and returns a list of dicts with the same shape as its input. Every input
dataset is a scaffold holding identity values and unraised flags
(`gains/datasets.py:assign_identity_values`). A term with `term.load_from is None` is
passed through untouched; a loaded term's scaffolds are returned with their values and
flags updated.

For a term which is loaded:

1. `term.load_from` is split into a store/group pair
   (`"::".join(term_path.rsplit('/', 1))`) and read with
   `daskms.experimental.zarr.xds_from_zarr`.
2. Every loaded dataset must carry the same `TYPE` attribute, and it must equal
   `term.type` — terms cannot change type across a load (`assert`).
3. The antenna axes are aligned (`align_antennas`, below).
4. `xarray.combine_by_coords` merges the datasets over their time/frequency coords.
   Overlapping spectral windows make this fail; the `ValueError` is re-raised with an
   instruction to split the data so that no SPWs overlap.
5. The merged dataset is rechunked to a single time/frequency chunk and one chunk per
   antenna — the interpolation is embarrassingly parallel over antennas. The loaded
   flags are reduced over time and frequency into `unsolved_antenna_mask`, which is set for each
   (antenna, direction) flagged everywhere (see "Flags" below).
6. `convert_native_to_interp` moves the values into the interpolation representation and
   replaces flagged values with NaN.
7. `interpolate_missing` fills the NaNs, then `linear2d_interpolate_gains` or
   `spline2d_interpolate_gains` evaluates the result on each target grid.
8. `convert_interp_to_native` moves back to the native representation,
   `reindex_and_rechunk` makes the correlation axis and the chunking match the reference
   dataset, and `assign_interpolated_arrays` writes the interpolated values onto the target
   dataset and ORs `unsolved_antenna_mask` into every flag variable it carries.
9. `compute_and_reload` triggers an early compute per term, writing to
   `{output_directory}/partials.tmp` (removed by an `atexit` hook) and reading back, so
   the interpolation graph is replaced by simple reads before the main compute.

## What is interpolated

`term.interpolation_targets` selects the variables: `["gains", "gain_flags"]` for `Gain`
subclasses, `["params", "param_flags"]` for `ParameterizedGain` subclasses. A
parameterised term therefore interpolates parameters, never gains, and its gains are
reconstructed from the parameters by the term's `init_term`/kernel.

Values are not interpolated in their native representation. `gains/conversion.py:Converter`
applies the term's declared `native_to_converted`/`converted_to_native` pairs, e.g.
`Complex` interpolates `(amplitude, cos phase, sin phase)` — the trig pair avoids phase
wraps — while a parameterised term's parameters pass through unchanged. Note that the
`interp_mode` option (`reim`/`ampphase`/`amp`/`phase` in `config/gain_schema.yaml`) is
read into `Gain.interp_mode` and **never used**: the representation is fixed per term
class. `interp_method` (`2dlinear`/`2dspline`) is live and selects the interpolant.

`interpolants.py` holds the interpolants:

- `_interpolate_missing` (numba) fills NaNs with linear interpolation and constant-value
  extrapolation, first along time then along frequency. A slice with no finite value at
  all after the time pass is **zeroed** — the "no information" fill. `interpolate_missing`
  also has a phase-aware `mode="phase"` variant, which nothing currently calls.
- `linear2d_interpolate_gains` uses linear interpolation inside the source domain and
  nearest-neighbour extrapolation outside it, and reindexes (rather than interpolates)
  along an axis of length one.
- `spline2d_interpolate_gains` needs at least four points along each axis of the merged
  source dataset and raises if it has fewer.

## Grid mismatch handling

- **Time and frequency**: handled by the interpolants — linear inside the loaded domain,
  nearest-neighbour outside it. No warning is emitted when the target grid extends well
  beyond the loaded one.
- **Correlations**: `reindex_and_rechunk` pads a narrower loaded axis with zeros and
  selects out of a wider one.
- **Antennas**: gain datasets span every antenna in the MS `ANTENNA` subtable, and the
  antenna axis is labelled with antenna *names* (`data_handling/ms_handler.py` assigns
  `ant`), so transferring between observations with different arrays gives differing
  antenna axes. `align_antennas` reindexes each loaded dataset onto the target antenna
  names: antennas which are absent from the target are dropped, antennas which are absent
  from the loaded solutions are added with every flag raised (so `unsolved_antenna_mask` is
  set for them), and
  — because the reindex is by label — differently *ordered* antenna axes are aligned rather
  than silently transposed.
- **Directions**: not handled. A direction axis mismatch is not detected here.

## Flags

The interpolation deliberately does not carry the loaded flags onto the target grid: a
flagged input value becomes a NaN which the interpolation fills from its neighbours, which
is the point of interpolating in the first place.

The exception is an (antenna, direction) whose loaded flags are raised at every time and
frequency — absent from the loaded solutions, or present but flagged throughout. There is
nothing to fill it from, so `_interpolate_missing` writes the "no information" zero (for
`Complex`, a null Jones matrix). `unsolved_antenna_mask` marks exactly these slices: the fill interpolates
along time and then along frequency, so it only falls back to zero where a whole
(antenna, direction) slice is empty, and flags raise a NaN in every parameter of an element
at once. It is computed from the merged loaded flags before interpolation, is
independent of the target grid, and so applies equally to the gain and parameter grids of
a parameterised term. `assign_interpolated_arrays` raises the target's flags there; the zero values
underneath are replaced by the identity in `init_term`. No NaN reaches the interpolants,
so `2dspline` (whose cubic fit rejects non-finite input) is unaffected.

## How loaded values and flags reach the solver

Every scaffold carries `gains` and `gain_flags` (plus `params` and `param_flags` for a
parameterised term), so `calibration/constructor.py:construct_solver` always adds
`{term}_initial_gain` and `{term}_initial_gain_flags` (plus `{term}_initial_params` and
`{term}_initial_param_flags` when the dataset has a `PARAM_SPEC`) as `Blocker` inputs.

`Gain.init_term`/`ParameterizedGain.init_term` start from those values and OR those flags
into the flags derived from the data by `general/flagging.py:init_flags`, whether or not
the term was loaded. For an unloaded term the scaffold holds the identity and no raised
flags, which reproduces a fresh start. Flagged gains and parameters are subsequently
overwritten with the identity element, so an unsolved antenna ends up holding an identity
gain, fully flagged. `load_from` still gates the initial estimate of the delay/TEC terms,
which only runs for an unloaded term.

### Zero-iteration terms

`solver.iter_recipe` entries of zero are the "apply, don't solve" case. The shared solver
loop (`gains/general/solver_loop.py`) still runs `finalize_gain_flags` and, for a
direction-independent term, `apply_gain_flags_to_flag_col` when `max_iter` is zero, so the
gain flags of a loaded term propagate into the MS `FLAG` column (subject to
`solver.propagate_flags`) even though nothing is solved. An unsolved antenna is therefore
flagged in the gain output *and* in the data.

## Tests

`testing/tests/interpolation/test_interpolants.py` covers the numba fill directly,
including the all-NaN-to-zeros archetype. `testing/tests/interpolation/test_interpolate.py`
drives `load_and_interpolate_gains` over synthetic gain datasets — five relative
time/frequency layouts (`GAIN_PROPERTIES`) crossed with the interpolation methods, plus the
antenna alignment cases (`ALIGNMENT_CASES`) and the unsolved-antenna cases (missing and
fully flagged, for a complex and a delay term). Neither needs a Measurement Set.
`testing/tests/gains/test_init_term.py` pins the `init_term` side: the identity fill for
flagged parameters, starting from the scaffold values, merging the scaffold flags, and
each term's `param_identity_fill` producing identity gains.
`testing/tests/gains/test_datasets.py` pins the scaffolds themselves, including that no two
scaffold chunks share a dask task.
