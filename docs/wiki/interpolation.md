---
type: architecture
title: Gain Interpolation
description: "Loading previously solved gains and interpolating them onto the current solution grid — the load path, grid mismatch handling, and how loaded values and flags reach the solver."
timestamp: 2026-08-25
last_verified_commit: b0dc5c5
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
the chain and returns a list of dicts with the same shape as its input. A term with
`term.load_from is None` is passed through untouched, so the returned datasets are a
mixture of empty scaffolds (to be solved) and populated datasets (loaded).

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
   antenna — the interpolation is embarrassingly parallel over antennas.
6. `convert_native_to_interp` moves the values into the interpolation representation and
   replaces flagged values with NaN.
7. `interpolate_missing` fills the NaNs, then `linear2d_interpolate_gains` or
   `spline2d_interpolate_gains` evaluates the result on each target grid.
8. `convert_interp_to_native` moves back to the native representation,
   `reindex_and_rechunk` makes the correlation axis and the chunking match the reference
   dataset, and `flag_missing_antennas` adds the flags.
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
  from the loaded solutions are added with a raised flag, and — because the reindex is by
  label — differently *ordered* antenna axes are aligned rather than silently transposed.
- **Directions**: not handled. A direction axis mismatch is not detected here.

## Flags

The interpolation deliberately does not carry the loaded flags onto the target grid: a
flagged input value becomes a NaN which the interpolation fills from its neighbours, which
is the point of interpolating in the first place. The consequence is that an input which is
flagged *everywhere* for some antenna is filled with the "no information" zero, which for
`Complex` means a null Jones matrix rather than an identity or a flag.

The exception is a *missing* antenna. `flag_missing_antennas` assigns `gain_flags` (and
`param_flags` for a parameterised term, on the parameter grid) which fully flag every
antenna that was absent from the loaded solutions, sized and chunked from the reference
dataset's `GAIN_SPEC`/`PARAM_SPEC`. These flags survive into the solve; see below.

## How loaded values and flags reach the solver

`calibration/constructor.py:construct_solver` inspects the data variables of each gain
dataset — present only for a loaded term, as the scaffolds are empty — and adds
`{term}_initial_gain`, `{term}_initial_params`, `{term}_initial_gain_flags` and
`{term}_initial_param_flags` as `Blocker` inputs.

`Gain.init_term`/`ParameterizedGain.init_term` then, when `self.load_from` is set, start
from the loaded values instead of the identity, and OR the loaded flags into the flags
derived from the data by `general/flagging.py:init_flags`. Flagged gains and parameters
are subsequently overwritten with the identity element, so a missing antenna ends up
holding an identity gain, fully flagged.

### Zero-iteration terms

`solver.iter_recipe` entries of zero are the "apply, don't solve" case. The shared solver
loop (`gains/general/solver_loop.py`) still runs `finalize_gain_flags` and, for a
direction-independent term, `apply_gain_flags_to_flag_col` when `max_iter` is zero, so the
gain flags of a loaded term propagate into the MS `FLAG` column (subject to
`solver.propagate_flags`) even though nothing is solved. A missing antenna is therefore
flagged in the gain output *and* in the data.

## Tests

`testing/tests/interpolation/test_interpolants.py` covers the numba fill directly,
including the all-NaN-to-zeros archetype. `testing/tests/interpolation/test_interpolate.py`
drives `load_and_interpolate_gains` over synthetic gain datasets — five relative
time/frequency layouts (`GAIN_PROPERTIES`) crossed with the interpolation methods, plus the
antenna alignment cases (`ALIGNMENT_CASES`). Neither needs a Measurement Set.
`testing/tests/gains/test_init_term.py` pins the `init_term` side: the identity fill for
flagged parameters and the merging of loaded flags.
