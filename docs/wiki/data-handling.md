---
type: stub
title: Data Handling
description: "Measurement Set I/O, chunking, selection, and derived quantities."
timestamp: 2026-10-07
last_verified_commit: 2c24468
---

# Data Handling

**STUB** — the predict section below is verified; the rest of the scope is still
pending. Fill in when working in this subsystem (see maintenance rule in CLAUDE.md).
Do not fabricate content to fill this page; document only what you have verified in
source.

## Intended scope

- `read_xds_list`/`write_xds_list` in `quartical/data_handling/ms_handler.py`; dask-ms
  usage.
- Chunking strategy: how config chunk specs become dask chunks.
- Selection (field/ddid/scan), weight/flag preprocessing.
- Parallactic angle machinery and BDA support.

## Table backend (casacure)

MS and subtable access goes through casacure, a Rust implementation of the casacore table
surface, rather than python-casacore. `data_handling/backend.py:activate_backend` — called
as the first statement of `quartical/__init__.py` — aliases `casacore`,
`casacore.tables`, `casacore.measures` and `casacore.quanta` onto the matching casacure
modules in `sys.modules` (and as attributes on the `casacore` module object, for
`import casacore; casacore.measures...` consumers). That is what makes dask-ms, which
reaches its backend as `lazy_import("casacore.tables")`, read and write tables through
casacure. Two failure modes are handled in place: if `casacore.tables` is already in
`sys.modules` the alias is skipped with a `RuntimeWarning` (a table handle from one
backend and a measures server from the other must not be mixed — import quartical first),
and if casacure is not importable the alias is skipped with a warning and python-casacore
is used. `BACKEND_NAME` records which backend is in effect.

casacore is reached through dask-ms as well as through QuartiCal's own imports, so the
alias also covers the `python-casacore` install that dask-ms 0.2.x requires: it is
installed but unused. dask-ms is pinned to 0.2.x for this reason — 0.3 opens tables
through arcae and offers no equivalent interception point. See the design-decisions entry
"casacure as the table and measures backend".

## Parallactic angles (`quartical/data_handling/angles.py`)

Angles are computed per unique time and antenna by `_make_parangles`, which runs lazily
via `da.blockwise` from `make_parangle_xds_list`. Per call it:

- creates a thread-local `casacure.measures.measures()` server (one per worker thread,
  since the server's frame is mutated by `do_frame`);
- makes the field centre a `J2000` direction, the zenith an `AZEL` direction at `(0, 90)`
  degrees and each antenna an `ITRF` position;
- per unique time does `do_frame(epoch('UTC', t seconds))`, per antenna
  `do_frame(position)`, and adds `posangle(field, zenith).get_value('rad')` onto the
  receptor angles.

The inputs arrive from `assign_parangle_data`, which attaches `RECEPTOR_ANGLE` (selected
from the FEED group matching the dataset's `SPECTRAL_WINDOW_ID`, with `-1` as the global
fallback), `POSITION`, `FEED_TYPE` and `FIELD_CENTRE` to each data dataset.
`FIELD_CENTRE` is the pointing, from `pointing.py:get_pointing_dir` (the first populated
of `REFERENCE_DIR`, `DELAY_DIR`, `PHASE_DIR`), which is what `predict.py` and the
`parallactic_angle` gain term (`meta["FIELD_CENTRE"]`) use. `make_parangle_xds_list`
does not read that attribute: it re-derives the field centre from
`FIELD.PHASE_DIR[xds.FIELD_ID, 0]` (line 157), so on a rephased MS the
`PARALLACTIC_ANGLES` datasets describe the phase centre while every other parallactic
angle consumer describes the pointing. That is a live defect against the intent of
ratt-ru/QuartiCal#441; the fix is to take the field centre from `xds.attrs` the way
`assign_parangle_data` sets it.

casacure's `AZEL` and `AZELGEO` both use the geodetic (WGS84) latitude, the vertical
astropy's `AltAz` uses; casacore's `AZEL` uses the geocentric latitude instead. This call
chain therefore returns angles ~2 deg higher than casacore's at MeerKAT (133.5507 vs
131.1922 deg for a MeerKAT antenna with the 3C147-like `J2000 (0.5, -0.5)` field centre at
MJD 57844.5). Against casacore's `AZELGEO` frame casacure agrees to a few arcsec over a
MeerKAT field/epoch grid; `testing/tests/data_handling/test_backend.py` asserts both the
frame choice and that agreement.

`apply_parangles` applies the angles to model columns under `input_model.apply_p_jones`
and derotates outputs under `output.apply_p_jones_inv`; the `parallactic_angle` gain term
consumes `meta["FIELD_CENTRE"]` instead and computes the angles again inside its kernel.
Both paths share `get_pointing_dir`, so they cannot disagree about the direction.

## Predict (`quartical/data_handling/predict.py`)

`predict` builds model visibilities from Tigger sky models using africanus' fused RIME
(`africanus.experimental.rime.fused`). The shape of a predict is:

- `parse_sky_models` reads the LSM into per-model, per-tag-cluster dictionaries of source
  parameters; `daskify_sky_model_dict` turns each into an `xarray.Dataset` of dask arrays
  chunked over `source` (`input_model.source_chunks`).
- `get_support_tables` reads ANTENNA, DATA_DESCRIPTION, FIELD, SPECTRAL_WINDOW,
  POLARIZATION and FEED. Only ANTENNA and FEED stay lazy; the rest are computed eagerly,
  so their variables hold numpy arrays and can be inspected while the graph is built.
- `build_rime_spec` assembles the RIME string per source type — `Kpq`/`Bpq` always,
  `Cpq` for gaussians, `Lp`/`Lq` for parallactic-angle feed rotation
  (`input_model.apply_p_jones`), `Ep`/`Eq` for the beam — and returns a
  `RimeSpecification`.
- The per-dataset `extras` dict supplies everything that is not per-source: `phase_dir`,
  `chan_freq`, `antenna_position`, `receptor_angle`, the `convention`
  (`input_model.invert_uvw` selects casa over fourier) and, with a beam, the beam cube.
  Inputs reach a term by *name*, whether they come from the `extras` dict or from a
  Dataset variable — africanus' `consolidate_args` flattens both into one mapping.

### Directions: phase centre versus pointing

`phase_dir` has three consumers inside the fused RIME, which is why it cannot simply be
repointed:

- `LMTransformer` turns `radec` + `phase_dir` into `lm`, consumed by the `Phase` term.
  This must be `FIELD.PHASE_DIR`, since it is what the MS uvw coordinates are referenced
  to.
- `ParallacticTransformer` uses it for `feed_parangle`/`beam_parangle`.
- `BeamCubeDDE` samples the beam cube at the *same* `lm`.

The beam and the parallactic angles belong at the pointing, not the phase centre, and the
two differ on a rephased MS. `pointing.py:get_pointing_dir` picks the first populated
column of `POINTING_DIR_COLUMNS` (`REFERENCE_DIR` -> `DELAY_DIR` -> `PHASE_DIR`, skipping
absent or non-finite ones). `predict` logs its separation from the phase centre
(astropy's `angular_separation`) so that a rephased MS is visible in the output.

`predict` then supplies both lm arrays itself, as variables on the sky model Dataset:
`lm` about `PHASE_DIR` for the fringe, and `beam_lm` about the pointing for
`PointedBeamCubeDDE` (a `BeamCubeDDE` subclass registered via
`RimeSpecification(terms=...)`). A transformer only runs for arguments that are missing,
so supplying `lm` stops `LMTransformer` running and leaves `phase_dir` reaching only
`ParallacticTransformer` — which is why `extras["phase_dir"]` holds the *pointing*.

`angles.py:assign_parangle_data` uses the same selection for `FIELD_CENTRE`, the one
direction QuartiCal's own parallactic angle machinery reads. `make_parangle_xds_list`
computes `PARALLACTIC_ANGLES` about it, which `apply_parangles` uses for model columns under
`input_model.apply_p_jones` and for `output.apply_p_jones_inv`. The `parallactic_angle`
gain term computes its angles about it too, via `meta["FIELD_CENTRE"]`. Both call
`_make_parangles`, which uses the same casacore measures chain as the africanus
`casa_parallactic_angles` behind the predict's P-Jones, so all three P-Jones paths agree.
The test MS's FIELD directions all coincide, so no integration test can tell the pointing
from the phase centre. `test_angles.py` therefore pins the agreement on a synthetic
rephased FIELD table. See the design-decisions entry "The pointing, not the phase centre,
drives the beam and the parallactic angles" for the rejected alternatives and the africanus
internals this leans on.

### Beams

`load_beams` expands a MeqTrees-style filename schema
(`input_model.beam`, e.g. `beam_$(corr)_$(reim).fits`) into one real/imaginary FITS pair
per correlation, checks that all headers agree, and returns the stacked cube, its lm
extents and its frequency grid. `input_model.beam_l_axis`/`beam_m_axis` set the axis
orientation (`~` stands in for `-`). Files are opened through a small wrapper so the FITS
handles are closed when the last reference is dropped, and the reads are `dask.delayed`.
