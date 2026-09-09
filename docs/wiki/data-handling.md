---
type: stub
title: Data Handling
description: "Measurement Set I/O, chunking, selection, and derived quantities."
timestamp: 2026-08-31
last_verified_commit: b1a0cb0
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

`angles.py:assign_parangle_data` uses the same selection for `FIELD_CENTRE`, which drives
QuartiCal's own parallactic angle machinery: `apply_parangles` for model columns under
`input_model.apply_p_jones`, `output.apply_p_jones_inv`, and the `parallactic_angle` gain
term (via `meta["FIELD_CENTRE"]`). Both parangle paths must agree, so they share the
helper. See the design-decisions entry "The pointing, not the phase centre, drives the
beam and the parallactic angles" for the rejected alternatives and the africanus internals
this leans on.

### Beams

`load_beams` expands a MeqTrees-style filename schema
(`input_model.beam`, e.g. `beam_$(corr)_$(reim).fits`) into one real/imaginary FITS pair
per correlation, checks that all headers agree, and returns the stacked cube, its lm
extents and its frequency grid. `input_model.beam_l_axis`/`beam_m_axis` set the axis
orientation (`~` stands in for `-`). Files are opened through a small wrapper so the FITS
handles are closed when the last reference is dropped, and the reads are `dask.delayed`.
