# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

QuartiCal is a calibration suite for radio interferometer data (successor to CubiCal). It
solves for chains of Jones terms (gains) over Measurement Set data using complex
optimisation, with dask for parallelism and numba for hot loops.

## Commands

All Python invocations must use the project venv: `source .venv/bin/activate`. The venv
is populated with an editable dev install: `uv pip install -e .` (CI instead mimics a
user install with plain `pip install .`).

```bash
# Run the test suite (from the repo root — conftest imports fixtures as `testing.fixtures.*`)
python -m pytest testing/tests

# Run a single test file / test
python -m pytest testing/tests/gains/test_delay.py
python -m pytest testing/tests/calibration/test_calibrate.py -k <pattern>

# Skip slow tests; other markers: data_handling, preprocess, predict, model_handler, calibrate
python -m pytest -m "not slow" testing/tests
```

Test-data caveat: `testing/conftest.py` downloads test Measurement Sets and beams
(~tarballs from Dropbox) at session start and **deletes them at session end**, so every
pytest session re-downloads. The data lands in `testing/data/`. Most tests are
integration-style and need this data; there is no way to run them offline the first time.

CLI entry points (defined in `pyproject.toml`): `goquartical` (main),
`goquartical-config`, `goquartical-backup`, `goquartical-restore`,
`goquartical-summary`, `goquartical-plot`.

Releases: `tbump <new-version>` bumps `pyproject.toml`, commits, and pushes a `v*` tag,
which triggers PyPI deploy + GitHub release via `.github/workflows/ci.yaml`.

## Dependency pins

Several upper bounds in `pyproject.toml` are deliberate workarounds — read the inline
comments before touching them. Notably: `dask<=2024.10.0` (later graph optimisers break
dask-ms read graphs) and `bokeh<3.7` (template rename breaks distributed's performance
report). `stimela` must never get an upper bound.

## Architecture

### Pipeline (quartical/executor.py)

`goquartical` → `executor:execute()` runs the whole pipeline lazily, then computes once:

1. `config.parser:parse_inputs` — merge YAML config file(s) + CLI args (OmegaConf).
2. `data_handling.ms_handler:read_xds_list` — read the MS into a list of xarray
   Datasets backed by dask arrays (via dask-ms); `preprocess_xds_list` initialises
   weights/flags.
3. `data_handling.model_handler:add_model_graph` — attach model visibilities
   (MS column and/or predicted from a Tigger sky model).
4. `calibration.calibrate:add_calibration_graph` — the core: builds per-chunk solver
   tasks and produces gain datasets + residuals/corrected data.
5. `gains.datasets:write_gain_datasets` (zarr) and `ms_handler:write_xds_list` (MS
   columns), then a single `dask.compute()` inside `utils/dask.py:compute_context`
   (local threads or distributed cluster).

### Gain terms — the central abstraction

- Users specify a chain via `solver.terms=['G','B',...]`; each name becomes a dynamic
  config section whose `type` selects a class from the `TERM_TYPES` registry in
  `quartical/gains/__init__.py`.
- All terms inherit from `Gain` (`gains/gain.py`); terms solved via parameters (delay,
  phase, tec, rotation, ...) inherit from `ParameterizedGain`
  (`gains/parameterized_gain.py`).
- Each term lives in `quartical/gains/<type>/` with an `__init__.py` (class: mapping
  methods, interpolation modes, param names) and a `kernel.py` (numba-jitted solver,
  assigned as `solver = staticmethod(...)`). Shared jitted machinery is in
  `gains/general/` (`factories.py`, `flagging.py`).
- To add a gain type: new `quartical/gains/<type>/` package (class + kernel), register
  it in `TERM_TYPES`, and expose its options in `quartical/config/gain_schema.yaml`.
  Add a `testing/tests/gains/test_<type>.py` mirroring the existing per-type tests.

### Config system (quartical/config/)

Schemas live in YAML (`argument_schema.yaml` for top-level sections,
`gain_schema.yaml` for per-term options) and are materialised into dataclasses at
runtime: `external.py:finalize_structure` builds the final config class with one field
per gain term; `internal.py:gains_to_chain` turns the config into instantiated Gain
objects. Validation hooks live in `config_classes.py`.

### Dask specifics

- `utils/dask.py:Blocker` builds custom blockwise HighLevelGraphs — this is how
  `calibration.constructor:construct_solver` wires `solver.py:solver_wrapper` over
  chunks with per-term time/freq interval mappings (`calibration/mapping.py`).
- `quartical/scheduling/` is a distributed `SchedulerPlugin` (`AutoRestrictor`) that
  pins task subtrees to workers to reduce data movement.

### Supporting modules

- `data_handling/` — MS I/O, chunking, selection, parallactic angles, BDA.
- `flagging/` — flag init/propagation and MAD-based flagging kernels.
- `weights/` — weight init and robust (residual-based) reweighting.
- `interpolation/` — load previously solved gains and interpolate onto the current grid.
- `statistics/` — per-chunk chi-squared tracking and post-solve logging.
- `apps/` — the auxiliary CLI entry points (backup/restore, summary, plotter).

## LLM wiki

`docs/wiki/` holds deep internal documentation written for LLM consumption, structured as
an Open Knowledge Format (OKF v0.1) bundle — read the relevant page BEFORE reading source
in that area. `docs/wiki/index.md` maps pages to tasks; highlights:

- Solver/gains/calibration work → `docs/wiki/solver-architecture.md`
- Graph construction, scheduling, performance → `docs/wiki/dask-machinery.md`
- Radio-interferometry or optimisation background → `docs/wiki/domain-primer.md`
- Before proposing structural changes → `docs/wiki/design-decisions.md`

**Maintenance rule (update-as-you-touch):** if a change you make invalidates or extends
a wiki page, update that page in the same session and refresh its frontmatter
`last_verified_commit` (`git rev-parse --short HEAD`) and `timestamp` (date). If you work
in a subsystem whose page is a stub, fill in what you verified while it is fresh in
context (and change its `type` from `stub`). New design decisions get an entry in
`design-decisions.md`.
