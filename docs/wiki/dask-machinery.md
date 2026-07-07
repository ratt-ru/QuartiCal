---
type: architecture
title: Dask Machinery
description: "How QuartiCal builds and executes its dask graphs — chunking, Blocker, single-compute, AutoRestrictor, and why the dask/bokeh pins exist."
timestamp: 2026-07-07
last_verified_commit: 50207c9
---

# Dask Machinery

The published description of this machinery is Kenyon et al. 2025, "Africanus II. QuartiCal"
(Astronomy and Computing 52, 100962; arXiv:2412.10072) — Section 4.2 for graph construction
(including Blocker) and Section 4.5 for distributed execution and the scheduler plugin.

## Chunking model

A "chunk" is one `(row, chan)` block of a data xarray.Dataset. QuartiCal reads the MS into a
list of datasets (`quartical/data_handling/ms_handler.py:read_xds_list`) whose `DATA` and `WEIGHT`
variables are dask arrays with dims `("row", "chan", "corr")`, while `FLAG` has dims `("row", "chan")`
— it has already collapsed the correlation axis. Only `row` and `chan` are chunked; `corr` is always
a single chunk. The solver runs one task per `(row, chan)` block.

Chunk sizes are decided in `quartical/data_handling/chunking.py:compute_chunking`, driven by the
config `input_ms.time_chunk` and `input_ms.freq_chunk` values. The chunk spec is interpreted by
type: an **int** `time_chunk` means a number of unique timeslots per chunk
(`row_chunking:integer_chunking`), while a **float** means a duration in seconds accumulated from
the `INTERVAL` column (`row_chunking:interval_chunking`); an **int** `freq_chunk` is a channel
count and a **float** is a bandwidth in Hz accumulated from `CHAN_WIDTH`
(`chan_chunking:integer_chunking`/`interval_chunking`). `compute_chunking` computes eagerly
(`compute=True` → `da.compute`) because the row-chunk boundaries must be known before the data
columns are opened; it returns `utime_chunking_per_data_xds` (unique-time counts per chunk),
`chunking_per_data_xds` (`{"row": ..., "chan": ...}` fed to `xds_from_storage_ms(chunks=...)`),
and `chunking_per_spw_xds`. Row chunks are always aligned to unique-time boundaries: chunk
lengths come from `np.add.reduceat(ucounts, chunk_starts)`, so all rows of a given timeslot land
in the same chunk.

**Invariant — solution intervals never straddle chunk boundaries.** This holds by construction,
not by an explicit guard. Each gain term's solution grid is built per data block:
`quartical/gains/gain.py:make_time_bins` calls `_make_time_bins` via `da.map_blocks` with
`chunks=(data_xds.UTIME_CHUNKS,)`, and `_make_time_bins` numbers its bins from `0` within each
block (see the `bin_num` accumulator). The gain array's own time chunking is then
`make_time_chunks`/`_make_time_chunks`, which returns `time_bins.max() + 1` **per block** — i.e.
one gain-time chunk per data-time chunk, sized to the number of solution intervals found inside
that data chunk (`quartical/gains/datasets.py:scaffold_from_data_xds`,
`make_gain_xds_lod`). Because binning restarts at each block boundary, a solution interval cannot
span two data chunks; the coarsest a solve can get is "one interval per chunk" (a `time_interval`
larger than the chunk). The same construction applies in frequency (`make_freq_map`/
`make_freq_chunks`). `respect_scan_boundaries` additionally forces a new bin at scan edges within
a block. See [solver-architecture.md](solver-architecture.md) for the full interval-mapping
semantics.

Consequence: to solve on a coarser grid you make solution intervals larger, not chunks smaller;
chunk size is a memory/parallelism knob, and each chunk is solved independently.

## Blocker

The solver task is a many-input/many-output blockwise operation: `solver_wrapper` consumes MS
columns, several interval-mapping arrays, per-chunk spec objects and scalar config, and returns a
**dictionary** with one entry per output (`weights` dims `("row","chan","corr")`;
`flags` dims `("row","chan")`; `presolve_chisq` and `postsolve_chisq` nominally
`("row","chan")` but one scalar value per chunk — they land on the stats datasets as
`("t_chunk","f_chunk")`;
per-term `<name>_gains`, `<name>_gain_flags`, `<name>_jhj`, `<name>_conviter`, `<name>_convperc`,
and the `param` variants).
Plain `dask.array.blockwise` is awkward for this shape: it addresses each output by positional
index and expects a fixed, tuple-shaped return, so a variable-length dict of heterogeneously-shaped
outputs would be error-prone. QuartiCal's `Blocker` (`quartical/utils/dask.py`) exists to remove
that reliance on output index — its docstring states the called function "is expected to return a
dictionary of outputs" precisely "to avoid error-prone reliance on output index."

`Blocker` API (all in `quartical/utils/dask.py`):

- `Blocker(func, index_string)` — `func` is applied per block; `index_string` names the **chunked
  output axes** and is stored as `self.func_axes = list(index_string)`. The solver passes the
  tuple `("row", "chan")` (`construct_solver`), so `func_axes == ["row", "chan"]`.
- `add_input(name, value, index_string=None)` — register a kwarg. `value` may be a `da.Array`
  (its `numblocks` are recorded against the axes in `index_string`), a nested list (a "list of
  lists" whose lengths are measured with `_len_at_depth`), or, when `index_string` is `None`, an
  unblocked scalar passed through verbatim. For dask-array inputs it calls `_check_axis`, which
  raises `ValueError` if any axis **not** in `func_axes` (a contraction axis) has more than one
  chunk. This is the invariant that forces antenna/direction/correlation to be single-chunk so one
  solver task sees a whole slice.
- `add_output(name, index_string, chunks, dtype)` — declare an output array: its axes, its
  `chunks` tuple and `dtype`. Outputs are created strictly from these specs, independent of the
  `index_string` given to the constructor.
- `get_dask_outputs()` — materialises the graph and returns `{name: da.Array}`.

`get_dask_outputs` builds a `HighLevelGraph` by hand. It tokenizes all input values into a stable
`layer_name`, then for every point in the cartesian product of the chunked axes emits one task
`(apply, self.func, [], (dict, kwargs))` — every argument is passed as a **kwarg**, so argument
order is irrelevant and adding inputs is cheap. Each declared output gets its own
`MaterializedLayer` whose tasks are `(getitem, <func-task-key>, o.name)`, pulling that output out
of the per-chunk result dict. The input arrays' existing layers and dependencies are folded in
(`inp.__dask_graph__().layers`), the whole thing is assembled into one `HighLevelGraph`, and each
output is wrapped as a `da.Array(hlg, name=..., chunks=o.chunks, dtype=o.dtype)`.

`construct_solver` (`quartical/calibration/constructor.py`) is the primary consumer: per data xds
it creates `Blocker(solver_wrapper, ("row", "chan"))`, adds the required MS columns, mapping arrays
(time-dimensioned inputs relabelled to `row`), the per-chunk `term_spec_list`, and scalar config,
declares the outputs listed above, and calls `get_dask_outputs()`. `blockwise_unique`
(`quartical/utils/dask.py`, alongside `Blocker`) is a second, self-contained consumer. See [solver-architecture.md](solver-architecture.md) for how
the outputs are assigned back onto the gain/data/stats datasets.

## Single-compute design

The entire pipeline is assembled lazily. `quartical/executor.py:_execute` builds the read graph,
model graph, calibration graph (`add_calibration_graph`), flag/postprocess graphs and the MS/gain
write graphs, then triggers exactly **one** materialising `dask.compute()` for the solve at
`quartical/executor.py:179`:

```python
with compute_context(dask_opts, output_opts, time_str):
    _, _, stats_xds_list, _ = dask.compute(
        ms_writes, gain_writes, stats_xds_list, bl_corr_writes,
        num_workers=dask_opts.threads,
        optimize_graph=True,
        scheduler=dask_opts.scheduler,
    )
```

Bundling the MS writes, gain writes, stats and baseline-correction writes into a single
`dask.compute` call means shared intermediates (read visibilities, residuals, gains) are computed
once and reused across all outputs — the MS is not read twice to produce both the written columns
and the gain datasets. (Nuance: this is the only compute of the *solve* graph, but not literally
the only `da.compute` in the process — `compute_chunking` and `make_gain_xds_lod` each run a small
eager compute earlier to reify chunk sizes and gain-dataset scaffolds before the main graph is
built.)

`compute_context` (`quartical/utils/dask.py`) is a context manager that switches on
`dask_opts.scheduler`. If it equals `"distributed"` it opens a
`dask.distributed.performance_report(filename=...)` that writes an HTML report (path derived from
`output_opts.log_directory` via `DaskMSStore`) when the block exits; otherwise it returns a
`contextlib.nullcontext()` (no-op). The `scheduler=` kwarg passed to `dask.compute` is
`dask_opts.scheduler` itself — so a non-distributed run uses dask's local thread scheduler with
`num_workers=dask_opts.threads`, and a distributed run dispatches to the `Client`/`LocalCluster`
set up earlier in `_execute`.

## AutoRestrictor

`AutoRestrictor` (`quartical/scheduling/__init__.py`) is a `distributed.SchedulerPlugin` that
reduces inter-worker data movement on the distributed scheduler. QuartiCal's chunks are
independent — each `(row, chan)` block's solve, residual and write form a self-contained subtree —
so if each such subtree is pinned to a single worker there is no need to shuffle intermediates
between workers. `AutoRestrictor` finds those subtrees and pins them.

It is opt-in and distributed-only: `_execute` installs it via `client.run_on_scheduler(
install_plugin)` only when `dask_opts.scheduler == "distributed"` **and**
`dask_opts.scheduler_plugin` is set (`install_plugin` calls
`dask_scheduler.add_plugin(AutoRestrictor(**kwargs), idempotent=True)`).

Mechanism — it implements the plugin's `update_graph(self, scheduler, dsk, keys, restrictions,
**kw)` hook, which the scheduler calls as each graph is submitted. Inside it:

- Derives `dependents = reverse_dict(dependencies)` and computes `ndependencies` and per-node
  `graph_metrics` (a function **vendored from an older `dask/order.py`** — see the note below).
- Identifies **root nodes** (no dependencies) and **terminal/partition nodes** (no dependents),
  and computes each task's depth from the roots (`get_node_depths`/`unravel_deps`, a recursive
  dependency walk). If there are fewer partition nodes than workers (a reduction), it walks back
  up the graph to a depth with enough independent work to fill every worker.
- Tokenizes each partition node's set of root ancestors, groups partition nodes that share (or are
  subsets of) the same roots into `task_groups`, and assigns each group to the currently
  least-loaded worker (`min(worker_loads, key=worker_loads.get)`). For every task in the group it
  sets `task.worker_restrictions |= {assignee}` and `task.loose_restrictions = False` (a **hard**
  restriction).

Limitations, per the code's own comments and structure: root-set matching is by exact tokenization
(`# This is very strict. What about nodes with very similar roots?`), so subtrees with only
partially-overlapping roots are not merged; it early-`return`s and falls back to default scheduling
if it cannot find a workable depth (`if max_depth <= 0: return`); load balancing is by task
**count** per group, not by data size or runtime; and installing it via `run_on_scheduler` is
flagged in `_execute` as "controversial from a security POV" since that is really a debugging
entry point (the documented alternative being `dask-scheduler --preload`).

## Dependency pins

The authoritative source for these is the inline comments in `pyproject.toml`; `git log` on that
file only surfaces squashed merge commits (`#416`, `#349`), so the root-cause detail below is
attributed to those comments rather than to per-pin history.

- **`dask[diagnostics]>=2023.5.0,<=2024.10.0`** (and matching `distributed>=2023.5.0,<=2024.10.0`).
  Symptom when unpinned: dask ≥ 2024.11 breaks dask-ms read graphs. Root cause per the comment:
  dask 2024.12.0 introduced a `fuse_linear_task_spec` array optimizer that fails on dask-ms read
  graphs (specifically the `clone()[0]` pattern used in `ms_handler`, e.g. the `clone(
  spw_xds.CHAN_FREQ.data)` calls in `read_xds_list`), and 2024.11.x is incompatible with dask-ms's
  task inlining; the comment points at the failing graph fusion around `make_gain_xds_lod`. To lift
  it, dask-ms's read graphs (or the `clone`-based indexing) would need to survive the newer
  optimizers, or QuartiCal would have to disable those optimizers for its read graphs. The precise
  upstream fix is not documented in-repo.
- **`bokeh>=2.4.2,<3.7`**. Symptom when unpinned: `compute_context`'s `performance_report` raises
  `TemplateNotFound('file.html')` when it writes its HTML report on exit (distributed-scheduler
  runs only). Root cause per the comment: bokeh 3.7.0 renamed its bundled templates to add a
  `.jinja` suffix (`file.html` → `file.html.jinja`), but distributed 2024.10.0's
  `performance_report` template still does `{% extends "file.html" %}`. Lifting this requires a
  distributed release whose `performance_report.html` references the renamed template — which is
  itself coupled to the dask pin above, since `distributed` is capped at the same `2024.10.0`.
- **`stimela>=2.1.4` — must never get an upper bound.** The comment is explicit: `# Do not include
  an upper bound on stimela.` No root cause is recorded in the file; the standing instruction is a
  deliberate policy (stimela is an orchestration front-end and QuartiCal must remain installable
  alongside whatever current stimela a user has), so no upper bound should be added.

Related self-imposed workaround (not a `pyproject.toml` pin but part of the same dask-version
story): `quartical/scheduling/__init__.py` vendors `graph_metrics` verbatim from an old
`dask/order.py` commit because the function "was removed upstream but is currently critical for the
scheduler plugin" — a temporary measure until the plugin is reworked. Bumping dask would not
restore that symbol, so the copy must stay until `AutoRestrictor` is rewritten.
