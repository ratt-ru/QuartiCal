# QuartiCal LLM Wiki — Index

Deep internal knowledge for LLM agents (and humans) working on QuartiCal. Read the
matching page *before* spelunking source. Pages marked (stub) have a defined scope but
no content yet — fill them in as you work in that subsystem (see the maintenance rule
in CLAUDE.md).

| Page | Covers | Read when |
| --- | --- | --- |
| [domain-primer.md](domain-primer.md) | RIME, Jones chains, gain solving as complex NLLS, parameterised terms, vocabulary | You need the radio-interferometry/optimisation background the code assumes |
| [solver-architecture.md](solver-architecture.md) | TERM_TYPES registry, Gain/ParameterizedGain contracts, calibration graph construction, interval mappings, numba factory pattern | Working on anything under `quartical/gains/` or `quartical/calibration/` |
| [dask-machinery.md](dask-machinery.md) | Chunking model, Blocker, single-compute design, AutoRestrictor, dependency pins | Working on graph construction, scheduling, or performance |
| [design-decisions.md](design-decisions.md) | Decision ledger: context, decision, rationale, consequences | Before proposing structural changes or "fixing" something surprising |
| [data-handling.md](data-handling.md) | (stub) MS I/O via dask-ms, chunking, selection, parallactic angles, BDA | Working under `quartical/data_handling/` |
| [config-system.md](config-system.md) | (stub) YAML schemas → runtime dataclasses, per-term sections, validation | Working under `quartical/config/` |
| [weights.md](weights.md) | (stub) Weight initialisation and robust reweighting | Working under `quartical/weights/` |
| [flagging.md](flagging.md) | (stub) Flag init/propagation, MAD flagging, gain flagging vs data flagging | Working under `quartical/flagging/` or `quartical/gains/general/flagging.py` |
| [interpolation.md](interpolation.md) | (stub) Loading solved gains, interpolating onto new grids | Working under `quartical/interpolation/` |
| [statistics.md](statistics.md) | (stub) Chi-squared tracking and post-solve logging | Working under `quartical/statistics/` |
| [apps.md](apps.md) | (stub) backup/restore, summary, plotter CLIs | Working under `quartical/apps/` |
