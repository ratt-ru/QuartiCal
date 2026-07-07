---
okf_version: "0.1"
---

# QuartiCal LLM Wiki — Index

Deep internal knowledge for LLM agents (and humans) working on QuartiCal. Read the
matching page *before* spelunking source. Pages typed `stub` have a defined scope but no
content yet — fill them in as you work in that subsystem (see the maintenance rule in
CLAUDE.md). This bundle follows the Open Knowledge Format v0.1: every page carries YAML
frontmatter (`type`, `title`, `description`, `timestamp`, plus `last_verified_commit`, the
commit against which the page's claims were last checked).

## Core pages

- [domain-primer.md](domain-primer.md) — RIME, Jones chains, gain solving as complex NLLS,
  parameterised terms, vocabulary. *Read when:* you need the radio-interferometry or
  optimisation background the code assumes.
- [solver-architecture.md](solver-architecture.md) — TERM_TYPES registry,
  Gain/ParameterizedGain contracts, calibration graph construction, interval mappings,
  numba factory pattern. *Read when:* working on anything under `quartical/gains/` or
  `quartical/calibration/`.
- [dask-machinery.md](dask-machinery.md) — chunking model, Blocker, single-compute design,
  AutoRestrictor, dependency pins. *Read when:* working on graph construction, scheduling,
  or performance.
- [design-decisions.md](design-decisions.md) — decision ledger (context, decision,
  rationale, consequences) plus known debt and recurring gotchas. *Read when:* before
  proposing structural changes or "fixing" something surprising.

## Stubs

- [data-handling.md](data-handling.md) — MS I/O via dask-ms, chunking, selection,
  parallactic angles, BDA. *Read when:* working under `quartical/data_handling/`.
- [config-system.md](config-system.md) — YAML schemas → runtime dataclasses, per-term
  sections, validation. *Read when:* working under `quartical/config/`.
- [weights.md](weights.md) — weight initialisation and robust reweighting. *Read when:*
  working under `quartical/weights/`.
- [flagging.md](flagging.md) — flag init/propagation, MAD flagging, gain flagging vs data
  flagging. *Read when:* working under `quartical/flagging/` or
  `quartical/gains/general/flagging.py`.
- [interpolation.md](interpolation.md) — loading solved gains, interpolating onto new
  grids. *Read when:* working under `quartical/interpolation/`.
- [statistics.md](statistics.md) — chi-squared tracking and post-solve logging. *Read
  when:* working under `quartical/statistics/`.
- [apps.md](apps.md) — backup/restore, summary, plotter CLIs. *Read when:* working under
  `quartical/apps/`.
