---
okf_version: "0.1"
type: index
title: LLM Wiki Index
description: "Map of the QuartiCal LLM wiki — which page answers which task, plus the writing and verification conventions every page follows."
timestamp: 2026-08-26
last_verified_commit: 2cc9557
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
  chain collapsing, numba factory pattern. *Read when:* working on anything under
  `quartical/gains/` or `quartical/calibration/`.
- [dask-machinery.md](dask-machinery.md) — chunking model, Blocker, single-compute design,
  AutoRestrictor, dependency pins. *Read when:* working on graph construction, scheduling,
  or performance.
- [design-decisions.md](design-decisions.md) — decision ledger (context, decision,
  rationale, consequences) plus known debt and recurring gotchas. *Read when:* before
  proposing structural changes or "fixing" something surprising.
- [config-system.md](config-system.md) — YAML schemas to runtime dataclasses, the
  dynamic per-term sections, merge order, validation, and how an option reaches a gain
  term. *Read when:* working under `quartical/config/` or adding an option.

## Stubs

- [data-handling.md](data-handling.md) — MS I/O via dask-ms, chunking, selection,
  parallactic angles, BDA. *Read when:* working under `quartical/data_handling/`.
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

## Writing conventions

The reader is an agent with no other context, so pages are written for retrieval rather
than for narrative:

- Dense factual prose. Facts over story, no marketing language, no restating what the
  page is about to say.
- Real symbol names and `path/to/file.py` anchors instead of vague description — a claim
  the reader cannot locate in source is a claim they cannot check.
- Invariants and non-obvious behaviour stated outright, including the reason. A page
  earns its keep on what is expensive to re-derive from source, not on what is
  rediscoverable by reading one function.
- A stub carries its scope statement and nothing else. Never fabricate content to fill
  one out.

A page is verified by checking every claim against source, and can be tested by handing
a fresh agent nothing but `CLAUDE.md` and the single page, then asking it the questions
the page exists to answer. Wrong answers are page defects.

There is deliberately no tooling here: no doc generation, no CI staleness check, no
Sphinx integration (the user-facing docs under `docs/source/` are separate and
untouched). The `last_verified_commit` stamp makes staleness visible and the
update-as-you-touch rule in CLAUDE.md is what keeps it honest.
