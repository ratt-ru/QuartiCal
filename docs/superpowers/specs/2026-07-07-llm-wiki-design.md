# LLM Wiki ("Second Brain") for QuartiCal — Design

**Date:** 2026-07-07
**Status:** Approved pending spec review
**Branch:** `llm-wiki` (cut from `v0.2.8-dev`; not merged to dev until ready)

## Purpose

A committed, in-repo wiki of markdown pages capturing the deep knowledge about
QuartiCal that is expensive or impossible to re-derive from source: internal
architecture, domain/maths background, design decisions, and task recipes. The
primary reader is an LLM agent (Claude Code sessions); humans are a secondary
audience. The goal is that a fresh session can load one wiki page instead of
spelunking the codebase.

## Non-goals (YAGNI)

- No doc-generation tooling, CI checks, or auto-staleness detection.
- No Sphinx integration; the existing `docs/source/*.rst` user docs are untouched.
- No attempt at full subsystem coverage upfront — stubs fill in organically.

## Layout

```
docs/wiki/
├── INDEX.md              # one line per page: what it covers, when to read it
├── domain-primer.md      # core page (written now)
├── solver-architecture.md# core page (written now)
├── dask-machinery.md     # core page (written now)
├── design-decisions.md   # core page (written now, seeded via user interview)
├── weights.md            # stub
├── flagging.md           # stub
├── interpolation.md      # stub
├── data-handling.md      # stub
├── config-system.md      # stub
├── apps.md               # stub
└── statistics.md         # stub
```

## Page format and writing conventions

Every page begins with a header block:

- One-line purpose statement.
- `Last verified: <short-hash>, <YYYY-MM-DD>` — the commit against which the
  page's claims were last checked.

Body conventions (LLM-first writing):

- Dense factual prose; facts over narrative, no marketing language.
- Real symbol names and `path/to/file.py` anchors instead of vague descriptions.
- Invariants and non-obvious behaviour stated explicitly.
- Stubs contain only the header plus a scope statement of what the page should
  eventually cover — nothing fabricated.

## Core page content

### `domain-primer.md`

The knowledge the code assumes but never states: the RIME and where QuartiCal
sits in it; Jones chains and term ordering; gain solving as a complex NLLS
problem; direction-dependent vs -independent terms; parameterised terms
(delay/phase/TEC/rotation) and why they solve in parameter space; vocabulary
mapping ("term" ↔ config section ↔ `Gain` subclass ↔ zarr dataset).

### `solver-architecture.md`

The gains/calibration core as a system: the `TERM_TYPES` registry and what a
term package must provide; `Gain` vs `ParameterizedGain` contracts; the
`add_calibration_graph` → `construct_solver` → `solver_wrapper` fan-out over
chunks; time/freq interval mappings; the numba factory pattern in
`quartical/gains/general/factories.py` and why it exists; kernel structure and
flagging-machinery hooks.

### `dask-machinery.md`

How QuartiCal drives dask: the chunking model and per-term interval mappings;
`Blocker` and why plain `blockwise` was insufficient; the single-`compute()`
design and `compute_context`; the `AutoRestrictor` scheduler plugin; the
dependency-pin rationale (`dask<=2024.10.0`, `bokeh<3.7`) in more depth than
the pyproject comments.

### `design-decisions.md`

A ledger, one entry per decision: context, decision, rationale, consequences.
Seeded from code comments, git history, and a brief interview with the user for
decisions only they know (CubiCal lessons, rejected alternatives). Grows as new
decisions land.

## CLAUDE.md wiring

A new section in the project `CLAUDE.md`:

- A pointer table mirroring `INDEX.md`: "working on solver internals → read
  `docs/wiki/solver-architecture.md` first".
- The maintenance rule (update-as-you-touch): if a change invalidates or
  extends a wiki page, update that page in the same session and refresh its
  `Last verified` stamp; when touching a subsystem whose page is a stub,
  consider filling in what was learned.

## Build order

1. Skeleton commit: `docs/wiki/` with `INDEX.md`, all stubs, CLAUDE.md wiring.
2. One commit per core page, in order: `solver-architecture.md`,
   `dask-machinery.md`, `domain-primer.md`, `design-decisions.md` (last, since
   it needs the user interview).
3. Per-page process: draft from actual source reading → self-verify every
   factual claim (paths, symbols, behaviour) against the code → present to the
   user for review → revise → commit. Pages go to the user one at a time.

## Acceptance test

Spawn a fresh agent whose only QuartiCal context is `CLAUDE.md` plus one wiki
page, and quiz it on questions the page should answer (e.g. "where do
time/freq interval mappings get built and consumed?", "what must a new gain
type provide?"). Answers must be correct against the actual source. Applied to
each core page before it is considered done.

## Maintenance

Update-as-you-touch (the CLAUDE.md rule) is the whole system. `Last verified`
stamps make staleness detectable; no other infrastructure.
