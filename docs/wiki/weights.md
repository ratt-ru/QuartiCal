# Weights

> **Purpose:** How QuartiCal initialises visibility weights and performs robust
> reweighting.
> **Last verified:** 602d584, 2026-07-07

**STUB** — scope defined, content pending. Fill in when working in this subsystem
(see maintenance rule in CLAUDE.md). Do not fabricate content to fill this page;
document only what you have verified in source.

## Intended scope

- Weight initialisation from MS columns (`quartical/weights/`).
- Robust (residual-based) reweighting: algorithm, when it runs, config knobs.
- How weights flow into the solver kernels and interact with flags.
