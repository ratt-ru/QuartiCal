# Flagging

> **Purpose:** Data flagging (init, propagation, MAD) and its relationship to
> solver-internal gain flagging.
> **Last verified:** 602d584, 2026-07-07

**STUB** — scope defined, content pending. Fill in when working in this subsystem
(see maintenance rule in CLAUDE.md). Do not fabricate content to fill this page;
document only what you have verified in source.

## Intended scope

- Flag initialisation and propagation in `quartical/flagging/`.
- MAD-based flagging kernels: algorithm, thresholds, config.
- Distinction between data flags and gain flags (`quartical/gains/general/flagging.py`).
- How flags round-trip to the MS on write.
