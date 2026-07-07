# Config System

> **Purpose:** How YAML schemas become runtime dataclasses with dynamic per-term
> sections.
> **Last verified:** 602d584, 2026-07-07

**STUB** — scope defined, content pending. Fill in when working in this subsystem
(see maintenance rule in CLAUDE.md). Do not fabricate content to fill this page;
document only what you have verified in source.

## Intended scope

- `argument_schema.yaml` and `gain_schema.yaml` structure.
- `quartical/config/external.py:finalize_structure` — dynamic config class
  construction.
- `quartical/config/internal.py:gains_to_chain` — config → Gain instances.
- Validation hooks in `config_classes.py`; OmegaConf merge order (YAML files vs CLI).
