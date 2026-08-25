---
type: stub
title: Config System
description: "How YAML schemas become runtime dataclasses with dynamic per-term sections."
timestamp: 2026-07-07
last_verified_commit: 602d584
---

# Config System

**STUB** — scope defined, content pending. Fill in when working in this subsystem
(see maintenance rule in CLAUDE.md). Do not fabricate content to fill this page;
document only what you have verified in source.

## Intended scope

- `argument_schema.yaml` and `gain_schema.yaml` structure.
- `quartical/config/external.py:finalize_structure` — dynamic config class
  construction.
- `quartical/config/internal.py:gains_to_chain` — config → Gain instances.
- Validation hooks in `config_classes.py`; OmegaConf merge order (YAML files vs CLI).
