# Data Handling

> **Purpose:** Measurement Set I/O, chunking, selection, and derived quantities.
> **Last verified:** 602d584, 2026-07-07

**STUB** — scope defined, content pending. Fill in when working in this subsystem
(see maintenance rule in CLAUDE.md). Do not fabricate content to fill this page;
document only what you have verified in source.

## Intended scope

- `read_xds_list`/`write_xds_list` in `quartical/data_handling/ms_handler.py`; dask-ms
  usage.
- Chunking strategy: how config chunk specs become dask chunks.
- Selection (field/ddid/scan), weight/flag preprocessing.
- Parallactic angle machinery and BDA support.
