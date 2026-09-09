---
type: reference
title: Config System
description: "How YAML schemas become runtime dataclasses with dynamic per-term sections, in what order sources merge, and where per-term options are validated and consumed."
timestamp: 2026-09-09
last_verified_commit: 2b4e420
---

# Config System

Two YAML files are the single source of truth for every option QuartiCal accepts:
`quartical/config/argument_schema.yaml` (fixed top-level sections — `input_ms`,
`input_model`, `solver`, `output`, `mad_flags`, `dask`) and
`quartical/config/gain_schema.yaml` (the options one gain term accepts, under a single
`gain:` key). Nothing else enumerates options: the CLI help, the stimela schema, and the
user-facing `docs/source/options.rst` page (jinja over the schemas) are all generated from
them, so adding an option means editing one of these files and nothing else in the docs.

Each entry is a `scabha.cargo.Parameter` mapping: `dtype` (a type *string*, e.g. `bool`,
`str`, `List[int]`, `Optional[str]`), an optional `default`, an optional `choices` or
`element_choices`, and `info` (the help text, which becomes the option's docs entry
verbatim).

## Schema to dataclass

`quartical/config/__init__.py` runs once at first import and builds two things:

- `BaseConfig` — `scabha.schema_utils.nested_schema_to_dataclass(base_schema, ...)` over
  `argument_schema.yaml`, one nested dataclass per top-level section, each section class
  based on `config_classes.py:BaseConfigSection` and given the matching post-init from
  `POST_INIT_MAP`.
- `Gain` — `schema_to_dataclass(gain_schema, ...)` over `gain_schema.yaml`'s `gain` key,
  with `POST_INIT_MAP['gain']` as its post-init. The gain schema is first merged onto a
  `_GainSchema` dataclass so its entries type as `Parameter`; it is loaded separately from
  the base schema precisely because multiple instances of it are needed.

**Name collision worth knowing:** `quartical.config.Gain` is this options dataclass;
`quartical.gains.gain.Gain` is the gain *term* base class. The former is what
`term_opts` is throughout the gain classes.

The per-term sections are dynamic, because their names are the user's own term names.
`external.py:finalize_structure(additional_config)` walks the supplied configs in reverse
to find the last stated `solver.terms` (falling back to `BaseConfig().solver.terms`), then
`make_dataclass`es a `FinalConfig` with one `Gain`-typed field per term name, based on
`BaseConfig`. So `solver.terms=[G,B]` is what makes `G.type` and `B.type` exist at all —
the config class is constructed after the term list is known, not before.
`external.py:make_stimela_schema` mirrors this for stimela by injecting an
`f"{jones}.{key}"` input for every gain-schema key.

## Merge order and error surface

`parser.py:parse_inputs` scans `sys.argv` for `.yaml`/`.yml` arguments, removes them from
argv, and merges:

```
oc.structured(FinalConfig)  <  yaml file 1  <  yaml file 2  <  ...  <  oc.from_cli()
```

Later sources win, so the CLI always beats a config file. A duplicated config file is a
`ValueError`. Both failure modes of the merge are caught and rewritten into user-facing
advice: `ConfigKeyError` becomes "unrecognised parameter" (typo or deprecation) and
`ValidationError` becomes "value not understood" (wrong type — most often a list given
unbracketed).

`bypass_sysargv` exists for tests and **skips the CLI layer entirely**
(`cli_config = [] if bypass_sysargv else [oc.from_cli()]`). A dotlist passed through it is
used only for the config-file scan; it will *not* override a value. Tests that need an
override either mutate the returned object (`_opts.G.type = ...`, the common pattern in
`testing/tests/gains/`) or set `sys.argv` and call `parse_inputs()` with no bypass.

Finally `oc.to_object(config)` converts to real dataclass instances — which is what runs
the post-inits — followed by `internal.py:additional_validation`.

## Validation, in two places

Per-section validation lives in `config_classes.py:POST_INIT_MAP`, one function per
section, and runs at `to_object` time. Every one calls `__validate_choices__` and
`__validate_element_choices__` (both provided by `BaseConfigSection`, which also supplies
`__helpstr__` for the CLI help). Beyond that they do section-specific work;
`__gain_post_init__` is the one that matters for a per-term option:

- converts `time_interval`/`freq_interval` through `converters.py:as_time`/`as_freq`, which
  is why those two are declared `str` in the schema but arrive as `int` (bare integrations)
  or `float` (a value with a unit suffix);
- rejects `crosshand_phase` with `solve_per != "array"`;
- tuples `pinned_directions`.

Cross-section validation lives in `internal.py:additional_validation`, which needs the
whole config: the output gain store must not already exist unless `output.overwrite`, no
term's `load_from` may sit inside the output directory, and `mad_flags.whitening=robust`
requires `solver.robust`.

**A converter which cannot run at post-init time:** `solver.reference_antenna` is declared
`Union[int, str]` — an int is an antenna index, a str is an antenna name — but
`converters.py:as_antenna_index` needs the antenna names, which only exist once the MS has
been read. It therefore runs in `calibration/calibrate.py:add_calibration_graph`, which
resolves the value to an index and rebuilds the section with `dataclasses.replace`; the
config object itself is never mutated, so the resolution is idempotent across the repeated
`add_calibration_graph` calls the module-scoped test fixtures make. The solver kernels only
ever see the integer index.

The union is what keeps the two selections apart, and it survives every layer: YAML
distinguishes `5` from `"5"`, `oc.from_cli()` distinguishes `ref=5` from `ref='"5"'` (the
shell strips one level of quotes, so naive `ref="5"` is an int), and a direct
`_opts.solver.reference_antenna = 0` in a test fixture is an int by construction. This is
why the option is a union rather than a `str` carrying a disambiguating prefix. Only values
which OmegaConf's grammar reads as a non-str scalar need quoting — integers, and the
boolean/null literals, which fail loudly rather than silently. `Union` in a schema `dtype`
works because `scabha/cargo.py` evaluates the string against `vars(typing)`, and because
`pyproject.toml` pins `omegaconf>=2.3.0`.

## Config to gain terms

`internal.py:gains_to_chain(opts)` is the whole bridge:

```python
chain = [TERM_TYPES[getattr(opts, t).type](t, getattr(opts, t)) for t in terms]
```

— one instantiated term per name in `solver.terms`, its class selected by that section's
`type`, constructed with `(term_name, term_opts)`. `gains/gain.py:Gain.__init__` then
copies the options it needs onto the term object by hand. That copy is explicit and
exhaustive: an option added to `gain_schema.yaml` reaches `term_opts` automatically but
reaches the term object only if `Gain.__init__` (or a subclass) assigns it.

**Adding a per-term option, end to end:** add the entry to `gain_schema.yaml`; assign it in
`gains/gain.py:Gain.__init__`; consume it wherever it acts. If a solver kernel needs it, it
travels as a field of `calibration/solver.py:meta_args_nt`, which is built per term per
iteration from the term object's attributes — see the `referenced` option for a worked
example, including the constraint that
`gains/crosshand_phase/null_v_kernel.py` reconstructs that namedtuple field by field and so
has to be updated alongside it. Nothing needs to be added to the CLI, the help output, or
`docs/source/options.rst`.

`testing/tests/gains/test_init_term.py` builds `term_opts` as a hand-written
`SimpleNamespace` rather than through the config machinery, so a newly required attribute
has to be added there too.
