---
type: decision-ledger
title: Design Decisions
description: "Why QuartiCal is built the way it is — a ledger of decisions, their rationale, and their consequences. Append new entries as decisions land."
timestamp: 2026-09-09
last_verified_commit: f2ac75c
---

# Design Decisions

Entries follow a fixed shape (Context / Decision / Rationale / Consequences / Source) so a
reader can judge whether a decision's premises still hold before "fixing" what it produced.
Interview entries record the lead developer's testimony (2026-07-07); paper citations refer
to Kenyon et al. 2025, "Africanus II. QuartiCal" (Astronomy and Computing 52, 100962;
arXiv:2412.10072). The [Known debt](#known-debt-do-not-entrench) and
[Recurring gotchas](#recurring-gotchas) sections at the end are not decisions but belong
here: they mark what should *not* be entrenched and what repeatedly bites contributors.

## Lessons from CubiCal

- **Context:** QuartiCal succeeds CubiCal, and much of its design is a direct reaction to
  CubiCal's failure modes.
- **Decision:** Treat per-correlation weights as first-class in the formulation; prefer
  functions over stateful objects; ship tests; avoid data layouts and large intermediary
  products that inflate memory; formulate the chain mathematics so parameterised terms can
  appear at arbitrary positions.
- **Rationale:** CubiCal could not incorporate per-correlation weights — the relationship
  between the 2x2 matrix formulation and its 4x4 Mueller representation was not clear at
  the time it was developed. Its extensive OOP backfired: god classes and mutable state
  made changes very difficult. It lacked proper tests, so changes often broke things. To
  keep the implementation numpy-like, it packed data into arrays with a pair of N_ant-sized
  axes (all baselines slot in easily), doubling the memory footprint, and produced many
  large intermediaries. Its mathematics precluded parameterised terms at arbitrary points
  in the Jones chain.
- **Consequences:** QuartiCal's weighted 2x2 formulation, functional task style, test
  suite, memory frugality, and chain-rule parameterisation (see
  [domain-primer.md](domain-primer.md)) all trace back to these lessons.
- **Source:** interview 2026-07-07; Kenyon et al. 2025, Sections 1–2.

## Solution-interval independence as the architectural driver

- **Context:** Calibration data volume grows quadratically with antenna count; QuartiCal
  must scale from a laptop to a cluster — the gap CASA/CubiCal did not fill.
- **Decision:** Formulate calibration so each time/frequency solution interval is solved
  entirely independently of its neighbours, exposing an embarrassingly parallel task graph.
- **Rationale:** The update equations sum over samples inside an interval, so intervals
  share nothing. Chunks are sized to span whole solution intervals precisely so that no
  inter-chunk communication is ever needed.
- **Consequences:** Chunk size is a memory/parallelism knob, not a correctness knob;
  solution intervals must not straddle chunk boundaries (enforced by construction — see
  [dask-machinery.md](dask-machinery.md), Chunking model). Chunk sizes therefore place an
  upper limit on solution intervals (see Recurring gotchas).
- **Source:** Kenyon et al. 2025, Sections 2.3 and 4.1.

## AllJones diagonal approximation of JᴴWJ

- **Context:** An unmodified Gauss–Newton update is expensive: large matrix products plus
  the inverse of JᴴWJ.
- **Decision:** Adopt the most extreme approximation from Smirnov & Tasse (2015),
  "AllJones": discard all off-diagonal entries of JᴴWJ.
- **Rationale:** Reduces per-iteration cost — only the left half of J need be considered,
  no large matrix is ever explicitly constructed, and the diagonal is guaranteed
  invertible.
- **Consequences:** Trades per-iteration accuracy for cheaper iterations. The maths is
  summarised in [domain-primer.md](domain-primer.md).
- **Source:** Kenyon et al. 2025, Section 2.3.

## One-term-at-a-time chain updates

- **Context:** QuartiCal solves chains of Jones terms of arbitrary length, some
  parameterised.
- **Decision:** Only ever update a single term in the chain at a time: iterate a term to
  convergence, move to the next, optionally repeat over epochs.
- **Rationale:** Solving all terms jointly is computationally impractical and may exhibit
  degeneracies. The approach offers no convergence guarantee but works empirically.
- **Consequences:** No formal convergence guarantee; multiple `solver.epochs` are sometimes
  required (robust reweighting in particular only triggers across epochs).
- **Source:** Kenyon et al. 2025, Sections 2.2 and 4.3.

## Analytic Jacobian instead of autodiff

- **Context:** The Gauss–Newton update needs the Jacobian; autodiff (e.g. JAX) could
  construct it automatically.
- **Decision:** Derive Jacobian elements analytically via Wirtinger calculus.
- **Rationale:** Autodiff cannot exploit the problem's structure — in particular the
  AllJones diagonal approximation — with meaningful cost to both performance and memory.
- **Consequences:** Hand-derived, hand-maintained kernels; in exchange, the approximations
  above stay available.
- **Source:** Kenyon et al. 2025, Section 2.3.

## Numba kernels (over vectorised numpy, Cython, or multiprocessing)

- **Context:** The solver kernels are the hot loops. The Measurement Set is essentially a
  relational database whose columns may hold arrays — very general, frequently ragged with
  missing data.
- **Decision:** Implement solver kernels as numba-jitted loops.
- **Rationale:** Array-based numpy maths becomes very challenging in the presence of
  missing data; neatly packing an MS into numpy-friendly matrices was one of CubiCal's
  core problems, and not all operations are array-expressible anyway. Numba gives loops at
  C-like speed, releases the GIL (enabling thread-level parallelism where
  CubiCal/DDFacet needed multiprocessing + shared memory), and — as a JIT — makes the
  package easy to distribute with no wheel-building (the unsolved packaging problem that
  ruled out repeating CubiCal's Cython experiment).
- **Consequences:** Nested parallelism (dask threads over numba `prange` threads) with
  near-zero extra memory per core; numba parallelism only applies to code written in
  numba. Aspiration, not yet verified: getting numba to emit vectorised (SIMD) machine
  code.
- **Source:** interview 2026-07-07; Kenyon et al. 2025, Sections 3.4 and 4.3.

## Factory + literal-dispatch kernel pattern

- **Context:** QuartiCal supports 1, 2 and 4 correlations with markedly different maths;
  per-element runtime branching would be slow.
- **Decision:** Build kernel internals from factory functions
  (`quartical/gains/general/factories.py`) that select a closure body at compile time from
  a numba literal, so each correlation count compiles to specialised machine code.
- **Rationale:** An optimisation attempt: the introducing PR was a performance-parity
  exercise, iterating until the correlation-agnostic kernels matched the speed of the old
  hand-specialised code. The lead developer's own caveat: "it is not guaranteed that I
  succeeded."
- **Consequences:** Indirection (factories returning closures) is now the required
  extension pattern for new gain types — see
  [solver-architecture.md](solver-architecture.md), Numba kernel conventions.
- **Source:** interview 2026-07-07; commit 958eb83 (#63); Kenyon et al. 2025, Section 4.3.

## Tuple-based kernel maths in the complex solver (over array-buffer factories)

- **Context:** Benchmarking (2026-07-08, i7-1355U, single P-core, synthetic 28-antenna
  problem) confirmed the long-suspected code-generation problem: the hot
  `compute_jhj_jhr` loop emitted **zero** packed (SIMD) floating-point instructions in
  all correlation modes, and data-movement instructions outnumbered arithmetic ~2:1 —
  per-visibility intermediaries were round-tripping through the small `valloc` array
  buffers instead of staying in registers.
- **Decision:** Rewrite the complex kernel's per-visibility maths on new tuple-based
  factories (`tuple_*` in `quartical/gains/general/factories.py`): tuples are immutable
  SSA values, so intermediaries become register-resident. On top of that: (a) accumulate
  only the upper triangle of each (4, 4) JHJ element and mirror it once per solution
  interval instead of once per visibility; (b) use the Kronecker factorisation
  `jhj[2i+j, 2k+l] = sum_p a_ip conj(a_kp) C_jl^p` with real diagonal factors, roughly
  halving 4-corr JHJ flops; (c) add a `single_dir` fast path which accumulates each row's
  JHJ/JHr in registers and flushes once per row (the antenna pair is fixed along a row).
- **Rationale:** Measured, interleaved A/B benchmarks against the pre-change tree at every
  step; every retained change beat its predecessor. Cumulative single-thread speedup on the
  full solve (20 fixed iterations): **2.8x (1 corr), 2.5x (2 corr), 2.1x (4 corr)**; the
  direction-dependent (general) path also gained 1.6-1.9x. Outputs agree with the old
  kernel to ~1e-15 relative (fastmath reassociation only); `testing/tests/gains/
  test_complex.py` and the calibration suite pass.
- **Consequences:** Two factory styles now coexist (array-buffer and tuple); the rewrite
  was subsequently propagated to every other kernel via the shared accumulation loop
  (see the next entry), so tuple style is now the only accumulation-loop style. The
  chain-product block is duplicated between the fast and general paths inside
  `nb_compute_jhj_jhr` — a numba parfor bug prevents factoring it out (see Recurring
  gotchas: nested tuple returns). The code is still essentially scalar (no packed SIMD in
  the 4-corr hot loop); the remaining ~2x SIMD headroom would require cross-visibility
  vectorisation with flag masking, judged not worth the complexity yet.
- **Source:** benchmark session 2026-07-08 (this entry); design predecessor: commit
  958eb83 (#63).

## Shared hook-parameterised accumulation loop

- **Context:** The tuple-based rewrite of the complex kernel (previous entry) left every
  other solver on the slow array-buffer style, and the `compute_jhj_jhr` loop body was
  near-verbatim duplicated across all of them. Propagating the optimisation kernel-by-
  kernel would have written the same loop ~12 more times.
- **Decision:** ADOPTED (decision gate, 2026-07-13, after the first parameterised
  conversion — phase). The optimised loop lives once in
  `quartical/gains/general/solver_components.py` as `build_jhj_jhr_impl`, parameterised by
  per-term hook factories, passed keyword-only: `accumulate_jhj_jhr_factory` /
  `zero_jhj_jhr_factory` / `flush_jhj_jhr_factory` / `compute_residual_factory` /
  `compute_channel_coeffs_factory` / `mirror_jhj_factory`; each kernel keeps a ~15-line
  `compute_jhj_jhr` overload that binds its hooks. The hook contract is documented in
  solver-architecture.md ("Numba kernel conventions").
- **Rationale:** The gate required >= 1.10x at every supported correlation mode; phase
  measured **2.255x (1 corr), 1.979x (2 corr), 1.723x (4 corr)** (min/min, threads=1,
  full 20-iteration solve, interleaved same-day A/B vs the 9ae814e baseline). Complex
  itself moved onto the loop at parity (pure code motion, bitwise-identical checksums),
  and phase's single-pass jhj/jhr checksums are bit-identical to the old kernel.
- **Consequences:** Remaining kernels convert by writing hooks only — and all of them
  subsequently did: as of ec25545 every solvable kernel binds `build_jhj_jhr_impl`
  (leakage reuses complex's `compute_jhj_jhr` wholesale) and no array-buffer
  accumulation loop remains. Final sweep (2026-07-16, threads=1, single pinned core,
  full 20-iteration solve, interleaved same-day A/B vs the 9ae814e baseline; min/min
  speedups, medians within a few percent throughout):

  | kernel | corr 1 | corr 2 | corr 4 |
  |---|---|---|---|
  | complex (parity re-check) | 0.99 | 0.99 | 0.99 |
  | diag_complex | 2.78 | 2.49 | 1.91 |
  | phase | 2.28 | 2.00 | 1.76 |
  | amplitude | 2.29 | 2.31 | 1.77 |
  | delay | 2.19 | 1.98 | 1.80 |
  | delay_and_offset | 2.11 | 1.85 | 1.90 |
  | tec_and_offset | 2.00 | 1.97 | 1.81 |
  | delay_and_tec | 2.11 | 1.87 | 1.88 |
  | delay_tec_and_offset | 2.27 | 1.81 | 1.83 |
  | crosshand_phase | — | — | 1.58 |
  | crosshand_phase_null_v | — | — | 1.38 |
  | rotation | — | — | 1.88 |
  | rotation_measure | — | — | 1.85 |

  (4-corr-only terms have no 1/2-corr modes; complex's parity row also covers leakage.
  Full raw numbers in `~/claude_artifacts/quaritcal_optimisation/results/
  UNIFICATION_LOG.md`.) The shared loop later required a caching fix — see the
  "Per-kernel numba disk-cache namespaces" entry below. **Behavioural note
  (deliberate, verified):** for multi-direction (DD) solves of parameterised terms, the
  shared loop passes each direction's *own* active-term gain to the elem chain rule,
  whereas the legacy array-buffer kernels passed the *last* direction's gain to every
  direction (a stale per-visibility buffer — latent bug). Single-direction solves are
  numerically unaffected (solved gains/params agree elementwise to <= 4e-10).
  Localisation evidence: temporarily mimicking the stale-gain behaviour in the shared
  loop reproduced the legacy DD output to 1.6e-8 elementwise, with the pinned direction
  exact. Expect legacy-vs-new DD comparisons of parameterised terms to differ for this
  reason, not from the loop itself.
- **Source:** Task 4/6 of the kernel-unification plan (commits dee33f1, 04f5271);
  benchmarks and elementwise diagnostics in
  `~/claude_artifacts/quaritcal_optimisation/results/` (2026-07-13).
- **Later cleanup (2026-07-21):** the original contract had `resid` return a flat
  `(residual..., aux...)` tuple with an `n_resid_aux` parameter telling the loop where to
  split it, so the elem's `aux` was the residual-aux concatenated with the stage
  coefficients. The residual-aux values (per-corr `normf`) were never read by any elem —
  every parameterised elem recomputes its own operator-based normalisation — so this was
  vestigial from an earlier design. Removed: `resid` now returns only the residual, the
  `stage` (`channel_coeffs`) output IS the whole `aux`, and `n_resid_aux` is gone. Hook
  parameters were renamed for clarity (see the hook list above) and made keyword-only. Pure
  cleanup: bitwise-identical checksums across all 14 terms × every corr mode, and no
  measurable timing change on complex/delay_and_offset (numba already dead-code-eliminated
  the unused appends).

## Per-kernel numba disk-cache namespaces for the shared loop

- **Context:** With the shared accumulation loop (previous entry) adopted by all 13
  kernels, the loop closure returned by `build_jhj_jhr_impl` was lowered as a **single
  numba disk-cache unit** shared by every kernel. Numba keys on-disk cache entries by
  source location plus argument-type signature, discriminated only by a cloudpickle
  hash of the closure cells — and that hash is **nondeterministic per build**. Every
  kernel presents the loop with an identical signature, so a process compiling kernel B
  could load kernel A's machine code written by an earlier session and silently run the
  wrong maths. Production severity: a stale multi-session cache corrupted solves with
  no error raised (deterministic repro: delay_and_tec solving with 42.8% wrong gain
  elements). Caught at the unification plan's full-suite gate (25 test failures across
  delay_and_offset / delay_and_tec / rotation); all benchmark and checksum harnesses
  were immune because they always use a fresh `NUMBA_CACHE_DIR` per process.
- **Decision:** `build_jhj_jhr_impl` returns the loop wrapped in `qcjit`
  (`inline="always"`), so it is never lowered as a standalone cache unit; every
  kernel's `nb_compute_jhj_jhr` returns a **module-local trampoline** that inlines it,
  giving each kernel a private cache namespace. The constraint is documented as the
  CACHE CORRECTNESS CONSTRAINT in `solver_components.py`'s docstring.
- **Rationale:** The trampoline is the smallest change that makes the cache key unique
  per kernel (each trampoline has its own source location) without giving up disk
  caching or the shared single-source loop. `prange` survives the inlining (parfor
  diagnostics confirm the parallel loop in the trampoline). leakage needs no
  trampoline of its own: it imports complex's `compute_jhj_jhr` wholesale, so sharing
  that cache unit is byte-identical.
- **Consequences:** Any future factory that returns a jitted closure consumed by
  multiple kernel modules with identical signatures MUST be inlined into a
  module-local wrapper, never returned as a directly-lowered `@overload` impl.
  Verified at the fix: checksums bit-identical to the pre-fix tip, bench parity within
  noise, full suite green twice back-to-back (the second run on the previously failing
  warm cross-session cache), hermetic two-process repro green in both orders.
- **Source:** commit ec25545 (2026-07-16); forensics and evidence in
  `~/claude_artifacts/quaritcal_optimisation/results/UNIFICATION_LOG.md`
  ("Cache-collision fix").

## Shared hook-parameterised solver loop (the iteration above the accumulation loop)

- **Context:** After the accumulation-loop unification, each kernel's `*_solver_impl`
  body (intermediary setup, the Gauss-Newton iteration calling compute_jhj_jhr /
  compute_update / finalize_update / flagging, and the return) remained near-verbatim
  duplicated across 13 modules — ~120-180 lines each, with only a handful of known
  variance points.
- **Decision:** The outer loop lives once in `quartical/gains/general/solver_loop.py`
  as TWO builders: `build_gain_solver_impl` (complex, diag_complex, leakage) and
  `build_param_solver_impl` (the ten parameterised terms). Two builders, not one,
  because the families differ in jhj dtype/shape, params/param-flags plumbing, and
  flagging extras. Per-term rescaling stays in-kernel as opaque `pre_solve`/`post_solve`
  jitted closures (the delay/tec families' scaled-basis entry/exit, relocated verbatim)
  — a declarative/class-based rescaling abstraction was considered and REJECTED
  (it presumes any rescaling fits a single class shape). Optional hooks resolve to
  build-time no-ops; each kernel's `nb_<term>_solver_impl` returns the mandatory
  module-local trampoline (previous entry). `crosshand_phase_null_v` keeps a private
  loop: its inverse-gains machinery (typed-List build before the loop, extra leading
  compute_jhj_jhr argument, per-iteration refresh) is not expressible as verbatim code
  motion through the hooks. `build_param_solver_impl` takes its solution-interval extents from
  `param_freq_maps` unconditionally, since jhj/jhr/update are allocated on the parameter shape; a
  `solve_on_param_grid` build flag able to select `freq_maps` instead was dropped as dead, no term
  having needed the gain-grid path (the three that set it never differed from the param grid, and
  rotation already solved on the param grid despite matching grids).
- **Rationale:** Pure maintainability refactor — the hot loops did not move, so the
  gate was exactness and parity rather than speedup: checksums bitwise-identical to
  kernel-propagation (4b022f4) for every ported term and supported corr mode (full and
  jhj-only scenarios), parity spot-checks complex corr 4 at 0.983 and delay corr 4 at
  0.993 min/min. A prerequisite commit standardised `finalize_update` to one 7-arg
  signature across the parameterised family (rotation_measure now recomputes
  `lambda_sq` inside its finalize impl) and `reference_params` to a 4-arg form.
- **Consequences:** A new gain type writes only hooks at BOTH levels (accumulation
  elem/resid/etc. + the solver-loop bindings) plus its finalize/reference machinery;
  the iteration logic cannot drift per-kernel any more. The plan's assumption that
  diag_complex shared complex's solver object was false — it had its own near-identical
  loop and was absorbed via two extra optional inputs (`collapse_to_scalar_jhj_jhr`
  — named `scalar_jhj_jhr` until 2026-07-29, when it was renamed to stop it shadowing
  the unrelated two-arg `generics.scalar_jhj_jhr` — and `reference_gains`). Known
  variance-inventory correction: every parameterised kernel passes `numbness=1e9`
  explicitly except amplitude (default 1e-6).
- **Source:** branch kernel-unification, commits 2d9d8d2..ce9de22 (2026-07-17 to
  2026-07-20); verification numbers in
  `~/claude_artifacts/quaritcal_optimisation/results/UNIFICATION_LOG.md`
  ("Solver-loop unification").

## Stock hook implementations in modules named for the hook, not the builder

- **Context:** With both loops shared, the surviving duplication was in the hooks themselves.
  `compute_residual_factory` was AST-identical across six kernels (phase and the delay/TEC
  families), and the other two variants had already leaked into cross-kernel imports — `rotation`,
  `rotation_measure` and `crosshand_phase_null_v` importing `complex.kernel`'s,
  `crosshand_phase` importing `phase.kernel`'s. `get_identity_params` had 11 copies, 5 distinct.
- **Decision:** Stock hook implementations live in `general/` modules named for the hook family:
  `general/residuals.py` (`standard_residual_factory`, `phase_only_residual_factory`,
  `amplitude_only_residual_factory` — between them they cover all 13 solvers) and
  `general/parameters.py` (`get_identity_params(corr_mode, params_per_corr, fill=)`).
  Placing them beside the builder that declares the hook — `solver_components.py` for the
  residual, `solver_loop.py` for the identity params — was considered and REJECTED: it loads the
  two loop modules with per-term maths, and it splits one concern across two files on an axis
  (which builder consumes it) that a reader looking for "the residual hooks" does not think in.
  The six `compute_channel_coeffs_factory` copies were deliberately LEFT in their kernels: only
  two pairs are duplicates (48 lines, 23 of code), the other two are unique, and the
  band-scaling convention they encode is restated in `pre_solve`/`post_solve` regardless — so
  unifying them belongs with the JHJ-scaling decision, not here.
- **Rationale:** Pure de-duplication, so the gate was exactness rather than performance: each
  moved residual body is AST-identical (docstrings stripped) to all nine originals it replaces,
  and the shared `get_identity_params` reproduces all 11 originals across corr modes 1, 2 and 4
  including which combinations raise.
- **Consequences:** No term takes a residual hook out of another term's kernel module (`leakage`
  importing complex's `compute_jhj_jhr` is unaffected). A new gain type names a residual hook and
  calls
  `get_identity_params` with its parameter count, so that count is stated once per kernel instead
  of appearing both as `params_per_corr` and as a hardcoded array length. The
  whole-2x2 rule — one parameter set acting on the full 2x2, four correlations only
  (crosshand_phase, rotation, rotation_measure) — is now carried by `params_per_corr=None` rather
  than a per-kernel `if corr_mode == 4` branch.
- **Source:** branch kernel-unification-tidying-7-8 (2026-07-31), addressing items 7 and 8 of the
  branch review.

## Accumulator layout generated from n_param, with literal slot indices

- **Context:** The three hooks describing the flat JHJ/JHr accumulator —
  `zero_jhj_jhr_factory`, `flush_jhj_jhr_factory`, `mirror_jhj_factory` — were hand-written per
  term: 12, 12 and 8 copies. They are not maths. For a parameterised term the layout is fixed by
  one number, how many real parameters the term solves per element, because JHJ is then a real
  symmetric `(n_param, n_param)` matrix. `delay_tec_and_offset`'s three copies alone were 137
  lines, of which 21 `jhj[i, j] += jhj_jhr[k]` lines and 15 mirror lines were a hand-typed
  triangular index table — the class of table a previous commit had already had to correct.
- **Decision:** `general/accumulator.py:triangular_accumulator_factories(params_per_corr)`
  returns the trio for the 11 parameterised solvers. Each kernel declares the count once as a
  module-level `PARAMS_PER_CORR`, feeding the accumulator, `get_identity_params` and
  `build_param_solver_impl`'s own `params_per_corr` from one statement; before this the same fact
  was written out separately for each consumer. `parameters.py:get_n_param(corr_mode,
  params_per_corr)` turns it into the flat parameter count, so the accumulator dimension and the
  identity vector cannot disagree, and `testing/tests/gains/test_parameters.py` pins that count
  against each gain class's `make_param_names`. `None` marks the whole-2x2 terms and is the same
  `None` `build_param_solver_impl` already took, since a scalar collapse is possible exactly when
  there is a parameter set per correlation. `complex` and `diag_complex` keep their own
  trio: their JHJ is a correlation-space block with mixed real and complex slots and a Hermitian
  (conjugating) mirror, none of which follows from a parameter count.
- **Rationale:** The generated hooks must not merely be correct but compile to the same machine
  code, and the obvious implementation does not. Writing `flush` as the natural loop over the
  triangle — constant trip counts, `jhj_jhr[slot]` with a loop-carried `slot` — measured **21%
  slower on `phase`** (accumulation pass 42.1 -> 50.9 ms, corr 4, single direction) while
  `delay_tec_and_offset` stayed at parity. Numba lowers a computed index into a homogeneous tuple
  through memory, so the accumulator stops being register-resident, and the cost lands on every
  visibility in the accumulate loop rather than on the flush. Terms with a large accumulator
  already spill and so hide it; the small ones do not. `zero` meets the same constraint from the
  other side: nothing writable in nopython mode builds a tuple whose length is only known at
  compile time — tuple repetition is unsupported and a tuple cannot be grown in a loop.

  So both tuple-facing hooks are built as `@intrinsic`s, which is what makes the layout expressible
  as an ordinary loop: `codegen` is plain python running at compile time, so a flat `for` over the
  triangle emits fully-unrolled statements that each name their slot by a literal. The alternative,
  composing one inlined single-slot closure per entry, works and was measured at parity, but it
  expresses code generation as runtime function composition — a reader has to simulate a closure
  tree to know what is emitted, and the emitted form is neither flat nor a direct statement of
  intent. The intrinsic form also compiles the largest hook (27 slots) in 0.045s against 0.468s,
  and is 10x cheaper across the whole trio. `mirror` stays an ordinary jitted loop — it only
  indexes arrays, and `complex`'s hand-written mirror was already a loop.

  Generating the flat source and `exec`ing it — the one option that would give literally the code
  the kernels used to carry — is not available: `qcjit` sets `cache=True` and numba refuses to
  cache a function with no real source file (`no locator available for file '<flush n=6>'`).

  With all of this, every measured term/correlation/direction configuration is within 1% of the
  hand-written versions and every checksum is bitwise identical.
- **Consequences:** A new parameterised gain type states its parameter count once and inherits the
  layout, and `flush`'s typing phase now polices it: an accumulator of the wrong length, or one
  holding anything but real slots, fails to compile with a message naming both counts instead of
  silently writing the wrong entries. That check has no equivalent in the hand-written kernels.

  Three prices. `cgutils.get_item_pointer` and `context.make_tuple` are numba internals rather
  than public API. `make_tuple` increfs nothing and `flush` emits `fadd` directly, both safe only
  while every accumulator slot is a real scalar — hence the typing guards. And an intrinsic cannot
  execute in pure python, so the hooks can no longer be driven by a harness that stubs
  `factories.qcjit` to the identity; `zero` and `flush` are instead verified by compiling them,
  flushing an accumulator of distinct values, and reading the index table back off the arrays,
  which tests the compiled artefact rather than a python simulation of it.

  The upper-triangle packing convention (JHJ block row-major, then JHr) is now defined in
  exactly one place, which the per-term `accumulate_jhj_jhr` hooks must agree with — those hooks
  are still hand-written maths and still name their slots by hand, so the convention has to be
  read out of `accumulator.py` when writing one. The general rule this entry establishes: inside
  the accumulation loop, an accumulator slot must be named by a literal, never by a computed
  index.
- **Source:** branch kernel-unification-tidying-10 (2026-07-31), addressing item 10 of the branch
  review.

## Referencing shared for the frequency-dependent parameterised terms

- **Context:** `reference_params` — the stage which subtracts the reference antenna's parameters to
  fix the per-antenna gauge freedom — was written out six times, 258 lines. Five of the six (delay,
  delay_and_offset, delay_and_tec, tec_and_offset, delay_tec_and_offset) were identical in every
  character but the `*_params_to_gains` symbol they call. The branch review's item 11 recorded a
  second axis of variation, an extra zero-mean step in the two offset terms; that is wrong —
  the step it meant was only ever called from `pre_solve`/`post_solve`, and has since been
  deleted outright.
- **Decision:** `general/parameters.py:reference_params_factory(params_to_gains)` returns the hook
  for those five, which each bind it at module level in one statement. It is a `qcjit` closure
  rather than its own `@njit` cache unit: five kernels binding a different `params_to_gains` at one
  shared source location is the aliasing the per-kernel namespace rule exists to prevent, and
  inlining into each kernel's solver trampoline avoids it without needing a second trampoline.
  `phase` keeps its own copy, because `phase_params_to_gains(params, gains)` takes no frequency
  arguments; an adapter discarding four of them would have made phase's kernel advertise a band
  dependence it does not have.
- **Consequences:** Referencing is stated once for the terms that share it, and the single remaining
  copy in `phase` is not duplication. The hook is no longer separately cached, so it compiles as
  part of each kernel's solver rather than once on its own. A new frequency-dependent parameterised
  term inherits referencing by giving its `params_to_gains` the shared signature.
- **Source:** branch kernel-unification-tidying-11 (2026-08-04), addressing item 11 of the branch
  review.

## The zero-mean ordering fix carried on the refactor branch, not the release branch

- **Context:** `tec_and_offset`'s `pre_solve` rescaled the TEC by the bandwidth before applying
  `apply_zero_mean_correction`, whose coefficient is defined on the TEC in native units. The
  forward correction was therefore under-applied by a factor of the bandwidth while `post_solve`'s
  inverse was applied in full, so the two were not inverses and the offset handed to the solver was
  short by `2*pi*tec_factor*TEC` - of order a radian at a TEC contributing a radian across the band.
  A converged solve was unaffected, since the solver converges on the gains regardless of where the
  offset starts and the inverse is correct; the cost was a worse starting point and the attendant
  phase-wrap risk. `delay_tec_and_offset` already ordered the two correctly.
- **Decision:** fix it on the branch that restructures the code rather than on `v0.2.8-dev`.
- **Rationale:** the fix moves two statements and adds no net lines. On `v0.2.8-dev` they live inline
  in the solver body; here they live in an extracted `pre_solve` hook. Merging the release branch in
  conflicts on the whole deleted solver body against the four-line trampoline that replaced it, and
  the natural resolution - take the refactored side - discards the fix silently, because the reorder
  reads as a comment edit. Landing it here also makes the regression test able to drive the real
  hooks, which is impossible while the statements are buried in the solver body.
- **Consequences:** this branch is no longer output-neutral for `tec_and_offset`: a solve starting
  from non-zero parameters, which includes the default path since the term makes an initial TEC
  estimate, now begins from the correct offset. Converged results move only within solver tolerance.
  The wider basis inconsistency this uncovered is recorded under Known debt and is not addressed.
- **Source:** branch kernel-unification-tidying-9 (2026-08-05), arising from item 9 of the branch
  review.
- **Superseded** by "The zero-mean offset shear removed from the delay/tec pre/post-solve hooks"
  below, which deletes the correction whose ordering this entry fixed.

## One jhj unscaling convention, correct on the diagonal only

- **Context:** the five frequency-dependent parameterised terms unscaled jhj in `post_solve` two
  different ways — `jhj[..., i::ppc]` in delay_and_offset and delay_and_tec, `jhj[..., i::ppc,
  i::ppc]` in tec_and_offset and delay_tec_and_offset, with delay_and_tec using both two lines
  apart. The branch review's item 9 asked which is right. Neither is, in full: a term solving in
  the basis `p' = Sp` has `jhj = S jhj' S`, so an element needs the product of its two indices'
  factors, and both forms leave the blocks coupling rescaled to unrescaled parameters in the solver
  basis. Measured against `S jhj' S`, delay is exact; the other four are wrong in 8 to 24 entries
  per correlation pair. Both forms are exact on the diagonal, and `calibration/solver.py` reduces a
  6-dim jhj to its diagonal before storing it.
- **Decision:** state the two-axis form everywhere it applies and record why the rest is not scaled,
  rather than compute the full `S jhj' S`. delay keeps `jhj[:] *= mid_freq ** 2`: every one of its
  parameters carries the same factor, so the whole array genuinely wants `mid_freq ** 2` and
  striding it would say less.
- **Rationale:** the off-diagonal blocks have no consumer, so making them correct would buy nothing
  observable. It would also cost bit-identity — the honest form scales by a product of per-slot
  factors, and `x * (1/bw) * (1/bw)` is not `x / bw ** 2` — and would need a per-slot multiplier and
  divisor recipe to avoid that. If a caller ever needs the full matrix, the relation above is the
  specification, and the reduction in `calibration/solver.py` is the place to start.
- **Consequences:** the stored jhj is unchanged to the bit, since the diagonal and the operation
  applied to it are the same in both forms. The divergence a reader would otherwise have to
  adjudicate is gone. jhj remains a diagnostic, not a covariance: its off-diagonal entries are not
  in native units.
- **Source:** branch kernel-unification-tidying-9 (2026-08-05), addressing item 9 of the branch
  review.

## Chain regression tests behind a slow marker, asserted on the net gain

- **Context:** Until 2026-07, no test ever *computed* a multi-term chain solve: every
  per-type gain test uses `solver.terms=['G']`, and `test_calibrate.py`'s G,B chain is
  asserted lazily (graph metadata only). The kernel-unification work exposed the gap — a
  full-pipeline A/B probe (three-term chains incl. a DD term, new vs pre-unification
  baseline) had to be improvised to show chain mechanics were preserved. Two obstacles
  kept chains out of the suite: (1) kernel compilation depends on the full chain signature
  (every term's gain dtypes enter each kernel's numba tuple argument), so each distinct
  chain composition compiles fresh and the suite already compiles for ages from scratch;
  (2) effects bleed between terms (e.g. a sufficiently resolved complex term absorbs a
  delay), so per-term truth assertions are ill-posed in a chain.
- **Decision:** One `@pytest.mark.slow` module, `testing/tests/gains/test_chain.py`, run by
  CI along with everything else; the marker exists so a *local* run can skip the compile
  cost (`pytest -m "not slow"`). It solves a single three-term chain mirroring a real
  setup - diag_complex G (time-dependent gain) + delay K + complex B (bandpass and
  leakage) - at one correlation mode, in two
  variants (DI, and DD via a two-direction synthetic model with B direction-dependent)
  that deliberately share the same chain signature — one extra set of chain compilations
  total. Assertions target the **net gain product** (via `output.net_gains`) against the
  composed true Jones product, plus residual magnitude and flag invariance — never
  individual terms.
- **Rationale:** The net product is what the data constrains, so it is invariant to
  inter-term bleed; right-referencing to antenna 0 removes the per-(t,f,dir) gauge
  ambiguity. Two well-posedness constraints were established empirically during
  bring-up: (1) the truth must be **diagonal with an unpolarised model** — diagonal-type
  terms (delay, diag_complex) are constrained by parallel-hand data only, so per-channel
  crosshand phase in the truth is gauge-free and stalls the solve with the entire
  cross-hand power left in the residual (~1% chi-squared plateau) while the
  freq-constant complex term fits junk leakage; (2) the DD term must be
  frequency-constant, otherwise only the sum over directions is constrained and the
  per-direction net is not unique. Chain solves also need many one-term-at-a-time
  cycles (20 in the test; converged terms exit immediately, so cycles are cheap — the
  module runs in ~100 s cold / ~20 s with a warm numba cache). Chain coverage is the
  coverage most worth having in CI, so the compile cost is paid there and the marker
  serves the local iteration loop instead.
- **Consequences:** Chain mechanics regressions (operator products, per-term
  time/freq/dir maps, DD stacking) are caught by CI instead of ad-hoc A/B probes, at the
  cost of the chain compilations on every run. A future test marked slow is therefore
  opting out of local runs, not out of CI. Markers are registered in `pyproject.toml`
  with `strict_markers`, so a marker misspelt on a test fails collection instead of
  leaving that test unmarked and silently selected by an `-m` expression meant to exclude
  it. That option is read by pytest >= 9 only, and it does not check the `-m` expression
  itself, which pytest matches against each test's own marks; CI resolves the top of the
  dev pin and so enforces it. Fixing this test also surfaced a latent `n_dir > 1` bug in
  `testing/utils/gains.py:reference_gains` (antenna loop outside the direction loop —
  the same stale-direction shape as the legacy kernel bug).
- **Source:** chain-mechanics A/B probe and design discussion 2026-07-16
  (`~/claude_artifacts/quaritcal_optimisation/results/UNIFICATION_LOG.md`, "Chain-mechanics
  A/B probe"); user requirement that compilation cost stay bounded and per-term
  assertions be avoided.

## A diagonal term's X-Y phase is gauge, so a chain truth must hold it constant in time

- **Context:** Giving the chain test a realistic split (diag_complex G for the
  time-dependent gain, complex B for the bandpass *and leakage*) made it fail: the solve
  settled at ~1% residual power having recovered only ~36% of the true leakage. The
  fraction was invariant to the iteration budget (2, 20 and 60 epochs; 25 and 200
  iterations per term-turn all agreed to six figures), to the bandpass phase range (down
  to exactly zero), and to the leakage amplitude - while the same B alone, at the same
  resolution against the same model, recovered its truth in 11 iterations.
- **Decision:** The truth's G carries a per-antenna X-Y phase difference which is
  **constant in time**. Its overall phase still varies per time interval.
- **Rationale:** Parallel-hand visibilities constrain only `phiX_p - phiX_q` and
  `phiY_p - phiY_q`; the per-antenna difference `phiX_p - phiY_p` appears solely in the
  cross-hands, which a diagonal term discards by construction (off-diagonal weights are
  treated as zero - see the accumulate hook in `complex/diag_kernel.py`). That difference
  is therefore gauge to the term, and `reference_gains` re-fixes it - zeroing the
  reference antenna's X and Y phases - independently in *every* solution interval. The
  solved chain consequently differs from the truth by `diag(exp(i*psi0(t)), 1)`, where
  `psi0` is the truth's reference-antenna X-Y phase. When `psi0` varies with time nothing
  in the chain can represent the leftover: G treats it as gauge and B is time-constant, so
  B fits the time average and the leakage is scaled by `|E[exp(i*psi0)]|` - the measured
  ~0.36, coherent in phase. Holding `psi0` constant lets B absorb it and the net is
  recovered to machine precision.
- **Consequences:** The chain test asserts the cross-hand residual and the net leakage,
  not just the parallel hands. Any chain needing a genuinely time-variable X-Y phase needs
  a term varying on that timescale (a crosshand phase term); a time-constant leakage term
  cannot stand in for one. Two dead ends are worth not repeating: propagating the
  cross-hand residual into the diagonal kernel's accumulation does not help (the extra
  terms were verified algebraically equivalent to the complex kernel's own factorisation
  for those entries, and the leftover remains unrepresentable), and an unpolarised model
  removes the *residual* - the leftover becomes a unitary, data-invariant gauge - but
  still leaves the net leakage unidentifiable.
- **Source:** investigation of the chain test's leakage recovery, 2026-08-12; minimal
  reproducer was a two-term `[diag_complex, complex]` chain.

## Dask for parallelism and distribution

- **Context:** QuartiCal needed one codebase that scales from laptop to cluster/cloud.
  CASA's MPI support requires manual partitioning; CubiCal's stateful shared-memory design
  could not be adapted.
- **Decision:** Express the pipeline as a dask task graph, using the distributed scheduler
  for multi-node execution.
- **Rationale:** The promise of a single graph is that dask maps it onto the hardware for
  you — resilience, node death, scheduling all handled. The reality was not quite so rosy
  (see AutoRestrictor below), but the graph also clearly encodes every dependency between
  functions and their outputs.
- **Consequences:** Forfeits fine-grained task placement (partially clawed back via the
  scheduler plugin); inherits the distributed scheduler's greedy, memory-unpredictable
  behaviour; couples QuartiCal to dask's upper-bound pins. Dask is now considered one of
  the project's biggest liabilities — upstream has stagnated and the plan is to move away
  from it eventually (see Known debt).
- **Source:** interview 2026-07-07; Kenyon et al. 2025, Sections 1, 3.1 and 4.5.

## Single end-to-end dask.compute()

- **Context:** Reads, model, calibration, flagging, MS writes and gain writes are all
  assembled lazily; shared intermediates (visibilities, residuals, gains) feed multiple
  outputs.
- **Decision:** Trigger exactly one materialising `dask.compute()` for the whole pipeline
  (`quartical/executor.py`), bundling all writes.
- **Rationale:** Shared intermediates are computed once and reused — the MS is not read
  twice — and the end-to-end graph encodes the entire calibration process and its
  dependencies explicitly.
- **Consequences:** One very large graph, hard to inspect or describe. Two small earlier
  eager computes (chunking, gain scaffolds) are needed to reify shapes first — see
  [dask-machinery.md](dask-machinery.md), Single-compute design.
- **Source:** interview 2026-07-07; Kenyon et al. 2025, Section 4.2.

## Blocker: hand-built graphs instead of dask blockwise

- **Context:** The solver is a many-input/many-output per-chunk operation returning a
  variable set of heterogeneously shaped arrays.
- **Decision:** Build the solver layer's HighLevelGraph by hand via a custom `Blocker`
  class (`quartical/utils/dask.py`) whose tasks return dicts keyed by output name.
- **Rationale:** `dask.array.blockwise` never supported many-to-many mappings, at least
  not transparently; Blocker is the custom workaround for that deficiency. Named dict
  outputs also avoid error-prone reliance on output position.
- **Consequences:** Full mechanics in [dask-machinery.md](dask-machinery.md), Blocker.
  Complicates the graph, but makes the many-output solver expressible at all.
- **Source:** interview 2026-07-07; Kenyon et al. 2025, Section 4.2.

## Pure task functions, paid for with guard copies

- **Context:** Dask graph nodes must not mutate their inputs — a mutated shared input
  corrupts every other task that consumes it.
- **Decision:** Write graph functions as pure functions; defensively copy inputs the solver
  would otherwise mutate (`WEIGHT` and `FLAG` are `.copy()`-ed at solver entry —
  `quartical/calibration/solver.py:96`, "a necessary evil").
- **Rationale:** A deliberate rejection of CubiCal's mutable-state style; the paper calls
  the functional constraint a double-edged sword — it prevents hard-to-debug stateful
  behaviour but forces spurious copies.
- **Consequences:** Higher peak memory in places; part of the peak-vs-average memory gap
  seen in benchmarks.
- **Source:** Kenyon et al. 2025, Sections 4.2, 4.3 and 5.2; file comment
  `quartical/calibration/solver.py:96`.

## Zarr-backed gain outputs via xarray

- **Context:** There is no universally accepted gain-solution format; every package rolls
  its own. Gains live naturally on a labelled (time, freq, antenna, direction, correlation)
  grid.
- **Decision:** Store gains as xarray Datasets written to zarr, one group per term.
- **Rationale:** xarray arrived in the project via its main data-ingest dependency,
  dask-ms; no backends beyond xarray's defaults were ever considered. Zarr won on merit:
  parallel reads *and* writes, thread safety, object-store (cloud) compatibility, and a
  chunked layout that maps directly onto dask's processing model. Self-describing labelled
  gains then make transfer calibration and on-the-fly interpolation simple.
- **Consequences:** A non-standard format whose interoperability depends on adoption
  (pfb-imaging already consumes it). Loading machinery lives in `quartical/interpolation/`.
- **Source:** interview 2026-07-07; Kenyon et al. 2025, Sections 3.3 and 4.4.

## Zarr-backed Measurement Sets as a CTDS escape hatch

- **Context:** The CASA Table Data System (CTDS) behind normal MSs is not thread-safe and
  exposes no explicit time axis, so parallel I/O stalls and time chunking needs a
  preprocessing pass.
- **Decision:** Support (via dask-ms) converting an MS to a zarr-backed equivalent, used
  interchangeably.
- **Rationale:** Zarr reads/writes are parallel and thread-safe and work against object
  stores. Benchmarks show zarr is faster than CTDS but uses *more* memory at a given thread
  count — slow CTDS access effectively starves the pipeline, keeping less in flight.
- **Consequences:** An extra conversion step; the memory-vs-throughput trade-off is
  workload dependent.
- **Source:** Kenyon et al. 2025, Sections 3.2, 3.4 and 5.3.

## AutoRestrictor: pinning subtrees after annotations failed

- **Context:** Dask's distributed scheduler is greedy and does no static graph analysis to
  minimise data movement or memory pressure, so it shuffles intermediates between workers
  even though QuartiCal's per-chunk subtrees are fully independent.
- **Decision:** An opt-in `SchedulerPlugin` (`quartical/scheduling/`) that pins each
  independent subtree of the graph to a specific worker. This *replaced* an earlier
  annotation-based attempt.
- **Rationale:** "Curbs dask's enthusiasm": guaranteeing each parallel stream stays on one
  node curtailed unnecessary data movement and capped the memory footprint. The annotation
  approach was abandoned because dask task fusion clobbers annotations (dask/dask#7036).
- **Consequences:** Hard pinning reduces resilience and caps dask-level parallelism at the
  number of data partitions; load is balanced by task count, not data size. Mechanics in
  [dask-machinery.md](dask-machinery.md), AutoRestrictor. A less strict successor that pins
  only the solver tasks (the paper's "Solver Restrictor", implemented as a `ScatterSolvers`
  plugin) exists on the unmerged `v0.2.4-simple-scheduling` branch but was never released.
- **Source:** interview 2026-07-07; commit 5e4cc28 (#62); Kenyon et al. 2025, Section 4.5;
  branch `v0.2.4-simple-scheduling`.

## Vendored graph_metrics from old dask

- **Context:** `AutoRestrictor` depends on dask's `graph_metrics`, which was removed
  upstream.
- **Decision:** Copy `graph_metrics` verbatim into `quartical/scheduling/__init__.py` from
  a pinned old dask commit.
- **Rationale:** In-code: "removed upstream but currently critical for the scheduler
  plugin... a temporary work around while we consider revised strategies."
- **Consequences:** Bumping dask will not bring the symbol back; the copy stays until the
  plugin is reworked.
- **Source:** file comment `quartical/scheduling/__init__.py:163`.

## Scheduler plugin installed via run_on_scheduler

- **Context:** The standard way to install a scheduler plugin is
  `dask-scheduler --preload install_plugin.py`, which is awkward for QuartiCal's
  single-command workflow.
- **Decision:** Install the plugin at runtime with `client.run_on_scheduler(...)`, gated
  behind `dask.scheduler_plugin`.
- **Rationale:** The preload pattern presumes an independently managed cluster, which the
  average user neither runs nor understands; `run_on_scheduler` is the easiest way to get
  the plugin into every user's hands regardless of how their cluster was started. The
  in-code comment flags the trade-off ("Controversial from a security POV,
  run_on_scheduler is a debugging function").
- **Consequences:** Works uniformly for local and user-supplied clusters at the cost of
  using a debugging hook in production; the security caveat stands.
- **Source:** interview 2026-07-07; file comment `quartical/executor.py:94`.

## Disable dask memory management (no spill, no pause, no limit)

- **Context:** By default, distributed workers spill data to disk under memory pressure
  and pause or kill workers approaching their memory limit.
- **Decision:** Opt out of dask's memory management entirely: QuartiCal routinely disables
  spill-to-disk when deploying, and the internally spawned `LocalCluster` starts workers
  with `memory_limit=0` (`quartical/executor.py:87`).
- **Rationale:** Spilling is "the death-knell" of a pipeline at this data volume, and
  dask's memory management has a tendency to pause and stall tasks. Disabling it and
  monitoring memory usage directly — tuning chunk sizes as needed — proved more robust in
  practice.
- **Consequences:** Memory control is the operator's job: chunk sizing is the knob, and
  there is no graceful degradation to disk. Larger problems need genuinely more RAM or
  more nodes.
- **Source:** interview 2026-07-07; Kenyon et al. 2025, Section 4.5; `quartical/executor.py:87`.

## Dependency upper bounds as a deliberate policy

- **Context:** dask releases after 2024.10.0 break dask-ms read graphs; bokeh 3.7 breaks
  distributed's performance report; stimela is the orchestration front-end that must always
  co-install.
- **Decision:** Pin `dask[distributed]<=2024.10.0` and `bokeh<3.7`; never give `stimela` an
  upper bound.
- **Rationale and mechanics:** documented in full in
  [dask-machinery.md](dask-machinery.md), Dependency pins — read that before touching any
  pin. The pyproject comments are the canonical statement of intent.
- **Consequences:** The bokeh cap is coupled to the dask cap; lifting dask requires dask-ms
  read graphs to survive the newer optimisers.
- **Source:** `pyproject.toml` inline comments.

## No TAQL: zero autocorrelation weights instead of deselecting rows

- **Context:** Early code used TAQL (CASA table query language) row selection, e.g. to drop
  autocorrelations.
- **Decision:** Remove all TAQL; keep autocorrelation rows but set their weights to zero.
- **Rationale:** QuartiCal supports two storage backends — the Measurement Set (CTDS) and
  its zarr equivalent — and zarr does not support TAQL. Removing TAQL entirely ensures both
  formats behave identically (lead developer's recollection).
- **Consequences:** Autocorrelations remain in the data, silently down-weighted to zero;
  nothing in the read path may reintroduce TAQL without breaking the zarr backend.
- **Source:** commit 291f7c5 (#117); interview 2026-07-07.

## An in-repo LLM wiki with no supporting tooling

- **Context:** The knowledge needed to work on QuartiCal safely — the maths the code
  assumes, why the kernels are shaped as they are, which surprises are deliberate — was
  recoverable only by reading source or asking the lead developer. Every fresh agent
  session paid that cost again.
- **Decision:** Commit a wiki of markdown pages under `docs/wiki/`, written for an agent
  reader, indexed from `index.md` and wired into `CLAUDE.md`. Build no tooling for it: no
  generation, no CI staleness check, no Sphinx integration. Pages carry a
  `last_verified_commit` stamp and are kept current by the update-as-you-touch rule in
  `CLAUDE.md`.
- **Rationale:** The expensive part is the knowledge, not the plumbing, and doc tooling
  tends to acquire maintenance cost faster than it repays it. A stamp plus a rule that
  fires while the author is already in the relevant code is the cheapest thing that makes
  staleness visible. Keeping the pages in-repo means they are reviewed and versioned with
  the code they describe. The user-facing docs under `docs/source/` are a separate
  audience and stay untouched.
- **Consequences:** Staleness is detectable but not enforced — a stamp is only as good as
  the session that refreshed it, and refreshing one without re-verifying the page is worse
  than leaving it stale. Coverage grows organically: stubs state their scope and are
  filled in by whoever next works in that subsystem, never speculatively.
- **Source:** the wiki design, 2026-07-07; conventions now stated in
  [index.md](index.md).

## The zero-mean offset shear removed from the delay/tec pre/post-solve hooks

- **Context:** `tec_and_offset` and `delay_tec_and_offset` ran an `apply_zero_mean_correction`
  step in `pre_solve`/`post_solve` which shifted the offset parameter by `2*pi*mid_freq*delay`
  and `-2*pi*log(nu_min/nu_max)/bandwidth*TEC`. It was described as making the offset
  consistent with the solver's zero-mean corrections, but those corrections are already part of
  each term's model: `params_to_gains` subtracts the band's mean frequency from a delay's
  coefficient and the band's mean of `1/nu` from a TEC's, in **both** `rescaled` modes. The step
  therefore applied the shift a second time, so `params_to_gains` in its native mode - the mode
  `init_term` calls - described gains that differed from the solver's by that constant, of order
  a radian at parameters contributing a radian across the band. `delay`, `delay_and_offset` and
  `delay_and_tec` never had the step and were self-consistent throughout.
- **Decision:** delete the step and both copies of the function. `pre_solve`/`post_solve` in all
  five frequency-dependent terms now change units and nothing else.
- **Rationale:** the shift is a shear, not a scaling — it mixes two parameters rather than
  rescaling one — and it was living in a hook whose job is units. Naming what it actually did
  settles where it belongs. Referencing a frequency-dependent coefficient to the band is what
  decorrelates it from the offset: for a delay the Jacobian columns `1` and `2*pi*(nu - nu_c)`
  are exactly orthogonal on an unflagged uniform grid, against a column correlation of 0.982 for
  `2*pi*nu` over MeerKAT L-band. That decorrelation is a property of the model the term fits, so
  it belongs in `params_to_gains` and the accumulate hooks, which is where it already was. The
  offset consequently means the phase at the band reference point in every offset term — the
  quantity the data constrains, rather than an extrapolation to zero frequency which for
  realistic delays is not even determined modulo 2*pi.
- **Consequences:** the value written to `phase_offset` changes for these two terms; the gains do
  not. A `load_from` of a gain zarr written before this reads its offsets in the new basis, which
  for a stored solution differs by the shift; both terms carry an experimental warning in
  `docs/source/gain_types.rst`. `init_term`'s gains now agree with the parameters beside them, so
  `load_from` and `initial_estimate` no longer displace the offset before iteration 0.
  `testing/tests/gains/test_{tec,delay_tec}_and_offset.py` truth gains carry the band-referenced
  coefficients, matching what `test_delay.py`, `test_delay_and_offset.py` and
  `test_delay_and_tec.py` already wrote, and `docs/source/gain_types.rst` needed no change
  because it already published this convention.
- **Not done:** centring on the *weighted sample* mean rather than the analytic band mean would
  make the decorrelation exact under flagging (at 30% edge flagging the column correlation is
  0.60 rather than 0). Rejected for now: the centring constant would become data-dependent per
  antenna and per interval and would drift as mid-solve flagging fires, which would make stored
  offsets incomparable across antennas and break parameter interpolation in `interpolation/`.
  The residual correlation costs conditioning, which `compute_update`'s exact block inversion
  already absorbs.
- **Source:** branch followups/zero-mean-shear (2026-08-20), addressing item 1 of the second
  branch review.

## Band constants left inline in the channel-coefficient hooks

- **Context:** the shared solver loop calls `compute_channel_coeffs(ms_inputs, meta_inputs, f)`
  from inside the row/channel loop, after the flag check, so once per unflagged visibility. The
  hook receives only the channel index, so the five frequency-dependent terms build their band
  constants in the hook body: `bandwidth = cf_max - cf_min`, `cf_mid = (cf_min + cf_max)/2` and,
  for the three TEC terms, `np.log(cf_min/cf_max)`. Before the loop was extracted these lived
  once per `compute_jhj_jhr` call, above the `prange`. The second branch review's item 8 asked
  whether the move costs a libm call per visibility, which at the benchmark dimensions
  (759k unflagged visibilities, 20 iterations) would be 15.2M `log` evaluations per solve.
- **Decision:** leave the hooks as they are. Do not add a per-chunk precompute hook, and do not
  thread the invariants in as extra arguments.
- **Rationale:** measured, not assumed. LLVM's LICM hoists every one of these out of the entire
  loop nest inside the parfor body, into the preheader of the interval loop — one evaluation per
  thread per call, which is the pre-extraction placement. Verified by dumping the
  `compute_jhj_jhr` parfor gufunc IR for `delay`, `delay_and_tec`, `tec_and_offset` and
  `delay_tec_and_offset` and classifying each block as loop-carried or not: exactly one
  `llvm.log.f64`, in the outer loop's preheader, none in any loop. It also beats a hand-hoist:
  for the four terms dividing by `cf_mid`, LLVM hoists a reciprocal (`fdiv 1.0, cf_mid`) and
  leaves a multiply in the loop, so writing the invariants out by hand would keep a division
  per visibility that the compiler removes. `rotation_measure` has nothing to hoist —
  `(c/chan_freq[f])**2` is per-channel in full.
- **Consequences:** the hook signature stays at three arguments and the seven kernels binding
  `compute_channel_coeffs_factory=None` stay untouched. The cost is a dependency on an optimiser
  pass: a numba or LLVM bump could stop hoisting and reintroduce the per-visibility libm call,
  worth roughly 10% of a `tec_and_offset` solve. `build_jhj_jhr_impl`'s docstring states the
  guarantee and the IR check that confirms it, so the recipe is in the tree rather than only
  here. No regression test guards it: the check needs a cold kernel compile (~45 s) and asserts
  on LLVM block naming, which is too brittle a thing to fail CI on for a performance property.
- **Source:** branch followups/channel-coeff-invariants (2026-08-21), addressing item 8 of the
  second branch review.

## Per-direction operator buffers allocated unconditionally

- **Context:** the shared solver loop `valloc`s four per-direction operator accumulators
  (`lop_pq_arr`, `rop_pq_arr`, `lop_qp_arr`, `rop_qp_arr`) at the top of each interval's
  `prange` body, above the `single_dir` branch that never reads them. Only the multi-direction
  path writes them, through `iunpack`/`iadd`; the single-direction fast path keeps its
  accumulation in registers and flushes once per row. The second branch review's item 9 read
  that as four wasted heap allocations per solution interval per solver iteration in the common
  direction-independent case.
- **Decision:** allocate all four unconditionally, outside any `single_dir` guard.
- **Rationale:** numba's parfor loop-invariant code motion already hoists all four out of the
  interval loop, so the cost is one allocation each per thread per `compute_jhj_jhr` call, not
  per interval. Verified with `NUMBA_PARALLEL_DIAGNOSTICS=4` over the bench harness for
  `complex` (corr 4) and `delay` (corr 4, parameterised and frequency-dependent): four
  "hoisted out of the parallel loop labelled #0 ... and reused inside the loop" reports against
  `valloc`'s `np.empty` in each. Hoisting cannot be circumvented by specialisation either —
  `n_dir` is a runtime dimension, so the DI and DD paths compile to one function. Guarding the
  allocations on `single_dir` would make their shape branch-dependent, defeat the hoist and pay
  four allocations per interval to save a 64 byte payload, since a zero leading dimension still
  costs a full NRT meminfo. Sinking them into the multi-direction path is worse again: that
  branch sits inside the row loop, so they would be allocated per row.
- **Consequences:** reuse across intervals is safe only because the multi-direction path zeroes
  all four before every visibility; anything added to that path must keep writing them before
  reading. As with the band constants above, this depends on an optimiser pass, and a numba bump
  could stop hoisting — the site comment carries the diagnostic recipe. No regression test
  guards it: the check needs a cold kernel compile and asserts on numba's diagnostic text.
- **Source:** branch followups/valloc-hoisting (2026-08-21), addressing item 9 of the second
  branch review.

## Referencing made a per-term option, and extended to the 2x2 complex term

- **Context:** referencing was wired in at build time and could not be turned off: `diag_complex`
  and the parameterised phase/delay/tec terms always referenced, and the full 2x2 `complex` term
  never did. There is no reason the gauge fix should be a property of the term type rather than of
  the run: referencing is what a chain wants nearly always, and the exceptions are properties of
  the run, not of the type.
- **Decision:** `referenced` is a per-term option (`gain_schema.yaml`, default `true`), read into
  `Gain.referenced`, carried into `meta_args_nt` and checked at runtime by the two solver-loop
  builders. `complex` now passes `general/referencing.py:reference_gains`, the hook `diag_complex`
  already used, which is hoisted out of `complex/diag_kernel.py` so both bind the same routine.
  Terms with no referencing stage still resolve to a build-time no-op and discard the option
  silently.
- **Rationale:** the gauge freedom is `G_p -> G_p X` for a constant X, and V_pq = G_p M_pq G_q^H is
  unchanged exactly when `X M_pq X^H = M_pq` for every baseline. Which X satisfy that is a property
  of the *model*, not of the term, and the count is easy to get wrong. When the model is one
  constant coherency on every baseline - an unresolved calibrator at phase centre - the stabiliser
  is the M-unitary group `X = M^(1/2) U M^(-1/2)`, `U` in U(2), which is **four** real parameters
  for any positive-definite M, not one. Only a model whose `M_pq` vary enough between baselines
  narrows it towards `e^{i phi} I`. What matters here is not the size of that group but whether a
  *diagonal unitary* lies in it: `diag(a, b) M diag(a, b)^H = M` needs `a b* M01 = M01`, so the
  answer is yes exactly when `M01 = 0`, and otherwise only for `a = b`. Referencing is therefore
  free whenever the model has no cross-hand coherency - every Stokes I model - and moves the fit
  whenever it does, independent of how degenerate the solve already was.
  The shared hook uses the conjugated unit-modulus diagonal of the reference antenna's gain, which
  spends exactly those two parameters and no more. Three constraints follow and are worth stating
  because each rules
  out a plausible alternative. X must be a pure phase: `X = cI` with `|c| != 1` scales `M` by
  `|c|^2`, and amplitude is fixed by the model, never gauge. X must discard the reference gain's
  off-diagonal elements, because fixing the two remaining SU(2) directions is legitimate only for
  an exactly unpolarised model, which is precisely the case where a full-Jones solve cannot
  determine leakage at all. And X must not be the reference gain's inverse: polar decomposition
  gives `G^-1 = H^-1 U^H`, whose Hermitian factor is not a symmetry of any model.
  `testing/utils/gains.py:reference_gains` does use the inverse, but applies it to truth and
  solution alike purely as a comparison device.

  Referencing the 2x2 term is a deliberate choice, not a concession to the Stokes I case: pinning
  both diagonal phases forces the cross-hand phase *out* of the complex term, which is what a real
  chain wants, because that phase belongs to a dedicated `crosshand_phase` term rather than being
  absorbed by G or B. `referenced=false` exists for the run that genuinely wants the complex term
  to carry it. The two properties are mutually exclusive and no implementation reconciles them:
  zeroing both reference-antenna phases is exactly the choice that makes `X = diag(a, b)` have
  `a != b`, and `X M X^H = M` for a model with cross-hand power requires `a = b`. Forcing the
  phase out is therefore the same act as moving the fit, whenever `M01 != 0`.
- **Consequences:** `complex` solves are referenced by default, which changes their output - the
  gauge is now pinned rather than wherever the iteration stopped, so results are reproducible
  between runs. `solve_per="array"` needs no special case: the stabiliser condition does not
  mention the antenna index, so an array-wide term's diagonal phases are gauge under exactly the
  same condition and driving them real is the correct fix, not a loss.
  `testing/tests/gains/test_complex.py` had to change its model from `[1, 0.1, 0.1, 1]` to
  `[1, 0, 0, 0.8]` - still non-singular, which is all the original scaling was for, but now
  diagonal, so the referencing transform is a symmetry of it and the residuals stay at zero. The
  gains keep their leakage, so the module still tests full-Jones recovery, and it now does so on
  the default referenced path. Two dead ends are worth not repeating. Setting the *truth's*
  reference-antenna cross-hand phase to zero does not work: with a constant coherency the fit is
  degenerate over that four-parameter group, the solve lands at an arbitrary point in it, and the
  *solved* reference-antenna cross-hand phase was measured wandering up to 0.86 rad across
  solution intervals regardless of the truth. Nor is scalar-phase referencing for the 2x2 term the
  answer - it would keep every model safe, but it leaves the cross-hand phase in the term, which
  is the thing referencing exists to remove. `test_diag_complex.py` needed no change: its model is
  already diagonal, and a diagonal term discards cross-hand data by construction anyway. Covered by
  `testing/tests/gains/test_referencing.py`, which asserts the reference antenna is pinned with
  the option on and free with it off, for one gain-referenced and one parameter-referenced term.
- **Source:** branch add-per-term-referenced-option (2026-08-26).

## Reference antenna selection typed as a union, not a prefixed string

- **Context:** `solver.reference_antenna` accepted only an integer index. Selecting by name
  needs a second kind of value in one option, and the two kinds collide: an MS whose
  `ANTENNA.NAME` values are integer strings (`'1'`..`'28'` is a real case) makes `5` both a
  valid index and a valid name, meaning different antennas.
- **Decision:** the schema types the option `Union[int, str]`. An int is an index, a str is a
  name, and `converters.py:as_antenna_index` does nothing but dispatch on the type. The
  rejected alternative was `dtype: str` plus `name:`/`index:` prefixes to disambiguate.
- **Rationale:** the union pushes the disambiguation into the type system, where every config
  layer already carries it: YAML separates `5` from `"5"`, `oc.from_cli()` separates `ref=5`
  from `ref='"5"'`, and a fixture's `_opts.solver.reference_antenna = 0` is an int by
  construction. A prefixed string has to re-derive that distinction by parsing, needs a
  precedence rule for bare values, and — the decisive point — silently accepts a bare `5` on
  an MS where it is ambiguous. The union also makes the option strictly better typed than the
  int it replaces: `reference_antenna=5.0`, `=true` and `=null` now fail in OmegaConf with
  QuartiCal's "value not understood" message instead of reaching the converter. Old configs
  are unaffected, because an unquoted `5` is still an int and still means index 5.
- **Consequences:** an antenna whose name is an integer can only be selected by quoting, and
  on the command line the shell eats one level of quotes, so `ref="5"` is an index and
  `ref='"5"'` is the name. That trap is real but confined: a sweep of MeerKAT, VLA, ALMA,
  ASKAP, LOFAR, ATCA and GMRT naming conventions found none that need quoting, and the other
  values OmegaConf's grammar claims (`true`, `false`, `on`, `off`, `yes`, `no`, `null`, `1e5`)
  fail loudly rather than silently. The remaining silent case keeps a warning: an int index
  which is also an antenna name says so and points at quoting, and an out-of-range int which
  is a name says so in the error. `Union` in a schema `dtype` is safe because
  `scabha/cargo.py` evaluates the dtype string against `vars(typing)` and `pyproject.toml`
  pins `omegaconf>=2.3.0`; note that `scabha`'s `clickify_parameters` has no union branch and
  degrades one to `str`, which matters only if QuartiCal is ever driven through that path.
  Resolution runs in `calibration/calibrate.py:add_calibration_graph` rather than a post-init
  because it needs the antenna table. Covered by
  `testing/tests/config/test_converters.py`.
- **Source:** branch v0.2.8-refant-name (2026-09-09).

## The pointing, not the phase centre, drives the beam and the parallactic angles

- **Context:** africanus' fused RIME derives one `lm` array from `phase_dir` (its
  `LMTransformer`) and hands it to both the `Phase` term, where it must be referenced to the
  visibilities' phase centre, and to `BeamCubeDDE`, where it is the point at which the beam
  cube is sampled; its `ParallacticTransformer` reads `phase_dir` too. QuartiCal fed
  `FIELD.PHASE_DIR` to all of them, and `data_handling/angles.py` set `FIELD_CENTRE` from
  `PHASE_DIR` as well. Rephasing tools (`chgcentre`, `phaseshift`) move `PHASE_DIR` and rotate
  the uvw coordinates to match, but leave `REFERENCE_DIR`/`DELAY_DIR` at the pointing, so on a
  rephased MS the beam was applied around a point the dishes were never pointed at —
  silently, and by up to a beam width (ratt-ru/QuartiCal#439 measured a model ~50x too bright
  at the new centre) — and the parallactic angles described an orientation no antenna had.
- **Decision:** select the pointing once, in `data_handling/pointing.py:get_pointing_dir`, as
  the first populated column of `REFERENCE_DIR` -> `DELAY_DIR` -> `PHASE_DIR`, and use it for
  everything that describes the dishes. `predict` supplies both lm arrays itself — `lm` about
  `PHASE_DIR` for the fringe, `beam_lm` about the pointing for `PointedBeamCubeDDE`, a
  `BeamCubeDDE` subclass registered through `RimeSpecification(terms={"E": ...})` — and passes
  the pointing as africanus' `phase_dir`, which then reaches only the parallactic angle
  transformer. `angles.py` sets `FIELD_CENTRE` to the same direction, covering
  `input_model.apply_p_jones` on model columns, `output.apply_p_jones_inv` and the
  `parallactic_angle` term.
- **Rationale:** the beam and the parallactic angles are properties of where the dishes point;
  the phase centre is a freely-shiftable convention, and the two coincide only on an
  unrephased MS. Supplying `lm` is what frees `phase_dir` to mean the pointing: a transformer
  only runs for arguments that are missing, and a hand-supplied `lm` is bit-identical to the
  transformer's (verified, max abs difference 0.0). Shifting `beam_lm_extents` by the pointing
  offset instead looks like a one-liner and is wrong: the term samples at `R(pa)·lm` and
  *then* indexes the cube, so a shifted cube puts the beam centre at `R(-pa)·lm_p` — it orbits
  the phase centre as the parallactic angle swings, an error as large as the offset itself for
  alt-az dishes. Supplying `feed_parangle`/`beam_parangle` ready-made is not viable either: no
  `dask_schema` declares dims for them, and africanus' dask wrapper sums over every dim
  outside `(source, row, chan, corr)`, so a per-chunk lookup table cannot be expressed — they
  would be broadcast whole while the samplers index them by the chunk's own unique times.
  `POINTING.DIRECTION` is the better truth (it is the only one that captures on-the-fly
  mosaicking) but is per-antenna and per-dump, and africanus' per-antenna hook
  (`beam_point_errors`) is commented out upstream, so only a single field-level direction is
  representable; it would have to be reduced, and that table is frequently empty or very
  large. A candidate is skipped when absent or non-finite because some writers never populate
  `REFERENCE_DIR`; `(0, 0)` is a real sky position and cannot serve as the sentinel.
- **Consequences:** on an MS where all the FIELD directions agree the change is an exact
  no-op — `test_predict` still reproduces the MeqTrees `MODEL_DATA` with a beam and
  `apply_p_jones` both active. Repointing the parallactic angles is a second-order correction
  next to the beam (a median of 0.2-2 degrees of angle error for a 1 degree offset at
  MeerKAT's latitude, rising without bound for a field transiting near zenith, where the angle
  flips through 180 degrees), but both parangle paths had to move together or the P-Jones
  applied during the predict would disagree with the inverse applied on output. Two things are
  now QuartiCal's to maintain: the lm projection itself, since `LMTransformer` no longer runs
  (a future africanus change to that convention would not reach us), and the fact that
  `extras["phase_dir"]` holds the pointing — africanus' name, our meaning, flagged at the
  site. No `MEASINFO` frame check is performed: dask-ms does not surface column keywords, so
  an AZEL `REFERENCE_DIR` — a time-dependent direction neither a fixed beam centre nor
  `_make_parangles`' hardcoded J2000 can represent — would be used as though it were J2000.
  `PointedBeamCubeDDE` leans on two africanus internals: that the constructor returned by
  `init_fields` is called positionally, and that its signature is checked against the declared
  inputs (hence the `co_varnames` rename, which avoids duplicating ~200 lines of jitted beam
  sampling). An africanus release that accepts a beam centre and a pointing direction of its
  own should retire both the subclass and the supplied `lm`.
- **Source:** branch fix-beam-centre-on-rephased-ms (2026-08-31), addressing
  ratt-ru/QuartiCal#439.

## Known debt (do not entrench)

Testimony from the lead developer (interview 2026-07-07). An LLM extending QuartiCal should
treat these as scars, not patterns to replicate:

- **The CLI / config system.** Dynamically expanding config (one section per gain term) is
  not something argparse or click support, and no simpler-but-equal design was ever found.
  The complexity is tolerated, not endorsed.
- **Per-solver code duplication.** Solver kernels were deliberately kept separate so each
  could evolve independently, avoiding "hideous if-else ladders" across correlation and
  parameterisation variants — but the result was that adding a feature across all solvers
  was painful, and the original choice "may have been misguided". **Largely addressed
  2026-07:** the `compute_jhj_jhr` accumulation loop now lives once in
  `gains/general/solver_components.py` and all 13 solvable kernels bind it through hook
  factories (leakage reuses complex's binding; no fallback holdouts remain), and
  `compute_update` lives once in `gains/general/solver_components.py`. What stays per-kernel is
  the per-term maths (elem/flush/resid/stage hooks and `finalize_update`) — which is the
  part that *should* vary. New gain types should bind the shared loop, not copy one.
- **The offset parameter means the same thing in every term. Resolved 2026-08-20** — see
  "The zero-mean offset shear removed from the delay/tec pre/post-solve hooks". All three
  offset terms now store the phase at their band reference point, `pre_solve`/`post_solve`
  are pure unit changes in all five frequency-dependent terms, and
  `testing/tests/gains/test_solver_basis.py` pins the two bases against each other.
  What remains is the `jhj` treatment: `post_solve` unscales only the diagonal blocks, so
  the exported parameter precisions for the blocks coupling rescaled to unrescaled
  parameters stay in the solver basis. That is a reporting gap, not a solve error, and is
  untouched here.
- **Dask itself.** No longer improving upstream and largely fallen out of favour; the
  project will almost certainly move away from it at some point. Avoid deepening dask
  coupling in new code where a scheduler-agnostic seam is possible.
- **Numba code generation.** The factory pattern's optimisation payoff was never verified,
  and the kernels do not reliably produce vectorised (SIMD) machine code — an acknowledged
  improvement area. Largely addressed 2026-07: assembly inspection (2026-07-08) confirmed
  the hot loops were 100% scalar and memory-bound; the **complex** kernel was rewritten on
  tuple-based factories for a 2.1-2.8x measured speedup (see "Tuple-based kernel maths in
  the complex solver"), and the optimisation was then propagated to **every** other
  solvable kernel via the shared accumulation loop for measured 1.4-2.8x speedups (see
  "Shared hook-parameterised accumulation loop"). All kernel accumulation is now
  tuple-based; the array-buffer factories survive only outside the hot accumulation loop
  (e.g. `general/generics.py` residual computation, inversion buffers). The remaining
  known headroom is cross-visibility SIMD, judged not worth the complexity yet.

## Recurring gotchas

Things that have repeatedly bitten the developers and contributors (interview 2026-07-07):

- Adding a new solver requires touching all the registration locations — follow
  [solver-architecture.md](solver-architecture.md), "Adding a gain type", to the letter.
- Returning nested tuples from an inlined (`qcjit`) helper called inside a `prange` body
  crashes compilation: numba's parfor array analysis misreads tuple-of-tuples returns as
  array shapes and fails with `AssertionError: Dimension mismatch` (seen with numba
  0.65.1). Return a single flat tuple and index it with literals instead — this is why the
  shared accumulation loop's JHJ/JHr accumulator is one flat tuple and why its chain-product
  block is inlined rather than factored into a helper returning four operator tuples.
- Accidentally introducing shared root nodes into the dask graph sends task ordering
  haywire.
- CASA table caching underneath dask-ms can produce suspicious memory footprints depending
  on how the data is tiled.
- A direction-dependent gain needs *both* a direction-dependent model *and* the
  direction-dependent label on the term itself.
- The model specification syntax (`input_model.recipe`) is complicated; expect user error.
- MAD flagging overflags when the model is incomplete and the data is poorly calibrated.
- Robust reweighting only triggers across multiple solver epochs — one epoch silently does
  no reweighting.
- Gain terms cannot change type when loaded from disk (arguably should be allowed between
  amplitude/phase/diag_complex — an open improvement).
- Chunk sizes place an upper limit on solution intervals, and solving over all time will
  likely load the entire dataset into memory.
