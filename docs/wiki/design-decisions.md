---
type: decision-ledger
title: Design Decisions
description: "Why QuartiCal is built the way it is — a ledger of decisions, their rationale, and their consequences. Append new entries as decisions land."
timestamp: 2026-07-20
last_verified_commit: ce9de22
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
  `quartical/gains/general/accumulation.py` as `build_jhj_jhr_impl`, parameterised by
  per-term hook factories (elem / acc_zeros / flush / resid / stage / mirror); each
  kernel keeps a ~15-line `compute_jhj_jhr` overload that binds its hooks. The hook
  contract is documented in solver-architecture.md ("Numba kernel conventions").
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
  CACHE CORRECTNESS CONSTRAINT in `accumulation.py`'s docstring.
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
  motion through the hooks.
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
  loop and was absorbed via two extra optional inputs (`scalar_jhj_jhr`,
  `reference_gains`). Known variance-inventory correction: every parameterised kernel
  passes `numbness=1e9` explicitly except amplitude (default 1e-6).
- **Source:** branch kernel-unification, commits 2d9d8d2..ce9de22 (2026-07-17 to
  2026-07-20); verification numbers in
  `~/claude_artifacts/quaritcal_optimisation/results/UNIFICATION_LOG.md`
  ("Solver-loop unification").

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
- **Decision:** One `@pytest.mark.slow` module, `testing/tests/gains/test_chain.py`,
  excluded from CI by default (`pytest -m "not slow"` in `ci.yaml`). It solves a single
  three-term chain (complex G + delay K + diag_complex B) at one correlation mode, in two
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
  module runs in ~100 s cold / ~20 s with a warm numba cache). The slow marker keeps
  the compile cost out of the default CI matrix while leaving the coverage one
  `-m slow` away.
- **Consequences:** Chain mechanics regressions (operator products, per-term
  time/freq/dir maps, DD stacking) are now caught by an opt-in test instead of ad-hoc A/B
  probes. CI no longer runs anything marked slow — genuinely slow future tests can use
  the marker freely, but a periodic/manual slow run is needed for their coverage to
  count. Fixing this test also surfaced a latent `n_dir > 1` bug in
  `testing/utils/gains.py:reference_gains` (antenna loop outside the direction loop —
  the same stale-direction shape as the legacy kernel bug).
- **Source:** chain-mechanics A/B probe and design discussion 2026-07-16
  (`~/claude_artifacts/quaritcal_optimisation/results/UNIFICATION_LOG.md`, "Chain-mechanics
  A/B probe"); user requirement that compilation cost stay bounded and per-term
  assertions be avoided.

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
  `gains/general/accumulation.py` and all 13 solvable kernels bind it through hook
  factories (leakage reuses complex's binding; no fallback holdouts remain), and
  `compute_update` lives once in `gains/general/solver_ops.py`. What stays per-kernel is
  the per-term maths (elem/flush/resid/stage hooks and `finalize_update`) — which is the
  part that *should* vary. New gain types should bind the shared loop, not copy one.
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
