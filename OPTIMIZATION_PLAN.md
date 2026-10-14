# Optimization Plan — NSS Sampling Performance

Status: **analysis only, no code changed yet**. This document records the current performance profile of the nested-slice-sampling (NSS) pipeline and candidate optimizations, to be picked up if sampling throughput becomes a bottleneck (e.g. bulk fitting of thousands of spectra). None of the items below are implemented.

Only optimizations that are **mathematically equivalent** (for a fixed seed, the sampler follows the identical deterministic trajectory) are admissible; see "Root cause and constraint" for why anything that attacks the bottleneck directly changes the sampling semantics.

## Current implementation

`NSSampler.run` (`src/bigspy/mcmc/sampler.py`) drives `blackjax.nss` from a Python `while` loop:

- each loop iteration calls one `jax.jit`-compiled step that deletes and replaces `num_delete` (default 100) live points; inside that step, `blackjax` replaces the points in parallel (`jax.vmap` over the particles) and runs `num_inner_steps` (default 10) constrained slice moves per particle sequentially (`jax.lax.scan`), with data-dependent stepping-out/shrinkage loops;
- the loop condition `bool(logZ_live - logZ >= -delta_logz)` forces a host–device synchronization at every iteration;
- the run therefore consists of roughly `n_dead / num_delete` (~140 for the reference run below) device launches paced by the Python host, with the device idle between them.

## Measured profile

Reference run: MaNGA spectrum 7443-12703-28-28 (SpecFit mode-2 result, DPL SFH), `n_live=1000, num_delete=100, num_inner_steps=10, seed=0`; ~13,800 likelihood calls to produce 14,000 dead points; jax 0.6.2 (cuda12), blackjax 1.7.1, float32; Tesla V100 32 GB (driver 550.54.15) versus an 80-core CPU host.

| Quantity | GPU (V100) | CPU (80 threads) |
|---|---|---|
| Full NSS run, wall time | 21.8 s | 84.3 s (3.9× slower) |
| Likelihood throughput, batch=64 | 12,600 ll/s | 7,700 ll/s (1.6×) |
| First JIT compile of the likelihood | 1.43 s | 0.46 s |

GPU utilization during sampling stays near 0% (one ~87% spike observed): the run is **host/launch-bound** — the Python loop and the per-step synchronization, not device compute, set the pace. Two observations bracket the headroom: at batch=64 the device manages only 1.6× the throughput of the full 80-thread CPU host (1.4× against a 16-thread pinned CPU), and the end-to-end 3.9× advantage over the default CPU setup shrinks to 3.0× once the CPU is pinned to its 16-thread optimum (see the table below). That is: roughly a third of the nominal speedup is CPU oversubscription, and the remaining 3.0× is achieved with the GPU essentially idle — far from saturation.

CPU thread scaling on the same run (affinity pinned with `taskset -c`; clean, otherwise-idle host):

| CPU threads | Wall time (s) | Relative to 80 threads |
|---|---|---|
| 80 | 84.3 | 1.00 |
| 32 | 68.8 | 0.82 |
| 16 | 64.4 | 0.76 |
| 8 | 71.7 | 0.85 |
| 4 | 93.5 | 1.11 |

Every configuration follows the identical deterministic trajectory (same 13,800 likelihood calls, same `logZ`), so the table isolates pure threading overhead. With batch-1-sized kernels the XLA CPU thread pool is mostly synchronization overhead past ~16 threads: the default "all cores" configuration is ~24% slower than the optimum on this host, and pinning to 16 threads recovers more time than the entire GPU speedup gives (21.8 s GPU vs 64.4 s tuned CPU = 3.0×, versus 3.9× against the untuned CPU).

## Root cause and constraint

The essential bottleneck is the **adaptivity** of the inner slice moves, not the likelihood kernel or the host loop alone. Stepping-out and shrinkage are data-dependent `while_loop`s whose trip counts differ per particle; under the `vmap` over replacements every loop therefore runs to the slowest particle, and each iteration is a batch-`num_delete` likelihood kernel with only microseconds of work. This adaptive kernel chain is what keeps the device idle between launches — and the adaptivity itself is the part any speedup would have to remove or fix, which is exactly what changes the sampler's mathematics.

Constraint: only **mathematically equivalent** optimizations are admissible — for a fixed seed the sampler must follow the identical deterministic trajectory, i.e. the same RNG consumption order, the same per-particle adaptive loop trip counts, and the same per-step termination. Making the adaptive loops faster under this constraint is an **open problem**; no candidate is currently identified. Ideas that change the RNG stream, replace the adaptive loops with fixed proposal counts, or alter termination timing are rejected by this constraint, however large their speedup.

## Candidate optimizations

### CPU thread pinning (no algorithm change, mathematically equivalent)

What: on many-core CPU hosts, cap the threads visible to the process (e.g. `taskset -c 0-15 python ...` or `os.sched_setaffinity`) around the measured optimum (~16 for the reference host; re-measure per machine). Could be surfaced as a README note or a helper in the benchmark tooling.

Expected effect: see the CPU scaling table — on the reference host this is of the same order as the entire GPU speedup.

Risk: none (pure environment configuration); the optimum is host-dependent. Every pinned configuration follows the identical deterministic trajectory, so mathematical equivalence is preserved.

## Benchmarking protocol

Any change claiming a speedup must be measured: on an otherwise-idle machine (no concurrent test runs or other CPU load), with fixed seed and identical `n_live / num_delete / num_inner_steps / delta_logz`, CPU affinity pinned and reported, and the GPU/CPU comparison run back-to-back on the same host state. Report wall time, likelihood-call count, `logZ`, and GPU utilization.

## Non-goals

- No change to the two-stage (SpecFit → MCMC) architecture or to the sampling semantics (priors, evidence definition, termination defaults).
- Only mathematically equivalent optimizations (identical deterministic trajectory for a fixed seed); proposals that alter the RNG stream, the slice-loop adaptivity, or the termination timing are rejected.
- float64 mode (`jax_enable_x64`) is slower on GPU and out of scope for throughput work.
