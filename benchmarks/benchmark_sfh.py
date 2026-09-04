"""Benchmark P4: DelayedExponentialSFH.evaluate_batch — old (archive.v0_6d70457) vs new.

Run from repo root:  python benchmarks/benchmark_sfh.py

Parity between old and new outputs is ASSERTED (tolerance 1e-15, i.e.
bit-exact up to a single ulp; measured 0 on the reference machine).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np

if not os.path.isdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")):
    print("archive/ not found — old-vs-new benchmarks require it (pre-cleanup state).")
    sys.exit(0)

import bigspy.mcmc.sfh as new_sfh
import archive.v0_6d70457.bigspy.mcmc.sfh as old_sfh

N_REPEAT = 5


def bench_one(mod, t, params, n_repeat=N_REPEAT):
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        out = mod.DelayedExponentialSFH.evaluate_batch(t, params)
        times.append(time.perf_counter() - t0)
    return out, np.mean(times)


def main():
    t = np.linspace(0.5, 13.8, 196)
    rng = np.random.RandomState(0)

    print(f"{'N':>6} | {'old (s)':>10} | {'new (s)':>10} | {'speedup':>8} | {'max|dSFR|':>10}")
    print("-" * 60)
    for n in (10, 100, 1000, 10000):
        params = np.column_stack([rng.uniform(0.1, 13.5, n),
                                  rng.uniform(0.1, 10.0, n)])
        out_old, t_old = bench_one(old_sfh, t, params)
        out_new, t_new = bench_one(new_sfh, t, params)
        max_diff = float(np.max(np.abs(out_old - out_new)))
        assert max_diff <= 1e-15, \
            f"parity violated at N={n}: max|dSFR|={max_diff:.3e} > 1e-15"
        print(f"{n:>6} | {t_old:>10.4f} | {t_new:>10.4f} | "
              f"{t_old / t_new:>7.1f}x | {max_diff:>10.1e}")


if __name__ == "__main__":
    main()
