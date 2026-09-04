"""Benchmark P1: convolution-matrix build + batch convolution — old vs new.

Run from repo root:  python benchmarks/benchmark_conv_matrix.py

Parity between old and new outputs is ASSERTED (tolerance 1e-12, matching
tests/test_mcmc.py::TestConvolutionMatrix; measured 0 on the reference machine).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if not os.path.isdir(os.path.join(REPO, "archive")):
    print("archive/ not found — old-vs-new benchmarks require it (pre-cleanup state).")
    sys.exit(0)

import bigspy.mcmc.kinematics as new_kin
import archive.v0_6d70457.bigspy.mcmc.kinematics as old_kin

N_REPEAT = 5


def main():
    # ── A) Matrix construction: old double loop vs vectorized scatter ──
    print("== A) _build_convolution_matrix: old double loop vs vectorized ==")
    print(f"{'n_pix':>6} | {'sigma':>6} | {'x0':>4} | {'old (s)':>9} | "
          f"{'new (s)':>9} | {'speedup':>8} | {'max|dK|':>8}")
    print("-" * 70)
    for n_pix in (150, 3129):
        for sigma in (1.74, 4.3, 8.0):
            for x0 in (0.0, 1.3):
                times_old, times_new = [], []
                for _ in range(N_REPEAT):
                    t0 = time.perf_counter()
                    K_old = old_kin._build_convolution_matrix(n_pix, sigma, x0)
                    times_old.append(time.perf_counter() - t0)
                    t0 = time.perf_counter()
                    K_new = new_kin._build_convolution_matrix(n_pix, sigma, x0)
                    times_new.append(time.perf_counter() - t0)
                max_diff = float(np.max(np.abs(K_old - K_new)))
                assert max_diff <= 1e-12, \
                    f"parity violated (matrix, n={n_pix}, s={sigma}, x0={x0}): {max_diff:.3e}"
                print(f"{n_pix:>6} | {sigma:>6.2f} | {x0:>4.1f} | "
                       f"{np.mean(times_old):>9.4f} | {np.mean(times_new):>9.4f} | "
                       f"{np.mean(times_old) / np.mean(times_new):>7.1f}x | {max_diff:>8.1e}")

    # ── B) Steady-state batch convolution: rebuild-per-call vs cached ──
    print("\n== B) gauss_convolve_batch (N, 3129): old rebuild-every-call vs new cached ==")
    print(f"{'N':>6} | {'old (s)':>9} | {'new 1st (s)':>11} | {'new cached (s)':>14} | "
          f"{'speedup':>8} | {'max|dout|':>9}")
    print("-" * 78)
    rng = np.random.RandomState(0)
    for n in (10, 50, 200, 400):
        spectra = rng.normal(0, 1, (n, 3129))
        times_old, times_new = [], []
        out_old = out_new = None
        for rep in range(N_REPEAT):
            t0 = time.perf_counter()
            out_old = old_kin.gauss_convolve_batch(spectra, 1.74)
            times_old.append(time.perf_counter() - t0)
            new_kin.clear_convolution_cache()
            t0 = time.perf_counter()
            out_new = new_kin.gauss_convolve_batch(spectra, 1.74)
            first = time.perf_counter() - t0
            if rep == 0:
                t0 = time.perf_counter()
                out_new = new_kin.gauss_convolve_batch(spectra, 1.74)
                cached = time.perf_counter() - t0
            times_new.append((first, cached))
        max_diff = float(np.max(np.abs(out_old - out_new)))
        assert max_diff <= 1e-12, \
            f"parity violated (batch, N={n}): max|dout|={max_diff:.3e} > 1e-12"
        mean_old = np.mean(times_old)
        mean_first = np.mean([t[0] for t in times_new])
        mean_cached = np.mean([t[1] for t in times_new])
        print(f"{n:>6} | {mean_old:>9.4f} | {mean_first:>11.4f} | {mean_cached:>14.4f} | "
              f"{mean_old / mean_cached:>7.1f}x | {max_diff:>9.1e}")


if __name__ == "__main__":
    main()
