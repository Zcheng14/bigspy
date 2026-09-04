"""Benchmark P3: CSPBuilder.build_batch — old (archive.v0_6d70457) vs new.

Run from repo root:  python benchmarks/benchmark_csp.py

Parity between old and new outputs is ASSERTED (tolerance atol=1e-12, matching
tests/test_csp.py; measured 0 on the reference machine).
"""
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                        # repo root (for archive.v0_6d70457)
sys.path.insert(0, os.path.join(REPO, "tests"))  # tests/ for _synth

import numpy as np

if not os.path.isdir(os.path.join(REPO, "archive")):
    print("archive/ not found — old-vs-new benchmarks require it (pre-cleanup state).")
    sys.exit(0)

import _synth
import bigspy.mcmc.csp as new_csp
import bigspy.mcmc.ssp as new_ssp
from bigspy.mcmc.sfh import DelayedExponentialSFH as NewSFH
import archive.v0_6d70457.bigspy.mcmc.csp as old_csp
import archive.v0_6d70457.bigspy.mcmc.ssp as old_ssp
from archive.v0_6d70457.bigspy.mcmc.sfh import DelayedExponentialSFH as OldSFH

N_REPEAT = 5
REAL_SSP = os.path.join(REPO, "template", "SSP_BC03_Padova1994_chab.fits")


def make_params(rng, n):
    return np.column_stack([rng.uniform(0.5, 13.0, n), rng.uniform(0.3, 8.0, n)])


def run_section(title, ssp_path, n_list, n_repeat=N_REPEAT):
    print(f"\n== {title} ==")
    b_old = old_csp.CSPBuilder(old_ssp.SSPLibrary(ssp_path))
    b_new = new_csp.CSPBuilder(new_ssp.SSPLibrary(ssp_path))
    rng = np.random.RandomState(0)
    # Warm-up (first-call numpy/BLAS initialization, excluded from timing)
    rng_warm = np.random.RandomState(1)
    warm = make_params(rng_warm, 16)
    b_old.build_batch(rng_warm.uniform(-2.5, 0.5, 16), warm, OldSFH)
    b_new.build_batch(rng_warm.uniform(-2.5, 0.5, 16), warm, NewSFH)
    print(f"{'N':>6} | {'old (s)':>10} | {'new (s)':>10} | {'speedup':>8} | "
          f"{'max|dCSP|':>10} | {'rel':>8}")
    print("-" * 70)
    for n in n_list:
        logZ = rng.uniform(-2.5, 0.5, n)
        params = make_params(rng, n)
        times_old, times_new = [], []
        for _ in range(n_repeat):
            t0 = time.perf_counter()
            out_old = b_old.build_batch(logZ, params, OldSFH)
            times_old.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            out_new = b_new.build_batch(logZ, params, NewSFH)
            times_new.append(time.perf_counter() - t0)
        max_diff = float(np.max(np.abs(out_old - out_new)))
        rel = float(np.max(np.abs(out_old - out_new) / np.maximum(np.abs(out_old), 1e-300)))
        assert max_diff <= 1e-12, \
            f"parity violated at N={n}: max|dCSP|={max_diff:.3e} > 1e-12"
        print(f"{n:>6} | {np.mean(times_old):>10.4f} | {np.mean(times_new):>10.4f} | "
              f"{np.mean(times_old) / np.mean(times_new):>7.1f}x | {max_diff:>10.1e} | {rel:>8.1e}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        synth_path = os.path.join(tmp, "synth_ssp.fits")
        _synth.make_ssp_fits(synth_path)
        run_section("Synthetic SSP (3 metal x 6 age x 761 px)", synth_path,
                    (10, 50, 200, 400, 1000))

    if os.path.isfile(REAL_SSP):
        run_section("Real SSP (6 x 196 x 3129 px)", REAL_SSP, (10, 50, 200, 400))
    else:
        print("\nReal SSP FITS not found — skipping real-data section.")


if __name__ == "__main__":
    main()
