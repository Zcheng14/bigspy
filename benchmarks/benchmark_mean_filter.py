"""Benchmark P5: specfit.mean_filter — old per-pixel loop vs vectorized.

Run from repo root:  python benchmarks/benchmark_mean_filter.py

Parity between old and new outputs is ASSERTED (tolerance 1e-13; measured
bit-identical on the reference machine).
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

import bigspy.specfit as new_sf
import archive.v0_6d70457.bigspy.specfit as old_sf

N_REPEAT = 5


def main():
    wave = np.linspace(3600, 7000, 3401)  # internal 1 A grid
    rng = np.random.RandomState(0)
    flux = rng.uniform(0.5, 2.0, 3401)
    mask_all = np.ones(3401)
    mask_rand = (rng.uniform(0, 1, 3401) > 0.2).astype(float)  # ~20% zeroed

    print(f"{'case':>18} | {'win':>4} | {'old (s)':>9} | {'new (s)':>9} | "
          f"{'speedup':>8} | {'max|d|':>8}")
    print("-" * 72)
    for wave_win in (200.0, 400.0):
        for label, mask in (("unmasked", None), ("mask=all-1", mask_all),
                            ("mask=random", mask_rand)):
            times_old, times_new = [], []
            out_old = out_new = None
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                out_old = old_sf.mean_filter(flux, wave, wave_win, mask=mask)
                times_old.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                out_new = new_sf.mean_filter(flux, wave, wave_win, mask=mask)
                times_new.append(time.perf_counter() - t0)
            max_diff = float(np.max(np.abs(out_old - out_new)))
            assert max_diff <= 1e-13, \
                f"parity violated ({label}, win={int(wave_win)}): {max_diff:.3e} > 1e-13"
            print(f"{label:>18} | {int(wave_win):>4} | {np.mean(times_old):>9.4f} | "
                   f"{np.mean(times_new):>9.4f} | "
                   f"{np.mean(times_old) / np.mean(times_new):>7.1f}x | {max_diff:>8.1e}")


if __name__ == "__main__":
    main()
