"""Benchmark P2: call_batch interpolation precompute — old vs new.

Section A isolates the interpolation step (interp1d rebuilt per call vs a
precomputed gather/lerp plan). Section B times the full Likelihood.call_batch
with vd=0 (broadening disabled) to isolate P2 from P1.

Run from repo root:  python benchmarks/benchmark_likelihood_interp.py

Parity between old and new outputs is ASSERTED (A: 1e-15, i.e. bit-exact up to
one ulp; B: rel 1e-10 matching tests/test_likelihood.py; both measured 0).
"""
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                         # repo root (for archive.v0_6d70457)
sys.path.insert(0, os.path.join(REPO, "tests"))  # tests/ for _synth

import numpy as np
from scipy.interpolate import interp1d

if not os.path.isdir(os.path.join(REPO, "archive")):
    print("archive/ not found — old-vs-new benchmarks require it (pre-cleanup state).")
    sys.exit(0)

import _synth
import bigspy.mcmc.likelihood as new_like_mod
import bigspy.mcmc.ssp as new_ssp
from bigspy.mcmc.sfh import DelayedExponentialSFH as NewSFH
from bigspy.mcmc.dust import DustAttenuation as NewDust
import archive.v0_6d70457.bigspy.mcmc.likelihood as old_like_mod
import archive.v0_6d70457.bigspy.mcmc.ssp as old_ssp
from archive.v0_6d70457.bigspy.mcmc.sfh import DelayedExponentialSFH as OldSFH
from archive.v0_6d70457.bigspy.mcmc.dust import DustAttenuation as OldDust

N_REPEAT = 5


def main():
    # ── A) Isolated interpolation: interp1d per call vs precomputed plan ──
    print("== A) batch interpolation: interp1d (rebuilt per call) vs plan.apply ==")
    x_src = np.linspace(3600, 7400, 3129)
    x_new = np.linspace(3650, 7350, 2600)
    plan = new_like_mod._LinearInterpPlan(x_src, x_new)
    rng = np.random.RandomState(1)
    print(f"{'N':>6} | {'old (s)':>10} | {'new (s)':>10} | {'speedup':>8} | {'max|d|':>8}")
    print("-" * 60)
    for n in (10, 100, 400, 1000):
        y = rng.normal(0, 1, (n, len(x_src)))
        times_old, times_new = [], []
        out_old = out_new = None
        for _ in range(N_REPEAT):
            t0 = time.perf_counter()
            out_old = interp1d(x_src, y, axis=1, kind='linear',
                               bounds_error=False, fill_value=0.0)(x_new)
            times_old.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            out_new = plan.apply(y)
            times_new.append(time.perf_counter() - t0)
        max_diff = float(np.max(np.abs(out_old - out_new)))
        assert max_diff <= 1e-15, \
            f"parity violated (interp, N={n}): max|d|={max_diff:.3e} > 1e-15"
        print(f"{n:>6} | {np.mean(times_old):>10.4f} | {np.mean(times_new):>10.4f} | "
              f"{np.mean(times_old) / np.mean(times_new):>7.1f}x | {max_diff:>8.1e}")

    # ── B) Full call_batch, vd=0 (isolates P2 from P1) ──
    print("\n== B) Likelihood.call_batch, vd=0, offset obs grid (synthetic SSP) ==")
    with tempfile.TemporaryDirectory() as tmp:
        ssp_path = os.path.join(tmp, "synth_ssp.fits")
        _synth.make_ssp_fits(ssp_path)
        ssp_new = new_ssp.SSPLibrary(ssp_path)
        ssp_old = old_ssp.SSPLibrary(ssp_path)
        obs_wave = np.concatenate([[3900.0], ssp_new.wave[::2] + 2.3, [7100.0]])
        rng2 = np.random.RandomState(2)
        obs_flux = rng2.uniform(0.5, 2.0, len(obs_wave))
        err = np.full(len(obs_wave), 0.02)
        mask = np.ones(len(obs_wave), dtype=bool)

        like_new = new_like_mod.Likelihood(
            ssp_new, obs_wave, obs_flux, err, mask, 0.0, 0.0,
            NewDust.from_mode2(ssp_new.wave, 0.03, -0.001))
        like_old = old_like_mod.Likelihood(
            ssp_old, obs_wave, obs_flux, err, mask, 0.0, 0.0,
            OldDust.from_mode2(ssp_old.wave, 0.03, -0.001))

        # Warm-up (excluded)
        like_old.call_batch(np.array([0.0]), OldSFH, np.array([[1.0, 3.0]]))
        like_new.call_batch(np.array([0.0]), NewSFH, np.array([[1.0, 3.0]]))

        print(f"{'N':>6} | {'old (s)':>10} | {'new (s)':>10} | {'speedup':>8} | "
              f"{'max|dchi2|':>10} | {'rel':>8}")
        print("-" * 70)
        for n in (10, 100, 400, 1000):
            logZ = rng2.uniform(-2.0, 0.2, n)
            params = np.column_stack([rng2.uniform(0.5, 13.0, n),
                                      rng2.uniform(0.3, 8.0, n)])
            times_old, times_new = [], []
            for _ in range(N_REPEAT):
                t0 = time.perf_counter()
                c_old = like_old.call_batch(logZ, OldSFH, params)
                times_old.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                c_new = like_new.call_batch(logZ, NewSFH, params)
                times_new.append(time.perf_counter() - t0)
            max_diff = float(np.max(np.abs(c_old - c_new)))
            rel = float(np.max(np.abs(c_old - c_new) / np.maximum(np.abs(c_old), 1e-300)))
            assert rel <= 1e-10, \
                f"parity violated (call_batch, N={n}): rel={rel:.3e} > 1e-10"
            print(f"{n:>6} | {np.mean(times_old):>10.4f} | {np.mean(times_new):>10.4f} | "
                  f"{np.mean(times_old) / np.mean(times_new):>7.1f}x | "
                  f"{max_diff:>10.1e} | {rel:>8.1e}")


if __name__ == "__main__":
    main()
