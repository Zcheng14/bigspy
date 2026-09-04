"""Benchmark (composite): full MCMC hot path Likelihood.call_batch — old vs new.

Runs a real SpecFit (mode2) on the MaNGA test spectrum, builds both the
archived (old) and current (new) NumPy Likelihood objects from the same
SSP file and SpecFit outputs, then times batched chi2 evaluation at the
UltraNest vectorized batch sizes. This is the headline number for the
whole optimization round (P1+P2+P3+P4 combined).

Run from repo root:  python benchmarks/benchmark_mcmc_pipeline.py

Parity between old and new chi2 values is ASSERTED (rel 1e-10, matching
tests/test_mcmc.py::TestVectorized; measured exactly 0 on the reference
machine).
"""
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)                         # repo root (for archive.v0_6d70457)
sys.path.insert(0, os.path.join(REPO, "tests"))  # tests/ for _synth

import numpy as np

if not os.path.isdir(os.path.join(REPO, "archive")):
    print("archive/ not found — old-vs-new benchmarks require it (pre-cleanup state).")
    sys.exit(0)

import _synth
import bigspy.specfit as new_sf
import bigspy.mcmc.likelihood as new_like_mod
import bigspy.mcmc.ssp as new_ssp
from bigspy.mcmc.dust import DustAttenuation as NewDust
from bigspy.mcmc.sfh import DelayedExponentialSFH as NewSFH
import archive.v0_6d70457.bigspy.mcmc.likelihood as old_like_mod
import archive.v0_6d70457.bigspy.mcmc.ssp as old_ssp
from archive.v0_6d70457.bigspy.mcmc.dust import DustAttenuation as OldDust
from archive.v0_6d70457.bigspy.mcmc.sfh import DelayedExponentialSFH as OldSFH

N_REPEAT = 5
PCA_REAL = os.path.join(REPO, "template", "BC03_Padova1994_chab_PCA_extend_new.fits")
SSP_REAL = os.path.join(REPO, "template", "SSP_BC03_Padova1994_chab.fits")
PKL_REAL = os.path.join(REPO, "tests", "manga-7443-12703-28-28.pkl")


def run_section(title, ssp_path, wave, flux, err, mask, ve, vd, p1, p2, n_list):
    print(f"\n== {title} ==")
    ssp_old = old_ssp.SSPLibrary(ssp_path)
    ssp_new = new_ssp.SSPLibrary(ssp_path)
    like_old = old_like_mod.Likelihood(
        ssp_old, wave, flux, err, mask, ve, vd,
        OldDust.from_mode2(ssp_old.wave, p1, p2))
    like_new = new_like_mod.Likelihood(
        ssp_new, wave, flux, err, mask, ve, vd,
        NewDust.from_mode2(ssp_new.wave, p1, p2))
    print(f"  vd={vd:.1f} km/s (sigma_pix={vd / new_ssp_vel:.2f}), "
          f"n_wave_ssp={ssp_new.n_wave}, n_obs={len(wave)}")

    # Warm-up (fills the new convolution-matrix cache; excluded from timing)
    like_old.call_batch(np.array([0.0]), OldSFH, np.array([[1.0, 3.0]]))
    like_new.call_batch(np.array([0.0]), NewSFH, np.array([[1.0, 3.0]]))

    rng = np.random.RandomState(0)
    print(f"{'N':>6} | {'old (s)':>10} | {'new (s)':>10} | {'speedup':>8} | "
          f"{'max|dchi2|':>10} | {'rel':>8}")
    print("-" * 70)
    for n in n_list:
        logZ = rng.uniform(-2.5, 0.5, n)
        params = np.column_stack([rng.uniform(0.5, 13.0, n),
                                  rng.uniform(0.3, 8.0, n)])
        times_old, times_new = [], []
        c_old = c_new = None
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
            f"parity violated at N={n}: max|dchi2|={max_diff:.3e}, rel={rel:.3e} > 1e-10"
        print(f"{n:>6} | {np.mean(times_old):>10.4f} | {np.mean(times_new):>10.4f} | "
              f"{np.mean(times_old) / np.mean(times_new):>7.1f}x | "
              f"{max_diff:>10.1e} | {rel:>8.1e}")


new_ssp_vel = (10 ** 0.0001 - 1) * 299792.458  # for the info print only


def main():
    if all(os.path.isfile(p) for p in (PCA_REAL, SSP_REAL, PKL_REAL)):
        data = new_sf.load_test_spectrum(PKL_REAL)
        res = new_sf.SpecFit(PCA_REAL).fit(
            wave=data["wave_obs"], flux=data["flux_obs"],
            error=data["error_obs"], mask=data["mask_obs"],
            z_sys=data["z"], mode="mode2")
        run_section("Real MaNGA spectrum via SpecFit(mode2) + real SSP",
                    SSP_REAL, res.wave_prep, res.flux_prep, res.error_prep,
                    res.mask_prep, res.ve[0], res.vd[0], res.p1, res.p2,
                    (10, 50, 200, 400, 1000))
    else:
        print("LFS reference data not found — falling back to synthetic setup.")
        with tempfile.TemporaryDirectory() as tmp:
            ssp_path = os.path.join(tmp, "synth_ssp.fits")
            _synth.make_ssp_fits(ssp_path)
            ssp = new_ssp.SSPLibrary(ssp_path)
            rng = np.random.RandomState(0)
            wave = ssp.wave[::2] + 2.3
            flux = rng.uniform(0.5, 2.0, len(wave))
            err = np.full(len(wave), 0.02)
            mask = np.ones(len(wave), dtype=bool)
            run_section("Synthetic fallback", ssp_path, wave, flux, err, mask,
                        0.0, 120.0, 0.03, -0.001, (10, 50, 200, 400, 1000))


if __name__ == "__main__":
    main()
