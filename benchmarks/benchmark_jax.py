"""Benchmark NumPy vs JAX likelihood performance."""

import time, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")

from bigspy.specfit import load_test_spectrum
from bigspy import SpecFit
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.dust import DustAttenuation
from bigspy.mcmc.likelihood import Likelihood
from bigspy.mcmc.likelihood_jax import JAXLikelihood
from bigspy.mcmc.sfh import DelayedExponentialSFH


def setup():
    """Load test data and build both likelihoods."""
    pca = "template/BC03_Padova1994_chab_PCA_extend_new.fits"
    ssp_file = "template/SSP_BC03_Padova1994_chab.fits"
    tst = "tests/manga-7443-12703-28-28.pkl"

    dt = load_test_spectrum(tst)
    r = SpecFit(pca).fit(wave=dt["wave_obs"], flux=dt["flux_obs"],
                          error=dt["error_obs"], mask=dt["mask_obs"],
                          z_sys=dt["z"], mode="mode2")

    ssp = SSPLibrary(ssp_file, wave_range=(3600, 7400))
    dust = DustAttenuation.from_mode2(ssp.wave, r.p1, r.p2)
    ve, vd = r.ve[0], r.vd[0]
    w, fl, er = r.wave_prep, r.flux_prep, r.error_prep
    mask = np.asarray(r.mask_prep, dtype=bool)

    like_np = Likelihood(ssp, w, fl, er, mask, ve, vd, dust)
    like_jx = JAXLikelihood(ssp, w, fl, er, mask, ve, vd, dust)

    return like_np, like_jx


def run_benchmark(like_np, like_jx, N, n_repeat=5):
    """Benchmark NumPy vs JAX."""
    rng = np.random.RandomState(0)
    logZ_arr = rng.uniform(-2.5, 0.5, N)
    t0_arr = rng.uniform(0.5, 13.0, N)
    tau_arr = rng.uniform(0.3, 8.0, N)
    sfh_params = np.column_stack([t0_arr, tau_arr])

    sfh = DelayedExponentialSFH

    # ── NumPy ──
    times_np = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        chi2_np = like_np.call_batch(logZ_arr, sfh, sfh_params)
        t1 = time.perf_counter()
        times_np.append(t1 - t0)

    # ── JAX (first call includes compilation) ──
    t0 = time.perf_counter()
    chi2_jx = like_jx.call_batch(logZ_arr, sfh, sfh_params)
    t_first = time.perf_counter() - t0

    times_jx = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        _ = like_jx.call_batch(logZ_arr, sfh, sfh_params)
        t1 = time.perf_counter()
        times_jx.append(t1 - t0)

    # ── Numerical consistency ──
    max_diff = np.max(np.abs(chi2_np - chi2_jx))
    mean_chi2 = np.mean(chi2_np)

    return {
        "N": N,
        "numpy_mean": np.mean(times_np),
        "jax_first": t_first,
        "jax_mean": np.mean(times_jx),
        "speedup": np.mean(times_np) / np.mean(times_jx),
        "max_diff": max_diff,
        "rel_diff": max_diff / mean_chi2 if mean_chi2 > 0 else 0,
    }


if __name__ == "__main__":
    print("Setting up...")
    like_np, like_jx = setup()
    print(f"  NumPy ndof = {like_np.ndof}")
    print(f"  JAX  ndof = {like_jx.ndof}")
    print()

    print(f"{'N':>6s}  {'NumPy (s)':>10s}  {'JAX 1st (s)':>12s}  "
          f"{'JAX mean (s)':>13s}  {'Speedup':>8s}  {'Δchi2':>10s}  {'Δrel':>8s}")
    print("-" * 76)

    for N in [10, 50, 200, 500, 1000]:
        res = run_benchmark(like_np, like_jx, N)
        print(f"{res['N']:>6d}  {res['numpy_mean']:>10.4f}  "
              f"{res['jax_first']:>12.4f}  {res['jax_mean']:>13.4f}  "
              f"{res['speedup']:>7.1f}x  {res['max_diff']:>10.2e}  "
              f"{res['rel_diff']:>8.2e}")

    print()
    print("Done — JAX JIT working.")
