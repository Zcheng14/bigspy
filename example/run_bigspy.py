#!/usr/bin/env python3
"""bigspy demo -- two-stage spectral fitting with Nested Slice Sampling.

Stages:
    1. SpecFit : PCA fit for kinematics (ve, vd) and dust (mode 2).
    2. MCMC    : blackjax Nested Slice Sampling of the double power-law SFH
                 (tau, alpha, beta) plus metallicity logZsun.

Usage:
    python example/run_bigspy.py

Outputs (figures and data) are written under ``out/``.
"""
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bigspy.specfit import load_test_spectrum
from bigspy import SpecFit, MCMCFitter

# ── Paths ──────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCA_FILE = os.path.join(REPO, "template", "BC03_Padova1994_chab_PCA_extend_new.fits")
SSP_FILE = os.path.join(REPO, "template", "SSP_BC03_Padova1994_chab.fits")
TEST_FILE = os.path.join(REPO, "tests", "manga-7443-12703-28-28.pkl")
OUT_DIR = os.path.join(REPO, "out")
STATE_DIR = os.path.join(OUT_DIR, "state_dpl")
os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)

# NSS controls (run() defaults): n_live=1000, num_delete=100, num_inner_steps=10.
N_LIVE = 1000

print("=" * 60)
print("  bigspy -- Bayesian Inference of Galaxy Spectra")
print("=" * 60)

# ── 1. Load observed spectrum ───────────────────────────────────────
print("\n[1/4] Load observed spectrum")
data = load_test_spectrum(TEST_FILE)
print(f"  z = {data['z']:.4f}   pixels = {len(data['wave_obs'])}")

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(data["wave_obs"], data["flux_obs"], "k-", lw=0.5)
ax.set_xlabel(r"$\lambda_{\rm obs}\ (\mathrm{\AA})$")
ax.set_ylabel(r"$F_\lambda$")
ax.set_title("Observed Spectrum (observed frame)")
fig.savefig(os.path.join(OUT_DIR, "figs", "01_observed_spectrum.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)

# ── 2. SpecFit: kinematics + dust ───────────────────────────────────
print("\n[2/4] SpecFit -- kinematics + dust")
t0 = time.perf_counter()
specfit = SpecFit(PCA_FILE).fit(
    wave=data["wave_obs"], flux=data["flux_obs"], error=data["error_obs"],
    mask=data["mask_obs"], z_sys=data["z"], mode="mode2")
print(f"  elapsed {time.perf_counter() - t0:.1f}s")
print(f"  v_e = {specfit.ve[0]:.1f} +/- {specfit.ve[1]:.1f} km/s")
print(f"  v_d = {specfit.vd[0]:.1f} +/- {specfit.vd[1]:.1f} km/s")
print(f"  E(B-V) = {specfit.ebv[0]:.4f} +/- {specfit.ebv[1]:.4f}   "
      f"p1={specfit.p1:.4f}  p2={specfit.p2:.4f}")
specfit.plot_fit(os.path.join(OUT_DIR, "figs", "02_specfit.png"))
specfit.plot_dust(os.path.join(OUT_DIR, "figs", "02b_dust_curve.png"))

# ── 3. MCMC: NSS with the double power-law SFH ──────────────────────
print(f"\n[3/4] MCMC -- Nested Slice Sampling ({N_LIVE} live points, DPL)")
mc = MCMCFitter(ssp_fits=SSP_FILE, specfit_result=specfit,
                sfh_model="dpl", wave_range=(3600, 7400))
t0 = time.perf_counter()
result = mc.run(n_live=N_LIVE, out_dir=STATE_DIR, seed=0)
print(f"  elapsed {time.perf_counter() - t0:.1f}s")
print(f"  log Z = {result.log_evidence:.2f}")
for name, value in result.bestfit.items():
    print(f"  best-fit {name:10s} = {value:.4f}")

# ── 4. Figures + results ────────────────────────────────────────────
print("\n[4/4] Figures + results")
result.plot_corner(os.path.join(OUT_DIR, "figs", "03_corner.png"))
result.plot_bestfit(os.path.join(OUT_DIR, "figs", "04_bestfit_csp.png"))
result.plot_sfh(os.path.join(OUT_DIR, "figs", "05_sfh_ci.png"))
result.save_result(os.path.join(OUT_DIR, "data", "mcmc_bestfit.fits"))
specfit.save(os.path.join(OUT_DIR, "data", "specfit_result.fits"))
np.save(os.path.join(OUT_DIR, "data", "posterior_samples.npy"), result.posterior)
print(f"  wrote {OUT_DIR}/figs/*.png and {OUT_DIR}/data/*")
print("\nbigspy demo complete")
