#!/usr/bin/env python3
"""
bigspy Demo — step-by-step two-stage spectral fitting pipeline

Usage:
    python example/run_bigspy.py

Sections:
    1. Imports & data paths
    2. Load observed spectrum
    3. SpecFit — kinematics + dust fitting
    4. MCMC  — stellar population inference
    5. Visualization
    6. Custom SFH model
    7. Save results
"""
import os, sys, time
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # example/ → bigs_v2/

PCA_FILE  = os.path.join(REPO, "template", "BC03_Padova1994_chab_PCA_extend_new.fits")
SSP_FILE  = os.path.join(REPO, "template", "SSP_BC03_Padova1994_chab.fits")
TEST_FILE = os.path.join(REPO, "tests", "manga-7443-12703-28-28.pkl")
OUT_DIR   = os.path.join(REPO, "out")
CHAIN_DIR = os.path.join(OUT_DIR, "chains_manga-7443")
os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "figs"), exist_ok=True)

N_LIVE = 200      # UltraNest live points (more = more accurate, slower)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  1. Imports                                                     ║
# ╚══════════════════════════════════════════════════════════════════╝
print("=" * 60)
print("  bigspy — Bayesian Inference of Galaxy Spectra")
print("=" * 60)

from bigspy.specfit import load_test_spectrum
from bigspy import SpecFit, MCMCFitter
from bigspy.mcmc.sfh import SFHBase, DelayedExponentialSFH
from bigspy.mcmc.priors import UniformPrior, LogUniformPrior, FixedPrior
import matplotlib.pyplot as plt

print("\nAll imports OK")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  2. Load observed spectrum                                      ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "-" * 40)
print("  2. Load Observed Spectrum")
print("-" * 40)

data = load_test_spectrum(TEST_FILE)
print(f"  Redshift           z = {data['z']:.4f}")
print(f"  Pixels               = {len(data['wave_obs'])}")
print(f"  Wavelength range     = {data['wave_obs'][0]:.0f} – {data['wave_obs'][-1]:.0f} Å")
print(f"  Dispersion (DAP)     = {data['sigma_dap']:.0f} km/s")

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(data["wave_obs"], data["flux_obs"], "k-", lw=0.5)
ax.set_xlabel(r"$\lambda_{\rm obs}\ (\mathrm{\AA})$")
ax.set_ylabel(r"$F_\lambda$")
ax.set_title("Observed Spectrum (observed frame)")
fig.savefig(os.path.join(OUT_DIR, "figs", "01_observed_spectrum.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  → figs/01_observed_spectrum.png")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  3. SpecFit — kinematics + dust                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "-" * 40)
print("  3. SpecFit — PCA fitting for kinematics + dust")
print("-" * 40)

t0 = time.perf_counter()
sf = SpecFit(PCA_FILE)
specfit = sf.fit(
    wave=data["wave_obs"],
    flux=data["flux_obs"],
    error=data["error_obs"],
    mask=data["mask_obs"],
    z_sys=data["z"],
    mode="mode2",
)
t1 = time.perf_counter()

print(f"  Elapsed: {t1 - t0:.1f}s")
print(f"  ┌─────────────────────────────────────────┐")
print(f"  │  v_e   = {specfit.ve[0]:8.1f}  ± {specfit.ve[1]:6.1f}  km/s  │")
print(f"  │  v_d   = {specfit.vd[0]:8.1f}  ± {specfit.vd[1]:6.1f}  km/s  │")
print(f"  │  E(B-V) = {specfit.ebv[0]:8.4f}  ± {specfit.ebv[1]:6.4f}        │")
print(f"  │  p1    = {specfit.p1:8.4f}    p2 = {specfit.p2:8.4f}        │")
print(f"  └─────────────────────────────────────────┘")

# Fit plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
w = specfit.wave_prep
ax1.plot(w, specfit.flux_prep, "k-", lw=0.5, label="Observed")
if specfit.bestfit is not None:
    ax1.plot(w, specfit.bestfit, "r-", lw=1, label="Best fit")
ax1.set_xlabel(r"$\lambda\ (\mathrm{\AA})$"); ax1.set_ylabel(r"$F_\lambda$")
ax1.legend(); ax1.set_title("SpecFit Result")

w_dust = np.linspace(3600, 7400, 500)
x_dust = 10000.0 / w_dust
xv = 10000.0 / 5500.0
A_dust = specfit.p1 * (x_dust - xv) + specfit.p2 * (x_dust**2 - xv**2)
if specfit._dust_data_wave is not None:
    ax2.scatter(specfit._dust_data_wave, specfit._dust_data_A, s=1, c='gray', alpha=0.4,
                rasterized=True, label="S/L data")
ax2.plot(w_dust, A_dust, "b-", lw=1.5, label="Polynomial fit")
ax2.axhline(0.0, color="k", ls="--", lw=0.8)
ax2.set_xlabel(r"$\lambda\ (\mathrm{\AA})$")
ax2.set_ylabel(r"$A_\lambda - A_V\ \mathrm{(mag)}$")
ax2.set_title(f"Dust Curve:  p1={specfit.p1:.4f}, p2={specfit.p2:.4f}")
ax2.legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "figs", "02_specfit.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  → figs/02_specfit.png")

# Separate dust curve plot
specfit.plot_dust(os.path.join(OUT_DIR, "figs", "02b_dust_curve.png"))
print("  → figs/02b_dust_curve.png")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  4. MCMC — stellar population inference                         ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "-" * 40)
print(f"  4. MCMC — UltraNest ({N_LIVE} live points)")
print("-" * 40)

mc = MCMCFitter(
    ssp_fits=SSP_FILE,
    specfit_result=specfit,
    sfh_model="delayed",
    wave_range=(3600, 7400),
)
print(f"  Likelihood ndof = {mc.likelihood.ndof}")

t0 = time.perf_counter()
mcmc_result = mc.run(
    n_live=N_LIVE,
    chain_dir=CHAIN_DIR,
    frac_remain=0.5,
)
t1 = time.perf_counter()

post  = mcmc_result.posterior
names = mc._sampler.param_names
best  = mcmc_result.bestfit

print(f"\n  Elapsed: {t1 - t0:.1f}s  ({(t1 - t0) / 60:.1f} min)")
print(f"  Posterior samples: {len(post)}")
print(f"  log Z = {mcmc_result.log_evidence:.2f}")
print(f"  ┌─────────────────────────────────────────┐")
for i, name in enumerate(names):
    lo, med, hi = np.percentile(post[:, i], [16, 50, 84])
    print(f"  │ {name:12s} = {med:8.4f}  [+{hi-med:.4f} / -{med-lo:.4f}] │")
if "logZsun" in best:
    Z_solar = 0.02 * 10 ** best["logZsun"]
    print(f"  │ Z          = {Z_solar:8.5f}  (Z_solar = 0.02)     │")
print(f"  └─────────────────────────────────────────┘")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  5. Visualization                                               ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "-" * 40)
print("  5. Visualization")
print("-" * 40)

# ── 5a. Corner plot ──
from corner import corner as _corner
_label_map = {
    "logZsun": r"$\log(Z/Z_\odot)$",
    "t0":      r"$t_0\ \mathrm{(Gyr)}$",
    "tau":     r"$\tau\ \mathrm{(Gyr)}$",
}
labels = [_label_map.get(n, n) for n in names]
truths = [best[n] for n in names]

fig = _corner(post, labels=labels, truths=truths,
              quantiles=[0.16, 0.5, 0.84],
              show_titles=True, title_fmt=".4f")
fig.savefig(os.path.join(OUT_DIR, "figs", "03_corner.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  → figs/03_corner.png")

# ── 5b. Best-fit CSP ──
like = mc.likelihood
sfh_best = DelayedExponentialSFH(
    **{k: v for k, v in best.items() if k != "logZsun"}, age_universe=13.8
)
logZ = best.get("logZsun", 0.0)

csp = like.builder.build(logZ, sfh_best)
csp = like.broadener.apply(csp)
n = like._med5500(like.ssp.wave, csp, np.ones_like(csp, dtype=bool), like._n_range)
csp = csp / n
csp = like.dust.apply(csp)
csp_obs = np.interp(like.obs_wave, like.ssp.wave, csp, left=0.0, right=0.0)
n_obs = 1.0 / np.median(like.obs_flux[like.obs_mask])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(like.obs_wave, like.obs_flux * n_obs, "k-", lw=0.5, label="Observed")
ax.plot(like.obs_wave, csp_obs * n_obs, "r-", lw=1, label="Best-fit CSP")
ax.set_xlabel(r"$\lambda\ (\mathrm{\AA})$")
ax.set_ylabel(r"Normalized $F_\lambda$")
ax.legend()
_name_map = {"logZsun": r"\log(Z/Z_\odot)", "t0": "t_0", "tau": r"\tau"}
tparts = [rf"${_name_map.get(k,k)}={v:.3f}$" for k, v in best.items()]
ax.set_title(r"$\mathrm{CSP\ Best\ Fit:}\ $" + r"$,\ $".join(tparts))
fig.savefig(os.path.join(OUT_DIR, "figs", "04_bestfit_csp.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  → figs/04_bestfit_csp.png")

# ── 5c. SFH + 68% CI ──
cosmic_time = np.max(like.ssp.time) - like.ssp.time

n_use = min(300, len(post))
idx = np.random.choice(len(post), n_use, replace=False)
_sfh_names = [n for n in names if n != "logZsun"]
_sfh_idx = {n: i for i, n in enumerate(names) if n != "logZsun"}

sfr_grid = np.zeros((n_use, len(cosmic_time)))
for k in range(n_use):
    kw = {n: post[idx[k], j] for n, j in _sfh_idx.items()}
    s = DelayedExponentialSFH(**kw, age_universe=13.8)
    sfr_grid[k] = s.evaluate(like.ssp.time)

sfr_lo = np.percentile(sfr_grid, 16, axis=0)
sfr_med = np.percentile(sfr_grid, 50, axis=0)
sfr_hi = np.percentile(sfr_grid, 84, axis=0)

fig, ax = plt.subplots(figsize=(8, 4))
ax.fill_between(cosmic_time, sfr_lo, sfr_hi, color="b", alpha=0.2,
                label=r"$68\%$ CI")
ax.plot(cosmic_time, sfr_med, "b-", lw=1.5, label="Median")
ax.set_xlabel(r"$\mathrm{Age\ of\ Universe\ (Gyr)}$")
ax.set_ylabel(r"$\mathrm{SFR\ (arb.\ units)}$")
tparts_sfh = []
for i, n in enumerate(names):
    lo, med, hi = np.percentile(post[:, i], [16, 50, 84])
    lbl = _name_map.get(n, n)
    tparts_sfh.append(rf"${lbl} = {med:.3f}^{{+{hi-med:.3f}}}_{{-{med-lo:.3f}}}$")
ax.set_title(r"$\mathrm{SFR:}\ $" + r"$,\ $".join(tparts_sfh), fontsize=9)
ax.legend()
fig.savefig(os.path.join(OUT_DIR, "figs", "05_sfh_ci.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  → figs/05_sfh_ci.png")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  6. Custom SFH model                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "-" * 40)
print("  6. Custom SFH — DelayTau (t0 ≡ 0)")
print("-" * 40)


class DelayTauSFH(SFHBase):
    """SFR(t) = t · exp(−t/τ), t₀ = 0,  free param: τ"""
    n_params = 1
    param_names = ["tau"]
    default_priors = {"tau": LogUniformPrior(0.1, 10.0)}

    def __init__(self, tau, age_universe=13.8):
        self.tau = float(tau)
        self.age_universe = float(age_universe)

    def evaluate(self, timegrid):
        t = np.max(timegrid) - timegrid
        sfr = t * np.exp(-t / self.tau)
        sfr[timegrid > self.age_universe] = 0.0
        return sfr


mc_custom = MCMCFitter(
    ssp_fits=SSP_FILE,
    specfit_result=specfit,
    sfh_model=DelayTauSFH,
    wave_range=(3600, 7400),
)

chain_custom = os.path.join(OUT_DIR, "chains_custom_sfh")
res_custom = mc_custom.run(
    n_live=100,
    chain_dir=chain_custom,
    max_ncalls=3000,
    priors={
        "logZsun": UniformPrior(-2.5, 0.5),
        "tau":     LogUniformPrior(0.1, 10.0),
    },
)

pc = res_custom.posterior
print(f"  Active params: {mc_custom._sampler.param_names}  (N={len(pc)})")
for i, n in enumerate(mc_custom._sampler.param_names):
    lo, med, hi = np.percentile(pc[:, i], [16, 50, 84])
    print(f"  {n:10s} = {med:.4f}  [{lo:.4f}, {hi:.4f}]")

dlogZ = res_custom.log_evidence - mcmc_result.log_evidence
print(f"\n  ΔlogZ (2p − 3p) = {dlogZ:.1f}")
print(f"  ({'>0' if dlogZ > 0 else '<0'}: DelayTau {'favored' if dlogZ > 0 else 'disfavored'} vs DelayedExp)")

# ── Custom SFH corner plot ──
_names_c = mc_custom._sampler.param_names
_label_map_c = {"logZsun": r"$\log(Z/Z_\odot)$", "tau": r"$\tau\ \mathrm{(Gyr)}$"}
labels_c = [_label_map_c.get(n, n) for n in _names_c]
best_c = [res_custom.bestfit[n] for n in _names_c]

fig = _corner(pc, labels=labels_c, truths=best_c,
              quantiles=[0.16, 0.5, 0.84],
              show_titles=True, title_fmt=".4f")
fig.savefig(os.path.join(OUT_DIR, "figs", "06_corner_custom_sfh.png"),
            dpi=120, bbox_inches="tight")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  6b. Custom SFH — Double Power Law (3 params)                   ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "-" * 40)
print("  6b. Custom SFH — Double Power Law (3 params)")
print("-" * 40)


class DoublePowerLawSFH(SFHBase):
    """SFR(t) = 1 / ((t/τ)^α + (t/τ)^(−β)),  free params: τ, α, β"""
    n_params = 3
    param_names = ["tau", "alpha", "beta"]
    default_priors = {
        "tau":   LogUniformPrior(0.1, 13.0),
        "alpha": LogUniformPrior(0.1, 10.0),
        "beta":  LogUniformPrior(0.1, 10.0),
    }

    def __init__(self, tau, alpha, beta, age_universe=13.8):
        self.tau   = float(tau)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.age_universe = float(age_universe)

    def evaluate(self, timegrid):
        t = np.max(timegrid) - timegrid
        t = np.where(t <= 0, 1e-10, t)
        x = t / self.tau
        sfr = 1.0 / (x**self.alpha + x**(-self.beta))
        sfr[timegrid > self.age_universe] = 0.0
        return sfr


mc_dpl = MCMCFitter(
    ssp_fits=SSP_FILE, specfit_result=specfit,
    sfh_model=DoublePowerLawSFH, wave_range=(3600, 7400),
)

chain_dpl = os.path.join(OUT_DIR, "chains_dpl")
res_dpl = mc_dpl.run(
    n_live=200, chain_dir=chain_dpl,
    priors={
        "logZsun": UniformPrior(-2.5, 0.5),
        "tau":     LogUniformPrior(0.1, 13.0),
        "alpha":   LogUniformPrior(0.1, 10.0),
        "beta":    LogUniformPrior(0.1, 10.0),
    },
)

pd = res_dpl.posterior
print(f"  Active params: {mc_dpl._sampler.param_names}  (N={len(pd)})")
for i, n in enumerate(mc_dpl._sampler.param_names):
    lo, med, hi = np.percentile(pd[:, i], [16, 50, 84])
    print(f"  {n:10s} = {med:.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"\n  log Z = {res_dpl.log_evidence:.2f}")

# DPL corner plot
_names_d = mc_dpl._sampler.param_names
_label_map_d = {"logZsun": r"$\log(Z/Z_\odot)$", "tau": r"$\tau$ (Gyr)",
                "alpha": r"$\alpha$", "beta": r"$\beta$"}
labels_d = [_label_map_d.get(n, n) for n in _names_d]
best_d = [res_dpl.bestfit[n] for n in _names_d]

fig = _corner(pd, labels=labels_d, truths=best_d,
              quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt=".4f")
fig.savefig(os.path.join(OUT_DIR, "figs", "07_corner_dpl.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  → figs/07_corner_dpl.png")

# Model comparison
print(f"\n  {'Model':<18s}  {'Np':>4s}  {'log Z':>10s}  {'ΔlogZ':>8s}")
print("  " + "-" * 44)
for name, np_, lz in [
    ("DelayedExp", 3, mcmc_result.log_evidence),
    ("DelayTau",   2, res_custom.log_evidence),
    ("DPL",        4, res_dpl.log_evidence),
]:
    dlz = lz - mcmc_result.log_evidence
    print(f"  {name:<18s}  {np_:>4d}  {lz:>10.2f}  {dlz:>+8.1f}")

plt.close(fig)
print("  → figs/06_corner_custom_sfh.png")


# ╔══════════════════════════════════════════════════════════════════╗
# ║  7. Save results                                                ║
# ╚══════════════════════════════════════════════════════════════════╝
print("\n" + "-" * 40)
print("  7. Save Results")
print("-" * 40)

specfit.save(os.path.join(OUT_DIR, "data", "specfit_result.fits"))
mcmc_result.save_result(os.path.join(OUT_DIR, "data", "mcmc_bestfit.fits"))
np.save(os.path.join(OUT_DIR, "data", "posterior_samples.npy"), post)

print(f"  {OUT_DIR}/data/specfit_result.fits")
print(f"  {OUT_DIR}/data/mcmc_bestfit.fits")
print(f"  {OUT_DIR}/data/posterior_samples.npy")
print(f"\n{'=' * 60}")
print("  bigspy Demo Complete")
print(f"{'=' * 60}")
