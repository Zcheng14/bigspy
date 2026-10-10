# bigspy — Bayesian Inference of Galaxy Spectra (Python)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)

bigspy (**B**ayesian **I**nference of **G**alaxy **S**pectra, in **Py**thon)
fits galaxy spectra in two stages:

1. **SpecFit** — a PCA-template fit (via `lmfit` least squares) of the stellar
   kinematics ($v_e$, $v_d$) and the dust attenuation.
2. **MCMC** — Bayesian inference of the star formation history (SFH) and
   metallicity with Nested Slice Sampling (`blackjax.nss`), using a JAX-only
   likelihood. The kinematics and dust curve from stage 1 are held fixed;
   only the SFH parameters and `logZsun` are sampled.

## Installation

```bash
# 1. Clone with git-lfs (template & test data are LFS-tracked)
git lfs install
git lfs clone https://github.com/Zcheng14/bigspy.git

# Or, if already cloned without LFS:
git lfs pull

# 2. Install (Python >= 3.11)
cd bigspy
pip install -e .
```

Dependencies (auto-installed): `numpy`, `scipy`, `astropy`, `lmfit`,
`matplotlib`, `corner`, `jax`, `jaxlib`, `blackjax>=1.5`.
For development: `pip install -e ".[dev]"` (adds `pytest`, `pytest-cov`).

## Quick Start

```python
from bigspy import SpecFit, MCMCFitter

# ---- 1. SpecFit — kinematics + dust ----
specfit = SpecFit("template/BC03_Padova1994_chab_PCA_extend_new.fits").fit(
    wave=wave_obs,        # observed-frame wavelength (Angstrom)
    flux=flux_obs,        # flux
    error=error_obs,      # 1-sigma uncertainty
    mask=mask_obs,        # pixel mask, 1/True = good
    z_sys=z,              # systemic redshift
    ebv_mw=ebv_mw,        # Galactic foreground E(B-V) (default 0)
    mode="mode2",         # default; see "Dust modes" below
)
print(f"v_e = {specfit.ve[0]:.1f} +/- {specfit.ve[1]:.1f} km/s")
print(f"v_d = {specfit.vd[0]:.1f} +/- {specfit.vd[1]:.1f} km/s")
print(f"E(B-V) = {specfit.ebv[0]:.3f}")

# ---- 2. MCMC — SFH + metallicity (NSS) ----
mc = MCMCFitter(
    ssp_fits="template/SSP_BC03_Padova1994_chab.fits",
    specfit_result=specfit,
    sfh_model="dpl",          # default; "delayed" or a custom SFHBase subclass
)
result = mc.run(n_live=1000, seed=0, out_dir="out/state_galaxy")

print(result.bestfit)          # {'tau': ..., 'alpha': ..., 'beta': ..., 'logZsun': ...}
print(result.posterior.shape)  # (N, 4)
print(f"log Z = {result.log_evidence:.2f}")
```

## How the two stages connect

- `SpecFit.fit()` **always runs the Mode-1 fit** (Calzetti dust); `mode="mode2"`
  (the default) *additionally* runs the Mode-2 S/L dust fit on top of it.
- `MCMCFitter` holds the following fixed from the SpecFit result: the
  preprocessed rest-frame spectrum, the kinematics (`ve`, `vd`), and the dust
  curve. Only the SFH parameters and `logZsun` are sampled.
- **Which dust curve is used**: the Mode-2 (S/L) curve when Mode 2 produced one
  (`specfit.mode2_dust_ok is True`); otherwise the Mode-1 Calzetti curve with
  the fitted `E(B-V)`. Both are 1 at the 5500 Å normalization wavelength.

### Conventions

- **Mask**: `1`/`True` = good pixel, everywhere (input `mask`, FITS `MASK`
  HDU, `mask_prep`).
- **Sky lines**: `SpecFit.fit(mask_sky=True)` (default) additionally masks
  the [O I] 5577 night-sky line (±800 km/s, applied in the observed frame
  via the redshift), whose subtraction residuals can contaminate the fit.
- **Normalization**: spectra and models are normalized at 5500 Å; the dust
  curves used by the MCMC are anchored to 1 there.
- **Templates**: PCA/SSP templates live on a log-wavelength grid
  (`DLOGW = 1e-4` dex, ≈ 6.9 km/s per pixel); velocity convolution is done in
  pixel units.
- **Reproducibility**: `mc.run(seed=...)` is deterministic for a fixed seed.

## Dust modes (SpecFit)

| Mode | What it does | Free parameters |
|------|--------------|-----------------|
| `"mode1"` | Calzetti+2000 attenuation curve | `E(B-V)` |
| `"mode2"` (default) | S/L (smooth/line) separation; free-form dust curve `A(x) − A(x_v) = p1·(x−xv) + p2·(x²−xv²)`, `x = 10⁴/λ`, `xv = 10⁴/5500` | `p1`, `p2` |

`SpecFitResult` also reports `mode1_success` (whether the Mode-1 fit
converged) and `mode2_dust_ok` (whether Mode 2 produced a dust curve).

## SFH models

**`DoublePowerLawSFH(tau, alpha, beta)`** (default):

```
SFR(t) = 1 / ( (t/τ)^α + (t/τ)^(−β) )
```

`t` is the cosmic time since the Big Bang; `tau` is the turnover time,
`alpha`/`beta` the rising/falling slopes. Default priors:
`logZsun ~ U(−2.5, 0.5)`, `tau ~ LogU(0.1, 13)`, `alpha, beta ~ LogU(0.1, 1000)`.

**`DelayedExponentialSFH(t0, tau)`**: `SFR(t) = (t−t0)·exp(−(t−t0)/τ)` for
`t > t0`, else 0. Default priors: `logZsun ~ U(−2.5, 0.5)`,
`t0 ~ U(0.1, 13.5)`, `tau ~ LogU(0.1, 10)`.

**Custom SFH** — subclass `SFHBase`. The sampler JIT-compiles the whole
likelihood, so a custom model must provide both a NumPy `evaluate` (plotting /
model building) and a JAX `evaluate_batch_jax` (sampling):

```python
import numpy as np
import jax.numpy as jnp
from bigspy import UniformPrior, LogUniformPrior
from bigspy.mcmc.sfh import SFHBase

class MySFH(SFHBase):
    n_params = 2
    param_names = ["tau", "beta"]
    default_priors = {"logZsun": UniformPrior(-2.5, 0.5),
                      "tau":  LogUniformPrior(0.1, 10.0),
                      "beta": UniformPrior(0.0, 5.0)}

    def __init__(self, tau, beta):
        self.tau, self.beta = float(tau), float(beta)

    def evaluate(self, timegrid):                      # NumPy
        t = np.max(timegrid) - timegrid
        return t**self.beta * np.exp(-t / self.tau)

    @classmethod
    def evaluate_batch_jax(cls, timegrid, params_2d):  # JAX (required)
        tau  = params_2d[:, 0][:, None]
        beta = params_2d[:, 1][:, None]
        t = jnp.max(timegrid) - timegrid
        return t[None, :]**beta * jnp.exp(-t[None, :] / tau)

mc = MCMCFitter(..., sfh_model=MySFH)
```

Required interface: `n_params`, `param_names`, `default_priors`,
`__init__(**params)`, `evaluate(timegrid)`, `evaluate_batch_jax(timegrid,
params_2d)`.

## Priors

| Class | Meaning |
|-------|---------|
| `UniformPrior(lo, hi)` | Uniform on [lo, hi] |
| `LogUniformPrior(lo, hi)` | Uniform in log₁₀ |
| `GaussianPrior(mu, sigma)` | Gaussian (μ, σ) |
| `FixedPrior(value)` | Hold the parameter fixed |

User priors passed to `mc.run(priors={...})` are merged over the model's
`default_priors`. A `FixedPrior` removes the parameter from sampling: it is
absent from `posterior` but **is** reported in `bestfit` (with its fixed
value), so model spectra can always be rebuilt from `bestfit`.

## API reference

### `SpecFit` / `SpecFitResult`

| Method / attribute | Description |
|--------------------|-------------|
| `SpecFit(pca_fits)` | Load PCA templates (`pca_log`/`wave_log` HDUs) |
| `.fit(wave, flux, error, mask, z_sys, mode="mode2", emission_mask=None, neig=None, observed_fits=None, ebv_mw=0.0, mask_sky=True)` | Run the fit → `SpecFitResult` |
| `.ve`, `.vd` | Velocity shift / dispersion, `(value, error)` in km/s |
| `.ebv` | Mode-1 `E(B−V)`, `(value, error)` |
| `.p1`, `.p2` | Mode-2 dust polynomial coefficients |
| `.mode1_success` | Mode-1 convergence flag (`SpecFit.fit` warns on failure) |
| `.mode2_dust_ok` | Whether Mode 2 produced a dust curve |
| `.wave_prep` / `.flux_prep` / `.error_prep` / `.mask_prep` | Preprocessed rest-frame spectrum (consumed by `MCMCFitter`) |
| `.dust_curve(wave)` | Fitted dust curve as a callable (1.0 outside the fitted range) |
| `.save(path)` | FITS: `WAVE/FLUX/ERROR/PARAMS` (+ `BESTFIT`, `DUST`) |
| `.plot_fit(path)` / `.plot_dust(path)` | Fit overview / dust-curve figure |

### `MCMCFitter` / `MCMCResult`

| Method / attribute | Description |
|--------------------|-------------|
| `MCMCFitter(ssp_fits, specfit_result, sfh_model="dpl", wave_range=(3600, 7400), emission_mask=None)` | Build the model + JAX likelihood |
| `.run(n_live=1000, num_delete=100, num_inner_steps=10, priors=None, seed=0, max_ncalls=None, out_dir=None)` | Run NSS → `MCMCResult`; `out_dir` also saves `state.npz` |
| `.bestfit` | Max-likelihood parameter dict (includes fixed parameters) |
| `.posterior` | Posterior samples, `(N, n_active_params)` |
| `.log_evidence` | log Z |
| `.bestfit_model()` | Best-fit CSP on the observed wavelength grid |
| `.save_result(path)` | FITS: `BESTFIT/WAVE/FLUX/ERROR/MASK/CSP/CSP_OBS`, `LOGEVID` in header |
| `.plot_corner(path)` / `.plot_bestfit(path)` / `.plot_sfh(path)` | Corner / best-fit CSP / SFH (68% CI) figures |

Sampling defaults: `n_live=1000`, `num_delete=100`, `num_inner_steps=10`.
Use `num_inner_steps >= 2 * n_params` for harder posteriors.

## Running the demo and tests

```bash
python example/run_bigspy.py     # full pipeline on a bundled MaNGA spectrum
python -m pytest                 # test suite
```

The demo writes figures/data to `out/`; the notebook
`example/bigspy_demo.ipynb` covers the same workflow interactively (run
Jupyter from the same Python environment where `bigspy` is installed, so the
`python3` kernel can import it).

Tests use synthetic data where possible; the tests that need the LFS-tracked
reference data (templates, MaNGA pkl) are skipped automatically when those
files are missing (run `git lfs pull` to fetch them).

## License

MIT. See [LICENSE](LICENSE).

## References

**If you use this code, please cite:**

- Zhou S., Mo H. J., Li C., et al., 2019, MNRAS, 485, 5256 — *"SDSS-IV MaNGA: stellar initial mass function variation inferred from Bayesian analysis of the integral field spectroscopy of early-type galaxies"* — [2019MNRAS.485.5256Z](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.5256Z)
- Li N., Li C., Mo H. J., Hu J., Zhou S., Du C., 2020, ApJ, 896, 38 — *"Estimating Dust Attenuation from Galactic Spectra. I. Methodology and Tests"* — [2020ApJ...896...38L](https://ui.adsabs.harvard.edu/abs/2020ApJ...896...38L)
- Yallup D., Kroupa N., Handley W., 2026, TMLR — *"Nested Slice Sampling: Vectorized Nested Sampling for GPU-Accelerated Inference"* — [arXiv:2601.23252](https://arxiv.org/abs/2601.23252)

**Related work:**

- Cheng Z., Li C., Li N., Yan R., Mo H., 2024, ApJ, 961, 216 — *"Post-starburst Galaxies in SDSS-IV MaNGA: Two Broad Categories of Evolutionary Pathways"* — [2024ApJ...961..216C](https://ui.adsabs.harvard.edu/abs/2024ApJ...961..216C)
- Guo R., Li C., Zhou S., Li N., Jing T., Cheng Z., 2025, RAA, 25, 5017 — *"Mapping Dust Attenuation at Kiloparsec Scales. II. Attenuation Curves from Near-ultraviolet to Near-infrared"* — [2025RAA....25f5017G](https://ui.adsabs.harvard.edu/abs/2025RAA....25f5017G)
- Jing T., Li C., 2024, ApJ, 975, 17 — *"On the Origin of Quenched but Gas-rich Regions at Kiloparsec Scales in Nearby Galaxies"* — [2024ApJ...975...17J](https://ui.adsabs.harvard.edu/abs/2024ApJ...975...17J)
- Li N., Li C., Mo H., Zhou S., Liang F., Boquien M., Drory N., Fernández-Trincado J. G., Greener M., Riffel R., 2021, ApJ, 917, 72 — *"Estimating Dust Attenuation From Galactic Spectra. II. Stellar and Gas Attenuation in Star-forming and Diffuse Ionized Gas Regions in MaNGA"* — [2021ApJ...917...72L](https://ui.adsabs.harvard.edu/abs/2021ApJ...917...72L)
- Li N., Li C., 2023, ChPhB, 32, 9801 — *"Measuring stellar populations, dust attenuation and ionized gas at kpc scales in 10010 nearby galaxies using the integral field spectroscopy from MaNGA"* — [2023ChPhB..32c9801L](https://ui.adsabs.harvard.edu/abs/2023ChPhB..32c9801L)
- Li N., Li C., 2024, ApJ, 975, 234 — *"Estimating Dust Attenuation from Galactic Spectra. III. Radial Variations of Dust Attenuation Scaling Relations in MaNGA Galaxies"* — [2024ApJ...975..234L](https://ui.adsabs.harvard.edu/abs/2024ApJ...975..234L)
- Zhou S., Mo H. J., Li C., Boquien M., Rossi G., 2020, MNRAS, 497, 4753 — *"SDSS-IV MaNGA: Bayesian analysis of the star formation history of low-mass galaxies in the local Universe"* — [2020MNRAS.497.4753Z](https://ui.adsabs.harvard.edu/abs/2020MNRAS.497.4753Z)
- Zhou S., Li C., Hao C.-N., Guo R., Mo H., Xia X., 2021, ApJ, 916, 38 — *"Star Formation Histories of Massive Red Spiral Galaxies in the Local Universe"* — [2021ApJ...916...38Z](https://ui.adsabs.harvard.edu/abs/2021ApJ...916...38Z)
- Zhou S., Li C., Li N., Mo H., Yan R., Eracleous M., Molina M., Gronwall C., Ajgaonkar N., Cheng Z., Guo R., 2023, ApJ, 957, 75 — *"Mapping Dust Attenuation and the 2175 Å Bump at Kiloparsec Scales in Nearby Galaxies"* — [2023ApJ...957...75Z](https://ui.adsabs.harvard.edu/abs/2023ApJ...957...75Z)
