# bigspy — Bayesian Inference of Galaxy Spectra (Python)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

Two-stage spectral fitting of galaxy spectra:

1. **SpecFit** — PCA fitting for stellar kinematics ($v_e$, $v_d$) and dust attenuation
2. **MCMC** — Bayesian inference of stellar population parameters via UltraNest nested sampling

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9. Dependencies: `numpy`, `scipy`, `astropy`, `lmfit`, `ultranest`, `matplotlib`, `corner`, `jax`, `jaxlib`, `h5py`.

## Quick Start

```python
from bigspy import SpecFit, MCMCFitter

# ---- 1. SpecFit — kinematics + dust ----
sf = SpecFit(pca_fits="template/BC03_Padova1994_chab_PCA_extend_new.fits")
specfit = sf.fit(
    wave=wave_obs,          # observed wavelength (Å)
    flux=flux_obs,          # flux
    error=error_obs,        # uncertainty
    mask=mask_obs,          # bool, good pixels
    z_sys=your_redshift,    # systemic redshift (user-supplied)
    ebv_mw=ebv_mw,          # Galactic E(B-V) for MW extinction correction
    mode="mode2",           # S/L non-parametric dust (default)
)
print(f"v_e = {specfit.ve[0]:.1f} ± {specfit.ve[1]:.1f} km/s")
print(f"v_d = {specfit.vd[0]:.1f} ± {specfit.vd[1]:.1f} km/s")

# ---- 2. MCMC — stellar population inference ----
mc = MCMCFitter(
    ssp_fits="template/SSP_BC03_Padova1994_chab.fits",
    specfit_result=specfit,
    sfh_model="delayed",              # "delayed" or custom SFHBase subclass
    wave_range=(3600, 7400),
    use_jax=True,                     # JAX JIT acceleration (default; auto-fallback to NumPy)
)
mcmc_result = mc.run(
    n_live=400,
    chain_dir="out/chains_galaxy",
    frac_remain=0.5,
)

best = mcmc_result.bestfit       # dict: {"t0": ..., "tau": ..., "logZsun": ...}
post = mcmc_result.posterior     # (N, 3) ndarray
print(f"Best fit: {best}")
print(f"log Z    = {mcmc_result.log_evidence:.2f}")
```

## API Reference

### SpecFit — kinematics + dust

| Method / Property | Description |
|-------------------|-------------|
| `SpecFit(pca_fits)` | Load PCA templates from FITS |
| `sf.fit(wave, flux, error, mask, z_sys, ebv_mw=0.0, mode="mode2", ...)` | Run fitting, return `SpecFitResult` |
| `result.ve`, `result.vd` | Line-of-sight velocity & dispersion `(value, error)` in km/s |
| `result.ebv` | E(B−V) colour excess `(value, error)` |
| `result.p1`, `result.p2` | Mode 2 dust polynomial coefficients |
| `result.wave_prep`, `result.flux_prep`, `result.error_prep`, `result.mask_prep` | Preprocessed spectrum arrays |
| `result.dust_curve(wave)` | Callable dust attenuation factor (10^(−0.4 A_λ)) |
| `result.save(path)` | Save to FITS |
| `result.plot_fit(path)` | Fit spectrum + dust curve (A_λ − A_V, mag) |
| `result.plot_dust(path)` | Standalone dust plot: S/L data (scatter) + polynomial fit (line) |

### Dust Modes

| Mode | String | Description |
|------|--------|-------------|
| Mode 1 | `"mode1"` | Calzetti et al. (2000) attenuation curve, fits single parameter `E(B−V)` |
| Mode 2 | `"mode2"` | S/L (Smooth/Line) non-parametric dust curve. Separates smooth stellar continuum from emission lines, fits quadratic polynomial `A_λ − A_V = p1·(x−xv) + p2·(x²−xv²)` where `x = 10⁴/λ`. Returns S/L data points (`_dust_data_wave`, `_dust_data_A`) plus polynomial fit parameters. (default) |

### SFH Models

**Built-in**: `DelayedExponentialSFH(t0, tau, age_universe=13.8)`

```
SFR(t) = 0                         for t ≤ t₀
SFR(t) = (t − t₀) · exp(−(t−t₀)/τ)  for t > t₀
```

where $t$ is cosmic time (0 = Big Bang, max = present).

- `t0` — formation start time (Gyr after Big Bang). Smaller → earlier formation.
- `tau` — decay timescale (Gyr). Larger → SFR declines more slowly.

**Custom SFH** — subclass `SFHBase`:

```python
from bigspy.mcmc.sfh import SFHBase
from bigspy import LogUniformPrior

class MySFH(SFHBase):
    n_params = 2
    param_names = ["tau", "beta"]
    default_priors = {
        "tau":  LogUniformPrior(0.1, 10.0),
        "beta": UniformPrior(0.0, 5.0),
    }

    def __init__(self, tau, beta, age_universe=13.8):
        self.tau = float(tau)
        self.beta = float(beta)
        self.age_universe = float(age_universe)

    def evaluate(self, timegrid):
        t = np.max(timegrid) - timegrid
        sfr = t**self.beta * np.exp(-t / self.tau)
        sfr[timegrid > self.age_universe] = 0.0
        return sfr

mc = MCMCFitter(..., sfh_model=MySFH)
```

Required interface: `n_params`, `param_names`, `default_priors`, `__init__(**params)`, `evaluate(timegrid)`.

### Priors

| Class | Description |
|-------|-------------|
| `UniformPrior(lo, hi)` | Uniform on [lo, hi] |
| `LogUniformPrior(lo, hi)` | Uniform in log₁₀ space |
| `GaussianPrior(mu, sigma)` | Gaussian with mean μ, std σ |
| `FixedPrior(value)` | Fixed value (parameter frozen) |

```python
from bigspy import UniformPrior, LogUniformPrior, FixedPrior

mc.run(
    priors={
        "logZsun": UniformPrior(-2.5, 0.5),
        "t0":      UniformPrior(0.1, 13.5),
        "tau":     FixedPrior(5.0),       # freeze τ = 5
    },
    ...
)
```

### MCMCFitter — stellar population inference

| Method / Property | Description |
|-------------------|-------------|
| `MCMCFitter(ssp_fits, specfit_result, sfh_model, use_jax=True, ...)` | Set up MCMC. JAX JIT-accelerated by default |
| `mc.run(n_live, chain_dir, priors=..., ...)` | Run UltraNest, return `MCMCResult` |
| `result.bestfit` | Best-fit parameter dict |
| `result.posterior` | Posterior samples `(N, n_params)` ndarray |
| `result.log_evidence` | log(Z) model evidence |
| `result.save_result(path)` | Save best-fit params + CSP spectrum to FITS |
| `result.plot_corner(path)` | Corner plot |
| `result.plot_bestfit(path)` | Best-fit CSP vs observed |
| `result.plot_sfh(path)` | SFH with 68% CI |

### MCMC Result FITS Structure

`save_result()` writes a FITS file with these HDUs:

| HDU | Content |
|-----|---------|
| `PRIMARY` | Header with `LOGEVID` (log evidence) |
| `BESTFIT` | Parameter table (name, value) |
| `WAVE` | Observed wavelength grid (rest frame) |
| `FLUX` | Preprocessed flux |
| `ERROR` | Preprocessed error |
| `MASK` | Pixel mask (1 = good) |
| `CSP` | Best-fit CSP on SSP wavelength grid |
| `CSP_OBS` | Best-fit CSP interpolated to observed grid |

## JAX Acceleration

Add `use_jax=True` for JIT-compiled likelihood evaluation (12–80× faster on CPU):

```python
mc = MCMCFitter(..., use_jax=True)
mc.run(n_live=400, chain_dir="out/chains")
```

| Batch size N | NumPy | JAX JIT | Speedup |
|-------------|-------|---------|---------|
| 10 | 0.10 s | 0.008 s | 12× |
| 50 | 0.27 s | 0.011 s | 25× |
| 200 | 1.02 s | 0.017 s | 61× |
| 500 | 3.12 s | 0.039 s | 80× |
| 1000 | 5.05 s | 0.076 s | 66× |

Numerical agreement: Δχ²/χ² < 10⁻³. The NumPy backend is retained for plotting.

## Running the Demo

```bash
# Jupyter notebooks (recommended)
jupyter notebook example/bigspy_demo_jax.ipynb      # pkl test spectra
jupyter notebook example/bigspy_manga_demo.ipynb    # MaNGA datacube
```

Both walk through the full pipeline: load data → SpecFit → MCMC → visualization → custom SFH.

## License

MIT. See [LICENSE](LICENSE).

## References

**If you use this code, please cite:**

- Zhou S., Mo H. J., Li C., et al., 2019, MNRAS, 485, 5256 — *"SDSS-IV MaNGA: stellar initial mass function variation inferred from Bayesian analysis of the integral field spectroscopy of early-type galaxies"* — [2019MNRAS.485.5256Z](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.5256Z)
- Li N., Li C., Mo H. J., Hu J., Zhou S., Du C., 2020, ApJ, 896, 38 — *"Estimating Dust Attenuation from Galactic Spectra. I. Methodology and Tests"* — [2020ApJ...896...38L](https://ui.adsabs.harvard.edu/abs/2020ApJ...896...38L)
- Buchner J., 2021, JOSS, 6(60), 3001 — *"UltraNest — a robust, general purpose Bayesian inference engine"* — [10.21105/joss.03001](https://doi.org/10.21105/joss.03001)

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
