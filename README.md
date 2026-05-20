# bigspy — Bayesian Inference of Galaxy Spectra (Python)

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

Two-stage spectral fitting of galaxy spectra:

1. **SpecFit** — PCA fitting for stellar kinematics ($v_e$, $v_d$) and dust attenuation
2. **MCMC** — Bayesian inference of stellar population parameters via UltraNest nested sampling

## Installation

```bash
pip install -e .
```

Requires Python ≥ 3.9. Dependencies: `numpy`, `scipy`, `astropy`, `lmfit`, `ultranest`, `matplotlib`, `corner`, `jax`, `jaxlib`.

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
    z_sys=0.04,             # systemic redshift
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

### SpecFit

| Method / Property | Description |
|-------------------|-------------|
| `SpecFit(pca_fits)` | Load PCA templates from FITS |
| `sf.fit(wave, flux, error, mask, z_sys, mode, ...)` | Run fitting, return `SpecFitResult` |
| `result.ve`, `result.vd` | Line-of-sight velocity & dispersion `(value, error)` in km/s |
| `result.ebv` | E(B−V) colour excess `(value, error)` |
| `result.p1`, `result.p2` | Mode 2 dust polynomial coefficients |
| `result.wave_prep`, `result.flux_prep`, `result.error_prep`, `result.mask_prep` | Preprocessed spectrum arrays |
| `result.dust_curve(wave)` | Callable dust attenuation factor (10^(−0.4 A_λ)) |
| `result.save(path)` | Save to FITS |
| `result.plot_fit(path)` | Fit spectrum + dust curve (A_λ − A_V, mag) |
| `result.plot_dust(path)` | Standalone dust plot: S/L data (scatter) + polynomial fit (line) |

### MCMCFitter

| Method / Property | Description |
|-------------------|-------------|
| `MCMCFitter(ssp_fits, specfit_result, sfh_model, use_jax=False, ...)` | Set up MCMC. `use_jax=True` for JIT-compiled likelihood |
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

## Dust Modes

| Mode | String | Description |
|------|--------|-------------|
| Mode 1 | `"mode1"` | Calzetti et al. (2000) attenuation curve, fits single parameter `E(B−V)` |
| Mode 2 | `"mode2"` | S/L (Smooth/Line) non-parametric dust curve. Separates smooth stellar continuum from emission lines, fits quadratic polynomial `A_λ − A_V = p1·(x−xv) + p2·(x²−xv²)` where `x = 10⁴/λ`. Returns S/L data points (`_dust_data_wave`, `_dust_data_A`) plus polynomial fit parameters. (default) |

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
JAX demo: `python example/run_bigspy_jax.py` or `example/bigspy_demo_jax.ipynb`.

## Running the Demo

```bash
# Command-line script
python example/run_bigspy.py

# Jupyter notebook
jupyter notebook example/bigspy_demo.ipynb
```

Both walk through the full pipeline: load data → SpecFit → MCMC → visualization → custom SFH.

## License

MIT. See [LICENSE](LICENSE).

## References

This code implements the methodology described in:

- Zhou S., Mo H. J., Li C., et al., 2019, MNRAS, 485, 5256 — *"SDSS-IV MaNGA: stellar initial mass function variation inferred from Bayesian analysis of the integral field spectroscopy of early-type galaxies"* — [2019MNRAS.485.5256Z](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.5256Z)
- Li N., Li C., Mo H. J., Hu J., Zhou S., Du C., 2020, ApJ, 896, 38 — *"Estimating Dust Attenuation from Galactic Spectra. I. Methodology and Tests"* — [2020ApJ...896...38L](https://ui.adsabs.harvard.edu/abs/2020ApJ...896...38L)
