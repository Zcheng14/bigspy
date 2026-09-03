# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**bigspy** is a Python library for two-stage Bayesian fitting of galaxy spectra:

1. **SpecFit** (`src/bigspy/specfit.py`) — PCA-template fitting of stellar kinematics (`ve`, `vd`) and dust attenuation via `lmfit` least squares. Mode 1 = Calzetti curve; Mode 2 (default) = S/L (smooth/line) separation with a free-form quadratic dust curve.
2. **MCMC** (`src/bigspy/mcmc/`) — Bayesian inference of stellar-population parameters (SFH + metallicity) via UltraNest nested sampling, with an optional JAX-JIT likelihood backend.

Implements the methodology of Zhou et al. 2019 (MNRAS 485, 5256) and Li et al. 2020 (ApJ 896, 38).

## Environment & Commands

The package is installed editable in the `bigspy` conda env:

```bash
conda activate bigspy        # python = /home/jingtao/software/anaconda3/envs/bigspy/bin/python
pip install -e .             # after dependency changes
```

```bash
python -m pytest                                   # all tests (from repo root)
python -m pytest tests/test_mcmc.py                # one file
python -m pytest "tests/test_mcmc.py::TestPriors::test_uniform"   # one test
python example/run_bigspy_jax.py                   # full-pipeline demo (JAX)
python tests/benchmark_jax.py                      # NumPy vs JAX benchmark (run from repo root — relative paths)
```

No linter/formatter is configured. Tests in `tests/` import `from conftest import requires_data` and **auto-skip without the reference data** (LFS-tracked `template/*.fits` and `tests/*.pkl`); run `git lfs pull` if whole suites skip.

## Architecture

### Two-stage data flow

```
observed spectrum ──> SpecFit.fit() ──SpecFitResult──> MCMCFitter ──> UltraNest ──> MCMCResult
                      (ve, vd, dust,     (feeds preprocessed rest-frame spectrum,
                       preprocessed       kinematics, and dust curve forward)
                       spectrum)
```

`SpecFitResult` is the contract between stages: `MCMCFitter` takes its `wave_prep`/`flux_prep`/`error_prep`/`mask_prep` (rest-frame, MW-corrected, trimmed, 5500 Å–normalized), `ve`/`vd`, and Mode-2 dust coefficients (`p1`, `p2` → `DustAttenuation.from_mode2`).

### MCMC subpackage (`src/bigspy/mcmc/`)

- `ssp.py` — `SSPLibrary`: loads SSP FITS (HDUs: `WAVE`, `SPEC` (n_metal × n_age × n_wave), `MASS`, `TIME`, `DT`, `METAL`).
- `sfh.py` — pluggable SFH models. `SFHBase` subclass needs class attrs `n_params`, `param_names`, `default_priors`, an `evaluate(timegrid)` method, and a `evaluate_batch(timegrid, params_2d)` classmethod. `DelayedExponentialSFH` is the built-in.
- `csp.py` — `CSPBuilder`: weights SSP spectra by SFH and interpolates linearly in log-metallicity.
- `kinematics.py` — Gaussian velocity broadening as a precomputed convolution matrix (batch-capable).
- `dust.py` — `DustAttenuation` ("poly" mode-2 or "calzetti").
- `likelihood.py` / `likelihood_jax.py` — **dual backends with the same interface**: NumPy (`Likelihood`, chi² via `__call__`/`call_batch`) and JAX (`JAXLikelihood`, JIT + vmap, 12–80× faster). `MCMCFitter(use_jax=True)` samples with JAX but *always* builds the NumPy likelihood too — plots/saves run through it. JAX import failure falls back to NumPy silently.
- `priors.py` — `UniformPrior`, `LogUniformPrior`, `GaussianPrior`, `FixedPrior` map UltraNest's unit cube to physical space. `FixedPrior` parameters are held out of sampling.
- `sampler.py` — `UltraNestSampler`: resolves SFH class + priors, builds the parameter list (`logZsun` appended when in priors but not in SFH param_names), wires vectorized likelihood into `ultranest.ReactiveNestedSampler`.
- `fitter.py` — user-facing `MCMCFitter`/`MCMCResult` (FITS save with fixed HDU layout, corner/bestfit/SFH plots; matplotlib always `Agg`).

### Wavelength-grid invariant

Templates live on a log-wavelength grid, `DLOGW = 0.0001` dex → constant velocity scale ≈ 6.9 km/s/pixel. All convolution is done in **pixel units** (`sigma / velscale`). Keep observed spectra and template slices pixel-aligned (see `preprocess_spectrum`).

## Gotchas

- **Two `calz_unred` functions with opposite sign conventions**: `bigspy.specfit.calz_unred(wave, ebv)` returns `10^(+0.4·k·ebv)` (positive ebv = deredden/brighten, used to redden models with `-ebv`), while `bigspy.mcmc.dust.calz_unred` returns `10^(−0.4·k·ebv)`. Don't mix them.
- `SpecFit.fit()` temporarily mutates module-level globals (`FIT_NEIG`, `_EM_LINES`) and restores them — not thread-safe/reentrant.
- `MCMCFitter.run()` requires `chain_dir` (UltraNest writes chain files there); `out/chains_*/` is gitignored.
- Everything normalizes at 5500 Å (`WAVE_NORM`); dust curves are anchored at `xv = 10⁴/5500`.
- Default rest-frame fit range is (3600, 7400) Å; PCA templates use the first 10 of 20 components (`FIT_NEIG`).

## Repo Layout Notes

- Binary/FITS/PNG/pkl files are **git-lfs tracked** (`.gitattributes`).
- `example/` — runnable demos (`run_bigspy_jax.py`, notebooks). `mytest/` and `out*/` are scratch (self-ignored).
- Branches: development on `dev`, PRs target `master`.
