"""MCMC fitter -- high-level interface for Bayesian spectral fitting.

Sampling is done with blackjax Nested Slice Sampling (``blackjax.nss``); see
:mod:`bigspy.mcmc.sampler`.  The default model is the double power-law SFH.
"""

import os
import numpy as np

from .ssp import SSPLibrary
from .dust import DustAttenuation
from .model import ModelComponents
from .likelihood_jax import JAXLikelihood
from .sampler import (NSSampler, DEFAULT_N_LIVE, DEFAULT_NUM_DELETE,
                      DEFAULT_NUM_INNER_STEPS)


def _robust_ylim(arrays, mask, pct=(1, 99), pad=0.10):
    """Robust y-axis limits from good-pixel values (percentiles + padding).

    Keeps emission-line spikes / outliers from dominating the plot range.
    """
    vals = []
    for a in arrays:
        a = np.asarray(a, dtype=float)
        good = np.asarray(mask, dtype=bool) & np.isfinite(a)
        if good.any():
            vals.append(a[good])
    if not vals:
        return None
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, pct)
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        span = max(abs(lo), 1e-6)
    return lo - pad * span, hi + pad * span


class MCMCResult:
    """Container for MCMC fitting results and plotting/saving helpers."""

    def __init__(self, sampler, model):
        self._sampler = sampler
        self._model = model
        self.result = sampler.result

    @property
    def bestfit(self):
        """Best-fit (maximum-likelihood) parameter dict.

        Includes parameters held fixed via ``FixedPrior`` (filled with
        their fixed values), so the dict always covers the full parameter
        set needed to rebuild a model spectrum.  ``posterior`` still
        contains only the sampled (active) parameters.
        """
        point = self._sampler.get_bestfit()
        best = dict(zip(self._sampler.param_names, point))
        best.update(self._sampler.fixed_params)
        return best

    @property
    def posterior(self):
        """Posterior samples, shape (N_samples, n_params)."""
        return self._sampler.get_posterior()

    @property
    def log_evidence(self):
        """Log evidence log(Z)."""
        return self.result.get("logz", np.nan)

    @property
    def model(self):
        """NumPy model container (observed data + CSP pipeline)."""
        return self._model

    def bestfit_model(self):
        """Best-fit CSP interpolated to the observed wavelength grid."""
        best = self.bestfit
        sfh_cls = self._sampler.sfh_class
        sfh = sfh_cls(**{k: v for k, v in best.items() if k != "logZsun"})
        return self._model.build_model(best.get("logZsun", 0.0), sfh)

    def save_result(self, path):
        """Save best-fit parameters and CSP spectrum to FITS.

        HDU structure:
            PRIMARY   -- header with LOGEVID
            BESTFIT   -- parameter table (name, value)
            WAVE/FLUX/ERROR/MASK -- preprocessed observed data
            CSP       -- best-fit CSP on the SSP grid
            CSP_OBS   -- best-fit CSP on the observed grid
        """
        from astropy.io import fits

        best = self.bestfit
        model = self._model
        logZ = best.get("logZsun", 0.0)
        sfh_cls = self._sampler.sfh_class
        sfh = sfh_cls(**{k: v for k, v in best.items() if k != "logZsun"})
        csp = model.build_csp(logZ, sfh)
        csp_obs = model.build_model(logZ, sfh)

        cols = [fits.Column(name=n, format="D", array=[v]) for n, v in best.items()]

        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul[0].header["LOGEVID"] = (float(self.log_evidence), "log(Z) evidence")
        hdul.append(fits.BinTableHDU.from_columns(cols, name="BESTFIT"))
        hdul.append(fits.ImageHDU(model.obs_wave.astype(np.float64), name="WAVE"))
        hdul.append(fits.ImageHDU(model.obs_flux.astype(np.float64), name="FLUX"))
        hdul.append(fits.ImageHDU(model.obs_error.astype(np.float64), name="ERROR"))
        hdul.append(fits.ImageHDU(model.obs_mask.astype(np.uint8), name="MASK"))
        hdul.append(fits.ImageHDU(csp.astype(np.float64), name="CSP"))
        hdul.append(fits.ImageHDU(csp_obs.astype(np.float64), name="CSP_OBS"))

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        hdul.writeto(path, overwrite=True)

    def plot_corner(self, path):
        """Corner plot of the posterior with the best-fit point marked."""
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 10,
        })
        import corner
        import matplotlib.pyplot as plt
        samples = self.posterior
        # Truths/labels must align with the posterior columns (active
        # parameters only); fixed params are not plotted.
        raw_labels = list(self._sampler.param_names)
        best_dict = self.bestfit
        best = [best_dict[k] for k in raw_labels]
        _label_map = {
            "logZsun": r"$\log(Z/Z_\odot)$",
            "t0":      r"$t_0\ \mathrm{(Gyr)}$",
            "tau":     r"$\tau\ \mathrm{(Gyr)}$",
            "alpha":   r"$\alpha$",
            "beta":    r"$\beta$",
        }
        labels = [_label_map.get(k, k) for k in raw_labels]
        fig = corner.corner(samples, labels=labels, truths=best,
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_fmt=".4f",
                            label_kwargs={"fontsize": 12},
                            title_kwargs={"fontsize": 11})
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_bestfit(self, path):
        """Best-fit CSP vs the observed spectrum."""
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 12,
        })
        import matplotlib.pyplot as plt
        model = self._model
        best = self.bestfit
        logZ = best.get("logZsun", 0.0)
        csp_obs = self.bestfit_model()
        n_obs = 1.0 / np.median(model.obs_flux[model.obs_mask])

        Z = 0.02 * 10 ** logZ
        _name_map = {"logZsun": r"\log(Z/Z_\odot)", "t0": "t_0", "tau": r"\tau",
                     "alpha": r"\alpha", "beta": r"\beta"}
        title_parts = []
        for k, v in best.items():
            label = _name_map.get(k, k)
            title_parts.append(rf"${label} = {v:.3f}$")
        title_parts.append(rf"$Z = {Z:.5f}$")

        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]})

        obs = model.obs_flux * n_obs
        mod = csp_obs * n_obs
        resid = mod - obs
        good = model.obs_mask

        # Top: full range (observed + best-fit CSP).
        ax1.plot(model.obs_wave, obs, 'k-', lw=0.5, label=r'$\mathrm{Observed}$')
        ax1.plot(model.obs_wave, mod, 'r-', lw=1, label=r'$\mathrm{Best\ fit\ CSP}$')
        ax1.set_ylabel(r'$\mathrm{Normalized}\ F_\lambda$')
        ax1.set_title(r'$\mathrm{MCMC\ Best\ Fit:}\ $' + r'$,\ $'.join(title_parts))
        ax1.legend(frameon=True, fontsize=11)

        # Middle: same curves, y-axis zoomed on the good-pixel range.
        ax2.plot(model.obs_wave, obs, 'k-', lw=0.5)
        ax2.plot(model.obs_wave, mod, 'r-', lw=1)
        ax2.set_ylim(*_robust_ylim([obs, mod], good))
        ax2.set_ylabel(r'$\mathrm{Normalized}\ F_\lambda$')

        # Bottom: residual (CSP - observed), zoomed.
        ax3.axhline(0.0, color='k', ls='--', lw=0.8)
        ax3.plot(model.obs_wave, resid, 'k-', lw=0.5)
        ax3.set_ylim(*_robust_ylim([resid], good))
        ax3.set_ylabel(r'$\mathrm{CSP} - \mathrm{Observed}$')
        ax3.set_xlabel(r'$\lambda\ (\mathrm{\AA})$')
        ax3.set_xlim(model.obs_wave[0], model.obs_wave[-1])

        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_sfh(self, path, n_samples=500):
        """Star formation history: 68% credible interval + median."""
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 12,
        })
        import matplotlib.pyplot as plt
        model = self._model
        post = self.posterior
        names = self._sampler.param_names

        _sfh_param_idx = {name: i for i, name in enumerate(names)
                          if name != "logZsun"}
        cosmic_time = np.max(model.ssp.time) - model.ssp.time

        n_use = min(n_samples, len(post))
        idx = np.random.choice(len(post), n_use, replace=False)
        post_sub = post[idx]

        sfr_grid = np.zeros((n_use, len(cosmic_time)))
        for i in range(n_use):
            sfh_kwargs = {name: post_sub[i, j] for name, j in _sfh_param_idx.items()}
            sfh = self._sampler.sfh_class(**sfh_kwargs)
            sfr_grid[i] = sfh.evaluate(model.ssp.time)

        sfr_lo = np.percentile(sfr_grid, 16, axis=0)
        sfr_med = np.percentile(sfr_grid, 50, axis=0)
        sfr_hi = np.percentile(sfr_grid, 84, axis=0)

        _name_map = {"logZsun": r"\log(Z/Z_\odot)", "t0": "t_0", "tau": r"\tau",
                     "alpha": r"\alpha", "beta": r"\beta"}
        title_parts = []
        for name in names:
            col = names.index(name)
            med = np.percentile(post[:, col], 50)
            lo = np.percentile(post[:, col], 16)
            hi = np.percentile(post[:, col], 84)
            label = _name_map.get(name, name)
            title_parts.append(
                rf"${label} = {med:.3f}^{{+{hi-med:.3f}}}_{{-{med-lo:.3f}}}$"
            )

        best = self.bestfit
        # Full SFH parameter set (active + fixed) — ``bestfit`` covers both.
        best_sfr = self._sampler.sfh_class(
            **{name: best[name]
               for name in self._sampler.sfh_class.param_names}
        ).evaluate(model.ssp.time)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(cosmic_time, sfr_lo, sfr_hi, color='b', alpha=0.2,
                        label=r'$68\%\ \mathrm{CI}$')
        ax.plot(cosmic_time, sfr_med, 'b-', lw=1.5, label=r'$\mathrm{Median}$')
        ax.plot(cosmic_time, best_sfr, 'r--', lw=1.2, label=r'$\mathrm{Best-fit}$')
        ax.set_xlabel(r'$\mathrm{Age\ of\ Universe\ (Gyr)}$')
        ax.set_ylabel(r'$\mathrm{SFR\ (arbitrary\ units)}$')
        ax.set_title(r'$\mathrm{Star\ Formation\ History:}\ $'
                     + r'$,\ $'.join(title_parts), fontsize=10)
        ax.legend(frameon=True, fontsize=10, loc="upper left")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class MCMCFitter:
    """Bayesian MCMC spectral fitting with Nested Slice Sampling.

    Parameters
    ----------
    ssp_fits : str
        Path to the SSP template FITS file.
    specfit_result : SpecFitResult
        SpecFit results (kinematics, dust curve, preprocessed spectrum).
    sfh_model : str or type, optional
        ``"dpl"`` (default) or ``"delayed"``, or an ``SFHBase`` subclass.
    wave_range : tuple, optional
        SSP wavelength range (default: (3600, 7400)).
    emission_mask : list, optional
        Additional emission-line regions to mask.
    """

    def __init__(self, ssp_fits, specfit_result, sfh_model="dpl",
                 wave_range=(3600, 7400), emission_mask=None):
        self.ssp = SSPLibrary(ssp_fits, wave_range=wave_range)
        self._specfit = specfit_result
        self._sfh_model = sfh_model

        self._wave_obs = np.asarray(specfit_result.wave_prep, dtype=float)
        self._flux_obs = np.asarray(specfit_result.flux_prep, dtype=float)
        self._error_obs = np.asarray(specfit_result.error_prep, dtype=float)

        self._obs_mask = np.asarray(specfit_result.mask_prep, dtype=bool)
        if emission_mask is not None:
            from ..mask import build_emission_mask
            em = build_emission_mask(self._wave_obs, emission_mask)
            self._obs_mask = self._obs_mask & em

        p1 = getattr(specfit_result, 'p1', 0.0)
        p2 = getattr(specfit_result, 'p2', 0.0)
        self._dust = DustAttenuation.from_mode2(self.ssp.wave, p1, p2)

        ve = specfit_result.ve[0]
        vd = specfit_result.vd[0]

        # NumPy model container (plots / saving) and the JAX likelihood.
        self._model = ModelComponents(
            self.ssp, self._wave_obs, self._flux_obs, self._error_obs,
            self._obs_mask, ve, vd, self._dust,
        )
        self._likelihood = JAXLikelihood(
            self.ssp, self._wave_obs, self._flux_obs, self._error_obs,
            self._obs_mask, ve, vd, self._dust,
        )

    def run(self, n_live=DEFAULT_N_LIVE, num_delete=DEFAULT_NUM_DELETE,
            num_inner_steps=DEFAULT_NUM_INNER_STEPS,
            priors=None, seed=0, max_ncalls=None, out_dir=None):
        """Run Nested Slice Sampling.

        Parameters
        ----------
        n_live : int
            Number of live points (default 1000).
        num_delete : int
            Number of live points replaced per step (default 100).
        num_inner_steps : int
            Inner slice steps per replacement (default 10); use at least
            ``2 * n_params`` for harder posteriors.
        priors : dict, optional
            Parameter name -> :class:`Prior`; merged over the model defaults.
        seed : int
            PRNG seed (the run is deterministic for a fixed seed).
        max_ncalls : int, optional
            Soft cap on dead points.
        out_dir : str, optional
            If given, save the finalised sampling state (positions,
            log-likelihoods, configuration, best-fit, posterior) to
            ``<out_dir>/state.npz`` for later auditing / re-analysis.
        """
        sampler = NSSampler(
            self._likelihood, self._sfh_model, priors=priors,
            n_live=n_live, num_delete=num_delete,
            num_inner_steps=num_inner_steps,
        )
        sampler.run(max_ncalls=max_ncalls, seed=seed)

        if out_dir is not None:
            sampler.save_state(os.path.join(out_dir, "state.npz"))

        self._sampler = sampler
        return MCMCResult(sampler, self._model)

    @property
    def likelihood(self):
        """The JAX likelihood."""
        return self._likelihood

    @property
    def model(self):
        """The NumPy model container (plots / saving)."""
        return self._model
