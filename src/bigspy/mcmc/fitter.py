"""MCMC Fitter — high-level interface for Bayesian spectral fitting."""

import os
import numpy as np

from .ssp import SSPLibrary
from .dust import DustAttenuation
from .likelihood import Likelihood
from .sampler import UltraNestSampler


class MCMCResult:
    """Container for MCMC fitting results."""
    
    def __init__(self, sampler, likelihood_np=None):
        self._sampler = sampler
        self._likelihood_np = likelihood_np  # NumPy Likelihood for plotting/saving
        self.result = sampler.result  # Raw UltraNest result dict

    @property
    def _like(self):
        """NumPy Likelihood for plotting/saving (falls back to sampler's like)."""
        if self._likelihood_np is not None:
            return self._likelihood_np
        return self._like
    
    @property
    def bestfit(self):
        """Best-fit parameters dict."""
        point = self._sampler.get_bestfit()
        names = self._sampler.param_names
        return dict(zip(names, point))
    
    @property
    def posterior(self):
        """Posterior samples (N_samples, n_params)."""
        return self._sampler.get_posterior()
    
    @property
    def log_evidence(self):
        """Log evidence log(Z)."""
        return self.result.get("logz", np.nan)
    
    def save_result(self, path):
        """Save best-fit params + CSP spectrum to FITS.
        
        HDU structure:
            PRIMARY   — header info
            BESTFIT   — parameter table (name, value)
            WAVE      — observed wavelength grid (rest frame)
            FLUX      — observed flux (preprocessed)
            ERROR     — observed error (preprocessed)
            MASK      — pixel mask (1=good)
            CSP       — best-fit CSP spectrum (SSP grid)
            CSP_OBS   — best-fit CSP interpolated to observed grid
        """
        from astropy.io import fits
        import numpy as np
        
        best = self.bestfit
        like = self._like
        
        # Build best-fit CSP
        logZ = best.get("logZsun", 0.0)
        sfh_kw = {k: v for k, v in best.items() if k != "logZsun"}
        sfh_cls = self._sampler.sfh_class
        sfh = sfh_cls(**sfh_kw, age_universe=13.8)
        csp = like.builder.build(logZ, sfh)
        csp = like.broadener.apply(csp)
        n = like._med5500(like.ssp.wave, csp, np.ones_like(csp, dtype=bool), like._n_range)
        csp = csp / n
        csp = like.dust.apply(csp)
        csp_obs = np.interp(like.obs_wave, like.ssp.wave, csp, left=0.0, right=0.0)
        
        # Parameter table
        cols = [fits.Column(name=n, format="D", array=[v]) for n, v in best.items()]
        
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul[0].header["LOGEVID"] = (float(self.log_evidence), "log(Z) evidence")
        hdul.append(fits.BinTableHDU.from_columns(cols, name="BESTFIT"))
        hdul.append(fits.ImageHDU(like.obs_wave.astype(np.float64), name="WAVE"))
        hdul.append(fits.ImageHDU(like.obs_flux.astype(np.float64), name="FLUX"))
        hdul.append(fits.ImageHDU(like.obs_error.astype(np.float64), name="ERROR"))
        hdul.append(fits.ImageHDU(like.obs_mask.astype(np.uint8), name="MASK"))
        hdul.append(fits.ImageHDU(csp.astype(np.float64), name="CSP"))
        hdul.append(fits.ImageHDU(csp_obs.astype(np.float64), name="CSP_OBS"))
        
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        hdul.writeto(path, overwrite=True)
    
    def plot_corner(self, path):
        """Generate corner plot with mathematical labels."""
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
        best = list(self.bestfit.values())
        # Use mathematical labels
        raw_labels = list(self.bestfit.keys())
        _label_map = {
            "logZsun": r"$\log(Z/Z_\odot)$",
            "t0":      r"$t_0\ \mathrm{(Gyr)}$",
            "tau":     r"$\tau\ \mathrm{(Gyr)}$",
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
        """Plot best-fit CSP vs observed spectrum."""
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 12,
        })
        import matplotlib.pyplot as plt
        import numpy as np
        like = self._like
        best = self.bestfit
        sfh_cls = self._sampler.sfh_class
        sfh = sfh_cls(**{k: v for k, v in best.items() if k != "logZsun"},
                       age_universe=13.8)
        logZ = best.get("logZsun", 0.0)
        csp = like.builder.build(logZ, sfh)
        csp = like.broadener.apply(csp)
        n = like._med5500(like.ssp.wave, csp, np.ones_like(csp, dtype=bool), like._n_range)
        csp = csp / n
        csp = like.dust.apply(csp)
        csp_obs = np.interp(like.obs_wave, like.ssp.wave, csp, left=0.0, right=0.0)
        n_obs = 1.0 / np.median(like.obs_flux[like.obs_mask])

        # Build title with proper math notation
        Z = 0.02 * 10 ** logZ
        _name_map = {"logZsun": r"\log(Z/Z_\odot)", "t0": "t_0", "tau": r"\tau"}
        title_parts = []
        for k, v in best.items():
            label = _name_map.get(k, k)
            title_parts.append(rf"${label} = {v:.3f}$")
        title_parts.append(rf"$Z = {Z:.5f}$")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(like.obs_wave, like.obs_flux * n_obs, 'k-', lw=0.5,
                label=r'$\mathrm{Observed}$')
        ax.plot(like.obs_wave, csp_obs * n_obs, 'r-', lw=1,
                label=r'$\mathrm{Best\ fit\ CSP}$')
        ax.set_xlabel(r'$\lambda\ (\mathrm{\AA})$')
        ax.set_ylabel(r'$\mathrm{Normalized}\ F_\lambda$')
        ax.set_title(r'$\mathrm{MCMC\ Best\ Fit:}\ $' + r'$,\ $'.join(title_parts))
        ax.legend(frameon=True, fontsize=11)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_sfh(self, path, n_samples=500):
        """Plot star formation history with 68% confidence interval.
        
        Parameters
        ----------
        path : str
            Output file path.
        n_samples : int
            Number of posterior samples to use for CI computation.
        """
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 12,
        })
        import matplotlib.pyplot as plt
        import numpy as np
        like = self._like
        post = self.posterior
        names = self._sampler.param_names

        # Build SFH param name → posterior column index map
        _sfh_param_idx = {name: i for i, name in enumerate(names) if name != "logZsun"}
        _logZ_idx = names.index("logZsun") if "logZsun" in names else None

        cosmic_time = np.max(like.ssp.time) - like.ssp.time

        # Subsample posterior for efficiency
        n_use = min(n_samples, len(post))
        idx = np.random.choice(len(post), n_use, replace=False)
        post_sub = post[idx]

        # Evaluate SFH for each posterior sample
        sfr_grid = np.zeros((n_use, len(cosmic_time)))
        for i in range(n_use):
            sfh_kwargs = {name: post_sub[i, j] for name, j in _sfh_param_idx.items()}
            sfh = self._sampler.sfh_class(**sfh_kwargs, age_universe=13.8)
            sfr_grid[i] = sfh.evaluate(like.ssp.time)

        # Compute percentiles
        sfr_lo = np.percentile(sfr_grid, 16, axis=0)
        sfr_med = np.percentile(sfr_grid, 50, axis=0)
        sfr_hi = np.percentile(sfr_grid, 84, axis=0)

        # Build title with posterior medians
        _name_map = {"logZsun": r"\log(Z/Z_\odot)", "t0": "t_0", "tau": r"\tau"}
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

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(cosmic_time, sfr_lo, sfr_hi, color='b', alpha=0.2,
                        label=r'$68\%\ \mathrm{CI}$')
        ax.plot(cosmic_time, sfr_med, 'b-', lw=1.5, label=r'$\mathrm{Median}$')
        ax.set_xlabel(r'$\mathrm{Age\ of\ Universe\ (Gyr)}$')
        ax.set_ylabel(r'$\mathrm{SFR\ (arbitrary\ units)}$')
        ax.set_title(r'$\mathrm{Star\ Formation\ History:}\ $' + r'$,\ $'.join(title_parts),
                     fontsize=10)
        ax.legend(frameon=True, fontsize=10)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class MCMCFitter:
    """Bayesian MCMC spectral fitting for stellar population parameters.
    
    Parameters
    ----------
    ssp_fits : str
        Path to SSP template FITS file.
    specfit_result : SpecFitResult
        SpecFit results providing ve, vd, dust curve, and preprocessed spectrum.
    sfh_model : str or type, optional
        SFH model ("delayed" default, or SFHBase subclass).
    wave_range : tuple, optional
        SSP wavelength range (default: (3600, 7400)).
    emission_mask : list, optional
        Additional emission line regions to mask (default: uses SpecFit's mask).
    use_jax : bool, optional
        Use JAX-accelerated likelihood backend (default: False).
        Requires ``jax`` and ``jax.numpy`` installed.
    """

    def __init__(self, ssp_fits, specfit_result, sfh_model="delayed",
                 wave_range=(3600, 7400), emission_mask=None,
                 use_jax=True):
        self.ssp = SSPLibrary(ssp_fits, wave_range=wave_range)
        self._specfit = specfit_result
        self._sfh_model = sfh_model

        # Use preprocessed data from SpecFit (rest-frame, trimmed, MW corrected)
        self._wave_obs = np.asarray(specfit_result.wave_prep, dtype=float)
        self._flux_obs = np.asarray(specfit_result.flux_prep, dtype=float)
        self._error_obs = np.asarray(specfit_result.error_prep, dtype=float)

        # Build mask: combine SpecFit mask + optional additional emission mask
        self._obs_mask = np.asarray(specfit_result.mask_prep, dtype=bool)
        if emission_mask is not None:
            from ...mask import build_emission_mask
            em = build_emission_mask(self._wave_obs, emission_mask)
            self._obs_mask = self._obs_mask & em

        # Build dust from SpecFit
        p1 = getattr(specfit_result, 'p1', 0.0)
        p2 = getattr(specfit_result, 'p2', 0.0)
        self._dust = DustAttenuation.from_mode2(self.ssp.wave, p1, p2)

        ve = specfit_result.ve[0]
        vd = specfit_result.vd[0]

        # Always build NumPy Likelihood (for plotting, backward compat)
        self._likelihood = Likelihood(
            self.ssp, self._wave_obs, self._flux_obs, self._error_obs,
            self._obs_mask, ve, vd, self._dust,
        )

        # Optionally build JAX Likelihood (for faster sampling)
        self._use_jax = use_jax
        if use_jax:
            try:
                from .likelihood_jax import JAXLikelihood
                self._likelihood_jax = JAXLikelihood(
                    self.ssp, self._wave_obs, self._flux_obs, self._error_obs,
                    self._obs_mask, ve, vd, self._dust,
                )
            except ImportError:
                self._use_jax = False
    
    def run(self, n_live=400, chain_dir=None, priors=None,
            frac_remain=0.5, max_ncalls=None, dlogz=0.5,
            min_ess=400, Lepsilon=0.001, max_iters=None, **kwargs):
        """Run MCMC sampling.
        
        Parameters
        ----------
        n_live : int
            Number of live points.
        chain_dir : str, required
            Output directory for UltraNest chains.
        priors : dict, optional
            Parameter name -> Prior object. Uses SFH defaults if None.
        frac_remain : float
            Fraction of likelihood calls for posterior sampling.
        max_ncalls : int, optional
            Maximum likelihood evaluations.
        dlogz : float
            Evidence tolerance.
        min_ess : int
            Minimum effective sample size.
        Lepsilon : float
            Likelihood contour accuracy.
        max_iters : int, optional
            Maximum iterations.
        **kwargs
            Passed to ultranest.ReactiveNestedSampler.run().
        """
        if chain_dir is None:
            raise ValueError("chain_dir is required")

        likelihood = self._likelihood_jax if self._use_jax else self._likelihood

        sampler = UltraNestSampler(
            likelihood, chain_dir, self._sfh_model, priors=priors
        )
        
        sampler.run(
            min_live_points=n_live,
            max_ncalls=max_ncalls,
            frac_remain=frac_remain,
            dlogz=dlogz,
            min_ess=min_ess,
            Lepsilon=Lepsilon,
            max_iters=max_iters,
            **kwargs
        )
        
        self._sampler = sampler
        return MCMCResult(sampler, likelihood_np=self._likelihood)
    
    @property
    def likelihood(self):
        """NumPy Likelihood (for plotting and inspection)."""
        return self._likelihood
