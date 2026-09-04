"""Likelihood: chi-squared spectral fitting likelihood."""

import numpy as np
from .kinematics import VelocityBroadening
from .csp import CSPBuilder


class Likelihood:
    def __init__(self, ssp, ow, oflux, oerr, omask, ve, vd, dust, nr=(5450, 5550)):
        self.ssp, self.ve = ssp, ve
        self.obs_wave = np.asarray(ow)
        self.obs_mask = np.asarray(omask, dtype=bool)
        self._n_range = nr
        n = self._med5500(ow, oflux, omask, nr)
        self.obs_flux = np.asarray(oflux) / n
        self.obs_error = np.asarray(oerr) / n
        self.broadener = VelocityBroadening(vd)
        self.dust, self.builder = dust, CSPBuilder(ssp)

    @staticmethod
    def _med5500(w, f, m, r):
        mm = (w >= r[0]) & (w <= r[1]) & np.asarray(m, dtype=bool)
        return (
            np.median(np.asarray(f)[mm])
            if mm.sum() > 5
            else np.median(np.asarray(f)[np.asarray(m, dtype=bool)])
        )

    def __call__(self, logZsun, sfh):
        c = self.builder.build(logZsun, sfh)
        c = self.broadener.apply(c)
        n = self._med5500(self.ssp.wave, c, np.ones_like(c, dtype=bool), self._n_range)
        c = c / n
        c = self.dust.apply(c)
        m = np.interp(self.obs_wave, self.ssp.wave, c, left=0.0, right=0.0)
        r = (m - self.obs_flux) / self.obs_error
        return np.sum(r[self.obs_mask] ** 2)

    def call_batch(self, logZsun_arr, sfh_class, sfh_params_2d):
        """Batch chi-squared computation for UltraNest vectorized mode.
        
        Parameters
        ----------
        logZsun_arr : ndarray, shape (N,)
        sfh_class : type
            SFH model class (needed for evaluate_batch).
        sfh_params_2d : ndarray, shape (N, n_sfh_params)
        
        Returns
        -------
        chi2 : ndarray, shape (N,)
        """
        # Build CSP for all parameter sets
        csp = self.builder.build_batch(logZsun_arr, sfh_params_2d, sfh_class)  # (N, n_wave_ssp)
        
        # Apply velocity broadening (batch)
        csp = self.broadener.apply_batch(csp)
        
        # Normalize at 5500 (batch)
        if self._n_range is not None:
            nr_lo, nr_hi = self._n_range
            mm = (self.ssp.wave >= nr_lo) & (self.ssp.wave <= nr_hi)
            if mm.sum() > 5:
                norms = np.median(csp[:, mm], axis=1)
            else:
                norms = np.median(csp, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            csp = csp / norms[:, np.newaxis]
        
        # Apply dust (broadcast: (N, n_wave_ssp) * (n_wave_ssp,))
        csp = csp * self.dust._curve[np.newaxis, :]
        
        # Interpolate to observed wavelength grid (batch)
        from scipy.interpolate import interp1d
        interp = interp1d(self.ssp.wave, csp, axis=1, kind='linear',
                         bounds_error=False, fill_value=0.0)
        model = interp(self.obs_wave)  # (N, n_obs)
        
        # Compute chi2
        residuals = (model - self.obs_flux[np.newaxis, :]) / self.obs_error[np.newaxis, :]
        chi2 = np.sum(residuals[:, self.obs_mask] ** 2, axis=1)
        return chi2

    ndof = property(lambda s: s.obs_mask.sum() - 3)
