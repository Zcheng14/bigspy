"""Likelihood: chi-squared spectral fitting likelihood."""

import numpy as np
from .kinematics import VelocityBroadening
from .csp import CSPBuilder


class _LinearInterpPlan:
    """Precomputed linear-interpolation plan for a fixed (x_src, x_new) pair.

    Reproduces ``scipy.interpolate.interp1d(x, y, axis=1, kind='linear',
    bounds_error=False, fill_value=0.0)`` bit-for-bit (same formula as
    scipy's ``_call_linear``, including strict out-of-bounds zeroing),
    without rebuilding an interp1d object per call.
    """

    def __init__(self, x_src, x_new):
        x = np.asarray(x_src, dtype=float)
        xo = np.asarray(x_new, dtype=float)
        idx = np.clip(np.searchsorted(x, xo), 1, len(x) - 1)
        self._lo = idx - 1
        self._dx = x[idx] - x[idx - 1]          # (n_new,)
        self._dxo = xo - x[idx - 1]             # (n_new,)
        self._out = (xo < x[0]) | (xo > x[-1])  # strictly OOB -> 0.0

    def apply(self, y):                         # y: (N, n_src) -> (N, n_new)
        slope = (y[:, self._lo + 1] - y[:, self._lo]) / self._dx
        out = slope * self._dxo + y[:, self._lo]
        out[:, self._out] = 0.0
        return out


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
        # Fixed grids / curve for the lifetime of the object — precomputed once.
        self._interp = _LinearInterpPlan(ssp.wave, self.obs_wave)
        if nr is not None:
            nr_mask = (ssp.wave >= nr[0]) & (ssp.wave <= nr[1])
            self._nr_mask, self._nr_ok = nr_mask, bool(nr_mask.sum() > 5)
        else:
            self._nr_mask, self._nr_ok = None, False
        self._ssp_true = np.ones(ssp.wave.shape, dtype=bool)
        self._dust_curve = np.asarray(dust._curve)

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
        n = self._med5500(self.ssp.wave, c, self._ssp_true, self._n_range)
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
            if self._nr_ok:
                norms = np.median(csp[:, self._nr_mask], axis=1)
            else:
                norms = np.median(csp, axis=1)
            norms = np.where(norms == 0, 1.0, norms)
            csp = csp / norms[:, np.newaxis]

        # Apply dust (broadcast: (N, n_wave_ssp) * (n_wave_ssp,))
        csp = csp * self._dust_curve

        # Interpolate to observed wavelength grid (batch)
        model = self._interp.apply(csp)  # (N, n_obs)

        # Compute chi2
        residuals = (model - self.obs_flux[np.newaxis, :]) / self.obs_error[np.newaxis, :]
        chi2 = np.sum(residuals[:, self.obs_mask] ** 2, axis=1)
        # Guard against NaN/Inf (extreme parameter values)
        chi2 = np.where(np.isfinite(chi2), chi2, 1e30)
        return chi2

    ndof = property(lambda s: s.obs_mask.sum() - 3)
