"""CSPBuilder: Composite Stellar Population builder via SFH convolution."""

import numpy as np
from .ssp import SSPLibrary  # for type hints  # noqa: F401


class CSPBuilder:
    def __init__(self, ssp):
        self.ssp = ssp

    def build(self, logZsun, sfh):
        """Build CSP at log(Z/Z_sun)."""
        Z_grid = self.ssp.metal
        logZ_grid = np.log10(Z_grid / 0.02)
        logM = logZsun
        if logM <= logZ_grid[0]:
            return self._conv(0, sfh)
        if logM >= logZ_grid[-1]:
            return self._conv(len(Z_grid) - 1, sfh)
        i = np.searchsorted(logZ_grid, logM)
        f = (logM - logZ_grid[i - 1]) / (logZ_grid[i] - logZ_grid[i - 1])
        return (1 - f) * self._conv(i - 1, sfh) + f * self._conv(i, sfh)

    def build_batch(self, logZsun_arr, sfh_params_2d, sfh_class):
        """Build CSP for multiple parameter sets.

        Parameters
        ----------
        logZsun_arr : ndarray, shape (N,)
            log(Z/Z_sun) for each set.
        sfh_params_2d : ndarray, shape (N, n_sfh_params)
            SFH parameters for each set.
        sfh_class : type
            SFH model class.

        Returns
        -------
        csp : ndarray, shape (N, n_wave)
        """
        # Evaluate SFH for all parameter sets (classmethod)
        sfr = sfh_class.evaluate_batch(self.ssp.time, sfh_params_2d)  # (N, n_age)
        # sfr is (N, n_age), dt is (n_age,)
        w = sfr * self.ssp.dt  # (N, n_age)
        w = w / w.sum(axis=1, keepdims=True)

        Z_grid = self.ssp.metal
        logZ_grid = np.log10(Z_grid / 0.02)
        N = len(logZsun_arr)
        n_wave = self.ssp.n_wave
        result = np.zeros((N, n_wave))

        # Metal interpolation per sample (loop over samples is OK since N is moderate)
        for j in range(N):
            logM = logZsun_arr[j]
            if logM <= logZ_grid[0]:
                result[j] = np.dot(w[j], self.ssp._spec[0, :, :])
            elif logM >= logZ_grid[-1]:
                result[j] = np.dot(w[j], self.ssp._spec[-1, :, :])
            else:
                i = np.searchsorted(logZ_grid, logM)
                f = (logM - logZ_grid[i-1]) / (logZ_grid[i] - logZ_grid[i-1])
                spec_lo = np.dot(w[j], self.ssp._spec[i-1, :, :])
                spec_hi = np.dot(w[j], self.ssp._spec[i, :, :])
                result[j] = (1-f) * spec_lo + f * spec_hi

        return result

    def _conv(self, mi, sfh):
        w = sfh.evaluate(self.ssp.time) * self.ssp.dt
        w /= w.sum()
        return np.dot(w, self.ssp._spec[mi, :, :])
