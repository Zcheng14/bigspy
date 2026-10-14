"""CSPBuilder: Composite Stellar Population builder via SFH convolution."""

import numpy as np

from .ssp import SSPLibrary  # noqa: F401  (for type hints)


class CSPBuilder:
    def __init__(self, ssp):
        self.ssp = ssp
        # Fixed for the lifetime of the builder -- precomputed once.
        self._logZ_grid = np.log10(np.asarray(ssp.metal) / 0.02)

    def build(self, logZsun, sfh):
        """Build the CSP at ``log(Z/Z_sun)``.

        The metallicity grid is small, so the two bracketing metallicities are
        convolved and linearly interpolated in ``logZ`` (clamped outside).
        """
        grid = self._logZ_grid
        if logZsun <= grid[0]:
            return self._conv(0, sfh)
        if logZsun >= grid[-1]:
            return self._conv(len(grid) - 1, sfh)
        i = np.searchsorted(grid, logZsun)
        f = (logZsun - grid[i - 1]) / (grid[i] - grid[i - 1])
        return (1 - f) * self._conv(i - 1, sfh) + f * self._conv(i, sfh)

    def _conv(self, mi, sfh):
        w = sfh.evaluate(self.ssp.time) * self.ssp.dt
        w /= w.sum()
        return np.dot(w, self.ssp._spec[mi, :, :])
