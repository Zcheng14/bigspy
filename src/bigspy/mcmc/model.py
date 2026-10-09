"""NumPy model container for building model spectra (plots / saving).

This is the NumPy counterpart of :mod:`bigspy.mcmc.likelihood_jax`: it holds the
observed data and the CSP / broadening / dust pipeline, and builds a model
spectrum for a given set of parameters.  It contains no chi-squared; the
likelihood lives in ``likelihood_jax`` and is JAX-only.
"""

import numpy as np

from ..constants import DLOGW_VEL, NR_RANGE
from ..utils import median_in_window
from .csp import CSPBuilder
from .kinematics import VelocityBroadening


class ModelComponents:
    """Builds CSP model spectra on the observed grid (NumPy).

    Parameters
    ----------
    ssp : SSPLibrary
        Loaded SSP library.
    ow, oflux, oerr, omask : ndarray
        Observed spectrum arrays (rest-frame, trimmed).
    ve, vd : float
        Velocity shift / dispersion from SpecFit (km/s).
    dust : DustAttenuation
        Dust curve from SpecFit mode 2.
    nr : tuple
        5500 A normalization window.
    velscale : float, optional
        Velocity scale per pixel (km/s); default from constants.
    """

    def __init__(self, ssp, ow, oflux, oerr, omask, ve, vd, dust,
                 nr=NR_RANGE, velscale=None):
        if velscale is None:
            velscale = DLOGW_VEL

        self.ssp = ssp
        self.dust = dust
        self.builder = CSPBuilder(ssp)
        self.broadener = VelocityBroadening(vd, velscale)
        self.obs_wave = np.asarray(ow)
        self.obs_mask = np.asarray(omask, dtype=bool)
        self._n_range = nr

        n = median_in_window(ow, oflux, omask, nr)
        self.obs_flux = np.asarray(oflux) / n
        self.obs_error = np.asarray(oerr) / n

    def build_csp(self, logZsun, sfh):
        """CSP on the SSP grid: built -> broadened -> normalized -> dusted."""
        csp = self.builder.build(logZsun, sfh)
        csp = self.broadener.apply(csp)
        n = median_in_window(self.ssp.wave, csp, np.ones_like(csp, dtype=bool),
                             self._n_range)
        return self.dust.apply(csp / n)

    def build_model(self, logZsun, sfh):
        """CSP interpolated onto the observed wavelength grid."""
        csp = self.build_csp(logZsun, sfh)
        return np.interp(self.obs_wave, self.ssp.wave, csp, left=0.0, right=0.0)
