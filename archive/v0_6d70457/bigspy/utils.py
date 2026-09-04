"""Utility functions for spectral fitting and data manipulation."""

import numpy as np
from scipy import interpolate


def rebin(x, y, x0=None, dlogx=None):
    """Rebin spectrum to a new wavelength grid with flux conservation.

    Uses trapezoidal integration within each new bin to preserve total
    flux. Boundary values are linearly interpolated from the original grid.

    Parameters
    ----------
    x : ndarray
        Original wavelength grid (Angstrom).
    y : ndarray
        Flux values on the original grid.
    x0 : ndarray, optional
        Target wavelength grid. If None, a logarithmic grid is built
        from x[0] with step size dlogx.
    dlogx : float, optional
        Logarithmic step size for the target grid (default: 0.0001).

    Returns
    -------
    x0 : ndarray
        New wavelength grid.
    y_new : ndarray
        Rebinned flux.

    Notes
    -----
    When x0 is None, the resulting grid spans from x[0] to nearly
    x[-1] with constant logarithmic spacing dlogx.
    """
    if x0 is None:
        if dlogx is None:
            dlogx = 0.0001
        m = int((np.log10(x[-1]) - np.log10(x[0])) / dlogx)
        x0 = x[0] * 10 ** (np.arange(m) * dlogx)
    else:
        m = len(x0)

    y_new = np.zeros(m)
    x0_diff = np.diff(x0)

    for i in range(m):
        if i == 0:
            x1 = x[np.where(x <= x0[0])[0][-1]]
            x2 = x0[i] + x0_diff[i] / 2
        elif i == m - 1:
            x1 = x0[i] - x0_diff[i - 1] / 2
            x2 = x[np.where(x >= x0[-1])[0][0]]
        else:
            x1 = x0[i] - x0_diff[i - 1] / 2
            x2 = x0[i] + x0_diff[i] / 2

        dx = x2 - x1

        xi1 = np.where(x >= x1)[0][0]
        y1 = ((x[xi1] - x1) * y[xi1 - 1] + (x1 - x[xi1 - 1]) * y[xi1]) / (x[xi1] - x[xi1 - 1])

        xi2 = np.where(x >= x2)[0][0]
        y2 = ((x[xi2] - x2) * y[xi2 - 1] + (x2 - x[xi2 - 1]) * y[xi2]) / (x[xi2] - x[xi2 - 1])

        y_new[i] = (x2 - x1) * (y2 + y1) / 2 / dx

    return x0, y_new


def log_rebin(wave, spec, dv=None):
    """Log-rebin a spectrum to constant velocity resolution.

    For a fixed velocity resolution dv, dlog(lambda) is constant.
    Linear interpolation on a log-uniform wavelength grid is
    equivalent to log-rebinning.

    Parameters
    ----------
    wave : ndarray
        Wavelength vector (Angstrom), monotonically increasing.
    spec : ndarray
        Spectrum flux vector.
    dv : float or None, optional
        Target velocity resolution (km/s). If None, uses the maximum
        native dlog(lambda) of the input wavelength grid.

    Returns
    -------
    spec_rebin : ndarray
        Rebinned spectrum.
    loglam : ndarray
        Log-uniform wavelength grid (Angstrom).
    dv_used : float
        Actual velocity resolution used (km/s).
    """
    from .constants import C_LIGHT

    dlam_native = np.min(np.diff(np.log(wave)))

    if dv is None:
        dv = dlam_native * C_LIGHT
    else:
        dlam_native = dv / C_LIGHT

    # Build log-uniform wavelength grid
    N = int(np.floor((np.max(np.log(wave)) - np.log(wave[1])) / dlam_native))
    maxlam = np.log(wave[1]) + dlam_native * N
    loglam = np.exp(np.linspace(np.log(wave[1]), maxlam, N))

    # Linear interpolation to new grid
    interp_fn = interpolate.interp1d(wave, spec, bounds_error=False,
                                     fill_value=(spec[0], spec[-1]))
    spec_rebin = interp_fn(loglam)

    return spec_rebin, loglam, dv


def air_to_vacuum_wave(lam):
    """Convert air wavelengths to vacuum wavelengths.

    Parameters
    ----------
    lam : ndarray or float
        Air wavelengths in Angstrom.

    Returns
    -------
    ndarray or float
        Vacuum wavelengths in Angstrom.
    """
    lam = np.asarray(lam, dtype=float)
    sigma2 = (1e4 / lam) ** 2
    fact = (1 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma2) + 2.5540e-4 / (41.0 - sigma2))
    return lam * fact
