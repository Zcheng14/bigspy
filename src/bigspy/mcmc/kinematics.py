"""
Velocity broadening — Gaussian convolution for spectral kinematics.
Batch-capable for UltraNest vectorized sampling.
"""
import numpy as np


def _build_convolution_matrix(n_pix, sigma_pix, x0_pix=0.0):
    """Pre-compute Gaussian convolution matrix.
    
    Parameters
    ----------
    n_pix : int
        Number of wavelength pixels.
    sigma_pix : float
        Gaussian width in pixel units.
    x0_pix : float
        Kernel centre offset in pixel units.
    
    Returns
    -------
    K : ndarray, shape (n_pix, n_pix)
        Convolution matrix. convolved = K @ spectrum
    """
    if sigma_pix <= 0:
        return np.eye(n_pix)
    
    khalf = round(4 * sigma_pix + abs(x0_pix) + 3)
    xx = np.arange(khalf * 2 + 1) - khalf
    kernel = np.exp(-(xx - x0_pix) ** 2 / (2 * sigma_pix ** 2))
    kernel /= kernel.sum()
    
    # Build convolution matrix using "same" mode
    K = np.zeros((n_pix, n_pix))
    offset = khalf
    for i in range(len(kernel)):
        for j in range(n_pix):
            src = j + i - offset
            if 0 <= src < n_pix:
                K[j, src] += kernel[i]
    
    return K


def gauss_convolve(y, sigma, x0=0.0):
    """Convolve spectrum with a normalised Gaussian kernel (1D).
    
    Uses np.convolve("same") for single spectra.
    
    Parameters
    ----------
    y : array_like
        Input spectrum.
    sigma : float
        Gaussian width in pixel units. sigma <= 0 -> no-op.
    x0 : float
        Kernel centre offset in pixel units.
    
    Returns
    -------
    ndarray
        Convolved spectrum, same length as input.
    """
    if sigma <= 0:
        return np.asarray(y)
    khalfsz = round(4 * sigma + abs(x0) + 3)
    xx = np.arange(khalfsz * 2 + 1) - khalfsz
    kernel = np.exp(-(xx - x0) ** 2 / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return np.convolve(y, kernel, "same")


def gauss_convolve_batch(spectra, sigma_pix, x0_pix=0.0):
    """Convolve multiple spectra with the same Gaussian kernel.
    
    Parameters
    ----------
    spectra : ndarray, shape (N, n_pix) or (n_pix,)
        Input spectra.
    sigma_pix : float
        Gaussian width in pixel units.
    x0_pix : float
        Kernel centre offset.
    
    Returns
    -------
    convolved : ndarray, shape matching input
    """
    spectra = np.atleast_2d(np.asarray(spectra))
    n_pix = spectra.shape[1]
    K = _build_convolution_matrix(n_pix, sigma_pix, x0_pix)
    # (N, n_pix) @ (n_pix, n_pix).T -> but K is symmetric so K.T works
    return (K @ spectra.T).T


class VelocityBroadening:
    """Apply Gaussian velocity broadening to spectra (batch-capable).
    
    Parameters
    ----------
    vd : float
        Velocity dispersion in km/s.
    velscale : float
        Velocity spacing per pixel (km/s).
    """
    
    DLOGW_VEL = (10 ** 0.0001 - 1) * 299792.458
    
    def __init__(self, vd, velscale=None):
        if velscale is None:
            velscale = self.DLOGW_VEL
        self.sigma_pix = vd / velscale if vd > 0 else 0.0
        self._conv_matrix = None  # cached
    
    def _get_conv_matrix(self, n_pix):
        if self._conv_matrix is None or self._conv_matrix.shape[0] != n_pix:
            self._conv_matrix = _build_convolution_matrix(n_pix, self.sigma_pix)
        return self._conv_matrix
    
    def apply(self, spectrum):
        """Apply broadening to a single spectrum."""
        return gauss_convolve(spectrum, self.sigma_pix)
    
    def apply_batch(self, spectra):
        """Apply broadening to multiple spectra (N, n_pix)."""
        return gauss_convolve_batch(spectra, self.sigma_pix)
