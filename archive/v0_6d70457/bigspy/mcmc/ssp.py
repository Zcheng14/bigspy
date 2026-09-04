"""SSP (Simple Stellar Population) library loaded from FITS files.

Provides the SSPLibrary class for loading and querying BC03-style SSP spectra.
"""

import os

import numpy as np
from astropy.io import fits

C_LIGHT = 299792.458  # km/s


class SSPLibrary:
    """Loads an SSP spectral library from a FITS file.

    The FITS file must contain HDU extensions:
        WAVE  - wavelength grid (1D)
        SPEC  - spectra cube (N_metal x N_age x N_wave)
        MASS  - mass grid (N_metal x N_age)
        TIME  - age grid (1D)
        DT    - age bin widths (1D)
        METAL - metallicity grid (1D)
    """

    def __init__(self, path, wave_range=None):
        with fits.open(path) as h:
            wf = h["WAVE"].data
            sr = h["SPEC"].data
            self.mass, self.time, self.dt = h["MASS"].data, h["TIME"].data, h["DT"].data
            self.metal = h["METAL"].data
        if wave_range is not None:
            m = (wf >= wave_range[0]) & (wf <= wave_range[1])
            self.wave, self._spec = wf[m], sr[:, :, m]
        else:
            self.wave, self._spec = wf, sr
        self.n_metal, self.n_age, self.n_wave = self._spec.shape

    def get_spectrum(self, mi, ai):
        """Return the spectrum at metallicity index *mi* and age index *ai*."""
        return self._spec[mi, ai, :].copy()

    def get_mass(self, mi, ai):
        """Return the mass at metallicity index *mi* and age index *ai*."""
        return self.mass[mi, ai]

    def __repr__(self):
        return (
            f"SSPLibrary(n_metal={self.n_metal}, n_age={self.n_age}, "
            f"n_wave={self.n_wave}, wave=[{self.wave[0]:.0f},{self.wave[-1]:.0f}]A, "
            f"Z=[{self.metal[0]:.4f},{self.metal[-1]:.4f}])"
        )
