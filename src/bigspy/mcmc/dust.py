"""
Dust attenuation — Calzetti+2000 curve + polynomial mode.

The Calzetti law itself lives in :mod:`bigspy.utils` (``calz_klam`` /
``calz_attenuation`` / ``calz_deredden``); this module provides the
:class:`DustAttenuation` container that bundles a precomputed curve with
a wavelength grid.  All curves here are attenuation factors meant to
multiply MODEL flux (positive ebv dims).
"""
import numpy as np

from ..constants import C_LIGHT  # noqa: F401
from ..utils import calz_klam


class DustAttenuation:
    """
    Multiplicative dust attenuation curve.

    Two modes:
      "poly"     — A_lambda = p1*(x-x_v) + p2*(x**2 - x_v**2)
      "calzetti" — Calzetti+2000 with parameter ebv

    Construct normally via __init__, or through the convenience
    classmethods from_mode2 / from_calzetti.
    """

    def __init__(self, wave_grid, mode="poly", ebv=None, p1=None, p2=None):
        self.mode = mode
        self.wave_grid = np.asarray(wave_grid)
        self._ebv = ebv
        self._p1 = p1
        self._p2 = p2
        self._curve = self._compute(wave_grid, mode, ebv=ebv, p1=p1, p2=p2)

    @staticmethod
    def _compute(wave, mode, ebv=None, p1=None, p2=None):
        """Compute the attenuation curve (factor multiplying model flux)."""
        wave = np.asarray(wave, float)
        x = 10000.0 / wave
        xv = 10000.0 / 5500.0

        if mode == "poly":
            A = p1 * (x - xv) + p2 * (x ** 2 - xv ** 2)
        elif mode == "calzetti":
            A = calz_klam(wave) * ebv
        else:
            raise ValueError(f"Unknown dust mode: {mode}")

        return 10.0 ** (-0.4 * A)

    def apply(self, flux, wave=None):
        """
        Apply dust attenuation to flux.

        Parameters
        ----------
        flux : array_like
            Input spectrum.
        wave : array_like, optional
            Wavelength grid.  If different from the stored wave_grid,
            the curve is recomputed.

        Returns
        -------
        ndarray
            Attenuated flux = flux * correction_curve.
        """
        if wave is not None and not np.array_equal(wave, self.wave_grid):
            c = self._compute(
                wave,
                self.mode,
                ebv=getattr(self, "_ebv", None),
                p1=getattr(self, "_p1", None),
                p2=getattr(self, "_p2", None),
            )
        else:
            c = self._curve
        return flux * c

    @classmethod
    def from_mode2(cls, wg, p1, p2):
        """Construct a polynomial-mode instance from p1, p2."""
        o = cls.__new__(cls)
        o.mode = "poly"
        o._p1 = p1
        o._p2 = p2
        o.wave_grid = np.asarray(wg)
        o._curve = cls._compute(wg, "poly", p1=p1, p2=p2)
        return o

    @classmethod
    def from_calzetti(cls, wg, ebv):
        """Construct a Calzetti-mode instance from E(B-V)."""
        o = cls.__new__(cls)
        o.mode = "calzetti"
        o._ebv = ebv
        o.wave_grid = np.asarray(wg)
        o._curve = cls._compute(wg, "calzetti", ebv=ebv)
        return o
