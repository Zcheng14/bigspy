"""Emission line mask definitions for spectral fitting."""

# Default emission line regions to mask (Angstrom)
DEFAULT_EMISSION_MASK = [
    (3715, 3740),   # [OII] 3727
    (3850, 3910),   # Balmer break region
    (3940, 4020),   # Ca H&K + [NeIII]
    (4080, 4120),   # H-delta
    (4310, 4370),   # H-gamma + [OIII]
    (4830, 4890),   # H-beta
    (4940, 5020),   # [OIII] 4959, 5007
    (5850, 5910),   # HeI + NaD
    (6280, 6330),   # [OI] 6300
    (6510, 6610),   # H-alpha + [NII]
    (6700, 6770),   # [SII] 6717, 6731
]


def build_emission_mask(wave, regions=None):
    """Build boolean emission line mask for a wavelength grid.

    Parameters
    ----------
    wave : ndarray
        Wavelength grid (Angstrom).
    regions : list of (lo, hi) tuples, optional
        Emission line regions. Uses DEFAULT_EMISSION_MASK if None.

    Returns
    -------
    mask : ndarray of bool
        True = keep pixel, False = masked (emission line).
    """
    import numpy as np
    if regions is None:
        regions = DEFAULT_EMISSION_MASK
    mask = np.ones(len(wave), dtype=bool)
    for lo, hi in regions:
        mask[(wave >= lo) & (wave <= hi)] = False
    return mask
