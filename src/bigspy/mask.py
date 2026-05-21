"""Emission line mask definitions for spectral fitting.

Provides a detailed emission line list (~50 lines) from the original BIGS
SpecFit.py Emlines module, plus the original 11 broad default regions.
"""

# ── Detailed emission line list (Å) ──────────────────────────────────
# From BIGS SpecFit.py / Emlines: individual line centers ± FWHM equivalent
# Format: {label: [lo, hi]} in rest-frame Angstrom
EMISSION_LINES = {
    # Near-UV / Blue
    "l3710": [3710, 3737],
    "l3795": [3795, 3805],
    "l3812": [3812, 3828],
    "l3830": [3830, 3845],
    "l3855": [3855, 3876],
    "l3883": [3883, 3896],
    "l3960": [3960, 3980],
    "l4020": [4020, 4033],
    "l4063": [4063, 4077],
    "l4095": [4095, 4110],
    # Blue-green
    "l4333": [4333, 4350],
    "l4356": [4356, 4372],
    "l4384": [4384, 4395],
    "l4466": [4466, 4478],
    "l4571": [4571, 4578],
    "l4652": [4652, 4666],
    "l4681": [4681, 4698],
    "l4707": [4707, 4720],
    "l4734": [4734, 4747],
    # H-beta region
    "l4850": [4850, 4871],
    "l4916": [4916, 4928],
    "l4946": [4946, 4968],
    "l4983": [4983, 4993],
    "l4985": [4985, 5020],
    "l5000": [5000, 5014],
    "l5014": [5014, 5022],
    "l5033": [5033, 5057],
    "l5077": [5077, 5105],
    # Mg / Na region
    "l5185": [5185, 5210],
    "l5509": [5509, 5525],
    # Orange / Red
    "l5866": [5866, 5888],
    "l6288": [6288, 6320],
    "l6308": [6308, 6322],
    "l6355": [6355, 6375],
    # H-alpha region
    "l6535": [6535, 6555],
    "l6555": [6555, 6575],
    "l6575": [6575, 6595],
    "l6668": [6668, 6690],
    "l6700": [6700, 6725],
    "l6725": [6725, 6740],
    "l6745": [6745, 6755],
    # NIR
    "l7058": [7058, 7075],
    "l7128": [7128, 7146],
    "l7275": [7275, 7290],
    "l7313": [7313, 7327],
    "l7327": [7327, 7343],
    "l7370": [7370, 7390],
    "l7540": [7540, 7570],
    "l7741": [7741, 7763],
    "l8440": [8440, 8462],
    "l8492": [8492, 8516],
    "l8540": [8540, 8558],
    "l8590": [8590, 8612],
    "l8656": [8656, 8680],
    "l8742": [8742, 8764],
    "l8856": [8856, 8876],
    "l9006": [9006, 9028],
    "l9060": [9060, 9082],
}

# ── Default broad regions (original 11 groups) ───────────────────────
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


def build_emission_mask(wave, regions=None, detailed=False):
    """Build boolean emission line mask for a wavelength grid.

    Parameters
    ----------
    wave : ndarray
        Wavelength grid (Angstrom).
    regions : list of (lo, hi) tuples, optional
        Custom emission line regions. Uses DEFAULT_EMISSION_MASK if None.
    detailed : bool, optional
        If True, use the detailed EMISSION_LINES dict (~50 lines).
        If False (default), use DEFAULT_EMISSION_MASK (11 broad groups).

    Returns
    -------
    mask : ndarray of bool
        True = keep pixel, False = masked (emission line).
    """
    import numpy as np
    mask = np.ones(len(wave), dtype=bool)

    if regions is not None:
        for lo, hi in regions:
            mask[(wave >= lo) & (wave <= hi)] = False
    elif detailed:
        for lo, hi in EMISSION_LINES.values():
            mask[(wave >= lo) & (wave <= hi)] = False
    else:
        for lo, hi in DEFAULT_EMISSION_MASK:
            mask[(wave >= lo) & (wave <= hi)] = False

    return mask


def mask_emlines_detailed(wave, mask_in, mask_add=None):
    """Full detailed emission line masking (mimics original BIGS SpecFit logic).

    Parameters
    ----------
    wave : ndarray
        Rest-frame wavelength grid (Å).
    mask_in : ndarray of bool
        Input mask (True = good pixel).
    mask_add : dict, optional
        Additional mask regions e.g. {'my_line': [w1, w2]}.

    Returns
    -------
    mask_out : ndarray of bool
        Updated mask with emission lines masked.
    lines : dict
        The emission line table used.
    """
    import numpy as np
    lines = dict(EMISSION_LINES)

    mask_out = np.asarray(mask_in, dtype=bool).copy()

    if mask_add is not None:
        lines.update(mask_add)

    for lo, hi in lines.values():
        mask_out[(wave >= lo) & (wave <= hi)] = False

    return mask_out, lines
