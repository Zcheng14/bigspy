"""FITS and data I/O utilities for bigspy."""

import numpy as np
from astropy.io import fits


def read_observed_fits(path):
    """Read observed spectrum from a standard FITS file.
    
    Expected HDU structure:
        WAVE  - wavelength grid (Angstrom)
        FLUX  - flux
        ERROR - uncertainty
        MASK  - (optional) boolean mask (1=good pixel)
    
    Parameters
    ----------
    path : str
        Path to FITS file.
    
    Returns
    -------
    dict with keys: wave, flux, error, mask, header
    """
    with fits.open(path) as h:
        wave = h["WAVE"].data.astype(np.float64)
        flux = h["FLUX"].data.astype(np.float64)
        error = h["ERROR"].data.astype(np.float64)
        mask = h["MASK"].data.astype(bool) if "MASK" in h else np.ones_like(wave, dtype=bool)
        header = dict(h[0].header)
    return {"wave": wave, "flux": flux, "error": error, "mask": mask, "header": header}


def write_observed_fits(path, wave, flux, error, mask=None, header_kw=None, overwrite=True):
    """Write observed spectrum to FITS.
    
    Parameters
    ----------
    path : str
        Output path.
    wave, flux, error : ndarray
        Spectrum arrays.
    mask : ndarray, optional
        Boolean mask.
    header_kw : dict, optional
        Additional header keywords.
    overwrite : bool
    """
    hdul = [fits.PrimaryHDU()]
    if header_kw:
        for k, v in header_kw.items():
            hdul[0].header[k] = v
    hdul.append(fits.ImageHDU(wave.astype(np.float64), name="WAVE"))
    hdul.append(fits.ImageHDU(flux.astype(np.float64), name="FLUX"))
    hdul.append(fits.ImageHDU(error.astype(np.float64), name="ERROR"))
    if mask is not None:
        hdul.append(fits.ImageHDU(mask.astype(np.uint8), name="MASK"))
    fits.HDUList(hdul).writeto(path, overwrite=overwrite)


def read_specfit_fits(path):
    """Read a SpecFit FITS result back.
    
    Returns a dict with the stored data for inspection.
    """
    with fits.open(path) as h:
        result = {"wave": h["WAVE"].data, "flux": h["FLUX"].data,
                  "error": h["ERROR"].data}
        if "PARAMS" in h:
            result["params"] = {col.name: h["PARAMS"].data[col.name][0]
                                for col in h["PARAMS"].columns}
        if "BESTFIT" in h:
            result["bestfit"] = h["BESTFIT"].data
        if "DUST" in h:
            result["dust"] = h["DUST"].data
    return result
