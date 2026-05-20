#!/usr/bin/env python3
"""
SpecFit: PCA-based spectral fitting for stellar kinematics and dust attenuation.

Uses PCA templates (from compute_pca output) to fit observed spectra with:
  model = [Sigma a_i * PC_i] (*) gauss(vd, ve+vsys) * dust_curve(ebv)

Two fitting modes:
  Mode 1 (Calzetti): Direct fit with Calzetti+2000 attenuation curve.
  Mode 2 (S/L): Separate smooth/line components with free-form dust curve.

No emission-line fitting, no stellar-population decomposition.
"""
import pickle
import numpy as np
import astropy.io.fits as fits
import os
import lmfit
import matplotlib

# ── Constants ──────────────────────────────────────────────────
C = 299792.458            # speed of light (km/s)
DLOGW = 0.0001            # log-wavelength spacing of templates
NEIG = 20                 # number of PCA components available
FIT_NEIG = 10             # number of PCA components to use in fit
WAVE_NORM = 5500.0        # normalization wavelength

# ── Emission-line masks for preprocessing ──────────────────────
_EM_LINES = [
    [3715, 3740], [3850, 3910], [3940, 4020], [4080, 4120],
    [4310, 4370], [4830, 4890], [4940, 5020], [5850, 5910],
    [6280, 6330], [6510, 6610], [6700, 6770],
]


# ═══════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════
def air_to_vacuum_wave(lam):
    """Convert air wavelengths (A) to vacuum."""
    lam = np.asarray(lam, dtype=float)
    sigma2 = (1e4 / lam) ** 2
    fact = (1 + 6.4328e-5 + 2.94981e-2 / (146.0 - sigma2)
            + 2.5540e-4 / (41.0 - sigma2))
    return lam * fact


def gauss_convolve(y, sigma, x0=0.0):
    """Convolve spectrum with a Gaussian kernel.
    sigma, x0 are in pixel units."""
    if sigma <= 0:
        return y
    khalfsz = round(4 * sigma + abs(x0) + 3)
    xx = np.arange(khalfsz * 2 + 1) - khalfsz
    kernel = np.exp(-(xx - x0) ** 2 / sigma ** 2 / 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, "same")


def calz_unred(wave, ebv):
    """
    Calzetti+2000 attenuation curve A(lambda).
    Returns 10^(0.4 * k(lam) * ebv).
    ebv > 0 -> deredden (brighten); ebv < 0 -> redden (dim).
    """
    wave = np.asarray(wave, dtype=float)
    x = 10000.0 / wave
    klam = np.zeros_like(x)
    Rv = 4.05

    # 6300-22000 A
    w1 = (wave >= 6300) & (wave <= 22000)
    klam[w1] = 2.659 * (-1.857 + 1.040 * x[w1]) + Rv

    # 912-6300 A
    w2 = (wave >= 912) & (wave < 6300)
    c2 = np.array([-2.156, 1.509, -0.198, 0.011])
    p2 = np.poly1d(c2[::-1])
    klam[w2] = 2.659 * p2(x[w2]) + Rv

    return 10.0 ** (0.4 * klam * ebv)


def ccm_unred(wave, ebv):
    """Cardelli+Clayton+Mathis 1989 (O'Donnell 1994) MW extinction."""
    wave = np.asarray(wave, float)
    x = 10000.0 / wave
    a = np.zeros_like(x)
    b = np.zeros_like(x)
    Rv = 3.1
    g = (x > 0.3) & (x < 1.1)
    a[g] = 0.574 * x[g] ** 1.61
    b[g] = -0.527 * x[g] ** 1.61
    g = (x >= 1.1) & (x < 3.3)
    y = x[g] - 1.82
    c1 = [1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505]
    c2 = [0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347]
    a[g] = np.polyval(c1[::-1], y)
    b[g] = np.polyval(c2[::-1], y)
    g = (x >= 3.3) & (x < 8)
    y = x[g]
    Fa = np.zeros_like(y)
    Fb = np.zeros_like(y)
    g1 = y > 5.9
    Fa[g1] = -0.04473 * (y[g1] - 5.9) ** 2 - 0.009779 * (y[g1] - 5.9) ** 3
    Fb[g1] = 0.2130 * (y[g1] - 5.9) ** 2 + 0.1207 * (y[g1] - 5.9) ** 3
    a[g] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341) + Fa
    b[g] = -3.090 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263) + Fb
    g = (x >= 8) & (x <= 11)
    y = x[g] - 8.0
    a[g] = np.polyval([-0.070, 0.137, -0.628, -1.073], y)
    b[g] = np.polyval([0.374, -0.420, 4.257, 13.670], y)
    Alambda = Rv * ebv * (a + b / Rv)
    return 10.0 ** (0.4 * Alambda)


# ═══════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════
def load_pca_templates(pca_file):
    """Load PCA templates from a FITS file.

    Parameters
    ----------
    pca_file : str
        Path to the PCA FITS file (must contain 'pca_log' and 'wave_log' HDUs).

    Returns
    -------
    pca : ndarray (FIT_NEIG, n_wave)
        PCA component spectra in log-flux.
    wave_temp : ndarray (n_wave,)
        Template wavelength grid in Angstrom.
    velscale : float
        Velocity scale per pixel (km/s).
    """
    with fits.open(pca_file) as h:
        pca = h["pca_log"].data[:FIT_NEIG, :]
        wt = h["wave_log"].data
    return pca, wt, (10 ** DLOGW - 1) * C


def load_test_spectrum(path):
    """Load a pickled test spectrum.

    Parameters
    ----------
    path : str
        Path to the .pkl test spectrum file.

    Returns
    -------
    dict with keys: z, ebv_mw, wave_obs, flux_obs, mask_obs,
                    error_obs, sigma_dap.
    """
    with open(path, "rb") as f:
        z_gal, ebv_mw, w_obs, s_obs, m_obs, e_obs, sig_dap, _ = \
            pickle.load(f, encoding="latin1")
    return {
        "z": float(z_gal.item()),
        "ebv_mw": float(ebv_mw),
        "wave_obs": w_obs,
        "flux_obs": s_obs,
        "mask_obs": m_obs,
        "error_obs": e_obs,
        "sigma_dap": float(sig_dap),
    }


# ═══════════════════════════════════════════════════════════════
#  Preprocessing
# ═══════════════════════════════════════════════════════════════
def preprocess_spectrum(data, wave_temp, pca_all, fit_range=(3600, 7400)):
    """Preprocess an observed spectrum for PCA fitting.

    Steps:
      1. MW extinction correction (CCM).
      2. Shift to rest frame.
      3. Trim to fit_range and mask bad pixels.
      4. Mask emission-line regions.
      5. Normalize flux at 5500 A.
      6. Slice PCA templates to match wavelength range.

    Parameters
    ----------
    data : dict
        Output of load_test_spectrum().
    wave_temp : ndarray
        Template wavelength grid.
    pca_all : ndarray
        Full PCA component array (NEIG x n_wave).
    fit_range : tuple
        Rest-frame wavelength range for fitting.

    Returns
    -------
    dict with keys: wave, flux, flux_raw, error, error_raw, mask,
                    temp_pca, vsys, it1, npix, norm_f5500, sigma_dap.
    """
    z = data["z"]
    mw = ccm_unred(data["wave_obs"], data["ebv_mw"])
    flux = data["flux_obs"] * mw
    err = data["error_obs"] * mw
    wave_rest = data["wave_obs"] / (1.0 + z)
    mask_full = (data["mask_obs"] == 0).astype(float)
    ok = ((wave_rest >= fit_range[0]) & (wave_rest <= fit_range[1])
          & (mask_full == 1) & np.isfinite(err) & (err > 0))
    ok[:5] = False
    ok[-5:] = False
    i0 = np.where(ok)[0][0]
    i1 = np.where(ok)[0][-1] + 1
    npix = i1 - i0
    it1 = np.searchsorted(wave_temp, wave_rest[i0], side="right") - 1
    it1 = max(it1, 0)
    if it1 + npix > len(wave_temp):
        npix = len(wave_temp) - it1
        i1 = i0 + npix
    wf = wave_rest[i0:i1]
    ff = flux[i0:i1]
    ef = err[i0:i1]
    mf = np.ones(npix, dtype=float)
    for lo, hi in _EM_LINES:
        mf[(wf >= lo) & (wf <= hi)] = 0.0
    i55 = np.argmin(np.abs(wf - 5500))
    norm_f5500 = ff[i55]
    ff_raw, ef_raw = ff.copy(), ef.copy()
    ff /= norm_f5500
    ef /= norm_f5500
    tpca = pca_all[:, it1:it1 + npix].T
    vsys = C * np.log(wave_temp[it1] / wf[0])
    return {
        "wave": wf, "flux": ff, "flux_raw": ff_raw,
        "error": ef, "error_raw": ef_raw, "mask": mf,
        "temp_pca": tpca, "vsys": vsys, "it1": it1, "npix": npix,
        "norm_f5500": norm_f5500, "sigma_dap": data["sigma_dap"],
    }


# ═══════════════════════════════════════════════════════════════
#  Core fitting functions
# ═══════════════════════════════════════════════════════════════
def _fit_residual(params, flux, error, mask, temp_pca, wave_fit,
                  vsys, velscale):
    """
    Compute weighted residual: (model - flux) * mask / error.
    Model = linear PCA combo (*) Gauss-convolve (*) dust curve.
    """
    pv = params.valuesdict()
    neig = temp_pca.shape[1]

    # (a) Dust attenuation curve (redden the model)
    curve = calz_unred(wave_fit, -pv["ebv"])

    # (b) PCA linear combination
    coeffs = np.array([pv[f"a{i}"] for i in range(neig)])
    model = np.dot(temp_pca, coeffs)

    # (c) Velocity convolution
    model = gauss_convolve(model, pv["vd"] / velscale,
                           (pv["ve"] + vsys) / velscale)

    # (d) Apply dust
    model *= curve

    return (model - flux) * mask / error


# ═══════════════════════════════════════════════════════════════
#  Mode 1: Calzetti
# ═══════════════════════════════════════════════════════════════
def _m1_residual(params, flux, error, mask, temp_pca, wave_fit,
                 vsys, velscale):
    """Residual function for Mode 1 (Calzetti-only fit)."""
    pv = params.valuesdict()
    ncomp = temp_pca.shape[1]
    curve = calz_unred(wave_fit, -pv["ebv"])
    coeffs = np.array([pv[f"a{i}"] for i in range(ncomp)])
    model = np.dot(temp_pca, coeffs)
    model = gauss_convolve(model, pv["vd"] / velscale,
                           (pv["ve"] + vsys) / velscale)
    return (model * curve - flux) * mask / error


def run_mode1(flux, error, mask, temp_pca, wave_fit, vsys, velscale,
              sigma_dap):
    """Two-stage Mode 1 fit: initial with fixed vd, then free all."""
    x0 = np.linalg.lstsq(temp_pca, flux, rcond=None)[0]
    params = lmfit.Parameters()
    for i in range(FIT_NEIG):
        params.add(f"a{i}", value=x0[i])
    params.add("ve", value=0.0, min=-500, max=500)
    params.add("vd", value=sigma_dap, min=0, max=500)
    params.add("ebv", value=0.1, min=0.0, max=0.5)
    params["vd"].vary = False
    r = lmfit.Minimizer(_m1_residual, params,
                        fcn_args=(flux, error, mask, temp_pca, wave_fit,
                                  vsys, velscale)) \
          .minimize(method="leastsq")
    params = r.params
    params["vd"].vary = True
    return lmfit.Minimizer(_m1_residual, params,
                           fcn_args=(flux, error, mask, temp_pca, wave_fit,
                                     vsys, velscale)) \
                 .minimize(method="leastsq")


# ═══════════════════════════════════════════════════════════════
#  Mode 2: S/L (Smooth / Line) method
# ═══════════════════════════════════════════════════════════════
def mean_filter(flux, wave, wave_win, mask=None):
    """Running-mean filter. Returns DETAILED component (flux - smoothed)."""
    n_wave = len(wave)
    wmin = int(np.ceil(wave[0]))
    wmax = int(np.floor(wave[-1]))
    wave_tmp = np.arange(wmin, wmax + 1, dtype=float)
    flux_tmp = np.interp(wave_tmp, wave, flux)
    if mask is not None:
        mask_tmp = np.interp(wave_tmp, wave, mask)
        mask_tmp[mask_tmp > 0] = 1.0
        if mask_tmp.sum() == 0:
            mask_tmp[0] = 1.0
        u = np.where(mask_tmp == 1.0)[0]
        mask_tmp[:u[0] + 1] = 1.0
        mask_tmp[u[-1]:] = 1.0
    k = int(wave_win / 2)
    n = len(flux_tmp)
    unres = flux_tmp.copy()
    for i in range(k, n - k):
        if mask is not None:
            eff = (mask_tmp[i - k:i + k + 1] == 1.0).sum()
            if eff == 0:
                eff = 1
            unres[i] = (flux_tmp[i - k:i + k + 1]
                        * mask_tmp[i - k:i + k + 1]).sum() / eff
        else:
            unres[i] = np.mean(flux_tmp[i - k:i + k + 1])
            if unres[i] == 0:
                unres[i] = flux_tmp[i]
    return np.interp(wave, wave_tmp, flux_tmp - unres)


def _A_lambda_fcn(params, wave, data, fit=True):
    """Residual for S/L dust-curve fitting (quadratic in 1/lambda)."""
    pv = params.valuesdict()
    x = 10000.0 / wave
    xv = 10000.0 / 5500.0
    if fit:
        return pv["p1"] * x + pv["p2"] * x ** 2 - pv["p1"] * xv - pv["p2"] * xv ** 2 - data
    else:
        return pv["p1"] * x + pv["p2"] * x ** 2 - pv["p1"] * xv - pv["p2"] * xv ** 2


def run_mode2(flux, error, mask, temp_pca, wave_fit, vsys, velscale,
              ve0, vd0, result_m1, pca_full, it1, wave_temp, ebv_sl_n=9):
    """Mode 2 S/L fit: separate smooth and line components with free dust."""
    npx, ncomp = len(flux), temp_pca.shape[1]
    wave_c = wave_fit / (1.0 + ve0 / C)
    snr = (flux * mask).sum() / max((error * mask).sum(), 1e-10)
    wave_win = 200.0 if snr > 10 else 400.0

    # Observed S/L
    flux_s = mean_filter(flux, wave_c, wave_win)
    flux_s1 = mean_filter(flux, wave_c, wave_win, mask=mask)
    flux_L = flux - flux_s
    flux_L1 = flux - flux_s1
    bad = flux_L <= 0
    flux_L[bad] = 1.0
    mask[bad] = 0.0
    bad = flux_L1 <= 0
    flux_L1[bad] = 1.0
    mask[bad] = 0.0
    slr1 = flux_s / flux_L
    slr2 = flux_s1 / flux_L1
    slr_obs = np.where(slr1 > slr2, slr1, slr2)

    # Template S/L
    t_s = np.zeros((ncomp, npx))
    t_L = np.zeros((ncomp, npx))
    t_sLe = np.zeros((npx, ncomp))
    for i in range(ncomp):
        ssp = gauss_convolve(pca_full[i, :].copy(), vd0 / velscale,
                             (ve0 + vsys) / velscale)
        ssp = ssp[it1:it1 + npx]
        t_sLe[:, i] = ssp
        t_s[i, :] = mean_filter(ssp, wave_temp[it1:it1 + npx], wave_win)
        t_L[i, :] = ssp - t_s[i, :]

    mask[flux / (error + 1e-30) < (snr / 4)] = 0.0
    if mask.sum() == 0:
        mask[0] = 1.0

    # Iterative ebv scan
    ebv_m1 = result_m1.params["ebv"].value
    best_redchi = np.inf
    best_wei = None
    best_mask = mask.copy()
    for k in range(ebv_sl_n):
        tmp_ebv = ebv_m1 + (k - (ebv_sl_n - 1) / 2.0) * 0.02
        sLe_d = (t_sLe * calz_unred(wave_c, -tmp_ebv)[:, np.newaxis]
                 / (error[:, np.newaxis] + 1e-30))
        g = (mask == 1)
        if g.sum() < ncomp:
            continue
        wei_k = np.linalg.lstsq(sLe_d[g, :], flux[g] / error[g], rcond=None)[0]
        s_m = np.dot(wei_k, t_s)
        L_m = np.dot(wei_k, t_L)
        slr_m = s_m / np.where(np.abs(L_m) < 1e-30, 1e-30, L_m)
        es = error / np.abs(flux_L1 + 1e-30)
        gs = g & np.isfinite(slr_m) & np.isfinite(slr_obs)
        if gs.sum() < ncomp:
            continue
        rc = np.sum(((slr_m[gs] - slr_obs[gs]) / es[gs]) ** 2) / gs.sum()
        if rc < best_redchi and rc > 0:
            best_redchi = rc
            best_wei = wei_k
            best_mask = mask.copy()

    if best_wei is None:
        return {"p1": 0.0, "p2": 0.0, "ebv": ebv_m1, "chi2r": 0.0,
                "slr_flux": None}

    slr_flux = np.dot(best_wei, t_s) + np.dot(best_wei, t_L)

    # Normalize slr_flux to match observed flux level at 5500A
    m55 = (wave_fit > 5450) & (wave_fit < 5550) & (mask == 1)
    if m55.sum() > 5:
        nm = np.median(slr_flux[m55])
        nf = np.median(flux[m55])
    else:
        nm = np.median(slr_flux[mask == 1])
        nf = np.median(flux[mask == 1])
    slr_flux_norm = slr_flux * (nf / nm)

    # Normalize + Fr (fit dust curve to flux ratio)
    m5500 = (wave_c > 5450) & (wave_c < 5550)
    mm = m5500 & (mask == 1)
    if mm.sum() > m5500.sum() / 4:
        Fo = flux / np.median(flux[mm])
        Fm = slr_flux / np.median(slr_flux[mm])
        ok = (Fm > 0) & (Fo > 0) & (mask == 1)
        Fr = np.zeros_like(flux)
        if ok.sum() > (Fm > 0).sum() / 4:
            Fr[ok] = 2.5 * np.log10(Fm[ok] / Fo[ok])
            params_A = lmfit.Parameters()
            params_A.add("p2", value=0.0)
            params_A.add("delt", value=1e-5, min=1e-7)
            params_A.add("p1",
                         expr=(f"delt-2*p2*10000/{np.max(wave_c)} if p2>=0 "
                               f"else delt-2*p2*10000/{np.min(wave_c)}"))
            params_A.add("ebv",
                         expr="p1*10000/4400+p2*(10000/4400)**2"
                              "-p1*10000/5500-p2*(10000/5500)**2")
            rA = lmfit.Minimizer(_A_lambda_fcn, params_A,
                                 fcn_args=(wave_c[ok], Fr[ok])).minimize(
                                     method="leastsq")
            p1 = rA.params["p1"].value
            p2 = rA.params["p2"].value
            ebv2 = rA.params["ebv"].value
        else:
            p1 = p2 = 0.0
            ebv2 = ebv_m1
    else:
        p1 = p2 = 0.0
        ebv2 = ebv_m1

    return {"p1": p1, "p2": p2, "ebv": ebv2, "chi2r": best_redchi,
            "slr_flux": slr_flux_norm,
            "dust_wave": wave_c[ok], "dust_A": Fr[ok]}


# ═══════════════════════════════════════════════════════════════
#  Unified fit entry
# ═══════════════════════════════════════════════════════════════
def fit_spectrum(prep, pca_full, wave_temp, mode="both"):
    """Run the full SpecFit pipeline on preprocessed data.

    Parameters
    ----------
    prep : dict
        Output of preprocess_spectrum().
    pca_full : ndarray (NEIG, n_wave)
        Full PCA component array (all components, full wavelength).
    wave_temp : ndarray
        Template wavelength grid.
    mode : str
        "both", "m1", or "sl". "both" runs Mode 1 and Mode 2.

    Returns
    -------
    dict with keys: mode1_result, mode2_result, ve, vd, ebv_m1,
                    chi2r_m1, mode1_dust, mode1_model, mode1_residual,
                    mode2_dust, mode2_model, mode2_residual.
    """
    out = {}
    velscale = (10 ** DLOGW - 1) * C
    res1 = run_mode1(prep["flux"], prep["error"], prep["mask"],
                     prep["temp_pca"], prep["wave"], prep["vsys"],
                     velscale, prep["sigma_dap"])
    out["mode1_result"] = res1
    out["ve"] = np.array([res1.params["ve"].value,
                          res1.params["ve"].stderr or 0])
    out["vd"] = np.array([res1.params["vd"].value,
                          res1.params["vd"].stderr or 0])
    out["ebv_m1"] = np.array([res1.params["ebv"].value,
                              res1.params["ebv"].stderr or 0])
    out["chi2r_m1"] = 0.0

    # Build Mode 1 model
    pv = res1.params.valuesdict()
    ncomp = prep["temp_pca"].shape[1]
    coeffs = np.array([pv[f"a{i}"] for i in range(ncomp)])
    m1_intrinsic = np.dot(prep["temp_pca"], coeffs)
    m1_intrinsic = gauss_convolve(m1_intrinsic, pv["vd"] / velscale,
                                  (pv["ve"] + prep["vsys"]) / velscale)
    m1_dust = calz_unred(prep["wave"], -pv["ebv"])
    out["mode1_dust"] = m1_dust
    out["mode1_model"] = m1_intrinsic * m1_dust * prep["norm_f5500"]
    out["mode1_residual"] = prep["flux_raw"] - out["mode1_model"]

    if mode in ("sl", "both"):
        m2 = run_mode2(prep["flux"], prep["error"], prep["mask"].copy(),
                       prep["temp_pca"], prep["wave"], prep["vsys"],
                       velscale, out["ve"][0], out["vd"][0], res1,
                       pca_full, prep["it1"], wave_temp)
        out["mode2_result"] = m2
        # Build S/L model: slr_flux + polynomial dust
        slr = m2.get("slr_flux")
        if slr is not None and (m2.get("p1", 0) != 0 or m2.get("p2", 0) != 0):
            x = 10000.0 / prep["wave"]
            xv = 10000.0 / 5500.0
            A = m2["p1"] * (x - xv) + m2["p2"] * (x ** 2 - xv ** 2)
            m2_dust = 10.0 ** (-0.4 * A)
            out["mode2_dust"] = m2_dust
            out["mode2_model"] = slr * m2_dust * prep["norm_f5500"]
            out["mode2_residual"] = prep["flux_raw"] - out["mode2_model"]
    return out


# ═══════════════════════════════════════════════════════════════
#  Output
# ═══════════════════════════════════════════════════════════════
def save_results(prep, fit, out_dir, prefix="fit_result"):
    """Save fit results to a FITS file.

    Parameters
    ----------
    prep : dict
        Preprocessed data from preprocess_spectrum().
    fit : dict
        Fit results from fit_spectrum().
    out_dir : str
        Output directory path.
    prefix : str
        Filename prefix (extension .fits is added automatically).
    """
    path = os.path.join(out_dir, f"{prefix}.fits")
    m2 = fit.get("mode2_result", {})
    cols = [fits.Column(name=n, format="D", array=[v]) for n, v in zip(
        ["ve", "ve_err", "vd", "vd_err", "ebv_m1", "ebv_m1_err",
         "ebv_m2", "p1", "p2"],
        [fit["ve"][0], fit["ve"][1], fit["vd"][0], fit["vd"][1],
         fit["ebv_m1"][0], fit["ebv_m1"][1],
         m2.get("ebv", 0), m2.get("p1", 0), m2.get("p2", 0)])]
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(prep["wave"], name="WAVE"),
        fits.ImageHDU(prep["flux_raw"], name="FLUX"),
        fits.ImageHDU(prep["error_raw"], name="ERROR"),
        fits.BinTableHDU.from_columns(cols, name="PARAMS"),
    ])
    hdul.writeto(path, overwrite=True)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
#  User-facing API
# ═══════════════════════════════════════════════════════════════
class SpecFitResult:
    """Container for SpecFit results."""
    def __init__(self, fit_dict, prep_dict):
        self._fit = fit_dict
        self._prep = prep_dict
        self.ve = (fit_dict["ve"][0], fit_dict["ve"][1])
        self.vd = (fit_dict["vd"][0], fit_dict["vd"][1])
        self.ebv = (fit_dict["ebv_m1"][0], fit_dict["ebv_m1"][1])
        m2 = fit_dict.get("mode2_result", {})
        self.p1 = m2.get("p1", 0.0)
        self.p2 = m2.get("p2", 0.0)
        self.chi2 = fit_dict.get("chi2r_m1", 0.0)
        self._bestfit = fit_dict.get("mode1_model")
        # Dust curve as callable
        self._dust_wave = prep_dict["wave"]
        m2_dust = fit_dict.get("mode2_dust")
        m1_dust = fit_dict.get("mode1_dust")
        self._dust_curve = m2_dust if m2_dust is not None else m1_dust

        # Noisy dust data points from S/L method (Mode2 only)
        m2 = fit_dict.get("mode2_result", {})
        self._dust_data_wave = m2.get("dust_wave", None)
        self._dust_data_A    = m2.get("dust_A", None)

    @property
    def bestfit(self):
        return self._bestfit

    @property
    def dust_curve(self):
        """Return dust attenuation as callable function of wavelength."""
        if self._dust_curve is None:
            return lambda w: np.ones_like(w)
        from scipy.interpolate import interp1d
        return interp1d(self._dust_wave, self._dust_curve, kind="linear",
                        bounds_error=False, fill_value=1.0)

    @property
    def wave_prep(self):
        """Preprocessed wavelength grid (rest frame, trimmed)."""
        return self._prep["wave"]

    @property
    def flux_prep(self):
        """Preprocessed flux (MW corrected, trimmed)."""
        return self._prep["flux_raw"]

    @property
    def error_prep(self):
        """Preprocessed error (MW corrected, trimmed)."""
        return self._prep["error_raw"]

    @property
    def mask_prep(self):
        """Preprocessed pixel mask (1=good, 0=bad)."""
        return self._prep.get("mask", np.ones_like(self._prep["wave"]))

    def save(self, path, overwrite=True):
        from astropy.io import fits
        import os
        m2 = self._fit.get("mode2_result", {})
        cols = [fits.Column(name=n, format="D", array=[v]) for n, v in zip(
            ["ve", "ve_err", "vd", "vd_err", "ebv_m1", "ebv_m1_err", "ebv_m2", "p1", "p2"],
            [self.ve[0], self.ve[1], self.vd[0], self.vd[1],
             self.ebv[0], self.ebv[1], m2.get("ebv", 0), self.p1, self.p2])]
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(self._prep["wave"], name="WAVE"),
            fits.ImageHDU(self._prep["flux_raw"], name="FLUX"),
            fits.ImageHDU(self._prep["error_raw"], name="ERROR"),
            fits.BinTableHDU.from_columns(cols, name="PARAMS"),
        ])
        if self._bestfit is not None:
            hdul.append(fits.ImageHDU(self._bestfit, name="BESTFIT"))
        if self._dust_curve is not None:
            hdul.append(fits.ImageHDU(self._dust_curve, name="DUST"))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        hdul.writeto(path, overwrite=overwrite)

    def plot_fit(self, path):
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 12,
        })
        import matplotlib.pyplot as plt
        w = self._prep["wave"]
        flux = self._prep["flux_raw"]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(w, flux, 'k-', lw=0.5, label=r'$\mathrm{Observed}$')
        if self._bestfit is not None:
            ax.plot(w, self._bestfit, 'r-', lw=1, label=r'$\mathrm{Best\ fit}$')
        ax.set_xlabel(r'$\lambda\ (\mathrm{\AA})$')
        ax.set_ylabel(r'$F_\lambda$')
        ax.legend(frameon=True, fontsize=11)
        ax.set_title(
            rf'$\mathrm{{SpecFit:\ }} v_e = {self.ve[0]:.0f} \pm {self.ve[1]:.0f}\ \mathrm{{km\,s^{{-1}}}},'
            rf'\ v_d = {self.vd[0]:.0f} \pm {self.vd[1]:.0f}\ \mathrm{{km\,s^{{-1}}}},'
            rf'\ E(B-V) = {self.ebv[0]:.3f} \pm {self.ebv[1]:.3f}$'
        )
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def plot_dust(self, path):
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams.update({
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "serif",
            "font.size": 12,
        })
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 4))

        # Noisy dust data from S/L method (A_λ in magnitudes)
        if self._dust_data_wave is not None and self._dust_data_A is not None:
            ax.plot(self._dust_data_wave, self._dust_data_A, 'r-', lw=0.5, alpha=0.7,
                    label=r'$\mathrm{S/L\ dust\ data}$')

        # Smooth polynomial fit: A_λ = p1*(x - xv) + p2*(x^2 - xv^2)
        w_smooth = np.linspace(3600, 7400, 500)
        x_smooth = 10000.0 / w_smooth
        xv = 10000.0 / 5500.0
        A_smooth = self.p1 * (x_smooth - xv) + self.p2 * (x_smooth**2 - xv**2)
        ax.plot(w_smooth, A_smooth, 'b-', lw=1.5,
                label=r'$\mathrm{Polynomial\ fit}$')

        ax.axhline(0.0, color='k', ls='--', lw=0.8)
        ax.set_xlabel(r'$\lambda\ (\mathrm{\AA})$')
        ax.set_ylabel(r'$A_\lambda - A_V\ \mathrm{(mag)}$')
        ax.set_title(
            rf'$\mathrm{{Dust\ Curve:\ }} p_1 = {self.p1:.4f},\ p_2 = {self.p2:.4f}$'
        )
        ax.legend(frameon=True, fontsize=11)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)


class SpecFit:
    """PCA-based spectral fitting for kinematics and dust.

    Parameters
    ----------
    pca_fits : str
        Path to PCA template FITS file (from compute_pca output).
    """

    def __init__(self, pca_fits):
        self.pca_fits = pca_fits
        self._pca, self._wave_temp, self._velscale = load_pca_templates(pca_fits)

    def fit(self, wave=None, flux=None, error=None, mask=None, z_sys=None,
            mode="mode2", emission_mask=None, neig=None, observed_fits=None):
        """Fit observed spectrum.

        Parameters
        ----------
        wave, flux, error : ndarray
            Observed spectrum arrays (observed frame).
        mask : ndarray, optional
            Boolean or 0/1 mask (1=good pixel).
        z_sys : float
            Systemic redshift.
        mode : str
            "mode2" (default, S/L non-parametric dust) or "mode1" (Calzetti only).
        emission_mask : list, optional
            Custom emission line regions as [(lo, hi), ...].
        neig : int, optional
            Number of PCA components to use (default: FIT_NEIG=10).
        observed_fits : str, optional
            Path to FITS file with WAVE/FLUX/ERROR extensions.

        Returns
        -------
        SpecFitResult
        """
        import numpy as np
        import astropy.io.fits as astro_fits
        import pickle

        global FIT_NEIG
        if neig is not None:
            old_neig = FIT_NEIG
            FIT_NEIG = neig

        # Load from FITS if provided
        if observed_fits is not None:
            with astro_fits.open(observed_fits) as h:
                wave = h["WAVE"].data
                flux = h["FLUX"].data
                error = h["ERROR"].data
                if "MASK" in h:
                    mask = h["MASK"].data
                if "REDSHIFT" in h[0].header:
                    z_sys = h[0].header["REDSHIFT"]

        if wave is None or flux is None or error is None or z_sys is None:
            raise ValueError("wave, flux, error, z_sys are required")

        # Build data dict compatible with existing preprocess_spectrum
        if mask is None:
            mask = np.ones_like(flux, dtype=float)
        else:
            mask = np.asarray(mask, dtype=float)

        data = {
            "z": float(z_sys),
            "ebv_mw": 0.0,
            "wave_obs": np.asarray(wave, dtype=float),
            "flux_obs": np.asarray(flux, dtype=float),
            "mask_obs": mask,
            "error_obs": np.asarray(error, dtype=float),
            "sigma_dap": 100.0,
        }

        # Override emission mask if provided
        global _EM_LINES
        old_em = list(_EM_LINES)
        if emission_mask is not None:
            _EM_LINES[:] = emission_mask

        # Run fitting (translate mode names)
        _mode_map = {"mode1": "m1", "mode2": "sl", "both": "both", "m1": "m1", "sl": "sl"}
        fit_mode = _mode_map.get(mode, mode)
        prep = preprocess_spectrum(data, self._wave_temp, self._pca)
        fit = fit_spectrum(prep, self._pca, self._wave_temp, mode=fit_mode)

        # Restore globals
        _EM_LINES[:] = old_em
        if neig is not None:
            FIT_NEIG = old_neig

        return SpecFitResult(fit, prep)
