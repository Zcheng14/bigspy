"""Shared synthetic-data builders and stubs for bigspy tests (not collected).

All builders are deterministic (fixed seeds / closed-form arrays) so expected
values can be computed analytically in the tests.  No LFS reference data is
needed anywhere in this module.

Synthetic SSP design:
    spec[mi, ai, :] = (mi+1) * (ai+1) * wave / 5500
The spectra are linear in BOTH the metallicity index and the age index, so
CSPBuilder's metallicity interpolation and SFH-weighted age summation have
exact closed-form expectations (verified to ~1e-15).
"""

import numpy as np
from astropy.io import fits

from bigspy.mcmc.sfh import SFHBase

# ── Synthetic SSP library constants ────────────────────────────────
SSP_WAVE = np.linspace(4000.0, 7000.0, 761)      # 5 A pixels; 25 px in 5450-5550
SSP_TIME = np.linspace(0.5, 13.8, 6)             # Gyr
SSP_DT = np.gradient(SSP_TIME)                   # all positive
SSP_METAL = np.array([0.0004, 0.004, 0.02])      # increasing Z
SSP_LOGZ_GRID = np.log10(SSP_METAL / 0.02)       # what CSPBuilder interpolates in


def make_ssp_fits(path):
    """Write a synthetic SSP library FITS; return the written arrays."""
    n_metal, n_age, n_wave = len(SSP_METAL), len(SSP_TIME), len(SSP_WAVE)
    spec = (
        (np.arange(n_metal) + 1)[:, None, None]
        * (np.arange(n_age) + 1)[None, :, None]
        * SSP_WAVE[None, None, :] / 5500.0
    )
    mass = np.ones((n_metal, n_age))
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(SSP_WAVE, name="WAVE"),
        fits.ImageHDU(spec, name="SPEC"),
        fits.ImageHDU(mass, name="MASS"),
        fits.ImageHDU(SSP_TIME, name="TIME"),
        fits.ImageHDU(SSP_DT, name="DT"),
        fits.ImageHDU(SSP_METAL, name="METAL"),
    ])
    hdul.writeto(path, overwrite=True)
    return {"wave": SSP_WAVE, "time": SSP_TIME, "dt": SSP_DT,
            "metal": SSP_METAL, "mass": mass, "spec": spec}


def metal_value(logM):
    """Expected 'metallicity factor' (mi+1, linearly interpolated) at log(Z/Zsun)."""
    if logM <= SSP_LOGZ_GRID[0]:
        return 1.0
    if logM >= SSP_LOGZ_GRID[-1]:
        return float(len(SSP_METAL))
    i = np.searchsorted(SSP_LOGZ_GRID, logM)
    f = (logM - SSP_LOGZ_GRID[i - 1]) / (SSP_LOGZ_GRID[i] - SSP_LOGZ_GRID[i - 1])
    return (1 - f) * i + f * (i + 1)


def sfh_weights(sfh):
    """Independent reimplementation of CSPBuilder._conv weights."""
    w = sfh.evaluate(SSP_TIME) * SSP_DT
    return w / w.sum()


def expected_csp(logM, sfh=None):
    """Closed-form expectation of CSPBuilder.build on the synthetic library."""
    w = sfh_weights(sfh) if sfh is not None else SSP_DT / SSP_DT.sum()
    age_term = np.dot(w, (np.arange(len(SSP_TIME)) + 1)[:, None]
                      * SSP_WAVE[None, :] / 5500.0)
    return metal_value(logM) * age_term


# ── Synthetic PCA templates ────────────────────────────────────────
N_WAVE_LOG = 3310
PCA_WAVE_LOG = 3500.0 * 10 ** (np.arange(N_WAVE_LOG) * 1e-4)  # 3500-7498 A, DLOGW=1e-4
SYN_Z = 0.01


def pca_rows(n_comp=20):
    """Scaled Legendre basis on the log-lambda grid (deterministic)."""
    u = np.linspace(-1.0, 1.0, N_WAVE_LOG)
    rows = np.zeros((n_comp, N_WAVE_LOG))
    for k in range(n_comp):
        coeff = np.zeros(k + 1)
        coeff[k] = 1.0
        rows[k] = 0.1 * np.polynomial.legendre.legval(u, coeff) / (k + 1)
    return rows


def make_pca_fits(path, n_comp=20):
    """Write a synthetic PCA-template FITS (pca_log / wave_log HDUs)."""
    rows = pca_rows(n_comp)
    hdul = fits.HDUList([
        fits.PrimaryHDU(),
        fits.ImageHDU(rows, name="pca_log"),
        fits.ImageHDU(PCA_WAVE_LOG, name="wave_log"),
    ])
    hdul.writeto(path, overwrite=True)
    return rows


def make_synthetic_obs(noise=0.002, seed=42):
    """Build a synthetic observed spectrum + matching data dict for preprocess.

    Truth: flat-ish Legendre-PCA combo (first 10 comps), no dust, no
    broadening, z=0.01, all pixels good (mask 0 = good convention).
    Returns (data_dict, model_rest_frame_on_the_log_grid).
    """
    rng = np.random.RandomState(seed)
    rows = pca_rows()
    coeffs = 1.0 / (np.arange(10) + 1.0)
    model = coeffs @ rows[:10]
    i55 = int(np.argmin(np.abs(PCA_WAVE_LOG - 5500.0)))
    model = model / model[i55]
    data = {
        "z": SYN_Z,
        "ebv_mw": 0.0,
        "wave_obs": PCA_WAVE_LOG * (1.0 + SYN_Z),
        "flux_obs": model + rng.normal(0.0, noise, N_WAVE_LOG),
        "mask_obs": np.zeros(N_WAVE_LOG),          # 0 = good
        "error_obs": np.full(N_WAVE_LOG, noise),
        "sigma_dap": 100.0,
    }
    return data, model


# ── Stub objects ───────────────────────────────────────────────────
class ConstantSFH(SFHBase):
    """SFH with constant SFR (tests SFHBase subclass contract + weights)."""

    n_params = 0
    param_names = []

    def __init__(self, value=1.0):
        self.value = float(value)

    def evaluate(self, timegrid):
        return np.full(len(timegrid), self.value, dtype=float)


class StubLikelihood:
    """Analytic chi2 with a known optimum, for UltraNestSampler unit tests.

    chi2(logZ, sfh_params) = ((logZ - logZ_true)/0.05)^2
                           + ((sfh_params[:, 0] - t0_true)/0.2)^2
    Records the last call for parameter-order assertions.
    """

    def __init__(self, t0_true=5.0, logZ_true=-1.0):
        self.t0_true, self.logZ_true = t0_true, logZ_true
        self.last_logZ = None
        self.last_sfh_params = None

    def call_batch(self, logZsun_arr, sfh_class, sfh_params_2d):
        logZsun_arr = np.asarray(logZsun_arr, dtype=float)
        sfh_params_2d = np.atleast_2d(np.asarray(sfh_params_2d, dtype=float))
        self.last_logZ = logZsun_arr.copy()
        self.last_sfh_params = sfh_params_2d.copy()
        chi2 = ((logZsun_arr - self.logZ_true) / 0.05) ** 2
        if sfh_params_2d.shape[1] >= 1:
            chi2 = chi2 + ((sfh_params_2d[:, 0] - self.t0_true) / 0.2) ** 2
        return chi2


class StubSampler:
    """Minimal UltraNestSampler stand-in for MCMCResult tests."""

    def __init__(self, like=None, n_samples=200, seed=0):
        from bigspy.mcmc.sfh import DelayedExponentialSFH
        rng = np.random.RandomState(seed)
        self.like = like if like is not None else StubLikelihood()
        self.param_names = ["t0", "tau", "logZsun"]
        self.sfh_class = DelayedExponentialSFH
        self.result = {
            "maximum_likelihood": {"point": np.array([5.0, 3.0, -1.0])},
            "samples": np.column_stack([
                rng.normal(5.0, 1.0, n_samples),
                rng.normal(3.0, 0.5, n_samples),
                rng.normal(-1.0, 0.3, n_samples),
            ]),
            "logz": -4.5,
        }

    def get_bestfit(self):
        return self.result["maximum_likelihood"]["point"]

    def get_posterior(self):
        return self.result["samples"]


class StubSpecFitResult:
    """Duck-typed SpecFitResult stand-in for MCMCFitter tests."""

    def __init__(self, wave, flux=None, error=None, mask=None,
                 ve=(10.0, 2.0), vd=(120.0, 8.0), p1=0.05, p2=-0.003):
        rng = np.random.RandomState(7)
        self._wave = np.asarray(wave, dtype=float)
        if flux is None:
            flux = (1.0 + 0.1 * np.sin(self._wave / 900.0)
                    + 0.02 * rng.randn(len(self._wave)))
        self._flux = np.asarray(flux, dtype=float)
        self._error = (np.full(len(self._wave), 0.05) if error is None
                       else np.asarray(error, dtype=float))
        self._mask = (np.ones(len(self._wave)) if mask is None
                      else np.asarray(mask, dtype=float))
        self.ve, self.vd = ve, vd
        self.p1, self.p2 = p1, p2

    @property
    def wave_prep(self):
        return self._wave

    @property
    def flux_prep(self):
        return self._flux

    @property
    def error_prep(self):
        return self._error

    @property
    def mask_prep(self):
        return self._mask
