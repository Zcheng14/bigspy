"""JAX likelihood for bigspy MCMC.

Provides the JIT-compiled batch chi-squared core
(:func:`compute_chi2_batch_jax`) and the :class:`JAXLikelihood` wrapper used by
the NSS sampler.  Sampling is JAX-only; there is no NumPy chi-squared
implementation in the library.

The likelihood maps physical parameters (log metallicity + SFH parameters) to a
log-likelihood.  SFH weights are evaluated in JAX via
``SFHClass.evaluate_batch_jax`` so the entire computation can be traced by
blackjax.

The interpolation from the SSP grid to the observed grid uses a plan of gather
indices / weights precomputed once in ``__init__`` (both grids are fixed),
matching the NumPy ``_LinearInterpPlan`` boundary behaviour (strictly
out-of-range -> 0).
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap

from ..constants import DLOGW_VEL, NR_RANGE
from ..utils import median_in_window


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _build_conv_kernel(sigma_pix):
    """Pre-compute a normalised Gaussian convolution kernel (1D)."""
    if sigma_pix <= 0:
        return np.array([1.0])
    khalf = round(4 * sigma_pix + 3)
    xx = np.arange(khalf * 2 + 1) - khalf
    kernel = np.exp(-xx ** 2 / (2 * sigma_pix ** 2))
    kernel /= kernel.sum()
    return kernel


def _build_interp_plan(x_src, x_new):
    """Pre-compute the linear-interpolation gather plan for fixed grids.

    Reproduces the NumPy ``_LinearInterpPlan`` for a fixed ``(x_src, x_new)``
    pair: same index layout, same formula, and strictly out-of-range targets
    are zeroed.

    Returns
    -------
    lo : ndarray (n_new,), int
        Lower source index for each target.
    dx : ndarray (n_new,)
        ``x_src[lo + 1] - x_src[lo]``.
    dxo : ndarray (n_new,)
        ``x_new - x_src[lo]``.
    oob : ndarray (n_new,), bool
        True where ``x_new`` lies strictly outside the source range.
    """
    x = np.asarray(x_src, dtype=float)
    xo = np.asarray(x_new, dtype=float)
    idx = np.clip(np.searchsorted(x, xo), 1, len(x) - 1)
    lo = (idx - 1).astype(np.int32)
    dx = (x[idx] - x[idx - 1]).astype(np.float64)
    dxo = (xo - x[idx - 1]).astype(np.float64)
    oob = (xo < x[0]) | (xo > x[-1])
    return lo, dx, dxo, oob


# ═══════════════════════════════════════════════════════════════════
#  Core: JIT-compiled batch chi-squared
# ═══════════════════════════════════════════════════════════════════

@jit
def compute_chi2_batch_jax(
    logZ_arr, sfh_weights, spec_3d, metal_log_grid,
    conv_kernel, dust_curve,
    interp_lo, interp_dx, interp_dxo, interp_oob,
    obs_flux, obs_err, obs_mask,
    nr_indices,
):
    """JIT-compiled batch chi-squared.

    All inputs are JAX arrays; there are no Python loops.  ``sfh_weights`` are
    the normalised SFH weights, shape ``(N, n_age)``.
    """
    N = logZ_arr.shape[0]

    # ── 1. Metal interpolation ─────────────────────────────────────
    idx = jnp.clip(
        jnp.searchsorted(metal_log_grid, logZ_arr),
        1, len(metal_log_grid) - 1,
    )  # (N,)
    f = (logZ_arr - metal_log_grid[idx - 1]) / (
        metal_log_grid[idx] - metal_log_grid[idx - 1]
    )  # (N,)
    f = jnp.clip(f, 0.0, 1.0)

    # Compute the CSP for all metals, then interpolate (avoids materialising
    # spec_3d[idx] which would be (N, n_age, n_wave)).
    all_csp = jnp.einsum("na,maw->nmw", sfh_weights, spec_3d)  # (N, M, n_wave)
    N_range = jnp.arange(N)
    csp_lo = all_csp[N_range, idx - 1, :]
    csp_hi = all_csp[N_range, idx, :]
    csp = (1.0 - f[:, None]) * csp_lo + f[:, None] * csp_hi

    # ── 2. Velocity broadening ─────────────────────────────────────
    csp = vmap(lambda s: jnp.convolve(s, conv_kernel, mode="same"))(csp)

    # ── 3. Normalize at 5500 ────────────────────────────────────────
    csp_nr = csp[:, nr_indices]
    norms = jnp.median(csp_nr, axis=1)
    norms = jnp.where(norms == 0.0, 1.0, norms)
    csp = csp / norms[:, None]

    # ── 4. Dust attenuation ─────────────────────────────────────────
    csp = csp * dust_curve[None, :]

    # ── 5. Interpolate to the observed grid (precomputed gather + lerp) ──
    c_lo = csp[:, interp_lo]
    c_hi = csp[:, interp_lo + 1]
    slope = (c_hi - c_lo) / interp_dx[None, :]
    model = slope * interp_dxo[None, :] + c_lo
    model = jnp.where(interp_oob[None, :], 0.0, model)

    # ── 6. chi-squared ──────────────────────────────────────────────
    residuals2 = (model - obs_flux[None, :]) ** 2 / (obs_err[None, :] ** 2)
    masked_res2 = jnp.where(obs_mask[None, :], residuals2, 0.0)
    chi2 = jnp.sum(masked_res2, axis=1)
    chi2 = jnp.where(jnp.isfinite(chi2), chi2, 1e30)
    return chi2


# ═══════════════════════════════════════════════════════════════════
#  JAXLikelihood
# ═══════════════════════════════════════════════════════════════════

class JAXLikelihood:
    """JAX likelihood used by the NSS sampler.

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
        5500 normalization range.
    velscale : float
        Velocity scale per pixel (km/s); default from constants.
    """

    def __init__(self, ssp, ow, oflux, oerr, omask, ve, vd, dust,
                 nr=NR_RANGE, velscale=None):
        if velscale is None:
            velscale = DLOGW_VEL

        # ── Constants that do not depend on parameters ─────────────
        sigma_pix = vd / velscale if vd > 0 else 0.0
        self._conv_kernel_jax = jnp.asarray(_build_conv_kernel(sigma_pix))
        self._dust_curve_jax = jnp.asarray(np.asarray(dust._curve, dtype=np.float64))
        self._spec_3d_jax = jnp.asarray(np.asarray(ssp._spec, dtype=np.float64))
        self._metal_log_grid_jax = jnp.log10(
            jnp.asarray(np.asarray(ssp.metal, dtype=np.float64) / 0.02)
        )
        self._time_grid_jax = jnp.asarray(np.asarray(ssp.time, dtype=np.float64))
        self._dt_jax = jnp.asarray(np.asarray(ssp.dt, dtype=np.float64))

        # ── Normalize observed data at 5500 ────────────────────────
        ow_arr = np.asarray(ow, float)
        of_arr = np.asarray(oflux, float)
        oe_arr = np.asarray(oerr, float)
        om_arr = np.asarray(omask, bool)

        n = median_in_window(ow_arr, of_arr, om_arr, nr)

        self._obs_flux_jax = jnp.asarray(np.asarray(of_arr / n, dtype=np.float64))
        self._obs_err_jax = jnp.asarray(np.asarray(oe_arr / n, dtype=np.float64))
        self._obs_mask_jax = jnp.asarray(np.asarray(om_arr, dtype=bool))
        self.ndof = om_arr.sum() - 3

        # ── Fixed SSP -> observed interpolation plan ───────────────
        lo, dx, dxo, oob = _build_interp_plan(ssp.wave, ow_arr)
        self._interp_lo_jax = jnp.asarray(lo)
        self._interp_dx_jax = jnp.asarray(dx)
        self._interp_dxo_jax = jnp.asarray(dxo)
        self._interp_oob_jax = jnp.asarray(oob)

        # ── 5500 normalization mask indices ────────────────────────
        nr_mask = (ssp.wave >= nr[0]) & (ssp.wave <= nr[1])
        self._nr_indices = tuple(np.where(nr_mask)[0].tolist())

    def loglike_batch(self, logZ_arr, sfh_params_2d, sfh_class):
        """JAX log-likelihood for a batch of parameter sets.

        Parameters
        ----------
        logZ_arr : jax.Array, shape (N,)
        sfh_params_2d : jax.Array, shape (N, n_sfh_params)
        sfh_class : type
            SFH model class implementing ``evaluate_batch_jax``.

        Returns
        -------
        loglike : jax.Array, shape (N,)
        """
        sfr = sfh_class.evaluate_batch_jax(self._time_grid_jax, sfh_params_2d)
        w = sfr * self._dt_jax[None, :]
        w_sum = jnp.sum(w, axis=1, keepdims=True)
        w_sum = jnp.where(w_sum == 0.0, 1.0, w_sum)
        w = w / w_sum

        chi2 = compute_chi2_batch_jax(
            logZ_arr,
            w,
            self._spec_3d_jax,
            self._metal_log_grid_jax,
            self._conv_kernel_jax,
            self._dust_curve_jax,
            self._interp_lo_jax,
            self._interp_dx_jax,
            self._interp_dxo_jax,
            self._interp_oob_jax,
            self._obs_flux_jax,
            self._obs_err_jax,
            self._obs_mask_jax,
            self._nr_indices,
        )
        return -0.5 * chi2
