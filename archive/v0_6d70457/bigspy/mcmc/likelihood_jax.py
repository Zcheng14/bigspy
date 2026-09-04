"""JAX-accelerated likelihood for bigspy MCMC.

Provides `compute_chi2_batch_jax` — a pure, JIT-compiled function that
replaces `Likelihood.call_batch`.  All NumPy → JAX, Python loops → vmap.

Usage:
    from bigspy.mcmc.likelihood_jax import JAXLikelihood
    like = JAXLikelihood(ssp, ...)         # pre-loads SSP, builds conv matrix
    chi2 = like.call_batch(logZ_arr, sfh_params)  # JIT-compiled batch eval
"""

import numpy as np
import jax.numpy as jnp
from jax import jit, vmap


# ═══════════════════════════════════════════════════════════════════
#  Helper: build convolution matrix (run once in __init__)
# ═══════════════════════════════════════════════════════════════════

def _build_conv_matrix(n_pix, sigma_pix):
    """Pre-compute Gaussian convolution matrix (NumPy, run once)."""
    if sigma_pix <= 0:
        return np.eye(n_pix)
    khalf = round(4 * sigma_pix + 3)
    xx = np.arange(khalf * 2 + 1) - khalf
    kernel = np.exp(-xx ** 2 / (2 * sigma_pix ** 2))
    kernel /= kernel.sum()
    K = np.zeros((n_pix, n_pix))
    offset = khalf
    for i in range(len(kernel)):
        for j in range(n_pix):
            src = j + i - offset
            if 0 <= src < n_pix:
                K[j, src] += kernel[i]
    return K


# ═══════════════════════════════════════════════════════════════════
#  SFH weights computation (NumPy, calls SFH.evaluate_batch)
# ═══════════════════════════════════════════════════════════════════

def _compute_sfh_weights(sfh_class, sfh_params_2d, time_grid, dt):
    """Compute normalized SFH weights: (N, n_age)."""
    sfr = sfh_class.evaluate_batch(time_grid, sfh_params_2d)  # (N, n_age)
    w = sfr * dt[None, :]   # (N, n_age)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum == 0, 1.0, w_sum)
    return w / w_sum


# ═══════════════════════════════════════════════════════════════════
#  Core: JIT-compiled batch chi2 computation
# ═══════════════════════════════════════════════════════════════════

@jit
def compute_chi2_batch_jax(
    logZ_arr, sfh_weights, spec_3d, metal_log_grid,
    conv_matrix, dust_curve, ssp_wave,
    obs_wave, obs_flux, obs_err, obs_mask,
    nr_indices,  # tuple of ints for 5500 normalization region
):
    """JIT-compiled batch chi-squared computation.

    All inputs are JAX arrays.  No Python loops — fully traceable by JAX.
    """
    N = logZ_arr.shape[0]

    # ── 1. Metal interpolation ─────────────────────────────────
    idx = jnp.clip(
        jnp.searchsorted(metal_log_grid, logZ_arr),
        1, len(metal_log_grid) - 1,
    )  # (N,)
    f = (logZ_arr - metal_log_grid[idx - 1]) / (
        metal_log_grid[idx] - metal_log_grid[idx - 1]
    )  # (N,)
    f = jnp.clip(f, 0.0, 1.0)  # match NumPy edge-case handling

    # Batch dot: precompute CSP for ALL 6 metals, then interpolate.
    # This avoids materializing spec_3d[idx] which is (N, 196, 3129) — huge.
    all_csp = jnp.einsum("na,maw->nmw", sfh_weights, spec_3d)  # (N, 6, n_wave)
    N_range = jnp.arange(N)
    csp_lo = all_csp[N_range, idx - 1, :]  # (N, n_wave)
    csp_hi = all_csp[N_range, idx, :]      # (N, n_wave)
    csp = (1.0 - f[:, None]) * csp_lo + f[:, None] * csp_hi  # (N, n_wave_ssp)

    # ── 2. Velocity broadening ─────────────────────────────────
    csp = jnp.dot(csp, conv_matrix.T)  # (N, n_wave_ssp)

    # ── 3. Normalize at 5500 ────────────────────────────────────
    csp_nr = csp[:, nr_indices]  # (N, n_nr) — integer indexing is JIT-safe
    norms = jnp.median(csp_nr, axis=1)  # (N,)
    norms = jnp.where(norms == 0.0, 1.0, norms)
    csp = csp / norms[:, None]

    # ── 4. Dust attenuation ─────────────────────────────────────
    csp = csp * dust_curve[None, :]

    # ── 5. Interpolate to observed grid ─────────────────────────
    # jnp.interp is 1D; vmap over the batch dimension
    interp_fn = lambda single_csp: jnp.interp(obs_wave, ssp_wave, single_csp)
    model = vmap(interp_fn)(csp)  # (N, n_obs)

    # ── 6. chi-squared ──────────────────────────────────────────
    residuals2 = (model - obs_flux[None, :]) ** 2 / (obs_err[None, :] ** 2)  # (N, n_obs)
    masked_res2 = jnp.where(obs_mask[None, :], residuals2, 0.0)
    chi2 = jnp.sum(masked_res2, axis=1)  # (N,)
    return chi2


# ═══════════════════════════════════════════════════════════════════
#  JAXLikelihood class — NumPy-compatible wrapper
# ═══════════════════════════════════════════════════════════════════

class JAXLikelihood:
    """JAX-accelerated likelihood, API-compatible with `Likelihood`.

    Parameters
    ----------
    ssp : SSPLibrary
        Loaded SSP library.
    ow, oflux, oerr, omask : ndarray
        Observed spectrum arrays (rest-frame, trimmed).
    ve, vd : float
        Velocity shift / dispersion from SpecFit (km/s).
    dust : DustAttenuation
        Dust curve from SpecFit mode2.
    nr : tuple
        5500 normalization range.
    velscale : float
        Velocity scale per pixel (km/s). Default from constants.
    """

    def __init__(self, ssp, ow, oflux, oerr, omask, ve, vd, dust,
                 nr=(5450, 5550), velscale=None):
        from ..constants import DLOGW, C_LIGHT

        if velscale is None:
            velscale = (10 ** DLOGW - 1) * C_LIGHT

        # ── Pre-compute everything that doesn't depend on parameters ──
        sigma_pix = vd / velscale if vd > 0 else 0.0
        self._conv_matrix_jax = jnp.asarray(
            np.asarray(_build_conv_matrix(len(ssp.wave), sigma_pix), dtype=np.float64)
        )
        self._dust_curve_jax = jnp.asarray(np.asarray(dust._curve, dtype=np.float64))
        self._spec_3d_jax = jnp.asarray(np.asarray(ssp._spec, dtype=np.float64))
        self._ssp_wave_jax = jnp.asarray(np.asarray(ssp.wave, dtype=np.float64))
        self._metal_log_grid_jax = jnp.log10(
            jnp.asarray(np.asarray(ssp.metal, dtype=np.float64) / 0.02)
        )
        self._time_grid = ssp.time
        self._dt = ssp.dt

        # ── Normalize observed data ──
        ow_arr = np.asarray(ow, float)
        of_arr = np.asarray(oflux, float)
        oe_arr = np.asarray(oerr, float)
        om_arr = np.asarray(omask, bool)

        nr_mask = (ow_arr >= nr[0]) & (ow_arr <= nr[1]) & om_arr
        n = float(np.median(of_arr[nr_mask]) if nr_mask.sum() > 5
                   else np.median(of_arr[om_arr]))

        self._obs_wave_jax = jnp.asarray(np.asarray(ow_arr, dtype=np.float64))
        self._obs_flux_jax = jnp.asarray(np.asarray(of_arr / n, dtype=np.float64))
        self._obs_err_jax  = jnp.asarray(np.asarray(oe_arr / n, dtype=np.float64))
        self._obs_mask_jax = jnp.asarray(np.asarray(om_arr, dtype=bool))
        self.ndof = om_arr.sum() - 3

        # Pre-compute 5500 normalization mask indices (JAX needs concrete indices)
        nr_mask = (ssp.wave >= nr[0]) & (ssp.wave <= nr[1])
        self._nr_indices = tuple(np.where(nr_mask)[0].tolist())  # for integer indexing

    def call_batch(self, logZsun_arr, sfh_class, sfh_params_2d):
        """Vectorized chi-squared — NumPy in, NumPy out.

        Parameters
        ----------
        logZsun_arr : ndarray (N,)
        sfh_class : type
            SFH model class.
        sfh_params_2d : ndarray (N, n_sfh_params)

        Returns
        -------
        chi2 : ndarray (N,)
        """
        # Compute SFH weights on CPU (SFH.evaluate is not JAX)
        sfh_weights = _compute_sfh_weights(
            sfh_class, sfh_params_2d, self._time_grid, self._dt
        )

        # JIT-compiled core
        chi2 = compute_chi2_batch_jax(
            jnp.asarray(logZsun_arr),
            jnp.asarray(sfh_weights),
            self._spec_3d_jax,
            self._metal_log_grid_jax,
            self._conv_matrix_jax,
            self._dust_curve_jax,
            self._ssp_wave_jax,
            self._obs_wave_jax,
            self._obs_flux_jax,
            self._obs_err_jax,
            self._obs_mask_jax,
            self._nr_indices,
        )
        return np.asarray(chi2)
