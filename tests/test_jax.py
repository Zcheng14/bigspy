"""Tests for the JAX likelihood: parity with a NumPy reference, OOB handling.

The NumPy reference chi-squared is built from ``ModelComponents`` (the same
model chain used for plotting), giving an independent cross-check of the JAX
implementation.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # noqa: F841
import jax.numpy as jnp  # noqa: E402

import _synth  # noqa: E402
from bigspy.mcmc.ssp import SSPLibrary  # noqa: E402
from bigspy.mcmc.csp import CSPBuilder  # noqa: E402
from bigspy.mcmc.dust import DustAttenuation  # noqa: E402
from bigspy.mcmc.sfh import DelayedExponentialSFH  # noqa: E402
from bigspy.mcmc.model import ModelComponents  # noqa: E402
from bigspy.mcmc.likelihood_jax import JAXLikelihood, _build_conv_kernel  # noqa: E402
from bigspy.mcmc.kinematics import _build_convolution_matrix  # noqa: E402


def _make_pair(ssp, obs_wave, obs_flux, obs_err, obs_mask, vd=100.0):
    dust = DustAttenuation.from_mode2(ssp.wave, 0.05, -0.003)
    jl = JAXLikelihood(ssp, obs_wave, obs_flux, obs_err, obs_mask, 0.0, vd, dust)
    mc = ModelComponents(ssp, obs_wave, obs_flux, obs_err, obs_mask, 0.0, vd, dust)
    return jl, mc


def _numpy_loglike(mc, logZ, sfh):
    model = mc.build_model(logZ, sfh)
    r = (model - mc.obs_flux) / mc.obs_error
    return -0.5 * np.sum(r[mc.obs_mask] ** 2)


@pytest.fixture(scope="module")
def pair(synth_ssp_file):
    """(JAXLikelihood, ModelComponents) on an identical synthetic observation."""
    ssp = SSPLibrary(synth_ssp_file)
    sfh0 = DelayedExponentialSFH(t0=1.0, tau=3.0)
    model = CSPBuilder(ssp).build(_synth.SSP_LOGZ_GRID[1], sfh0)
    err = 0.02 * np.abs(model) + 1e-6
    mask = np.ones(ssp.n_wave, dtype=bool)
    return _make_pair(ssp, ssp.wave, model, err, mask)


class TestBuildConvKernel:
    def test_matches_dense_matrix_x0_zero(self):
        n, sigma = 150, 2.5
        kernel = _build_conv_kernel(sigma)
        dense = _build_convolution_matrix(n, sigma, 0.0)
        x = np.random.RandomState(0).randn(n)
        np.testing.assert_allclose(np.convolve(x, kernel, mode="same"),
                                   dense @ x, atol=1e-12)


class TestJAXParity:
    def test_loglike_matches_numpy(self, pair):
        _, mc = pair
        for logZ, t0, tau in [(-1.0, 1.0, 3.0), (-0.3, 5.0, 2.0), (0.0, 0.5, 6.0)]:
            ssp = mc.ssp
            jl, _ = _make_pair(ssp, ssp.wave, mc.obs_flux, mc.obs_error,
                               mc.obs_mask)
            jll = float(jl.loglike_batch(jnp.array([logZ]), jnp.array([[t0, tau]]),
                                         DelayedExponentialSFH)[0])
            nll = _numpy_loglike(mc, logZ, DelayedExponentialSFH(t0, tau))
            np.testing.assert_allclose(jll, nll, rtol=3e-3)

    def test_interp_oob_matches_numpy(self, synth_ssp_file):
        """Observed grid extending past the SSP range: OOB targets -> 0."""
        ssp = SSPLibrary(synth_ssp_file)
        obs_wave = np.concatenate([[3800.0], ssp.wave[::2] + 2.3, [7200.0]])
        rng = np.random.RandomState(6)
        obs_flux = rng.uniform(0.5, 2.0, len(obs_wave))
        err = np.full(len(obs_wave), 0.02)
        mask = np.ones(len(obs_wave), dtype=bool)
        jl, mc = _make_pair(ssp, obs_wave, obs_flux, err, mask, vd=50.0)

        rng = np.random.RandomState(0)
        logZ = rng.uniform(-2.0, 0.2, 8)
        t0 = rng.uniform(0.5, 10.0, 8)
        tau = rng.uniform(0.5, 8.0, 8)
        for lz, a, b in zip(logZ, t0, tau):
            jll = float(jl.loglike_batch(jnp.array([lz]), jnp.array([[a, b]]),
                                         DelayedExponentialSFH)[0])
            nll = _numpy_loglike(mc, lz, DelayedExponentialSFH(a, b))
            np.testing.assert_allclose(jll, nll, rtol=3e-3)


class TestEdgeMetallicity:
    def test_clamp_parity(self, pair):
        _, mc = pair
        ssp = mc.ssp
        jl, _ = _make_pair(ssp, ssp.wave, mc.obs_flux, mc.obs_error, mc.obs_mask)
        for logZ in (-5.0, 1.0):   # below / above the metallicity grid
            jll = float(jl.loglike_batch(jnp.array([logZ]), jnp.array([[2.0, 2.0]]),
                                         DelayedExponentialSFH)[0])
            nll = _numpy_loglike(mc, logZ, DelayedExponentialSFH(2.0, 2.0))
            np.testing.assert_allclose(jll, nll, rtol=3e-3)
