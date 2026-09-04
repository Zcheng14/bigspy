"""Tests for bigspy.mcmc.likelihood_jax parity with the NumPy backend.

Skipped entirely when jax is not installed.  Note: without
``jax_enable_x64`` the JAX pipeline runs in float32, so tolerances are
looser than the NumPy-vs-NumPy comparisons elsewhere.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # noqa: F841  (module-level skip)

import _synth  # noqa: E402
from bigspy.mcmc.ssp import SSPLibrary  # noqa: E402
from bigspy.mcmc.csp import CSPBuilder  # noqa: E402
from bigspy.mcmc.dust import DustAttenuation  # noqa: E402
from bigspy.mcmc.sfh import DelayedExponentialSFH  # noqa: E402
from bigspy.mcmc.likelihood import Likelihood  # noqa: E402
from bigspy.mcmc.likelihood_jax import JAXLikelihood, _build_conv_matrix  # noqa: E402
from bigspy.mcmc.kinematics import _build_convolution_matrix  # noqa: E402


@pytest.fixture(scope="module")
def pair(synth_ssp_file):
    """(NumPy Likelihood, JAXLikelihood) on identical synthetic observations."""
    ssp = SSPLibrary(synth_ssp_file)
    sfh0 = DelayedExponentialSFH(t0=1.0, tau=3.0)
    model = CSPBuilder(ssp).build(_synth.SSP_LOGZ_GRID[1], sfh0)
    err = 0.05 * np.abs(model)
    mask = np.ones(ssp.n_wave, dtype=bool)
    dust = DustAttenuation.from_mode2(ssp.wave, 0.05, -0.003)
    like = Likelihood(ssp, ssp.wave, model, err, mask, 0.0, 100.0, dust)
    jlike = JAXLikelihood(ssp, ssp.wave, model, err, mask, 0.0, 100.0, dust)
    return like, jlike


class TestBuildConvMatrix:
    def test_equivalence_kinematics_x0_zero(self):
        a = _build_conv_matrix(150, 2.5)
        b = _build_convolution_matrix(150, 2.5, 0.0)
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestJAXParity:
    def test_call_batch_parity(self, pair):
        like, jlike = pair
        logZ = np.array([-1.0, -0.3, _synth.SSP_LOGZ_GRID[1]])
        params = np.array([[1.0, 3.0], [5.0, 2.0], [0.5, 6.0]])
        chi2_np = like.call_batch(logZ, DelayedExponentialSFH, params)
        chi2_jax = jlike.call_batch(logZ, DelayedExponentialSFH, params)
        assert chi2_jax.shape == (3,)
        np.testing.assert_allclose(chi2_jax, chi2_np, rtol=3e-3)

    def test_edge_metallicity_clamp_parity(self, pair):
        like, jlike = pair
        logZ = np.array([-5.0, 1.0])   # below / above the metallicity grid
        params = np.array([[2.0, 2.0], [3.0, 1.0]])
        chi2_np = like.call_batch(logZ, DelayedExponentialSFH, params)
        chi2_jax = jlike.call_batch(logZ, DelayedExponentialSFH, params)
        # JAX clips the interpolation fraction, NumPy clamps via if/else:
        # identical values are expected at both edges
        np.testing.assert_allclose(chi2_jax, chi2_np, rtol=3e-3)
