"""Tests for bigspy.mcmc.csp.CSPBuilder on the synthetic SSP library.

The synthetic spectra spec[mi, ai, :] = (mi+1)(ai+1) wave/5500 make every
CSPBuilder operation analytically predictable (see tests/_synth.py).
"""

import numpy as np
import pytest

import _synth
from _synth import ConstantSFH
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.csp import CSPBuilder
from bigspy.mcmc.sfh import DelayedExponentialSFH


@pytest.fixture(scope="module")
def builder(synth_ssp_file):
    return CSPBuilder(SSPLibrary(synth_ssp_file))


class TestCSPBuilder:
    def test_grid_points_exact(self, builder):
        for logM in _synth.SSP_LOGZ_GRID:
            np.testing.assert_allclose(
                builder.build(logM, ConstantSFH()),
                _synth.expected_csp(logM), atol=1e-12)

    def test_interior_interpolation_exact(self, builder):
        # Midpoint between two metallicity grid points -> linear in (mi+1)
        g = _synth.SSP_LOGZ_GRID
        logM_mid = 0.5 * (g[0] + g[1])
        np.testing.assert_allclose(
            builder.build(logM_mid, ConstantSFH()),
            _synth.expected_csp(logM_mid), atol=1e-12)

    def test_edge_clamping(self, builder):
        # logZ below / above the grid clamps to the lowest / highest metallicity
        np.testing.assert_allclose(
            builder.build(-5.0, ConstantSFH()),
            _synth.expected_csp(_synth.SSP_LOGZ_GRID[0], ConstantSFH()), atol=1e-12)
        np.testing.assert_allclose(
            builder.build(+1.0, ConstantSFH()),
            _synth.expected_csp(_synth.SSP_LOGZ_GRID[-1], ConstantSFH()), atol=1e-12)

    def test_delayed_sfh_weights(self, builder):
        # Weights computed independently: w = SFR(time)*dt / sum
        sfh = DelayedExponentialSFH(t0=1.0, tau=3.0)
        logM = 0.5 * (_synth.SSP_LOGZ_GRID[1] + _synth.SSP_LOGZ_GRID[2])
        w = _synth.sfh_weights(sfh)
        age_term = np.dot(w, (np.arange(6) + 1)[:, None]
                          * _synth.SSP_WAVE[None, :] / 5500.0)
        expected = _synth.metal_value(logM) * age_term
        np.testing.assert_allclose(builder.build(logM, sfh), expected, atol=1e-12)

    def test_weights_normalized(self, builder):
        # CSP depends only on normalized weights: constant SFR of 7.3 == 1.0
        logM = -1.0
        np.testing.assert_allclose(
            builder.build(logM, ConstantSFH(7.3)),
            builder.build(logM, ConstantSFH(1.0)), atol=1e-14)

    def test_build_batch_equals_loop(self, builder):
        rng = np.random.RandomState(3)
        N = 8
        logZ = rng.uniform(-2.3, 0.3, N)
        t0 = rng.uniform(0.5, 10.0, N)
        tau = rng.uniform(0.5, 8.0, N)
        params = np.column_stack([t0, tau])

        batch = builder.build_batch(logZ, params, DelayedExponentialSFH)
        loop = np.array([
            builder.build(lz, DelayedExponentialSFH(to, ta))
            for lz, to, ta in zip(logZ, t0, tau)
        ])
        assert batch.shape == (N, 761)
        np.testing.assert_allclose(batch, loop, atol=1e-12)

    def test_build_batch_edge_metallicities(self, builder):
        logZ = np.array([-5.0, 1.0])
        params = np.array([[1.0, 3.0], [2.0, 2.0]])
        batch = builder.build_batch(logZ, params, DelayedExponentialSFH)
        for j in range(2):
            expected = builder.build(logZ[j], DelayedExponentialSFH(*params[j]))
            np.testing.assert_allclose(batch[j], expected, atol=1e-12)
