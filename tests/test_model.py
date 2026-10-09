"""Tests for bigspy.mcmc.model.ModelComponents (NumPy model container)."""

import numpy as np
import pytest

import _synth
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.csp import CSPBuilder
from bigspy.mcmc.dust import DustAttenuation
from bigspy.mcmc.sfh import DelayedExponentialSFH
from bigspy.mcmc.model import ModelComponents


@pytest.fixture(scope="module")
def synth_model(synth_ssp_file):
    ssp = SSPLibrary(synth_ssp_file)
    sfh = DelayedExponentialSFH(t0=1.0, tau=3.0)
    model = CSPBuilder(ssp).build(_synth.SSP_LOGZ_GRID[1], sfh)
    err = np.full(ssp.n_wave, 0.01)
    mask = np.ones(ssp.n_wave, dtype=bool)
    dust = DustAttenuation.from_mode2(ssp.wave, 0.0, 0.0)
    return ModelComponents(ssp, ssp.wave, model, err, mask, 0.0, 0.0, dust)


class TestModelComponents:
    def test_normalization_at_5500(self, synth_model):
        m = synth_model
        win = (m.obs_wave > 5450) & (m.obs_wave < 5550)
        np.testing.assert_allclose(np.median(m.obs_flux[win]), 1.0, rtol=1e-6)

    def test_build_shapes_finite(self, synth_model):
        m = synth_model
        csp = m.build_csp(_synth.SSP_LOGZ_GRID[1], DelayedExponentialSFH(1.0, 3.0))
        model_obs = m.build_model(_synth.SSP_LOGZ_GRID[1], DelayedExponentialSFH(1.0, 3.0))
        assert csp.shape == m.ssp.wave.shape
        assert model_obs.shape == m.obs_wave.shape
        assert np.all(np.isfinite(csp)) and np.all(np.isfinite(model_obs))

    def test_reproduces_observation(self, synth_model):
        # The synthetic observation is itself a DelayedExp CSP -> model == observed.
        m = synth_model
        model_obs = m.build_model(_synth.SSP_LOGZ_GRID[1], DelayedExponentialSFH(1.0, 3.0))
        np.testing.assert_allclose(model_obs, m.obs_flux, rtol=1e-6, atol=1e-6)
