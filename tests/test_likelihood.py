"""Tests for bigspy.mcmc.likelihood.Likelihood on synthetic data (no LFS).

The synthetic construction (obs grid == ssp.wave, obs == exact CSP model,
flat dust, vd=0) makes the full build -> broaden -> 5500-normalize -> dust ->
interp -> mask -> chi2 chain analytically predictable.
"""

import numpy as np
import pytest

import _synth
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.csp import CSPBuilder
from bigspy.mcmc.dust import DustAttenuation
from bigspy.mcmc.sfh import DelayedExponentialSFH
from bigspy.mcmc.likelihood import Likelihood


@pytest.fixture(scope="module")
def synth_setup(synth_ssp_file):
    """Likelihood whose observation IS the exact model -> chi2 == 0."""
    ssp = SSPLibrary(synth_ssp_file)
    logZ0 = _synth.SSP_LOGZ_GRID[1]
    sfh0 = DelayedExponentialSFH(t0=1.0, tau=3.0)
    model = CSPBuilder(ssp).build(logZ0, sfh0)
    err = np.full(ssp.n_wave, 0.01)
    mask = np.ones(ssp.n_wave, dtype=bool)
    dust = DustAttenuation.from_mode2(ssp.wave, 0.0, 0.0)
    like = Likelihood(ssp, ssp.wave, model, err, mask, 0.0, 0.0, dust)
    return like, logZ0, sfh0, model, err, mask


class TestLikelihoodSynthetic:
    def test_exact_model_zero_chi2(self, synth_setup):
        like, logZ0, sfh0, model, err, mask = synth_setup
        chi2 = like(logZ0, sfh0)
        assert chi2 == pytest.approx(0.0, abs=1e-10)

    def test_known_noise_chi2_scale(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        logZ0 = _synth.SSP_LOGZ_GRID[1]
        sfh0 = DelayedExponentialSFH(t0=1.0, tau=3.0)
        model = CSPBuilder(ssp).build(logZ0, sfh0)
        rng = np.random.RandomState(11)
        obs = model * (1.0 + rng.normal(0.0, 0.05, ssp.n_wave))
        err = 0.05 * np.abs(model)
        like = Likelihood(ssp, ssp.wave, obs, err,
                          np.ones(ssp.n_wave, dtype=bool), 0.0, 0.0,
                          DustAttenuation.from_mode2(ssp.wave, 0.0, 0.0))
        chi2 = like(logZ0, sfh0)
        N = ssp.n_wave
        # chi2-distributed with ~N dof -> loose sanity bounds
        assert 0.2 * N < chi2 < 3.0 * N

    def test_call_batch_matches_call_loop(self, synth_setup):
        like, logZ0, sfh0, model, err, mask = synth_setup
        rng = np.random.RandomState(5)
        N = 8
        logZ = rng.uniform(-2.0, 0.2, N)
        params = np.column_stack([rng.uniform(0.5, 10.0, N),
                                  rng.uniform(0.5, 8.0, N)])
        chi2_loop = np.array([
            like(lz, DelayedExponentialSFH(to, ta))
            for lz, to, ta in zip(logZ, params[:, 0], params[:, 1])
        ])
        chi2_batch = like.call_batch(logZ, DelayedExponentialSFH, params)
        assert chi2_batch.shape == (N,)
        np.testing.assert_allclose(chi2_batch, chi2_loop, rtol=1e-10)

    def test_masked_pixels_excluded(self, synth_setup):
        like, logZ0, sfh0, model, err, mask = synth_setup
        ssp = like.ssp
        mask2 = np.ones(ssp.n_wave, dtype=bool)
        mask2[[10, 20, 400, 700]] = False  # outliers land outside 5450-5550
        obs_clean = model.copy()
        obs_out = model.copy()
        obs_out[10] = 1.0e6
        obs_out[700] = -50.0
        dust = DustAttenuation.from_mode2(ssp.wave, 0.0, 0.0)
        like_clean = Likelihood(ssp, ssp.wave, obs_clean, err, mask2,
                                0.0, 0.0, dust)
        like_out = Likelihood(ssp, ssp.wave, obs_out, err, mask2,
                              0.0, 0.0, dust)
        assert like_out(logZ0, sfh0) == like_clean(logZ0, sfh0)

    def test_med5500_main_branch(self):
        w = np.linspace(4000, 7000, 761)  # dense grid, ~25 px in window
        f = w.copy()
        m = np.ones_like(w, dtype=bool)
        expected = np.median(f[(w >= 5450) & (w <= 5550) & m])
        assert Likelihood._med5500(w, f, m, (5450, 5550)) == expected

    def test_med5500_fallback_branch(self):
        w = np.linspace(4000, 7000, 31)  # only 1 px in window -> global median
        f = w.copy()
        m = np.ones_like(w, dtype=bool)
        assert Likelihood._med5500(w, f, m, (5450, 5550)) == np.median(f[m])

    def test_ndof(self, synth_setup):
        like, logZ0, sfh0, model, err, mask = synth_setup
        assert like.ndof == mask.sum() - 3

    def test_dust_changes_chi2(self, synth_setup):
        like, logZ0, sfh0, model, err, mask = synth_setup
        ssp = like.ssp
        like_dusty = Likelihood(ssp, ssp.wave, model, err, mask, 0.0, 0.0,
                                DustAttenuation.from_mode2(ssp.wave, 0.05, -0.003))
        # Observation is dust-free: applying attenuation must worsen chi2
        assert like_dusty(logZ0, sfh0) > 1.0
