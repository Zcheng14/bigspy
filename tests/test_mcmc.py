"""Tests for MCMC modules."""

import numpy as np
import pytest
from conftest import requires_data

from bigspy.mcmc.priors import UniformPrior, LogUniformPrior, GaussianPrior, FixedPrior
from bigspy.mcmc.sfh import DelayedExponentialSFH, SFHBase
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.dust import DustAttenuation, calz_unred
from bigspy.mcmc.kinematics import gauss_convolve, gauss_convolve_batch, VelocityBroadening
from bigspy.mcmc.likelihood import Likelihood
from bigspy import MCMCFitter


class TestPriors:
    def test_uniform(self):
        p = UniformPrior(-2.0, 3.0)
        c = np.array([0.0, 0.5, 1.0])
        result = p.transform(c)
        np.testing.assert_allclose(result, [-2.0, 0.5, 3.0])

    def test_loguniform(self):
        p = LogUniformPrior(0.1, 10.0)
        c = np.array([0.0, 0.5, 1.0])
        result = p.transform(c)
        np.testing.assert_allclose(result, [0.1, 1.0, 10.0], rtol=1e-10)

    def test_fixed(self):
        p = FixedPrior(5.0)
        c = np.array([0.0, 0.5, 1.0])
        result = p.transform(c)
        np.testing.assert_allclose(result, [5.0, 5.0, 5.0])

    def test_gaussian(self):
        p = GaussianPrior(0.0, 1.0)
        c = np.array([0.5])
        result = p.transform(c)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_batch_transform(self):
        p = UniformPrior(0.0, 10.0)
        cube = np.linspace(0, 1, 5).reshape(-1, 1)
        result = p.transform(cube)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, np.linspace(0, 10, 5))


class TestSFH:
    def test_delayed_exp_creation(self):
        sfh = DelayedExponentialSFH(t0=2.0, tau=5.0)
        assert sfh.t0 == 2.0
        assert sfh.tau == 5.0
        assert sfh.n_params == 2
        assert sfh.param_names == ["t0", "tau"]

    def test_delayed_exp_evaluate(self):
        sfh = DelayedExponentialSFH(t0=0.5, tau=3.0, age_universe=13.8)
        timegrid = np.linspace(0, 13.8, 50)
        sfr = sfh.evaluate(timegrid)
        assert sfr.shape == timegrid.shape
        # SFR > 0 at some points (delayed exponential has rising then falling SFR)
        assert np.any(sfr > 0)

    def test_evaluate_batch(self):
        t = np.linspace(0, 13.8, 30)
        params = np.array([[1.0, 3.0], [5.0, 2.0]])
        result = DelayedExponentialSFH.evaluate_batch(t, params)
        assert result.shape == (2, 30)


class TestSSPLibrary:
    @requires_data
    def test_load(self, ssp_file):
        ssp = SSPLibrary(ssp_file)
        assert ssp.n_metal == 6
        assert ssp.n_age > 0
        assert ssp.n_wave > 0

    @requires_data
    def test_wave_range(self, ssp_file):
        ssp = SSPLibrary(ssp_file, wave_range=(3600, 7400))
        assert ssp.wave[0] >= 3600
        assert ssp.wave[-1] <= 7400


class TestDust:
    def test_calz_unred_noop(self):
        w = np.linspace(4000, 7000, 50)
        c = calz_unred(w, 0.0)
        np.testing.assert_allclose(c, 1.0)

    def test_dust_from_mode2(self):
        w = np.linspace(3600, 7400, 100)
        d = DustAttenuation.from_mode2(w, p1=0.5, p2=-0.05)
        curve = d.apply(np.ones_like(w))
        assert curve.shape == w.shape


class TestKinematics:
    def test_convolve_noop(self):
        y = np.arange(100, dtype=float)
        result = gauss_convolve(y, 0.0)
        np.testing.assert_allclose(result, y)

    def test_convolve_batch(self):
        y = np.array([np.ones(100), np.arange(100, dtype=float)])
        result = gauss_convolve_batch(y, 3.0)
        assert result.shape == y.shape

    def test_velocity_broadening(self):
        vb = VelocityBroadening(100.0)
        spec = np.ones(200)
        result = vb.apply(spec)
        # Interior values should be normalised
        np.testing.assert_allclose(result[50:150], 1.0, atol=1e-6)


class TestLikelihood:
    @requires_data
    def test_chi2_positive(self, ssp_file, specfit_result):
        ssp = SSPLibrary(ssp_file, wave_range=(3600, 7400))
        dust = DustAttenuation.from_mode2(ssp.wave, specfit_result.p1, specfit_result.p2)
        w = specfit_result.wave_prep
        fl = specfit_result.flux_prep
        er = specfit_result.error_prep
        i55 = np.argmin(np.abs(w - 5500))
        fn, en = fl / fl[i55], er / fl[i55]
        mask = np.asarray(specfit_result.mask_prep, dtype=bool)
        like = Likelihood(ssp, w, fn, en, mask,
                          specfit_result.ve[0], specfit_result.vd[0], dust)
        sfh = DelayedExponentialSFH(t0=2.0, tau=5.0, age_universe=13.8)
        chi2 = like(-0.5, sfh)
        assert chi2 > 0
        assert like.ndof > 0


class TestVectorized:
    @requires_data
    def test_batch_equals_loop(self, ssp_file, specfit_result):
        """Phase 3 verification: call_batch matches scalar __call__."""
        ssp = SSPLibrary(ssp_file, wave_range=(3600, 7400))
        dust = DustAttenuation.from_mode2(ssp.wave, specfit_result.p1, specfit_result.p2)
        w, fl, er = (specfit_result.wave_prep, specfit_result.flux_prep,
                     specfit_result.error_prep)
        i55 = np.argmin(np.abs(w - 5500))
        mask = np.asarray(specfit_result.mask_prep, dtype=bool)
        like = Likelihood(ssp, w, fl / fl[i55], er / fl[i55], mask,
                          specfit_result.ve[0], specfit_result.vd[0], dust)

        N = 10
        rng = np.random.RandomState(0)
        logZ = rng.uniform(-2, 0, N)
        t0 = rng.uniform(0.5, 10, N)
        tau = rng.uniform(0.5, 5, N)

        chi2_loop = np.array([
            like(lz, DelayedExponentialSFH(to, ta, age_universe=13.8))
            for lz, to, ta in zip(logZ, t0, tau)
        ])
        chi2_batch = like.call_batch(
            logZ, DelayedExponentialSFH, np.column_stack([t0, tau])
        )
        np.testing.assert_allclose(chi2_batch, chi2_loop, rtol=1e-10)
