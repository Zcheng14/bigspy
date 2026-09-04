"""Tests for MCMC modules."""

import numpy as np
import pytest
from conftest import requires_data

from bigspy.mcmc.priors import UniformPrior, LogUniformPrior, GaussianPrior, FixedPrior
from bigspy.mcmc.sfh import DelayedExponentialSFH, SFHBase
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.dust import DustAttenuation, calz_unred
from bigspy.mcmc.kinematics import (
    gauss_convolve, gauss_convolve_batch, VelocityBroadening,
    _build_convolution_matrix,
)
from bigspy.mcmc import kinematics as _kin_mod
from bigspy.mcmc.likelihood import Likelihood
from bigspy.specfit import calz_unred as specfit_calz_unred
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

    def test_evaluate_batch_matches_loop_delayed(self):
        """Batch evaluation must reproduce the per-object loop exactly."""
        rng = np.random.RandomState(7)
        t = np.linspace(0.5, 13.8, 196)
        params = np.column_stack([rng.uniform(0.1, 13.5, 64),
                                  rng.uniform(0.1, 10.0, 64)])
        batch = DelayedExponentialSFH.evaluate_batch(t, params)
        loop = np.array([DelayedExponentialSFH(r[0], r[1]).evaluate(t)
                         for r in params])
        np.testing.assert_array_equal(batch, loop)

    def test_evaluate_batch_age_cutoff(self):
        """SFR is zero beyond age_universe=14 (default), batch == loop."""
        t = np.linspace(0.5, 15.5, 64)
        params = np.array([[1.0, 3.0], [5.0, 2.0]])
        batch = DelayedExponentialSFH.evaluate_batch(t, params)
        loop = np.array([DelayedExponentialSFH(r[0], r[1]).evaluate(t)
                         for r in params])
        np.testing.assert_array_equal(batch, loop)
        assert np.all(batch[:, t > 14.0] == 0.0)


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


class TestConvolutionMatrix:
    """K @ y must reproduce gauss_convolve(y, sigma, x0) for any x0."""

    def test_matrix_equals_convolve_x0_zero(self):
        rng = np.random.RandomState(0)
        n, sigma = 150, 2.5
        y = rng.normal(0, 1, n)
        K = _build_convolution_matrix(n, sigma, 0.0)
        np.testing.assert_allclose(K @ y, gauss_convolve(y, sigma, 0.0), atol=1e-12)

    def test_matrix_equals_convolve_with_offset(self):
        rng = np.random.RandomState(1)
        n, sigma, x0 = 150, 2.5, 1.3
        y = rng.normal(0, 1, n)
        K = _build_convolution_matrix(n, sigma, x0)
        np.testing.assert_allclose(K @ y, gauss_convolve(y, sigma, x0), atol=1e-12)
        # Direction check: a delta at pixel p peaks at p + x0 in both paths
        delta = np.zeros(n)
        delta[30] = 1.0
        assert np.argmax(K @ delta) == 30 + round(x0)
        assert np.argmax(gauss_convolve(delta, sigma, x0)) == 30 + round(x0)

    def test_interior_row_sums_one(self):
        n, sigma = 150, 2.5
        K = _build_convolution_matrix(n, sigma, 0.0)
        khalf = round(4 * sigma + 3)
        np.testing.assert_allclose(
            np.sum(K[khalf:n - khalf, :], axis=1), 1.0, atol=1e-12)

    def test_matrix_matches_naive_loop(self):
        """K must equal the original per-pixel double-loop construction exactly."""
        n, sigma, x0 = 200, 3.7, -1.9
        K = _build_convolution_matrix(n, sigma, x0)
        # Naive reference: verbatim copy of the original double loop
        khalf = round(4 * sigma + abs(x0) + 3)
        xx = np.arange(khalf * 2 + 1) - khalf
        kernel = np.exp(-(xx - x0) ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        K_ref = np.zeros((n, n))
        offset = khalf
        for i in range(len(kernel)):
            for j in range(n):
                src = j + offset - i
                if 0 <= src < n:
                    K_ref[j, src] += kernel[i]
        np.testing.assert_array_equal(K, K_ref)

    def test_convolve_batch_offset_matches_rows(self):
        """gauss_convolve_batch rows == per-row gauss_convolve, x0 != 0."""
        rng = np.random.RandomState(4)
        spectra = rng.normal(0, 1, (6, 500))
        sigma, x0 = 2.0, 1.7
        batch = gauss_convolve_batch(spectra, sigma, x0)
        for i in range(spectra.shape[0]):
            np.testing.assert_allclose(
                batch[i], gauss_convolve(spectra[i], sigma, x0), atol=1e-12)

    def test_matrix_cache_reused(self):
        """The convolution matrix is built once per (n_pix, sigma, x0) key."""
        _kin_mod.clear_convolution_cache()
        try:
            spectra = np.random.RandomState(5).normal(0, 1, (8, 300))
            out1 = gauss_convolve_batch(spectra, 2.5)
            info1 = _kin_mod._conv_matrix_cached.cache_info()
            assert info1.misses == 1 and info1.hits == 0
            out2 = gauss_convolve_batch(spectra, 2.5)
            info2 = _kin_mod._conv_matrix_cached.cache_info()
            assert info2.misses == 1 and info2.hits == 1  # no rebuild
            np.testing.assert_array_equal(out1, out2)
            gauss_convolve_batch(spectra, 1.5)  # different sigma -> rebuild
            assert _kin_mod._conv_matrix_cached.cache_info().misses == 2
        finally:
            _kin_mod.clear_convolution_cache()

    def test_cached_matrix_readonly(self):
        _kin_mod.clear_convolution_cache()
        try:
            K = _kin_mod._conv_matrix_cached(120, 2.5)
            assert K.flags.writeable is False
            out = gauss_convolve_batch(np.ones((3, 120)), 2.5)
            assert out.shape == (3, 120)
        finally:
            _kin_mod.clear_convolution_cache()

    def test_sigma_zero_identity(self):
        K = _build_convolution_matrix(50, 0.0)
        np.testing.assert_allclose(K, np.eye(50))
        y = np.arange(50, dtype=float)
        np.testing.assert_allclose(gauss_convolve(y, 0.0), y)

    def test_velocity_broadening_sigma_pix(self):
        vb = VelocityBroadening(34.5, velscale=6.9)
        assert vb.sigma_pix == pytest.approx(5.0, abs=1e-12)
        # Default velscale comes from the DLOGW log grid
        expected = (10 ** 0.0001 - 1) * 299792.458
        assert VelocityBroadening.DLOGW_VEL == pytest.approx(expected, rel=1e-12)
        assert VelocityBroadening(expected).sigma_pix == pytest.approx(1.0, abs=1e-12)

    def test_apply_batch_equals_rows(self):
        rng = np.random.RandomState(2)
        spectra = rng.normal(0, 1, (4, 120))
        vb = VelocityBroadening(50.0, velscale=10.0)
        batch = vb.apply_batch(spectra)
        for i in range(spectra.shape[0]):
            np.testing.assert_allclose(batch[i], vb.apply(spectra[i]), atol=1e-12)


class TestDustAttenuationCalzetti:
    def test_from_calzetti_apply(self):
        w = np.linspace(4000, 8000, 200)
        ebv = 0.15
        d = DustAttenuation.from_calzetti(w, ebv)
        curve = d.apply(np.ones_like(w))
        # Closed form: 10^(-0.4 k(lam) ebv), k from the two Calzetti branches
        x = 1e4 / w
        k = np.where(w >= 6300,
                     2.659 * (-1.857 + 1.040 * x) + 4.05,
                     2.659 * (0.011 * x ** 3 - 0.198 * x ** 2 + 1.509 * x - 2.156) + 4.05)
        np.testing.assert_allclose(curve, 10.0 ** (-0.4 * k * ebv), rtol=1e-12)
        # Matches the standalone calz_unred function (independent code path)
        np.testing.assert_allclose(curve, calz_unred(w, ebv), rtol=1e-12)

    def test_unknown_mode_raises(self):
        w = np.linspace(4000, 5000, 10)
        with pytest.raises(ValueError, match="Unknown dust mode"):
            DustAttenuation(w, mode="wedge")

    def test_apply_recomputes_on_new_grid(self):
        w1 = np.linspace(4000, 6000, 100)
        w2 = np.linspace(5000, 7000, 80)
        ebv = 0.15
        d = DustAttenuation.from_calzetti(w1, ebv)
        flux = np.full(len(w2), 2.0)
        # wave=w2 differs from the stored grid -> curve recomputed on w2
        np.testing.assert_allclose(d.apply(flux, wave=w2),
                                   flux * calz_unred(w2, ebv), rtol=1e-12)
        # wave=None -> cached curve on the stored grid
        f1 = np.full(len(w1), 2.0)
        np.testing.assert_allclose(d.apply(f1), f1 * calz_unred(w1, ebv), rtol=1e-12)

    def test_inverse_relation_with_specfit(self):
        # specfit.calz_unred uses +0.4*k*ebv, mcmc.dust.calz_unred uses -0.4:
        # for the same |k| their product must be exactly 1.
        w = np.linspace(4000, 8000, 300)
        e = 0.15
        np.testing.assert_allclose(
            specfit_calz_unred(w, e) * calz_unred(w, e), 1.0, rtol=1e-12)

    def test_calz_unred_continuity_at_6300(self):
        for ebv in (0.0, 0.1):
            lo = calz_unred(np.array([6299.5]), ebv)[0]
            hi = calz_unred(np.array([6300.5]), ebv)[0]
            assert abs(lo / hi - 1.0) < 0.003
