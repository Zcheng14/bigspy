"""Tests for MCMC building blocks: priors, SFH, SSP, CSP, kinematics, dust."""

import numpy as np
import pytest
from conftest import requires_data

import _synth
from _synth import ConstantSFH
from bigspy.mcmc.priors import (UniformPrior, LogUniformPrior, GaussianPrior,
                                FixedPrior)
from bigspy.mcmc.sfh import (SFHBase, DelayedExponentialSFH, DoublePowerLawSFH)
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.csp import CSPBuilder
from bigspy.mcmc.dust import DustAttenuation, calz_unred
from bigspy.mcmc.kinematics import (
    gauss_convolve, gauss_convolve_batch, VelocityBroadening,
    _build_convolution_matrix,
)
from bigspy.mcmc import kinematics as _kin_mod
from bigspy.specfit import calz_unred as specfit_calz_unred


# ═══════════════════════════════════════════════════════════════════
#  Priors
# ═══════════════════════════════════════════════════════════════════

class TestPriors:
    def test_uniform(self):
        p = UniformPrior(-2.0, 3.0)
        c = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(p.transform(c), [-2.0, 0.5, 3.0])

    def test_loguniform(self):
        p = LogUniformPrior(0.1, 10.0)
        c = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(p.transform(c), [0.1, 1.0, 10.0], rtol=1e-10)

    def test_fixed(self):
        p = FixedPrior(5.0)
        c = np.array([0.0, 0.5, 1.0])
        np.testing.assert_allclose(p.transform(c), [5.0, 5.0, 5.0])

    def test_gaussian(self):
        p = GaussianPrior(0.0, 1.0)
        np.testing.assert_allclose(p.transform(np.array([0.5])), [0.0], atol=1e-6)

    def test_batch_transform(self):
        p = UniformPrior(0.0, 10.0)
        cube = np.linspace(0, 1, 5).reshape(-1, 1)
        result = p.transform(cube)
        assert result.shape == (5,)
        np.testing.assert_allclose(result, np.linspace(0, 10, 5))

    def test_transform_jax_matches_numpy(self):
        import jax.numpy as jnp
        c = np.linspace(0.0, 1.0, 7)
        for p in (UniformPrior(-2.0, 3.0), LogUniformPrior(0.1, 13.0),
                  GaussianPrior(0.5, 1.2), FixedPrior(3.3)):
            np.testing.assert_allclose(
                np.asarray(p.transform_jax(jnp.asarray(c))), p.transform(c),
                rtol=1e-6, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════
#  Star formation histories
# ═══════════════════════════════════════════════════════════════════

class TestSFH:
    def test_delayed_exp_creation(self):
        sfh = DelayedExponentialSFH(t0=2.0, tau=5.0)
        assert sfh.t0 == 2.0 and sfh.tau == 5.0
        assert sfh.n_params == 2
        assert sfh.param_names == ["t0", "tau"]

    def test_delayed_exp_evaluate(self):
        sfh = DelayedExponentialSFH(t0=0.5, tau=3.0)
        timegrid = np.linspace(0, 13.8, 50)
        sfr = sfh.evaluate(timegrid)
        assert sfr.shape == timegrid.shape
        assert np.any(sfr > 0)

    def test_dpl_evaluate(self):
        sfh = DoublePowerLawSFH(tau=3.0, alpha=2.0, beta=1.0)
        sfr = sfh.evaluate(np.linspace(0.5, 13.8, 50))
        assert sfr.shape == (50,) and np.all(sfr >= 0)

    def test_evaluate_batch_jax_matches_numpy(self):
        import jax.numpy as jnp
        rng = np.random.RandomState(5)
        t = np.linspace(0.5, 13.8, 64)
        for cls, params in (
            (DelayedExponentialSFH, np.column_stack([rng.uniform(0.1, 13.0, 8),
                                                     rng.uniform(0.1, 8.0, 8)])),
            (DoublePowerLawSFH, np.column_stack([rng.uniform(0.1, 12.0, 8),
                                                 rng.uniform(0.5, 5.0, 8),
                                                 rng.uniform(0.2, 3.0, 8)])),
        ):
            jax_out = np.asarray(cls.evaluate_batch_jax(jnp.asarray(t),
                                                        jnp.asarray(params)))
            loop = np.array([cls(*row).evaluate(t) for row in params])
            np.testing.assert_allclose(jax_out, loop, rtol=1e-4, atol=1e-6)

    def test_evaluate_batch_jax_required(self):
        import jax.numpy as jnp

        class Bare(SFHBase):
            n_params = 1
            param_names = ["tau"]

            def evaluate(self, timegrid):
                return np.ones_like(timegrid)

        with pytest.raises(NotImplementedError):
            Bare.evaluate_batch_jax(jnp.ones(4), jnp.ones((2, 1)))


# ═══════════════════════════════════════════════════════════════════
#  SSP library
# ═══════════════════════════════════════════════════════════════════

class TestSSPLibrary:
    @requires_data
    def test_load(self, ssp_file):
        ssp = SSPLibrary(ssp_file)
        assert ssp.n_metal == 6
        assert ssp.n_age > 0 and ssp.n_wave > 0

    @requires_data
    def test_wave_range(self, ssp_file):
        ssp = SSPLibrary(ssp_file, wave_range=(3600, 7400))
        assert ssp.wave[0] >= 3600 and ssp.wave[-1] <= 7400


class TestSSPLibrarySynthetic:
    def test_load_shapes_and_values(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        assert ssp.n_metal == 3 and ssp.n_age == 6 and ssp.n_wave == 761
        np.testing.assert_array_equal(ssp.wave, _synth.SSP_WAVE)
        np.testing.assert_array_equal(ssp.metal, _synth.SSP_METAL)
        np.testing.assert_array_equal(ssp.time, _synth.SSP_TIME)
        np.testing.assert_array_equal(ssp.dt, _synth.SSP_DT)

    def test_wave_range_slice(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file, wave_range=(4500, 6000))
        assert ssp.wave[0] >= 4500 and ssp.wave[-1] <= 6000
        assert ssp._spec.shape == (3, 6, ssp.n_wave)
        full = SSPLibrary(synth_ssp_file)
        k = np.searchsorted(full.wave, ssp.wave[0])
        np.testing.assert_allclose(ssp.get_spectrum(1, 2),
                                   full.get_spectrum(1, 2)[k:k + ssp.n_wave],
                                   atol=1e-12)

    def test_get_spectrum_values_and_copy(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        expected = (1 + 1) * (2 + 1) * _synth.SSP_WAVE / 5500.0
        np.testing.assert_allclose(ssp.get_spectrum(1, 2), expected, atol=1e-12)
        spec = ssp.get_spectrum(1, 2)
        spec[:] = -1.0
        assert not np.any(ssp._spec[1, 2, :] == -1.0)

    def test_get_mass(self, synth_ssp_file):
        ssp = SSPLibrary(synth_ssp_file)
        np.testing.assert_allclose(ssp.get_mass(0, 0), 1.0)
        np.testing.assert_allclose(ssp.get_mass(2, 5), 1.0)

    def test_repr(self, synth_ssp_file):
        r = repr(SSPLibrary(synth_ssp_file))
        assert "SSPLibrary" in r and "n_metal=3" in r and "n_age=6" in r


# ═══════════════════════════════════════════════════════════════════
#  CSP builder
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def builder(synth_ssp_file):
    return CSPBuilder(SSPLibrary(synth_ssp_file))


class TestCSPBuilder:
    def test_grid_points_exact(self, builder):
        for logM in _synth.SSP_LOGZ_GRID:
            np.testing.assert_allclose(builder.build(logM, ConstantSFH()),
                                       _synth.expected_csp(logM), atol=1e-12)

    def test_interior_interpolation_exact(self, builder):
        g = _synth.SSP_LOGZ_GRID
        logM_mid = 0.5 * (g[0] + g[1])
        np.testing.assert_allclose(builder.build(logM_mid, ConstantSFH()),
                                   _synth.expected_csp(logM_mid), atol=1e-12)

    def test_edge_clamping(self, builder):
        np.testing.assert_allclose(
            builder.build(-5.0, ConstantSFH()),
            _synth.expected_csp(_synth.SSP_LOGZ_GRID[0], ConstantSFH()), atol=1e-12)
        np.testing.assert_allclose(
            builder.build(+1.0, ConstantSFH()),
            _synth.expected_csp(_synth.SSP_LOGZ_GRID[-1], ConstantSFH()), atol=1e-12)

    def test_delayed_sfh_weights(self, builder):
        sfh = DelayedExponentialSFH(t0=1.0, tau=3.0)
        logM = 0.5 * (_synth.SSP_LOGZ_GRID[1] + _synth.SSP_LOGZ_GRID[2])
        w = _synth.sfh_weights(sfh)
        age_term = np.dot(w, (np.arange(6) + 1)[:, None]
                          * _synth.SSP_WAVE[None, :] / 5500.0)
        expected = _synth.metal_value(logM) * age_term
        np.testing.assert_allclose(builder.build(logM, sfh), expected, atol=1e-12)

    def test_weights_normalized(self, builder):
        np.testing.assert_allclose(builder.build(-1.0, ConstantSFH(7.3)),
                                   builder.build(-1.0, ConstantSFH(1.0)), atol=1e-14)

    def test_build_batch_is_gone(self, builder):
        assert not hasattr(builder, "build_batch")


# ═══════════════════════════════════════════════════════════════════
#  Dust
# ═══════════════════════════════════════════════════════════════════

class TestDust:
    def test_calz_unred_noop(self):
        w = np.linspace(4000, 7000, 50)
        np.testing.assert_allclose(calz_unred(w, 0.0), 1.0)

    def test_dust_from_mode2(self):
        w = np.linspace(3600, 7400, 100)
        d = DustAttenuation.from_mode2(w, p1=0.5, p2=-0.05)
        assert d.apply(np.ones_like(w)).shape == w.shape


class TestDustAttenuationCalzetti:
    def test_from_calzetti_apply(self):
        w = np.linspace(4000, 8000, 200)
        ebv = 0.15
        d = DustAttenuation.from_calzetti(w, ebv)
        curve = d.apply(np.ones_like(w))
        x = 1e4 / w
        k = np.where(w >= 6300,
                     2.659 * (-1.857 + 1.040 * x) + 4.05,
                     2.659 * (0.011 * x ** 3 - 0.198 * x ** 2 + 1.509 * x - 2.156) + 4.05)
        np.testing.assert_allclose(curve, 10.0 ** (-0.4 * k * ebv), rtol=1e-12)
        np.testing.assert_allclose(curve, calz_unred(w, ebv), rtol=1e-12)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown dust mode"):
            DustAttenuation(np.linspace(4000, 5000, 10), mode="wedge")

    def test_apply_recomputes_on_new_grid(self):
        w1 = np.linspace(4000, 6000, 100)
        w2 = np.linspace(5000, 7000, 80)
        ebv = 0.15
        d = DustAttenuation.from_calzetti(w1, ebv)
        flux = np.full(len(w2), 2.0)
        np.testing.assert_allclose(d.apply(flux, wave=w2),
                                   flux * calz_unred(w2, ebv), rtol=1e-12)
        f1 = np.full(len(w1), 2.0)
        np.testing.assert_allclose(d.apply(f1), f1 * calz_unred(w1, ebv), rtol=1e-12)

    def test_inverse_relation_with_specfit(self):
        w = np.linspace(4000, 8000, 300)
        e = 0.15
        np.testing.assert_allclose(
            specfit_calz_unred(w, e) * calz_unred(w, e), 1.0, rtol=1e-12)

    def test_calz_unred_continuity_at_6300(self):
        for ebv in (0.0, 0.1):
            lo = calz_unred(np.array([6299.5]), ebv)[0]
            hi = calz_unred(np.array([6300.5]), ebv)[0]
            assert abs(lo / hi - 1.0) < 0.003


# ═══════════════════════════════════════════════════════════════════
#  Kinematics
# ═══════════════════════════════════════════════════════════════════

class TestKinematics:
    def test_convolve_noop(self):
        y = np.arange(100, dtype=float)
        np.testing.assert_allclose(gauss_convolve(y, 0.0), y)

    def test_convolve_batch(self):
        y = np.array([np.ones(100), np.arange(100, dtype=float)])
        assert gauss_convolve_batch(y, 3.0).shape == y.shape

    def test_velocity_broadening(self):
        vb = VelocityBroadening(100.0)
        result = vb.apply(np.ones(200))
        np.testing.assert_allclose(result[50:150], 1.0, atol=1e-6)


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
        delta = np.zeros(n)
        delta[30] = 1.0
        assert np.argmax(K @ delta) == 30 + round(x0)
        assert np.argmax(gauss_convolve(delta, sigma, x0)) == 30 + round(x0)

    def test_interior_row_sums_one(self):
        n, sigma = 150, 2.5
        K = _build_convolution_matrix(n, sigma, 0.0)
        khalf = round(4 * sigma + 3)
        np.testing.assert_allclose(np.sum(K[khalf:n - khalf, :], axis=1), 1.0,
                                   atol=1e-12)

    def test_matrix_matches_naive_loop(self):
        n, sigma, x0 = 200, 3.7, -1.9
        K = _build_convolution_matrix(n, sigma, x0)
        khalf = round(4 * sigma + abs(x0) + 3)
        xx = np.arange(khalf * 2 + 1) - khalf
        kernel = np.exp(-(xx - x0) ** 2 / (2 * sigma ** 2))
        kernel /= kernel.sum()
        K_ref = np.zeros((n, n))
        for i in range(len(kernel)):
            for j in range(n):
                src = j + khalf - i
                if 0 <= src < n:
                    K_ref[j, src] += kernel[i]
        np.testing.assert_array_equal(K, K_ref)

    def test_convolve_batch_offset_matches_rows(self):
        rng = np.random.RandomState(4)
        spectra = rng.normal(0, 1, (6, 500))
        batch = gauss_convolve_batch(spectra, 2.0, 1.7)
        for i in range(spectra.shape[0]):
            np.testing.assert_allclose(batch[i], gauss_convolve(spectra[i], 2.0, 1.7),
                                       atol=1e-12)

    def test_matrix_cache_reused(self):
        _kin_mod.clear_convolution_cache()
        try:
            spectra = np.random.RandomState(5).normal(0, 1, (8, 300))
            out1 = gauss_convolve_batch(spectra, 2.5)
            info1 = _kin_mod._conv_matrix_cached.cache_info()
            assert info1.misses == 1 and info1.hits == 0
            out2 = gauss_convolve_batch(spectra, 2.5)
            info2 = _kin_mod._conv_matrix_cached.cache_info()
            assert info2.misses == 1 and info2.hits == 1
            np.testing.assert_array_equal(out1, out2)
            gauss_convolve_batch(spectra, 1.5)
            assert _kin_mod._conv_matrix_cached.cache_info().misses == 2
        finally:
            _kin_mod.clear_convolution_cache()

    def test_cached_matrix_readonly(self):
        _kin_mod.clear_convolution_cache()
        try:
            K = _kin_mod._conv_matrix_cached(120, 2.5)
            assert K.flags.writeable is False
            assert gauss_convolve_batch(np.ones((3, 120)), 2.5).shape == (3, 120)
        finally:
            _kin_mod.clear_convolution_cache()

    def test_sigma_zero_identity(self):
        np.testing.assert_allclose(_build_convolution_matrix(50, 0.0), np.eye(50))
        y = np.arange(50, dtype=float)
        np.testing.assert_allclose(gauss_convolve(y, 0.0), y)

    def test_velocity_broadening_sigma_pix(self):
        vb = VelocityBroadening(34.5, velscale=6.9)
        assert vb.sigma_pix == pytest.approx(5.0, abs=1e-12)
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
