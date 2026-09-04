"""Tests for bigspy.utils (rebin, log_rebin, air_to_vacuum_wave)."""

import numpy as np
import pytest

from bigspy.utils import rebin, log_rebin, air_to_vacuum_wave
from bigspy.specfit import air_to_vacuum_wave as specfit_air_to_vacuum


class TestRebin:
    def test_constant_flux_conserved(self):
        x = np.linspace(4000, 5000, 501)
        y = np.full_like(x, 5.0)
        x0 = np.linspace(4010, 4990, 20)
        _, y_new = rebin(x, y, x0=x0)
        np.testing.assert_allclose(y_new, 5.0, atol=1e-12)

    def test_linear_flux_exact_interior(self):
        # Trapezoid of a linear function over a symmetric bin = bin-centre value
        x = np.linspace(4000, 5000, 501)
        a, b = 2.0, 0.001
        y = a + b * x
        x0 = np.linspace(4010, 4990, 20)
        _, y_new = rebin(x, y, x0=x0)
        np.testing.assert_allclose(y_new[1:-1], a + b * x0[1:-1], atol=1e-9)

    def test_default_log_grid(self):
        x = np.linspace(4000, 5000, 501)
        y = np.full_like(x, 3.0)
        dlogx = 1e-3
        x0, y_new = rebin(x, y, dlogx=dlogx)
        m = int((np.log10(x[-1]) - np.log10(x[0])) / dlogx)
        assert len(x0) == m
        assert x0[0] == x[0]
        np.testing.assert_allclose(np.diff(np.log10(x0)), dlogx, rtol=1e-12)
        np.testing.assert_allclose(y_new, 3.0, atol=1e-9)
        assert np.all(np.isfinite(y_new))


class TestLogRebin:
    def test_grid_starts_at_second_pixel(self):
        wave = np.linspace(4000, 5000, 500)
        spec = np.ones_like(wave)
        _, loglam, _ = log_rebin(wave, spec)
        # exp(log(w[1])) round-trip: compare with relative tolerance
        assert loglam[0] == pytest.approx(wave[1], rel=1e-12)
        assert loglam[0] < wave[2]

    def test_constant_spectrum(self):
        wave = np.linspace(4000, 5000, 500)
        spec = np.full_like(wave, 7.5)
        spec_rebin, _, _ = log_rebin(wave, spec)
        np.testing.assert_allclose(spec_rebin, 7.5, atol=1e-12)

    def test_identity_on_log_grid(self):
        # spec = wave is linear in wave -> linear interpolation is exact,
        # so the rebinned spectrum equals the returned log grid itself.
        n = 2001
        wave = 4000.0 * 10 ** (np.arange(n) * 5e-4)
        spec = wave.copy()
        dv = 5e-4 * 299792.458
        spec_rebin, loglam, dv_used = log_rebin(wave, spec, dv=dv)
        assert dv_used == dv
        np.testing.assert_allclose(spec_rebin, loglam, rtol=1e-10)
        steps = np.diff(np.log(loglam))
        assert np.ptp(steps) < 1e-12                      # log-uniform
        np.testing.assert_allclose(steps * 299792.458, dv_used, rtol=2e-3)


class TestAirToVacuum:
    def test_reference_value_5000(self):
        # Hand-derived: fact = 1 + 6.4328e-5 + 2.94981e-2/142 + 2.5540e-4/37
        #              = 1.000278964 -> 5000 A (air) -> 5001.3948 A (vacuum)
        assert air_to_vacuum_wave(5000.0) == pytest.approx(5001.3948, abs=1e-3)

    def test_offset_positive(self):
        w = np.linspace(3000, 10000, 71)
        vac = air_to_vacuum_wave(w)
        assert np.all(vac > w)

    def test_matches_specfit_duplicate(self):
        w = np.linspace(3000, 10000, 71)
        np.testing.assert_allclose(air_to_vacuum_wave(w),
                                   specfit_air_to_vacuum(w), rtol=1e-14)
