"""Tests for bigspy utilities: rebin / log_rebin / air_to_vacuum, FITS IO, mask."""

import numpy as np
import pytest
from astropy.io import fits

from bigspy.utils import rebin, log_rebin, air_to_vacuum_wave
from bigspy.io import write_observed_fits, read_observed_fits
from bigspy.mask import EMISSION_LINES, build_emission_mask, mask_emlines_detailed


# ═══════════════════════════════════════════════════════════════════
#  utils
# ═══════════════════════════════════════════════════════════════════

class TestRebin:
    def test_constant_flux_conserved(self):
        x = np.linspace(4000, 5000, 501)
        y = np.full_like(x, 5.0)
        x0 = np.linspace(4010, 4990, 20)
        _, y_new = rebin(x, y, x0=x0)
        np.testing.assert_allclose(y_new, 5.0, atol=1e-12)

    def test_linear_flux_exact_interior(self):
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
        assert loglam[0] == pytest.approx(wave[1], rel=1e-12)
        assert loglam[0] < wave[2]

    def test_constant_spectrum(self):
        wave = np.linspace(4000, 5000, 500)
        spec = np.full_like(wave, 7.5)
        spec_rebin, _, _ = log_rebin(wave, spec)
        np.testing.assert_allclose(spec_rebin, 7.5, atol=1e-12)

    def test_identity_on_log_grid(self):
        n = 2001
        wave = 4000.0 * 10 ** (np.arange(n) * 5e-4)
        spec = wave.copy()
        dv = 5e-4 * 299792.458
        spec_rebin, loglam, dv_used = log_rebin(wave, spec, dv=dv)
        assert dv_used == dv
        np.testing.assert_allclose(spec_rebin, loglam, rtol=1e-10)
        steps = np.diff(np.log(loglam))
        assert np.ptp(steps) < 1e-12
        np.testing.assert_allclose(steps * 299792.458, dv_used, rtol=2e-3)


class TestAirToVacuum:
    def test_reference_value_5000(self):
        assert air_to_vacuum_wave(5000.0) == pytest.approx(5001.3948, abs=1e-3)

    def test_offset_positive(self):
        w = np.linspace(3000, 10000, 71)
        assert np.all(air_to_vacuum_wave(w) > w)


# ═══════════════════════════════════════════════════════════════════
#  io
# ═══════════════════════════════════════════════════════════════════

class TestObservedFits:
    def test_roundtrip(self, tmp_path):
        rng = np.random.RandomState(0)
        wave = np.linspace(4000, 5000, 100)
        flux = 1.0 + 0.1 * rng.randn(100)
        error = np.full(100, 0.02)
        mask = np.ones(100, dtype=bool)
        mask[[5, 50]] = False
        p = str(tmp_path / "obs.fits")
        write_observed_fits(p, wave, flux, error, mask=mask,
                            header_kw={"REDSHIFT": 0.02, "OBJECT": "test"})
        d = read_observed_fits(p)
        np.testing.assert_array_equal(d["wave"], wave)
        np.testing.assert_array_equal(d["flux"], flux)
        np.testing.assert_array_equal(d["error"], error)
        np.testing.assert_array_equal(d["mask"], mask)
        assert d["header"]["REDSHIFT"] == 0.02
        assert d["header"]["OBJECT"] == "test"

    def test_mask_uint8_on_disk_and_hdu_order(self, tmp_path):
        wave = np.linspace(4000, 4100, 20)
        p = str(tmp_path / "obs.fits")
        write_observed_fits(p, wave, np.ones(20), np.full(20, 0.01),
                            mask=np.ones(20, dtype=bool))
        with fits.open(p) as h:
            assert [x.name for x in h] == ["PRIMARY", "WAVE", "FLUX", "ERROR", "MASK"]
            assert h["MASK"].data.dtype == np.uint8

    def test_no_mask_defaults_all_good(self, tmp_path):
        wave = np.linspace(4000, 4100, 20)
        p = str(tmp_path / "obs.fits")
        write_observed_fits(p, wave, np.ones(20), np.full(20, 0.01), mask=None)
        assert np.all(read_observed_fits(p)["mask"])


# ═══════════════════════════════════════════════════════════════════
#  mask
# ═══════════════════════════════════════════════════════════════════

class TestEmissionLines:
    def test_count_and_bounds(self):
        assert len(EMISSION_LINES) == 58
        for lo, hi in EMISSION_LINES.values():
            assert lo < hi
            assert 3000 < lo < hi < 10000


class TestBuildEmissionMask:
    def test_custom_regions_exact(self):
        wave = np.linspace(4000, 6000, 2001)
        regions = [(5000, 5010), (5500, 5510)]
        mask = build_emission_mask(wave, regions)
        expected = np.ones(len(wave), dtype=bool)
        for lo, hi in regions:
            expected[(wave >= lo) & (wave <= hi)] = False
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, expected)

    def test_default_list(self):
        wave = np.array([4000.0, 5400.0, 6560.0, 5450.0])
        mask = build_emission_mask(wave)
        assert mask.dtype == bool
        assert mask[1] is np.True_
        assert not mask[2]
        assert mask[3] is np.True_

    def test_no_overlap_all_true(self):
        mask = build_emission_mask(np.linspace(2000, 2500, 100))
        assert np.all(mask)


class TestMaskEmLinesDetailed:
    def test_masks_default_lines(self):
        wave = np.linspace(4000, 7000, 3001)
        mask_out, lines = mask_emlines_detailed(wave, np.ones(len(wave), dtype=bool))
        np.testing.assert_array_equal(mask_out, build_emission_mask(wave))
        assert lines == EMISSION_LINES

    def test_mask_add_extends(self):
        wave = np.linspace(4000, 6000, 2001)
        _, lines = mask_emlines_detailed(
            wave, np.ones(len(wave), dtype=bool), mask_add={"custom": [4999, 5001]})
        assert "custom" in lines
        assert len(lines) == len(EMISSION_LINES) + 1
        mask_out, _ = mask_emlines_detailed(
            wave, np.ones(len(wave), dtype=bool), mask_add={"custom": [4999, 5001]})
        assert not np.all(mask_out[(wave >= 4999) & (wave <= 5001)])

    def test_input_false_preserved(self):
        wave = np.linspace(4000, 6000, 2001)
        mask_in = np.ones(len(wave), dtype=bool)
        mask_in[[10, 20, 1500]] = False
        mask_out, _ = mask_emlines_detailed(wave, mask_in)
        assert not mask_out[10]
        assert not mask_out[20]
        assert not mask_out[1500]
