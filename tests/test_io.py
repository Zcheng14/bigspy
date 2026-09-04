"""Tests for bigspy.io observed-spectrum FITS round-trips (no LFS data)."""

import numpy as np
from astropy.io import fits

from bigspy.io import write_observed_fits, read_observed_fits


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
        flux = np.ones(20)
        error = np.full(20, 0.01)
        mask = np.ones(20, dtype=bool)
        p = str(tmp_path / "obs.fits")
        write_observed_fits(p, wave, flux, error, mask=mask)
        with fits.open(p) as h:
            names = [x.name for x in h]
            assert names == ["PRIMARY", "WAVE", "FLUX", "ERROR", "MASK"]
            assert h["MASK"].data.dtype == np.uint8

    def test_no_mask_defaults_all_good(self, tmp_path):
        wave = np.linspace(4000, 4100, 20)
        flux = np.ones(20)
        error = np.full(20, 0.01)
        p = str(tmp_path / "obs.fits")
        write_observed_fits(p, wave, flux, error, mask=None)
        d = read_observed_fits(p)
        assert np.all(d["mask"])
