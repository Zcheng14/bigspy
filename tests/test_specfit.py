"""Tests for SpecFit module."""

import numpy as np
import pytest
from conftest import requires_data

from bigspy import SpecFit, SpecFitResult
from bigspy.specfit import (
    calz_unred, ccm_unred, air_to_vacuum_wave, gauss_convolve,
    load_pca_templates, load_test_spectrum, preprocess_spectrum,
    fit_spectrum,
)


class TestUtilityFunctions:
    def test_air_to_vacuum(self):
        w = np.array([4000.0, 5500.0, 7000.0])
        wv = air_to_vacuum_wave(w)
        assert len(wv) == 3
        assert np.all(wv > w)

    def test_calz_unred_identity(self):
        w = np.linspace(1000, 8000, 100)
        c = calz_unred(w, 0.0)
        assert np.allclose(c, 1.0)

    def test_calz_unred_ebv_positive(self):
        w = np.linspace(4000, 7000, 50)
        c = calz_unred(w, 0.1)
        assert np.all(c > 1.0)  # deredden = brighten

    def test_gauss_convolve_noop(self):
        y = np.array([1.0, 2.0, 3.0])
        result = gauss_convolve(y, 0.0)
        assert np.allclose(result, y)

    def test_gauss_convolve_normalised(self):
        y = np.ones(200)
        result = gauss_convolve(y, 5.0)
        # Interior values should be 1.0, edges may deviate due to boundary
        mid = result[50:150]
        np.testing.assert_allclose(mid, 1.0, atol=1e-6)


class TestPCALoading:
    @requires_data
    def test_load_pca_templates(self, pca_file):
        pca, wave, velscale = load_pca_templates(pca_file)
        assert pca.ndim == 2
        assert pca.shape[0] == 10  # FIT_NEIG
        assert wave.ndim == 1
        assert velscale > 0


class TestPreprocessing:
    @requires_data
    def test_preprocess_returns_expected_keys(self, pca_file, test_data):
        pca, wave_temp, velscale = load_pca_templates(pca_file)
        prep = preprocess_spectrum(test_data, wave_temp, pca)
        for key in ["wave", "flux", "flux_raw", "error", "mask", "temp_pca", "npix"]:
            assert key in prep, f"Missing key: {key}"
        assert prep["npix"] > 0


class TestSpecFit:
    @requires_data
    def test_fit_mode2(self, pca_file, test_data):
        sf = SpecFit(pca_file)
        result = sf.fit(
            wave=test_data["wave_obs"],
            flux=test_data["flux_obs"],
            error=test_data["error_obs"],
            mask=test_data["mask_obs"],
            z_sys=test_data["z"],
            mode="mode2",
        )
        assert isinstance(result, SpecFitResult)
        assert result.ve[0] > 0
        assert result.vd[0] > 0
        assert 0.0 <= result.ebv[0] <= 1.0

    @requires_data
    def test_fit_mode1(self, pca_file, test_data):
        sf = SpecFit(pca_file)
        result = sf.fit(
            wave=test_data["wave_obs"],
            flux=test_data["flux_obs"],
            error=test_data["error_obs"],
            mask=test_data["mask_obs"],
            z_sys=test_data["z"],
            mode="mode1",
        )
        assert result.ve[0] > 0
        assert result.vd[0] > 0

    @requires_data
    def test_preprocessed_properties(self, specfit_result):
        assert specfit_result.wave_prep is not None
        assert specfit_result.flux_prep is not None
        assert specfit_result.error_prep is not None
        assert len(specfit_result.wave_prep) == len(specfit_result.flux_prep)

    @requires_data
    def test_dust_curve_callable(self, specfit_result):
        w = np.linspace(4000, 7000, 10)
        d = specfit_result.dust_curve(w)
        assert d.shape == (10,)
