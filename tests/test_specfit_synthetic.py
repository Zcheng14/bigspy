"""Tests for bigspy.specfit internals on synthetic data (no LFS reference data).

Synthetic truth: Legendre-PCA combo (first 10 components), z=0.01, no dust,
no broadening, all pixels good.  Expected values verified analytically.
"""

import os

import numpy as np
import pytest
import lmfit

import _synth
from bigspy import SpecFit, SpecFitResult
from bigspy.io import read_specfit_fits, write_observed_fits
from bigspy.specfit import (
    C, DLOGW, calz_unred, ccm_unred, gauss_convolve,
    load_pca_templates, preprocess_spectrum, fit_spectrum,
    run_mode2, mean_filter, _A_lambda_fcn, save_results,
)
from bigspy.mcmc.kinematics import gauss_convolve as mcmc_gauss_convolve


@pytest.fixture(scope="module")
def synth_prep(synth_pca_file):
    pca, wave_temp, velscale = load_pca_templates(synth_pca_file)
    data, model = _synth.make_synthetic_obs()
    prep = preprocess_spectrum(data, wave_temp, pca)
    return prep, pca, wave_temp, velscale, data


@pytest.fixture(scope="module")
def synth_fit(synth_prep):
    prep, pca, wave_temp, velscale, data = synth_prep
    return fit_spectrum(prep, pca, wave_temp, mode="both")


# ── ccm_unred ──────────────────────────────────────────────────────
class TestCCMUnred:
    def test_zero_ebv_identity(self):
        # includes wavelengths outside the CCM valid range (factor stays 1)
        w = np.array([200.0, 3000.0, 5500.0, 9000.0, 30000.0])
        np.testing.assert_allclose(ccm_unred(w, 0.0), 1.0)

    def test_positive_ebv_reddens_blue(self):
        w = np.array([3000.0, 10000.0])
        c = ccm_unred(w, 0.1)
        assert c[0] > c[1] > 1.0

    def test_branch_continuity(self):
        # Straddle the CCM branch joins (x = 1/lambda-micron): no jumps
        for xb in (1.1, 3.3, 8.0):
            lo = ccm_unred(np.array([1e4 / (xb - 5e-4)]), 0.1)[0]
            hi = ccm_unred(np.array([1e4 / (xb + 5e-4)]), 0.1)[0]
            assert abs(lo / hi - 1.0) < 0.03, f"discontinuity at x={xb}"


# ── gauss_convolve ─────────────────────────────────────────────────
class TestGaussConvolveSpecFit:
    def test_matches_mcmc_module(self):
        rng = np.random.RandomState(0)
        y = rng.normal(0, 1, 120)
        for sigma, x0 in [(2.0, 0.0), (2.0, 1.5), (1.5, -2.0)]:
            np.testing.assert_allclose(
                gauss_convolve(y, sigma, x0),
                mcmc_gauss_convolve(y, sigma, x0), atol=1e-14)

    def test_x0_shifts_peak(self):
        delta = np.zeros(80)
        delta[30] = 1.0
        out = gauss_convolve(delta, 1.5, 2.0)
        assert np.argmax(out) == 32  # peak moves to p + x0


# ── mean_filter ────────────────────────────────────────────────────
class TestMeanFilter:
    def test_linear_flux_detail_zero(self):
        wave = np.linspace(4000, 5000, 1001)  # matches the internal 1 A grid
        flux = 2.0 + 0.001 * (wave - 4000)
        detail = mean_filter(flux, wave, 200.0)
        assert np.max(np.abs(detail)) < 1e-9

    def test_constant_flux_detail_zero(self):
        wave = np.linspace(4000, 5000, 1001)
        detail = mean_filter(np.full(1001, 3.0), wave, 200.0)
        assert np.max(np.abs(detail)) < 1e-12

    def test_masked_delta_localized(self):
        wave = np.linspace(4000, 5000, 1001)
        p = 500
        flux = np.full(1001, 1.0)
        flux[p] += 5.0
        mask = np.ones(1001)
        mask[p] = 0.0  # hide the spike from the filter
        detail = mean_filter(flux, wave, 200.0, mask=mask)
        assert detail[p] > 1.0
        assert np.median(np.abs(detail)) < 1e-10


# ── _A_lambda_fcn ──────────────────────────────────────────────────
class TestALambdaFcn:
    @staticmethod
    def _model(p1, p2, wave):
        x = 1e4 / wave
        xv = 1e4 / 5500.0
        return p1 * x + p2 * x ** 2 - p1 * xv - p2 * xv ** 2

    def test_fit_true_residual(self):
        params = lmfit.Parameters()
        params.add("p1", value=0.5)
        params.add("p2", value=-0.05)
        wave = np.linspace(4000, 6000, 50)
        data = np.linspace(0.1, -0.1, 50)
        np.testing.assert_allclose(
            _A_lambda_fcn(params, wave, data, fit=True),
            self._model(0.5, -0.05, wave) - data, atol=1e-14)

    def test_fit_false_model(self):
        params = lmfit.Parameters()
        params.add("p1", value=0.5)
        params.add("p2", value=-0.05)
        wave = np.linspace(4000, 6000, 50)
        np.testing.assert_allclose(
            _A_lambda_fcn(params, wave, None, fit=False),
            self._model(0.5, -0.05, wave), atol=1e-14)


# ── load_pca_templates / preprocess_spectrum ───────────────────────
class TestLoadPCATemplatesSynthetic:
    def test_synthetic_fits(self, synth_pca_file):
        pca, wave_temp, velscale = load_pca_templates(synth_pca_file)
        np.testing.assert_allclose(pca, _synth.pca_rows()[:10], atol=1e-12)
        np.testing.assert_array_equal(wave_temp, _synth.PCA_WAVE_LOG)
        assert velscale == (10 ** DLOGW - 1) * C


class TestPreprocessSynthetic:
    def test_alignment_norm_and_mask(self, synth_prep):
        prep, pca, wave_temp, velscale, data = synth_prep
        assert prep["npix"] > 0
        # Observed grid is the template log grid redshifted by z -> aligned:
        # the pixel below wave[0] is the template pixel it1 -> vsys = 0
        assert prep["vsys"] == pytest.approx(
            C * np.log(wave_temp[prep["it1"]] / prep["wave"][0]), abs=1e-9)
        assert abs(prep["vsys"]) < 1e-6
        # Flux normalized at the pixel closest to 5500 A
        i55 = int(np.argmin(np.abs(prep["wave"] - 5500)))
        assert prep["flux"][i55] == pytest.approx(1.0, rel=1e-12)
        # H-alpha region masked
        m = (prep["wave"] >= 6555) & (prep["wave"] <= 6575)
        assert m.sum() > 0
        assert np.all(prep["mask"][m] == 0)
        # PCA slice is exact
        rows = _synth.pca_rows()
        np.testing.assert_allclose(
            prep["temp_pca"],
            rows[:10, prep["it1"]:prep["it1"] + prep["npix"]].T, atol=1e-12)

    def test_fit_range_trims(self, synth_prep):
        prep = synth_prep[0]
        assert prep["wave"][0] >= 3600
        assert prep["wave"][-1] <= 7400

    def test_mask_convention_zero_is_good(self, synth_pca_file):
        # preprocess_spectrum requires the input mask in 0=good convention
        # (as produced by load_test_spectrum / the MaNGA pipeline);
        # an all-ones mask leaves no good pixel.
        pca, wave_temp, _ = load_pca_templates(synth_pca_file)
        data, _ = _synth.make_synthetic_obs()
        data_bad = dict(data)
        data_bad["mask_obs"] = np.ones(len(data["wave_obs"]))
        with pytest.raises(IndexError):
            preprocess_spectrum(data_bad, wave_temp, pca)


# ── run_mode1 / run_mode2 ──────────────────────────────────────────
class TestRunMode1Synthetic:
    def test_recovers_model(self, synth_prep, synth_fit):
        prep = synth_prep[0]
        r1 = synth_fit["mode1_result"]
        assert r1.redchi < 1.5
        assert abs(r1.params["ebv"].value) < 0.05   # truth: dust-free
        corr = np.corrcoef(synth_fit["mode1_model"], prep["flux_raw"])[0, 1]
        assert corr > 0.999
        assert np.all(np.isfinite(synth_fit["ve"]))
        assert np.all(np.isfinite(synth_fit["vd"]))


class TestRunMode2Synthetic:
    def test_dust_branch_returns_full_dict(self, synth_prep, synth_fit):
        prep = synth_prep[0]
        m2 = synth_fit["mode2_result"]
        for key in ("p1", "p2", "ebv", "chi2r", "slr_flux", "dust_wave", "dust_A"):
            assert key in m2
        assert len(m2["dust_wave"]) > 100            # dust branch was taken
        assert len(m2["slr_flux"]) == prep["npix"]
        assert np.isfinite(m2["chi2r"]) and m2["chi2r"] > 0
        assert len(m2["dust_A"]) == len(m2["dust_wave"])

    def test_best_wei_none_early_return(self, synth_prep, synth_fit):
        prep, pca, wave_temp, velscale, data = synth_prep
        mask = np.zeros(prep["npix"])
        mask[100:105] = 1.0                          # fewer good px than ncomp
        m2 = run_mode2(prep["flux"], prep["error"], mask, prep["temp_pca"],
                       prep["wave"], prep["vsys"], velscale,
                       0.0, 0.0, synth_fit["mode1_result"],
                       pca, prep["it1"], wave_temp)
        assert m2 == {"p1": 0.0, "p2": 0.0,
                      "ebv": synth_fit["mode1_result"].params["ebv"].value,
                      "chi2r": 0.0, "slr_flux": None}

    def test_no_crash_when_5500_window_masked(self, synth_prep, synth_fit):
        # The 5450-5550 normalisation window fully masked must degrade
        # gracefully (p1 = p2 = 0, no dust data) instead of crashing.
        prep, pca, wave_temp, velscale, data = synth_prep
        mask = np.ones(prep["npix"])
        mask[(prep["wave"] > 5450) & (prep["wave"] < 5550)] = 0.0
        m2 = run_mode2(prep["flux"], prep["error"], mask, prep["temp_pca"],
                       prep["wave"], prep["vsys"], velscale,
                       0.0, 0.0, synth_fit["mode1_result"],
                       pca, prep["it1"], wave_temp)
        assert m2["p1"] == 0.0
        assert m2["p2"] == 0.0
        assert len(m2["dust_wave"]) == 0


# ── SpecFitResult direct construction ──────────────────────────────
@pytest.fixture()
def direct_result():
    w = np.linspace(4000, 5000, 500)
    rng = np.random.RandomState(2)
    fit_dict = {
        "ve": np.array([30.0, 5.0]), "vd": np.array([120.0, 8.0]),
        "ebv_m1": np.array([0.12, 0.02]), "chi2r_m1": 1.3,
        "mode1_model": np.full(500, 1.5),
        "mode1_dust": np.linspace(1.1, 0.9, 500),
        "mode2_dust": np.linspace(1.05, 0.95, 500),
        "mode2_result": {"p1": 0.05, "p2": -0.003, "ebv": 0.1,
                         "dust_wave": np.linspace(4000, 5000, 50),
                         "dust_A": np.linspace(0.0, 0.2, 50)},
    }
    prep_dict = {"wave": w, "flux_raw": 1.0 + 0.1 * rng.randn(500),
                 "error_raw": np.full(500, 0.05), "mask": np.ones(500)}
    return SpecFitResult(fit_dict, prep_dict), fit_dict, prep_dict


class TestSpecFitResultDirect:
    def test_attributes(self, direct_result):
        res, fit_dict, _ = direct_result
        assert res.ve == (30.0, 5.0)
        assert res.vd == (120.0, 8.0)
        assert res.ebv == (0.12, 0.02)
        assert res.p1 == 0.05
        assert res.p2 == -0.003
        assert res.chi2 == 1.3
        assert res.bestfit is fit_dict["mode1_model"]

    def test_dust_curve_interp(self, direct_result):
        res, fit_dict, prep_dict = direct_result
        w = prep_dict["wave"]
        curve = res.dust_curve
        np.testing.assert_allclose(curve(w), fit_dict["mode2_dust"], atol=1e-12)
        # Outside the prep wave grid the curve fills with 1.0
        np.testing.assert_allclose(curve(np.array([3000.0, 6000.0])), 1.0)

    def test_dust_curve_none_falls_back_to_ones(self):
        w = np.linspace(4000, 5000, 100)
        fit_dict = {"ve": np.array([0.0, 0.0]), "vd": np.array([0.0, 0.0]),
                    "ebv_m1": np.array([0.0, 0.0])}
        prep_dict = {"wave": w, "flux_raw": np.ones(100),
                     "error_raw": np.ones(100), "mask": np.ones(100)}
        res = SpecFitResult(fit_dict, prep_dict)
        np.testing.assert_allclose(res.dust_curve(w), 1.0)

    def test_save_layout_and_roundtrip(self, direct_result, tmp_path):
        res, fit_dict, prep_dict = direct_result
        p = str(tmp_path / "direct.fits")
        res.save(p)
        from astropy.io import fits
        with fits.open(p) as h:
            assert [x.name for x in h] == ["PRIMARY", "WAVE", "FLUX", "ERROR",
                                           "PARAMS", "BESTFIT", "DUST"]
            colnames = list(h["PARAMS"].data.columns.names)
        assert colnames == ["ve", "ve_err", "vd", "vd_err", "ebv_m1",
                            "ebv_m1_err", "ebv_m2", "p1", "p2"]
        loaded = read_specfit_fits(p)
        np.testing.assert_array_equal(loaded["wave"], prep_dict["wave"])
        np.testing.assert_array_equal(loaded["flux"], prep_dict["flux_raw"])
        np.testing.assert_array_equal(loaded["bestfit"], fit_dict["mode1_model"])
        np.testing.assert_array_equal(loaded["dust"], fit_dict["mode2_dust"])
        assert loaded["params"]["ve"] == 30.0
        assert loaded["params"]["p2"] == -0.003

    def test_plots_write_png(self, direct_result, tmp_path):
        res, _, _ = direct_result
        p1 = str(tmp_path / "fit.png")
        p2 = str(tmp_path / "dust.png")
        res.plot_fit(p1)
        res.plot_dust(p2)
        assert os.path.exists(p1) and os.path.getsize(p1) > 1000
        assert os.path.exists(p2) and os.path.getsize(p2) > 1000


# ── SpecFit.fit kwargs ─────────────────────────────────────────────
class TestSpecFitFitKwargs:
    @staticmethod
    def _obs():
        data, model = _synth.make_synthetic_obs()
        return data

    def test_missing_z_sys_raises(self, synth_pca_file):
        data = self._obs()
        sf = SpecFit(synth_pca_file)
        with pytest.raises(ValueError, match="required"):
            sf.fit(wave=data["wave_obs"], flux=data["flux_obs"],
                   error=data["error_obs"])

    def test_observed_fits_path_without_mask(self, synth_pca_file, tmp_path):
        # No MASK HDU -> every pixel is good
        data = self._obs()
        p = str(tmp_path / "obs.fits")
        write_observed_fits(p, data["wave_obs"], data["flux_obs"],
                            data["error_obs"], mask=None,
                            header_kw={"REDSHIFT": data["z"]})
        sf = SpecFit(synth_pca_file)
        res = sf.fit(observed_fits=p, mode="mode1")
        assert isinstance(res, SpecFitResult)
        assert np.isfinite(res.ve[0]) and np.isfinite(res.vd[0])
        assert len(res.bestfit) == len(res.wave_prep)

    def test_observed_fits_path_with_mask(self, synth_pca_file, tmp_path):
        # MASK HDU written by write_observed_fits uses 1=good booleans
        data = self._obs()
        p = str(tmp_path / "obs.fits")
        write_observed_fits(p, data["wave_obs"], data["flux_obs"],
                            data["error_obs"],
                            mask=np.ones(len(data["wave_obs"]), dtype=bool),
                            header_kw={"REDSHIFT": data["z"]})
        sf = SpecFit(synth_pca_file)
        res = sf.fit(observed_fits=p, mode="mode1")
        assert isinstance(res, SpecFitResult)

    def test_emission_mask_override(self, synth_pca_file):
        data = self._obs()
        sf = SpecFit(synth_pca_file)
        res = sf.fit(wave=data["wave_obs"], flux=data["flux_obs"],
                     error=data["error_obs"], mask=data["mask_obs"],
                     z_sys=data["z"], mode="mode1",
                     emission_mask=[(5000, 5100)])
        w = res.wave_prep
        m = res.mask_prep
        inside = (w >= 5000) & (w <= 5100)
        assert inside.sum() > 0
        assert np.all(m[inside] == 0)      # custom region masked
        # default line list replaced: H-alpha no longer masked
        i_ha = int(np.argmin(np.abs(w - 6560)))
        assert m[i_ha] == 1

    @pytest.mark.parametrize("neig", [5, 15])
    def test_neig(self, synth_pca_file, neig):
        data = self._obs()
        sf = SpecFit(synth_pca_file)
        res = sf.fit(wave=data["wave_obs"], flux=data["flux_obs"],
                     error=data["error_obs"], mask=data["mask_obs"],
                     z_sys=data["z"], mode="mode1", neig=neig)
        assert isinstance(res, SpecFitResult)
        assert np.isfinite(res.ve[0]) and np.isfinite(res.vd[0])

    def test_save_results(self, synth_prep, synth_fit, tmp_path):
        prep, pca, wave_temp, velscale, data = synth_prep
        out_dir = str(tmp_path)
        save_results(prep, synth_fit, out_dir)
        p = os.path.join(out_dir, "fit_result.fits")
        assert os.path.exists(p)
        from astropy.io import fits
        with fits.open(p) as h:
            colnames = list(h["PARAMS"].data.columns.names)
        assert colnames == ["ve", "ve_err", "vd", "vd_err", "ebv_m1",
                            "ebv_m1_err", "ebv_m2", "p1", "p2"]
