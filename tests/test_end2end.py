"""End-to-end pipeline tests (SpecFit -> NSS MCMC)."""

import os
import tempfile

import numpy as np
import pytest
from conftest import requires_data

pytest.importorskip("jax")

from bigspy import SpecFit, MCMCFitter  # noqa: E402


class TestEndToEnd:
    @requires_data
    def test_full_pipeline_specfit(self, pca_file, test_data):
        result = SpecFit(pca_file).fit(
            wave=test_data["wave_obs"], flux=test_data["flux_obs"],
            error=test_data["error_obs"], mask=test_data["mask_obs"],
            z_sys=test_data["z"], mode="mode2")
        assert result.ve[0] > 0 and result.vd[0] > 0 and result.ebv[0] >= 0

    @requires_data
    def test_full_pipeline_mcmc(self, pca_file, ssp_file, test_data):
        result = SpecFit(pca_file).fit(
            wave=test_data["wave_obs"], flux=test_data["flux_obs"],
            error=test_data["error_obs"], mask=test_data["mask_obs"],
            z_sys=test_data["z"], mode="mode2")

        mc = MCMCFitter(ssp_fits=ssp_file, specfit_result=result,
                        sfh_model="dpl", wave_range=(3600, 7400))
        with tempfile.TemporaryDirectory() as tmp:
            res = mc.run(out_dir=tmp, n_live=20, num_delete=10,
                         num_inner_steps=4, max_ncalls=800)
            assert {"tau", "alpha", "beta", "logZsun"} <= set(res.bestfit)
            assert res.posterior.shape[1] == 4
            assert np.isfinite(res.log_evidence)
            assert np.all(np.isfinite(res.posterior))

    @requires_data
    def test_save_and_load(self, pca_file, test_data, tmp_path):
        from bigspy.io import read_specfit_fits
        result = SpecFit(pca_file).fit(
            wave=test_data["wave_obs"], flux=test_data["flux_obs"],
            error=test_data["error_obs"], mask=test_data["mask_obs"],
            z_sys=test_data["z"], mode="mode2")
        p = os.path.join(tmp_path, "test_specfit.fits")
        result.save(p)
        assert os.path.exists(p)
        loaded = read_specfit_fits(p)
        assert "wave" in loaded and "params" in loaded

    @requires_data
    def test_custom_priors_fixed(self, pca_file, ssp_file, test_data):
        from bigspy import UniformPrior, LogUniformPrior, FixedPrior
        result = SpecFit(pca_file).fit(
            wave=test_data["wave_obs"], flux=test_data["flux_obs"],
            error=test_data["error_obs"], mask=test_data["mask_obs"],
            z_sys=test_data["z"], mode="mode2")

        mc = MCMCFitter(ssp_fits=ssp_file, specfit_result=result,
                        sfh_model="dpl", wave_range=(3600, 7400))
        with tempfile.TemporaryDirectory() as tmp:
            res = mc.run(out_dir=tmp, n_live=20, num_delete=10,
                         num_inner_steps=4, max_ncalls=600,
                         priors={"logZsun": UniformPrior(-2.5, 0.5),
                                 "tau": LogUniformPrior(0.1, 13.0),
                                 "alpha": LogUniformPrior(0.1, 1000.0),
                                 "beta": FixedPrior(0.2)})
            # Fixed params are reported in bestfit, not in the posterior.
            assert res.bestfit["beta"] == 0.2
            assert res.posterior.shape[1] == 3   # tau, alpha, logZsun
            assert len(res.bestfit_model()) == len(res.model.obs_wave)
