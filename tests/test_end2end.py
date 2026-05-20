"""End-to-end pipeline tests."""

import tempfile
import os
import numpy as np
import pytest
from conftest import requires_data

from bigspy import SpecFit, MCMCFitter


class TestEndToEnd:
    @requires_data
    def test_full_pipeline_specfit(self, pca_file, test_data):
        """SpecFit runs end-to-end without errors."""
        sf = SpecFit(pca_file)
        result = sf.fit(
            wave=test_data["wave_obs"],
            flux=test_data["flux_obs"],
            error=test_data["error_obs"],
            mask=test_data["mask_obs"],
            z_sys=test_data["z"],
            mode="mode2",
        )
        assert result.ve[0] > 0
        assert result.vd[0] > 0
        assert result.ebv[0] >= 0

    @requires_data
    def test_full_pipeline_mcmc(self, pca_file, ssp_file, test_data):
        """Full SpecFit + MCMC pipeline runs without errors."""
        sf = SpecFit(pca_file)
        result = sf.fit(
            wave=test_data["wave_obs"],
            flux=test_data["flux_obs"],
            error=test_data["error_obs"],
            mask=test_data["mask_obs"],
            z_sys=test_data["z"],
            mode="mode2",
        )

        mc = MCMCFitter(
            ssp_fits=ssp_file,
            specfit_result=result,
            sfh_model="delayed",
            wave_range=(3600, 7400),
        )

        with tempfile.TemporaryDirectory() as tmp:
            mcmc_result = mc.run(
                n_live=50,
                chain_dir=tmp,
                max_ncalls=2000,
            )
            best = mcmc_result.bestfit
            assert "logZsun" in best or "t0" in best
            assert mcmc_result.posterior.shape[1] >= 2
            assert np.isfinite(mcmc_result.log_evidence)

    @requires_data
    def test_save_and_load(self, pca_file, test_data, tmp_path):
        """SpecFitResult can be saved and FITS re-read."""
        from bigspy.io import read_specfit_fits

        sf = SpecFit(pca_file)
        result = sf.fit(
            wave=test_data["wave_obs"],
            flux=test_data["flux_obs"],
            error=test_data["error_obs"],
            mask=test_data["mask_obs"],
            z_sys=test_data["z"],
            mode="mode2",
        )
        p = os.path.join(tmp_path, "test_specfit.fits")
        result.save(p)
        assert os.path.exists(p)
        loaded = read_specfit_fits(p)
        assert "wave" in loaded
        assert "params" in loaded

    @requires_data
    def test_custom_priors(self, pca_file, ssp_file, test_data):
        """MCMC accepts custom priors."""
        from bigspy import UniformPrior, FixedPrior

        sf = SpecFit(pca_file)
        result = sf.fit(
            wave=test_data["wave_obs"],
            flux=test_data["flux_obs"],
            error=test_data["error_obs"],
            mask=test_data["mask_obs"],
            z_sys=test_data["z"],
            mode="mode2",
        )

        mc = MCMCFitter(
            ssp_fits=ssp_file,
            specfit_result=result,
            sfh_model="delayed",
            wave_range=(3600, 7400),
        )

        with tempfile.TemporaryDirectory() as tmp:
            mcmc_result = mc.run(
                n_live=50,
                chain_dir=tmp,
                max_ncalls=2000,
                priors={
                    "logZsun": UniformPrior(-1.0, 0.0),
                    "t0": UniformPrior(0.5, 10.0),
                    "tau": FixedPrior(5.0),  # freeze tau
                },
            )
            assert "tau" not in mcmc_result.bestfit
            assert mcmc_result.posterior.shape[1] == 2  # only logZsun, t0
