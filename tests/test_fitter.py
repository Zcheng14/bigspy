"""Tests for bigspy.mcmc.fitter (MCMCResult / MCMCFitter) on synthetic data."""

import os
import sys

import numpy as np
import pytest
from astropy.io import fits

import _synth
from bigspy.mcmc.ssp import SSPLibrary
from bigspy.mcmc.csp import CSPBuilder
from bigspy.mcmc.dust import DustAttenuation
from bigspy.mcmc.sfh import DelayedExponentialSFH
from bigspy.mcmc.likelihood import Likelihood
from bigspy.mcmc.fitter import MCMCResult, MCMCFitter


@pytest.fixture(scope="module")
def synth_like(synth_ssp_file):
    """Real NumPy Likelihood on the synthetic SSP library."""
    ssp = SSPLibrary(synth_ssp_file)
    sfh0 = DelayedExponentialSFH(t0=1.0, tau=3.0)
    model = CSPBuilder(ssp).build(_synth.SSP_LOGZ_GRID[1], sfh0)
    return Likelihood(ssp, ssp.wave, model, np.full(ssp.n_wave, 0.01),
                      np.ones(ssp.n_wave, dtype=bool), 0.0, 0.0,
                      DustAttenuation.from_mode2(ssp.wave, 0.0, 0.0))


class TestMCMCResult:
    def test_bestfit_dict(self, synth_like):
        res = MCMCResult(_synth.StubSampler(like=synth_like))
        assert res.bestfit == {"t0": 5.0, "tau": 3.0, "logZsun": -1.0}

    def test_posterior_and_log_evidence(self, synth_like):
        stub = _synth.StubSampler(like=synth_like)
        res = MCMCResult(stub)
        np.testing.assert_array_equal(res.posterior, stub.result["samples"])
        assert res.log_evidence == -4.5
        stub.result.pop("logz")
        assert np.isnan(res.log_evidence)

    def test_save_result_layout(self, synth_like, tmp_path):
        stub = _synth.StubSampler(like=synth_like)
        res = MCMCResult(stub, likelihood_np=synth_like)
        p = str(tmp_path / "mcmc.fits")
        res.save_result(p)
        with fits.open(p) as h:
            assert [x.name for x in h] == ["PRIMARY", "BESTFIT", "WAVE", "FLUX",
                                           "ERROR", "MASK", "CSP", "CSP_OBS"]
            assert h[0].header["LOGEVID"] == -4.5
            assert h["MASK"].data.dtype == np.uint8
            assert len(h["CSP"].data) == synth_like.ssp.n_wave
            assert len(h["CSP_OBS"].data) == len(synth_like.obs_wave)
            cols = list(h["BESTFIT"].data.columns.names)
        assert cols == ["t0", "tau", "logZsun"]

    def test_plot_corner(self, synth_like, tmp_path):
        pytest.importorskip("corner")
        res = MCMCResult(_synth.StubSampler(like=synth_like),
                         likelihood_np=synth_like)
        p = str(tmp_path / "corner.png")
        res.plot_corner(p)
        assert os.path.exists(p) and os.path.getsize(p) > 1000

    def test_plot_bestfit(self, synth_like, tmp_path):
        res = MCMCResult(_synth.StubSampler(like=synth_like),
                         likelihood_np=synth_like)
        p = str(tmp_path / "bestfit.png")
        res.plot_bestfit(p)
        assert os.path.exists(p) and os.path.getsize(p) > 1000

    def test_plot_sfh(self, synth_like, tmp_path):
        res = MCMCResult(_synth.StubSampler(like=synth_like),
                         likelihood_np=synth_like)
        p = str(tmp_path / "sfh.png")
        res.plot_sfh(p)
        assert os.path.exists(p) and os.path.getsize(p) > 1000

    def test_like_fallback_to_sampler(self, synth_like, tmp_path):
        """likelihood_np=None falls back to the sampler's like (no recursion)."""
        stub = _synth.StubSampler(like=synth_like)
        res = MCMCResult(stub, likelihood_np=None)
        assert res._like is stub.like
        p = str(tmp_path / "mcmc.fits")
        res.save_result(p)      # save/plot must work through the fallback
        assert os.path.exists(p)


class TestMCMCFitter:
    @staticmethod
    def _make(synth_ssp_file, **kwargs):
        stub_res = _synth.StubSpecFitResult(_synth.SSP_WAVE[::2])
        kwargs.setdefault("use_jax", False)
        return MCMCFitter(synth_ssp_file, stub_res,
                          sfh_model="delayed", **kwargs)

    def test_likelihood_property(self, synth_ssp_file):
        f = self._make(synth_ssp_file)
        assert isinstance(f.likelihood, Likelihood)
        assert f.likelihood.ssp is f.ssp
        assert f._use_jax is False

    def test_emission_mask_kwarg(self, synth_ssp_file):
        f = self._make(synth_ssp_file, emission_mask=[(5500, 5600)])
        w = f._wave_obs
        m = f._obs_mask
        inside = (w >= 5500) & (w <= 5600)
        assert inside.sum() > 0
        assert not np.any(m[inside])
        assert np.all(m[(w > 4000) & (w < 5000)])

    def test_use_jax_fallback_on_import_error(self, synth_ssp_file, monkeypatch):
        monkeypatch.setitem(sys.modules, "bigspy.mcmc.likelihood_jax", None)
        f = self._make(synth_ssp_file, use_jax=True)
        assert f._use_jax is False

    def test_use_jax_true_builds_jax_like(self, synth_ssp_file):
        pytest.importorskip("jax")
        from bigspy.mcmc.likelihood_jax import JAXLikelihood
        f = self._make(synth_ssp_file, use_jax=True)
        assert f._use_jax is True
        assert isinstance(f._likelihood_jax, JAXLikelihood)

    def test_run_requires_chain_dir(self, synth_ssp_file):
        f = self._make(synth_ssp_file)
        with pytest.raises(ValueError, match="chain_dir"):
            f.run()

    def test_run_with_stub_likelihood(self, synth_ssp_file, tmp_path):
        f = self._make(synth_ssp_file)
        f._likelihood = _synth.StubLikelihood(t0_true=5.0, logZ_true=-1.0)
        res = f.run(n_live=20, max_ncalls=800, chain_dir=str(tmp_path / "chains"))
        assert isinstance(res, MCMCResult)
        assert {"t0", "tau", "logZsun"} <= set(res.bestfit)
        assert np.all(np.isfinite(res.posterior))
