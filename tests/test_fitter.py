"""Tests for bigspy.mcmc.fitter (MCMCResult / MCMCFitter) with NSS."""

import os

import numpy as np
import pytest
from astropy.io import fits

jax = pytest.importorskip("jax")  # noqa: F841

import _synth  # noqa: E402
from bigspy.mcmc.ssp import SSPLibrary  # noqa: E402
from bigspy.mcmc.csp import CSPBuilder  # noqa: E402
from bigspy.mcmc.dust import DustAttenuation  # noqa: E402
from bigspy.mcmc.sfh import DelayedExponentialSFH  # noqa: E402
from bigspy.mcmc.model import ModelComponents  # noqa: E402
from bigspy.mcmc.likelihood_jax import JAXLikelihood  # noqa: E402
from bigspy.mcmc.fitter import MCMCResult, MCMCFitter  # noqa: E402


@pytest.fixture(scope="module")
def synth_model(synth_ssp_file):
    ssp = SSPLibrary(synth_ssp_file)
    sfh0 = DelayedExponentialSFH(t0=1.0, tau=3.0)
    model = CSPBuilder(ssp).build(_synth.SSP_LOGZ_GRID[1], sfh0)
    return ModelComponents(ssp, ssp.wave, model, np.full(ssp.n_wave, 0.01),
                           np.ones(ssp.n_wave, dtype=bool), 0.0, 0.0,
                           DustAttenuation.from_mode2(ssp.wave, 0.0, 0.0))


class _FakeSampler:
    """Duck-typed sampler matching the interface MCMCResult depends on."""

    def __init__(self, n=200, seed=0):
        rng = np.random.RandomState(seed)
        self.param_names = ["t0", "tau", "logZsun"]
        self.sfh_class = DelayedExponentialSFH
        self.fixed_params = {}
        self._best = np.array([5.0, 3.0, -1.0])
        self._post = np.column_stack([rng.normal(5.0, 1.0, n),
                                      rng.normal(3.0, 0.5, n),
                                      rng.normal(-1.0, 0.3, n)])
        self.result = {"logz": -4.5}

    def get_bestfit(self):
        return self._best

    def get_posterior(self):
        return self._post


class _FakeSamplerFixedLogZ(_FakeSampler):
    """Sampler with logZsun held fixed (excluded from the posterior)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.param_names = ["t0", "tau"]
        self.fixed_params = {"logZsun": -0.5}
        self._best = np.array([5.0, 3.0])
        self._post = self._post[:, :2]


class TestMCMCResult:
    def test_bestfit_dict(self, synth_model):
        res = MCMCResult(_FakeSampler(), synth_model)
        assert res.bestfit == {"t0": 5.0, "tau": 3.0, "logZsun": -1.0}

    def test_posterior_and_log_evidence(self, synth_model):
        stub = _FakeSampler()
        res = MCMCResult(stub, synth_model)
        np.testing.assert_array_equal(res.posterior, stub._post)
        assert res.log_evidence == -4.5
        stub.result.pop("logz")
        assert np.isnan(res.log_evidence)

    def test_save_result_layout(self, synth_model, tmp_path):
        res = MCMCResult(_FakeSampler(), synth_model)
        p = str(tmp_path / "mcmc.fits")
        res.save_result(p)
        with fits.open(p) as h:
            assert [x.name for x in h] == ["PRIMARY", "BESTFIT", "WAVE", "FLUX",
                                           "ERROR", "MASK", "CSP", "CSP_OBS"]
            assert h[0].header["LOGEVID"] == -4.5
            assert h["MASK"].data.dtype == np.uint8
            assert len(h["CSP"].data) == synth_model.ssp.n_wave
            assert len(h["CSP_OBS"].data) == len(synth_model.obs_wave)
            cols = list(h["BESTFIT"].data.columns.names)
        assert cols == ["t0", "tau", "logZsun"]

    def test_plots_write_png(self, synth_model, tmp_path):
        pytest.importorskip("corner")
        res = MCMCResult(_FakeSampler(), synth_model)
        for meth, name in ((res.plot_corner, "corner.png"),
                           (res.plot_bestfit, "bestfit.png"),
                           (res.plot_sfh, "sfh.png")):
            p = str(tmp_path / name)
            meth(p)
            assert os.path.exists(p) and os.path.getsize(p) > 1000


class TestMCMCResultFixedParams:
    """FixedPrior parameters are reported in bestfit but not posterior."""

    def test_bestfit_merges_fixed_params(self, synth_model):
        res = MCMCResult(_FakeSamplerFixedLogZ(), synth_model)
        assert res.bestfit == {"t0": 5.0, "tau": 3.0, "logZsun": -0.5}
        assert res.posterior.shape[1] == 2

    def test_bestfit_model_uses_fixed_value(self, synth_model):
        res = MCMCResult(_FakeSamplerFixedLogZ(), synth_model)
        expected = synth_model.build_model(
            -0.5, DelayedExponentialSFH(t0=5.0, tau=3.0))
        np.testing.assert_allclose(res.bestfit_model(), expected, rtol=1e-12)

    def test_plots_align_with_active_params(self, synth_model, tmp_path):
        pytest.importorskip("corner")
        res = MCMCResult(_FakeSamplerFixedLogZ(), synth_model)
        for meth, name in ((res.plot_corner, "corner.png"),
                           (res.plot_bestfit, "bestfit.png"),
                           (res.plot_sfh, "sfh.png")):
            p = str(tmp_path / name)
            meth(p)
            assert os.path.exists(p) and os.path.getsize(p) > 1000


class TestMCMCFitter:
    @staticmethod
    def _make(synth_ssp_file, **kwargs):
        stub_res = _synth.StubSpecFitResult(_synth.SSP_WAVE[::2])
        return MCMCFitter(synth_ssp_file, stub_res, sfh_model="delayed", **kwargs)

    def test_dust_from_mode2_when_available(self, synth_ssp_file):
        f = self._make(synth_ssp_file)
        expected = DustAttenuation.from_mode2(f.ssp.wave, 0.05, -0.003)
        np.testing.assert_allclose(f._dust._curve, expected._curve,
                                   rtol=1e-12)

    def test_dust_falls_back_to_mode1_calzetti(self, synth_ssp_file):
        stub_res = _synth.StubSpecFitResult(
            _synth.SSP_WAVE[::2], mode2_dust_ok=False, ebv=(0.15, 0.01))
        f = MCMCFitter(synth_ssp_file, stub_res, sfh_model="delayed")
        expected = DustAttenuation.from_calzetti(f.ssp.wave, 0.15,
                                                 anchor=5500.0)
        np.testing.assert_allclose(f._dust._curve, expected._curve,
                                   rtol=1e-12)
        # Anchored: the curve is 1 at the 5500 A normalization wavelength.
        i55 = np.argmin(np.abs(f.ssp.wave - 5500.0))
        assert f.ssp.wave[i55] == 5500.0
        assert f._dust._curve[i55] == pytest.approx(1.0, rel=1e-12)

    def test_likelihood_and_model(self, synth_ssp_file):
        f = self._make(synth_ssp_file)
        assert isinstance(f.likelihood, JAXLikelihood)
        assert isinstance(f.model, ModelComponents)
        assert f.likelihood is f._likelihood

    def test_emission_mask_kwarg(self, synth_ssp_file):
        f = self._make(synth_ssp_file, emission_mask=[(5500, 5600)])
        w, m = f._wave_obs, f._obs_mask
        inside = (w >= 5500) & (w <= 5600)
        assert inside.sum() > 0
        assert not np.any(m[inside])
        assert np.all(m[(w > 4000) & (w < 5000)])

    def test_run_without_out_dir(self, synth_ssp_file):
        f = self._make(synth_ssp_file)
        res = f.run(n_live=20, num_delete=10, num_inner_steps=4, max_ncalls=400)
        assert isinstance(res, MCMCResult)

    def test_run_returns_result(self, synth_ssp_file, tmp_path):
        f = self._make(synth_ssp_file)
        res = f.run(out_dir=str(tmp_path / "state"), n_live=20,
                    num_delete=10, num_inner_steps=4, max_ncalls=600)
        assert isinstance(res, MCMCResult)
        assert {"t0", "tau", "logZsun"} <= set(res.bestfit)
        assert np.all(np.isfinite(res.posterior))

    def test_run_saves_state(self, synth_ssp_file, tmp_path):
        from bigspy.mcmc.sampler import load_state
        f = self._make(synth_ssp_file)
        out = tmp_path / "state"
        f.run(out_dir=str(out), n_live=20, num_delete=10, num_inner_steps=4,
              max_ncalls=400)
        st = load_state(str(out / "state.npz"))
        assert "posterior" in st and "logz" in st

    def test_run_with_fixed_prior(self, synth_ssp_file):
        """Fixed SFH param: excluded from posterior, present in bestfit,
        and model rebuilding works with the full parameter set."""
        from bigspy import FixedPrior
        f = self._make(synth_ssp_file)
        res = f.run(n_live=20, num_delete=10, num_inner_steps=4,
                    max_ncalls=400, priors={"tau": FixedPrior(3.0)})
        assert res.bestfit["tau"] == 3.0
        assert set(res.bestfit) == {"t0", "tau", "logZsun"}
        assert res.posterior.shape[1] == 2   # t0, logZsun
        assert len(res.bestfit_model()) == len(res.model.obs_wave)
