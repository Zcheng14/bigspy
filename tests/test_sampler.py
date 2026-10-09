"""Tests for bigspy.mcmc.sampler.NSSampler (blackjax Nested Slice Sampling).

Uses the synthetic SSP library (no LFS data) and a real JAXLikelihood.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # noqa: F841

import _synth  # noqa: E402
from bigspy.mcmc.ssp import SSPLibrary  # noqa: E402
from bigspy.mcmc.csp import CSPBuilder  # noqa: E402
from bigspy.mcmc.dust import DustAttenuation  # noqa: E402
from bigspy.mcmc.sfh import (DelayedExponentialSFH, DoublePowerLawSFH)  # noqa: E402
from bigspy.mcmc.likelihood_jax import JAXLikelihood  # noqa: E402
from bigspy.mcmc.priors import UniformPrior, LogUniformPrior, FixedPrior  # noqa: E402
from bigspy.mcmc.sampler import NSSampler  # noqa: E402


@pytest.fixture(scope="module")
def jlike(synth_ssp_file):
    ssp = SSPLibrary(synth_ssp_file)
    sfh0 = DelayedExponentialSFH(t0=1.0, tau=3.0)
    model = CSPBuilder(ssp).build(_synth.SSP_LOGZ_GRID[1], sfh0)
    dust = DustAttenuation.from_mode2(ssp.wave, 0.0, 0.0)
    return JAXLikelihood(ssp, ssp.wave, model, np.full(ssp.n_wave, 0.01),
                         np.ones(ssp.n_wave, dtype=bool), 0.0, 0.0, dust)


def _sampler(jlike, tmp_path, sfh_model="delayed", priors=None, **kw):
    kw.setdefault("n_live", 20)
    kw.setdefault("num_delete", 10)
    kw.setdefault("num_inner_steps", 4)
    return NSSampler(jlike, sfh_model, priors=priors, **kw)


class TestConstruction:
    def test_delayed_string(self, jlike, tmp_path):
        s = _sampler(jlike, tmp_path, "delayed")
        assert s.sfh_class is DelayedExponentialSFH
        assert s.param_names == ["t0", "tau", "logZsun"]
        assert s.like is jlike

    def test_dpl_string_and_class(self, jlike, tmp_path):
        assert _sampler(jlike, tmp_path, "dpl").sfh_class is DoublePowerLawSFH
        assert _sampler(jlike, tmp_path, DoublePowerLawSFH).sfh_class is DoublePowerLawSFH

    @pytest.mark.parametrize("bad", ["foo", None, DelayedExponentialSFH(t0=1, tau=1)])
    def test_invalid_raises(self, jlike, tmp_path, bad):
        with pytest.raises(ValueError, match="Unknown sfh_model"):
            _sampler(jlike, tmp_path, bad)

    def test_priors_merge_over_defaults(self, jlike, tmp_path):
        # Override only tau; the other priors come from the model defaults.
        s = _sampler(jlike, tmp_path, "delayed", priors={"tau": FixedPrior(5.0)})
        assert s.param_names == ["t0", "logZsun"]
        assert s._fixed_params == {"tau": 5.0}

    def test_unknown_prior_warns(self, jlike, tmp_path):
        import warnings
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _sampler(jlike, tmp_path, "delayed",
                     priors={"not_a_param": UniformPrior(0.0, 1.0)})
        assert any("unknown parameters" in str(w.message) for w in rec)

    def test_fixed_prior_excluded(self, jlike, tmp_path):
        priors = {"logZsun": UniformPrior(-2.5, 0.5),
                  "t0": UniformPrior(0.1, 13.5),
                  "tau": FixedPrior(5.0)}
        s = _sampler(jlike, tmp_path, "delayed", priors=priors)
        assert s.param_names == ["t0", "logZsun"]
        assert s._fixed_params == {"tau": 5.0}

    def test_default_priors_resolved(self, jlike, tmp_path):
        s = _sampler(jlike, tmp_path, "delayed")
        assert len(s._active_priors) == 3 and s._fixed_params == {}


class TestTransforms:
    def test_physical_batch_values(self, jlike, tmp_path):
        import jax.numpy as jnp
        s = _sampler(jlike, tmp_path, "delayed")   # [t0, tau, logZsun]
        cube = jnp.array([[0.0, 0.0, 1.0], [1.0, 0.5, 0.5]])
        phys = s._physical_batch(cube)
        np.testing.assert_allclose(np.asarray(phys["t0"]), [0.1, 13.5], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(phys["tau"]), [0.1, 1.0], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(phys["logZsun"]), [0.5, -1.0], rtol=1e-6)

    def test_logprior_box(self, jlike, tmp_path):
        import jax.numpy as jnp
        s = _sampler(jlike, tmp_path, "delayed")
        assert float(s._logprior(jnp.array([0.5, 0.5, 0.5]))) == 0.0
        assert not np.isfinite(float(s._logprior(jnp.array([1.5, 0.5, 0.5]))))
        assert not np.isfinite(float(s._logprior(jnp.array([-0.1, 0.5, 0.5]))))


class TestRun:
    def test_tiny_run_shapes_and_finiteness(self, jlike, tmp_path):
        s = _sampler(jlike, tmp_path, "delayed")
        s.run(max_ncalls=1000, seed=0)
        best = s.get_bestfit()
        post = s.get_posterior()
        assert best.shape == (3,)
        assert post.ndim == 2 and post.shape[1] == 3 and post.shape[0] > 50
        assert np.all(np.isfinite(post))
        assert np.isfinite(s.result["logz"])
        # Priors respected: logZsun in [-2.5, 0.5], t0 in [0.1, 13.5], tau in [0.1, 10]
        assert -2.5 <= best[2] <= 0.5
        assert 0.1 <= best[0] <= 13.5
        assert 0.1 <= best[1] <= 10.0

    def test_deterministic_with_seed(self, jlike, tmp_path):
        a = _sampler(jlike, tmp_path).run(max_ncalls=400, seed=0)
        b = _sampler(jlike, tmp_path).run(max_ncalls=400, seed=0)
        np.testing.assert_array_equal(a.get_bestfit(), b.get_bestfit())

    def test_save_and_load_state(self, jlike, tmp_path):
        from bigspy.mcmc.sampler import load_state
        s = _sampler(jlike, tmp_path, "delayed")
        s.run(max_ncalls=400, seed=0)
        p = str(tmp_path / "state.npz")
        s.save_state(p)
        st = load_state(p)
        np.testing.assert_array_equal(st["bestfit"], s.get_bestfit())
        np.testing.assert_array_equal(st["posterior"], s.get_posterior())
        np.testing.assert_array_equal(st["position_cube"], s._state.particles.position)
        assert st["params"].shape == (s._state.particles.position.shape[0], len(s.param_names))
        assert int(st["n_live"]) == 20
        assert list(st["param_names"]) == s.param_names
        assert float(st["logz"]) == s.result["logz"]
