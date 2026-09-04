"""Tests for bigspy.mcmc.sampler.UltraNestSampler with a stub likelihood.

No LFS data and no SSP library needed: chi2 comes from StubLikelihood's
analytic function with a known optimum (t0=5, logZ=-1).
"""

import numpy as np
import pytest

import _synth
from bigspy.mcmc.priors import UniformPrior, LogUniformPrior, FixedPrior
from bigspy.mcmc.sfh import DelayedExponentialSFH
from bigspy.mcmc.sampler import UltraNestSampler


@pytest.fixture()
def stub():
    return _synth.StubLikelihood(t0_true=5.0, logZ_true=-1.0)


@pytest.fixture()
def sampler(stub, tmp_path):
    return UltraNestSampler(stub, str(tmp_path / "chains"), "delayed")


class TestSamplerConstruction:
    def test_sfh_model_delayed_string(self, stub, tmp_path):
        s = UltraNestSampler(stub, str(tmp_path / "c1"), "delayed")
        assert s.sfh_class is DelayedExponentialSFH
        assert s.param_names == ["t0", "tau", "logZsun"]
        assert s.like is stub

    def test_sfh_model_class_ok(self, stub, tmp_path):
        s = UltraNestSampler(stub, str(tmp_path / "c2"), DelayedExponentialSFH)
        assert s.param_names == ["t0", "tau", "logZsun"]

    @pytest.mark.parametrize("bad", ["foo", None, DelayedExponentialSFH(t0=1, tau=1)])
    def test_sfh_model_invalid_raises(self, stub, tmp_path, bad):
        with pytest.raises(ValueError, match="Unknown sfh_model"):
            UltraNestSampler(stub, str(tmp_path / "c3"), bad)

    def test_default_priors_resolved(self, sampler):
        assert len(sampler._active_priors) == 3
        assert sampler._fixed_params == {}

    def test_missing_prior_raises(self, stub, tmp_path):
        priors = {"logZsun": UniformPrior(-2.5, 0.5), "t0": UniformPrior(0.1, 13.5)}
        with pytest.raises(ValueError, match="No prior specified"):
            UltraNestSampler(stub, str(tmp_path / "c4"), "delayed", priors=priors)

    def test_fixed_prior_excluded(self, stub, tmp_path):
        priors = {
            "logZsun": UniformPrior(-2.5, 0.5),
            "t0": UniformPrior(0.1, 13.5),
            "tau": FixedPrior(5.0),
        }
        s = UltraNestSampler(stub, str(tmp_path / "c5"), "delayed", priors=priors)
        assert s.param_names == ["t0", "logZsun"]
        assert s._fixed_params == {"tau": 5.0}


class TestPriorTransform:
    def test_numeric_values(self, sampler):
        # columns: [t0, tau, logZsun] in default order
        cube = np.array([[0.0, 0.5, 1.0],
                         [0.5, 0.5, 0.5]])
        out = sampler.prior_transform(cube)
        # tau is log-uniform: c=0.5 -> geometric mean of 0.1 and 10 -> 1.0
        np.testing.assert_allclose(out[0], [0.1, 1.0, 0.5], rtol=1e-12)
        np.testing.assert_allclose(out[1], [6.8, 1.0, -1.0], rtol=1e-12)

    def test_loguniform_edges(self, sampler):
        cube = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        out = sampler.prior_transform(cube)
        np.testing.assert_allclose(out[:, 1], [0.1, 10.0], rtol=1e-12)


class TestLoglike:
    def test_reassembly_order_and_sign(self, stub, tmp_path):
        priors = {
            "logZsun": UniformPrior(-2.5, 0.5),
            "t0": UniformPrior(0.1, 13.5),
            "tau": FixedPrior(5.0),
        }
        s = UltraNestSampler(stub, str(tmp_path / "c6"), "delayed", priors=priors)
        out = s.loglike(np.array([[4.0, -0.5]]))
        # Fixed tau must be re-inserted: stub receives sfh params [t0, tau]=[[4.0, 5.0]]
        np.testing.assert_allclose(stub.last_sfh_params, [[4.0, 5.0]])
        np.testing.assert_allclose(stub.last_logZ, [-0.5])
        chi2 = ((-0.5 - (-1.0)) / 0.05) ** 2 + ((4.0 - 5.0) / 0.2) ** 2
        np.testing.assert_allclose(out, -0.5 * chi2, rtol=1e-12)

    def test_vectorized_shapes(self, sampler):
        out = sampler.loglike(np.zeros((3, 3)))
        assert out.shape == (3,)
        out0 = sampler.loglike(np.zeros((0, 3)))
        assert len(out0) == 0


class TestRun:
    def test_tiny_real_run(self, stub, tmp_path):
        """Mini UltraNest run recovers the analytic optimum."""
        s = UltraNestSampler(stub, str(tmp_path / "chains"), "delayed")
        result = s.run(min_live_points=20, max_ncalls=800)
        best = s.get_bestfit()
        assert abs(best[0] - 5.0) < 0.5      # t0
        assert abs(best[2] - (-1.0)) < 0.15  # logZsun
        post = s.get_posterior()
        assert post.ndim == 2 and post.shape[1] == 3 and post.shape[0] > 50
        assert np.isfinite(result["logz"])

    def test_fake_result_dict(self, stub, tmp_path):
        s = UltraNestSampler(stub, str(tmp_path / "c7"), "delayed")
        s.result = {
            "maximum_likelihood": {"point": np.array([1.0, 2.0, 3.0])},
            "samples": np.arange(12.0).reshape(4, 3),
        }
        np.testing.assert_array_equal(s.get_bestfit(), [1.0, 2.0, 3.0])
        assert s.get_posterior().shape == (4, 3)
