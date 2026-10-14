"""Star Formation History (SFH) models.

Provides the abstract :class:`SFHBase` and the built-in
:class:`DelayedExponentialSFH` / :class:`DoublePowerLawSFH` models.

Each model implements two evaluations:

* ``evaluate(timegrid)``      -- NumPy, used for plotting and model building.
* ``evaluate_batch_jax(...)`` -- JAX-traceable batch evaluation, required for
  NSS sampling (the whole likelihood is JIT-compiled by blackjax).
"""

from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp


class SFHBase(ABC):
    """Abstract base class for star formation history models.

    Subclasses must provide the class attributes
    ``n_params`` / ``param_names`` / ``default_priors`` and implement both

    * ``evaluate(self, timegrid)`` -- NumPy SFR (plotting / model building), and
    * ``evaluate_batch_jax(timegrid, params_2d)`` -- JAX SFR (NSS sampling).
    """

    n_params = 0
    param_names = []
    default_priors = {}

    def __init__(self, **params):
        """Store model parameters (subclasses usually set attributes directly)."""
        pass

    @abstractmethod
    def evaluate(self, timegrid):
        """Compute SFR on *timegrid* (NumPy).

        Parameters
        ----------
        timegrid : ndarray
            SSP time grid (0 = early universe, max = present).

        Returns
        -------
        sfr : ndarray, shape ``timegrid.shape``
        """
        pass

    @classmethod
    def evaluate_batch_jax(cls, timegrid, params_2d):
        """JAX-traceable batch evaluation of the SFR.

        Required for NSS sampling, where the whole likelihood is JIT-compiled
        by blackjax.  Built-in models implement it; custom subclasses must
        override it to be usable with the NSS backend.

        Parameters
        ----------
        timegrid : jax.Array, shape (n_age,)
        params_2d : jax.Array, shape (N, n_params)

        Returns
        -------
        sfr : jax.Array, shape (N, n_age)
        """
        raise NotImplementedError(
            f"{cls.__name__} does not implement evaluate_batch_jax; "
            "NSS sampling requires a JAX-traceable SFH model."
        )


class DelayedExponentialSFH(SFHBase):
    """Delayed exponentially declining SFH.

        SFR(t) = 0                             ,  t <= t0
        SFR(t) = (t - t0) * exp(-(t - t0)/tau) ,  t > t0

    where t = max(timegrid) - timegrid (cosmic time since the Big Bang),
    t0 is the formation start time, and tau is the decay timescale.
    """

    n_params = 2
    param_names = ["t0", "tau"]
    default_priors = {}  # Set at module level below

    def __init__(self, t0, tau):
        self.t0, self.tau = float(t0), float(tau)

    def evaluate(self, timegrid):
        t = np.max(timegrid) - timegrid
        dt = t - self.t0
        return np.where(dt > 0, dt * np.exp(-dt / self.tau), 0.0)

    @classmethod
    def evaluate_batch_jax(cls, timegrid, params_2d):
        t0 = params_2d[:, 0][:, None]     # param_names order: ["t0", "tau"]
        tau = params_2d[:, 1][:, None]
        t = jnp.max(timegrid) - timegrid
        dt = t[None, :] - t0
        return jnp.where(dt > 0, dt * jnp.exp(-dt / tau), 0.0)

    def __repr__(self):
        return f"DelayedExpSFH(t0={self.t0:.2f}, tau={self.tau:.2f})"


class DoublePowerLawSFH(SFHBase):
    """Double power-law SFH.

        SFR(t) = 1 / ( (t/tau)^alpha + (t/tau)^(-beta) )

    where t = max(timegrid) - timegrid (cosmic time since the Big Bang), and
    tau, alpha, beta are free parameters.
    """

    n_params = 3
    param_names = ["tau", "alpha", "beta"]
    default_priors = {}  # Set at module level below

    def __init__(self, tau, alpha, beta):
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.beta = float(beta)

    def evaluate(self, timegrid):
        # Extreme posterior samples (large beta) can overflow x**(-beta); the
        # resulting inf yields SFR = 0, which is the intended behaviour.
        with np.errstate(over="ignore", invalid="ignore"):
            t = np.max(timegrid) - timegrid
            t = np.where(t <= 0, 1e-10, t)
            x = t / self.tau
            return 1.0 / (x ** self.alpha + x ** (-self.beta))

    @classmethod
    def evaluate_batch_jax(cls, timegrid, params_2d):
        tau = params_2d[:, 0][:, None]
        alpha = params_2d[:, 1][:, None]
        beta = params_2d[:, 2][:, None]
        t = jnp.max(timegrid) - timegrid
        t = jnp.where(t <= 0, 1e-10, t)
        x = t[None, :] / tau
        return 1.0 / (x ** alpha + x ** (-beta))

    def __repr__(self):
        return (f"DoublePowerLawSFH(tau={self.tau:.2f}, "
                f"alpha={self.alpha:.2f}, beta={self.beta:.2f})")


# Set default priors after class definitions (lazy import to avoid circular deps)
from .priors import UniformPrior, LogUniformPrior, GaussianPrior  # noqa: E402

DelayedExponentialSFH.default_priors = {
    "logZsun": UniformPrior(-2.5, 0.5),
    "t0":      UniformPrior(0.1, 13.5),
    "tau":     LogUniformPrior(0.1, 10.0),
}

DoublePowerLawSFH.default_priors = {
    "logZsun": UniformPrior(-2.5, 0.5),
    "tau":     LogUniformPrior(0.1, 13.0),
    "alpha":   LogUniformPrior(0.1, 1000.0),
    "beta":    LogUniformPrior(0.1, 1000.0),
}
