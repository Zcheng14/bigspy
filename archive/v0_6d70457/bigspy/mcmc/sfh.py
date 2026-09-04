"""Star Formation History (SFH) models.

Provides an abstract base class SFHBase and the DelayedExponentialSFH
implementation for use in CSP (Composite Stellar Population) building.
"""

from abc import ABC, abstractmethod

import numpy as np


class SFHBase(ABC):
    """Abstract base class for star formation history models.

    Subclasses must implement:
        - n_params  (int): number of free parameters (class attribute)
        - param_names (list[str]): parameter name strings (class attribute)
        - evaluate(self, timegrid): compute SFR on the given time grid

    Class attributes:
        default_priors (dict): parameter name -> Prior object
        n_params (int): required on subclasses
        param_names (list[str]): required on subclasses
    """

    n_params = 0
    param_names = []
    default_priors = {}

    def __init__(self, **params):
        """Store model parameters.

        Concrete subclasses may call ``super().__init__(**kwargs)``
        or set attributes directly.
        """
        pass

    @abstractmethod
    def evaluate(self, timegrid):
        """Compute SFR on *timegrid*.

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
    def evaluate_batch(cls, timegrid, params_2d):
        """Evaluate the SFH for multiple parameter sets.
        
        Parameters
        ----------
        timegrid : ndarray
            SSP time grid.
        params_2d : ndarray, shape (N, n_params)
            Each row is one set of parameter values.
        
        Returns
        -------
        sfr : ndarray, shape (N, len(timegrid))
        """
        results = []
        for row in params_2d:
            kwargs = dict(zip(cls.param_names, row))
            sfh = cls(**kwargs)
            results.append(sfh.evaluate(timegrid))
        return np.array(results)


class DelayedExponentialSFH(SFHBase):
    """Delayed exponentially declining SFH.

        SFR(t) = 0                          ,  t <= t0
        SFR(t) = (t - t0) * exp(-(t - t0)/tau) ,  t > t0

    where t is SSP time (0 = early universe, max = present),
    t0 is the formation start time, and tau is the decay timescale.
    """

    n_params = 2
    param_names = ["t0", "tau"]
    default_priors = {}  # Set at module level below

    def __init__(self, t0, tau, age_universe=14.0):
        self.t0, self.tau, self.age_universe = float(t0), float(tau), float(age_universe)

    def evaluate(self, timegrid):
        t = np.max(timegrid) - timegrid
        dt = t - self.t0
        sfr = np.where(dt > 0, dt * np.exp(-dt / self.tau), 0.0)
        sfr[timegrid > self.age_universe] = 0.0
        return sfr

    def __repr__(self):
        return f"DelayedExpSFH(t0={self.t0:.2f}, tau={self.tau:.2f})"


# Set default priors after class definition (lazy import to avoid circular deps)
from .priors import UniformPrior, LogUniformPrior, GaussianPrior
DelayedExponentialSFH.default_priors = {
    "logZsun": UniformPrior(-2.5, 0.5),
    "t0":      UniformPrior(0.1, 13.5),
    "tau":     LogUniformPrior(0.1, 10.0),
}
