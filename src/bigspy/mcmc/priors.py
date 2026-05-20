"""Prior distributions for MCMC / nested-sampling parameter transforms.

Defines an abstract Prior base class and four concrete implementations
that map the unit hypercube [0, 1]^n to physical parameter space.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import norm


class Prior(ABC):
    """Abstract base class for prior transforms.

    Each prior maps a unit-cube coordinate (or vector) to a physical
    parameter value.
    """

    @abstractmethod
    def transform(self, cube):
        """Map unit-cube values to physical parameter values.

        Parameters
        ----------
        cube : ndarray, shape (N,) or (N, 1)
            Unit-cube coordinates.

        Returns
        -------
        ndarray, shape (N,)
            Physical parameter values.
        """
        ...


class UniformPrior(Prior):
    """Uniform prior on [lo, hi]."""

    def __init__(self, lo, hi):
        self.lo = float(lo)
        self.hi = float(hi)

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return self.lo + c * (self.hi - self.lo)


class LogUniformPrior(Prior):
    """Log-uniform prior on [lo, hi] (i.e., uniform in log10 space)."""

    def __init__(self, lo, hi):
        self.lo = np.log10(float(lo))
        self.hi = np.log10(float(hi))

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return 10.0 ** (self.lo + c * (self.hi - self.lo))


class GaussianPrior(Prior):
    """Gaussian prior with mean *mu* and standard deviation *sigma*.

    Uses ``scipy.stats.norm.ppf`` to transform the unit cube.
    """

    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return norm.ppf(c, loc=self.mu, scale=self.sigma)


class FixedPrior(Prior):
    """Fixed-value prior -- always returns *value* regardless of cube."""

    def __init__(self, value):
        self.value = float(value)

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return np.full_like(c, self.value)
