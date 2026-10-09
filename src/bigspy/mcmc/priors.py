"""Prior distributions for nested-sampling parameter transforms.

Defines an abstract :class:`Prior` and four concrete implementations that map
the unit hypercube [0, 1] to physical parameter space.

Each prior provides two transforms:

* ``transform(cube)``      -- NumPy version (tests, non-JAX callers).
* ``transform_jax(cube)``  -- JAX-traceable version (used by the NSS sampler,
  where the whole likelihood is JIT-compiled).

Both operate on 1-D arrays of unit-cube coordinates and return 1-D arrays of
physical values.
"""

from abc import ABC, abstractmethod

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import ndtri
from scipy.stats import norm


class Prior(ABC):
    """Abstract base class for prior transforms."""

    @abstractmethod
    def transform(self, cube):
        """Map unit-cube values (NumPy) to physical parameter values."""
        ...

    @abstractmethod
    def transform_jax(self, cube):
        """Map unit-cube values (JAX) to physical parameter values."""
        ...


class UniformPrior(Prior):
    """Uniform prior on [lo, hi]."""

    def __init__(self, lo, hi):
        self.lo = float(lo)
        self.hi = float(hi)

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return self.lo + c * (self.hi - self.lo)

    def transform_jax(self, cube):
        c = jnp.asarray(cube).ravel()
        return self.lo + c * (self.hi - self.lo)


class LogUniformPrior(Prior):
    """Log-uniform prior on [lo, hi] (i.e., uniform in log10 space)."""

    def __init__(self, lo, hi):
        self.lo = np.log10(float(lo))
        self.hi = np.log10(float(hi))

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return 10.0 ** (self.lo + c * (self.hi - self.lo))

    def transform_jax(self, cube):
        c = jnp.asarray(cube).ravel()
        return 10.0 ** (self.lo + c * (self.hi - self.lo))


class GaussianPrior(Prior):
    """Gaussian prior with mean *mu* and standard deviation *sigma*.

    Uses the inverse normal CDF (``scipy.stats.norm.ppf`` / ``ndtri``).
    """

    def __init__(self, mu, sigma):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return norm.ppf(c, loc=self.mu, scale=self.sigma)

    def transform_jax(self, cube):
        c = jnp.asarray(cube).ravel()
        return self.mu + self.sigma * ndtri(c)


class FixedPrior(Prior):
    """Fixed-value prior -- always returns *value* regardless of the cube."""

    def __init__(self, value):
        self.value = float(value)

    def transform(self, cube):
        c = np.asarray(cube).ravel()
        return np.full_like(c, self.value)

    def transform_jax(self, cube):
        c = jnp.asarray(cube).ravel()
        return jnp.full_like(c, self.value)
