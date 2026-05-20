"""bigspy — Bayesian Inference of Galaxy Spectra (Python)."""

from .specfit import SpecFit, SpecFitResult
from .mcmc.fitter import MCMCFitter, MCMCResult
from .mcmc.priors import UniformPrior, LogUniformPrior, GaussianPrior, FixedPrior
from .mcmc.sfh import SFHBase, DelayedExponentialSFH
from . import io, constants, mask, utils

__version__ = "0.1.0"
__all__ = [
    "SpecFit", "SpecFitResult",
    "MCMCFitter", "MCMCResult",
    "UniformPrior", "LogUniformPrior", "GaussianPrior", "FixedPrior",
    "SFHBase", "DelayedExponentialSFH",
    "io", "constants", "mask", "utils",
]
