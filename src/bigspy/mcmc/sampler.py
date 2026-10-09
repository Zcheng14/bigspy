"""NSS sampler -- blackjax Nested Slice Sampling backend.

Sampling runs on the unit hypercube with ``blackjax.nss``; the prior is
encoded entirely in the per-parameter transforms
(:meth:`Prior.transform_jax`), so the log-prior is a constant and the
evidence is computed with respect to the physical prior.
"""

import os
import warnings

import numpy as np
import jax
import jax.numpy as jnp

from .priors import FixedPrior


DEFAULT_N_LIVE = 1000
DEFAULT_NUM_DELETE = 100
DEFAULT_NUM_INNER_STEPS = 10

_SFH_ALIASES = {
    "dpl": "DoublePowerLawSFH",
    "double_power_law": "DoublePowerLawSFH",
    "delayed": "DelayedExponentialSFH",
}


class NSSampler:
    """Nested Slice Sampling wrapper (``blackjax.nss``).

    Parameters
    ----------
    likelihood : JAXLikelihood
        Prebuilt JAX likelihood.
    sfh_model : str or SFHBase subclass
        ``"dpl"`` (default model) / ``"delayed"`` / a custom ``SFHBase``
        subclass implementing ``evaluate_batch_jax``.
    priors : dict, optional
        Parameter name -> :class:`Prior`, merged over the model's
        ``default_priors`` (user entries override).
    n_live, num_delete, num_inner_steps : int
        NSS controls.  Defaults: ``1000 / 100 / 10``.
    """

    def __init__(self, likelihood, sfh_model, priors=None,
                 n_live=DEFAULT_N_LIVE, num_delete=DEFAULT_NUM_DELETE,
                 num_inner_steps=DEFAULT_NUM_INNER_STEPS):
        from . import sfh as _sfh

        self.like = likelihood
        self.n_live = int(n_live)
        self.num_delete = int(num_delete)
        self.num_inner_steps = int(num_inner_steps)

        # Resolve the SFH model.
        if isinstance(sfh_model, str):
            class_name = _SFH_ALIASES.get(sfh_model, sfh_model)
            try:
                self.sfh_class = getattr(_sfh, class_name)
            except AttributeError as exc:
                raise ValueError(f"Unknown sfh_model: {sfh_model}") from exc
        elif isinstance(sfh_model, type):
            self.sfh_class = sfh_model
        else:
            raise ValueError(f"Unknown sfh_model: {sfh_model}")

        # Resolve priors: model defaults overridden by user-provided entries.
        defaults = dict(getattr(self.sfh_class, "default_priors", {}))
        if priors:
            unknown = set(priors) - set(self.sfh_class.param_names) - {"logZsun"}
            if unknown:
                warnings.warn(
                    f"Ignoring priors for unknown parameters: {sorted(unknown)}",
                    stacklevel=2)
            defaults.update(priors)
        self.priors = defaults

        # Full parameter list: SFH params + optional logZsun.
        self._all_param_names = list(self.sfh_class.param_names)
        if "logZsun" in self.priors and "logZsun" not in self._all_param_names:
            self._all_param_names.append("logZsun")

        # Split fixed / active parameters.
        self._fixed_params = {}
        self._active_names = []
        self._active_priors = []
        self.param_names = []
        for name in self._all_param_names:
            prior = self.priors.get(name)
            if prior is None:
                raise ValueError(f"No prior specified for parameter '{name}'")
            if isinstance(prior, FixedPrior):
                self._fixed_params[name] = prior.value
            else:
                self._active_names.append(name)
                self._active_priors.append(prior)
                self.param_names.append(name)
        self._n_active = len(self._active_names)

        # Populated by ``run``.
        self.result = None
        self._best = None
        self._post = None
        self._state = None
        self._seed = None

    @property
    def fixed_params(self):
        """Parameters held fixed via :class:`FixedPrior` (name -> value).

        These are excluded from sampling (and from ``param_names`` /
        ``get_posterior()``) but reported by result containers so that
        model spectra can be rebuilt with the full parameter set.
        """
        return dict(self._fixed_params)

    # ── internal: cube -> physical ─────────────────────────────────
    def _physical_batch(self, cube):
        """Unit-cube batch (N, n_active) -> dict {name: (N,) physical}."""
        cube = jnp.atleast_2d(cube)
        n = cube.shape[0]
        phys = {}
        for i, name in enumerate(self._active_names):
            phys[name] = self._active_priors[i].transform_jax(cube[:, i])
        for name, value in self._fixed_params.items():
            phys[name] = jnp.full((n,), value)
        return phys

    def _loglike(self, cube):
        """Single-particle log-likelihood (JAX scalar) for blackjax.

        The unit-cube bounds are enforced by :meth:`_logprior`; here we only
        compute the likelihood.  ``loglike_batch`` already guards against
        non-finite chi-squared values from extreme parameters.
        """
        phys = self._physical_batch(jnp.atleast_2d(cube))
        log_z = phys["logZsun"][0] if "logZsun" in phys else jnp.asarray(0.0)
        sfh_params = jnp.stack([phys[name][0]
                                for name in self.sfh_class.param_names])
        ll = self.like.loglike_batch(jnp.atleast_1d(log_z),
                                     sfh_params[None, :], self.sfh_class)
        return ll[0]

    @staticmethod
    def _logprior(cube):
        """Uniform (constant) log-prior on the unit cube; -inf outside.

        blackjax uses this to keep the chain inside the prior support, so the
        box bounds are enforced here rather than in the likelihood.
        """
        u = jnp.asarray(cube)
        inside = jnp.all((u >= 0.0) & (u <= 1.0))
        return jnp.where(inside, jnp.asarray(0.0), jnp.asarray(-jnp.inf))

    # ── run ────────────────────────────────────────────────────────
    def run(self, max_ncalls=None, seed=0):
        """Run Nested Slice Sampling.

        Parameters
        ----------
        max_ncalls : int, optional
            Soft cap on the number of dead points.
        seed : int
            PRNG seed.

        The number of live points, deletion and inner steps are set on the
        sampler (``self.n_live`` / ``self.num_delete`` /
        ``self.num_inner_steps``).
        """
        import blackjax
        from blackjax.ns.utils import finalise, sample

        n_live = self.n_live
        n = self._n_active

        algo = blackjax.nss(
            logprior_fn=self._logprior,
            loglikelihood_fn=self._loglike,
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner_steps,
        )
        init_fn = jax.jit(algo.init)
        step_fn = jax.jit(algo.step)

        key = jax.random.PRNGKey(int(seed))
        key, sub = jax.random.split(key)
        particles = jax.random.uniform(sub, (n_live, n))

        live = init_fn(particles)
        step_fn(jax.random.PRNGKey(int(seed) + 1), live)  # warm-up / compile
        live = init_fn(particles)

        dead = []
        n_dead = 0
        while bool(live.integrator.logZ_live - live.integrator.logZ >= -3.0):
            key, sub = jax.random.split(key)
            live, info = step_fn(sub, live)
            dead.append(info)
            n_dead += self.num_delete
            if max_ncalls is not None and n_dead >= int(max_ncalls):
                break

        dead = finalise(live, dead)
        logz = float(live.integrator.logZ)
        self._state = dead
        self._seed = int(seed)

        # Best-fit (maximum-likelihood) point.
        ll = np.asarray(dead.particles.loglikelihood)
        pos = np.asarray(dead.particles.position)
        j = int(np.argmax(ll))
        phys = self._cube_to_physical(pos)
        self._best = np.array([float(phys[name][j]) for name in self.param_names])

        # Posterior samples (importance resampling).
        res = sample(jax.random.PRNGKey(int(seed) + 2), dead, 8000).position
        post_phys = self._cube_to_physical(np.asarray(res))
        self._post = np.column_stack([post_phys[name] for name in self.param_names])

        self.result = {
            "logz": logz,
            "ncall": int(dead.particles.loglikelihood.shape[0]),
            "n_live": n_live,
        }
        return self

    def _cube_to_physical(self, cube):
        """Unit-cube batch (N, n_active) NumPy -> dict {name: (N,) physical}."""
        cube = np.atleast_2d(cube)
        n = cube.shape[0]
        phys = {}
        for i, name in enumerate(self._active_names):
            phys[name] = np.asarray(self._active_priors[i].transform(cube[:, i]),
                                    dtype=float)
        for name, value in self._fixed_params.items():
            phys[name] = np.full(n, value, dtype=float)
        return phys

    # ── results ────────────────────────────────────────────────────
    def get_bestfit(self):
        """Best-fit parameter values (order: ``param_names``)."""
        return self._best

    def get_posterior(self):
        """Posterior samples, shape (N, len(param_names))."""
        return self._post

    # ── persistence ────────────────────────────────────────────────
    def save_state(self, path):
        """Persist the finalised nested-sampling state for later auditing.

        Writes an ``.npz`` with the dead + final live particles (unit-cube
        positions and their log-likelihoods), the run configuration, and the
        derived best-fit / posterior.  From this file the evidence, importance
        weights and posterior can be re-derived without re-running the
        likelihood (see :func:`load_state`).
        """
        if self._state is None:
            raise RuntimeError("call run() before save_state()")
        p = self._state.particles
        pos_cube = np.asarray(p.position)
        phys = self._cube_to_physical(pos_cube)
        params = np.column_stack([phys[name] for name in self.param_names])
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            params=params,                       # physical, per dead + live point
            position_cube=pos_cube,              # raw unit-cube positions
            loglikelihood=np.asarray(p.loglikelihood),
            loglikelihood_birth=np.asarray(p.loglikelihood_birth),
            logz=self.result["logz"],
            param_names=np.array(self.param_names, dtype=object),
            n_live=self.result["n_live"],
            num_delete=self.num_delete,
            num_inner_steps=self.num_inner_steps,
            seed=self._seed,
            bestfit=self._best,
            posterior=self._post,
        )


def load_state(path):
    """Load a state file written by :meth:`NSSampler.save_state`.

    Returns a dict with the stored arrays (``params`` (physical, per point),
    ``position_cube``, ``loglikelihood``, ``loglikelihood_birth``, ``logz``,
    ``param_names``, ``bestfit``, ``posterior`` and the run configuration).
    """
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}
