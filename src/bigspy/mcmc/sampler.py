# UltraNest wrapper with prior support
import os
import numpy as np

class UltraNestSampler:
    """UltraNest wrapper supporting Prior objects and vectorized likelihood."""

    def __init__(self, likelihood, out_dir, sfh_model, priors=None):
        """
        Parameters
        ----------
        likelihood : Likelihood
        out_dir : str
            UltraNest output directory.
        sfh_model : str or SFHBase subclass
            "delayed" for DelayedExponentialSFH, or a custom SFHBase subclass.
        priors : dict, optional
            Dict mapping param_name -> Prior object. If None, uses sfh_model.default_priors.
        """
        self.like = likelihood
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Resolve SFH model
        if isinstance(sfh_model, str) and sfh_model == "delayed":
            from .sfh import DelayedExponentialSFH
            self.sfh_class = DelayedExponentialSFH
        elif isinstance(sfh_model, type):
            self.sfh_class = sfh_model
        else:
            raise ValueError(f"Unknown sfh_model: {sfh_model}")

        # Resolve priors
        self.priors = priors if priors is not None else getattr(self.sfh_class, 'default_priors', {})

        # Build full parameter list (SFH params + optional logZsun)
        self._all_param_names = list(self.sfh_class.param_names)
        if 'logZsun' in self.priors and 'logZsun' not in self._all_param_names:
            self._all_param_names.append('logZsun')
        
        # Separate active (non-Fixed) and fixed parameters
        self._fixed_params = {}
        self._active_prior_names = []
        self._active_priors = []
        self.param_names = []
        for name in self._all_param_names:
            prior = self.priors.get(name)
            if prior is None:
                raise ValueError(f"No prior specified for parameter '{name}'")
            from .priors import FixedPrior
            if isinstance(prior, FixedPrior):
                self._fixed_params[name] = prior.value
            else:
                self._active_prior_names.append(name)
                self._active_priors.append(prior)
                self.param_names.append(name)

    def prior_transform(self, cube):
        """Prior transform for UltraNest. cube: (N, n_active) unit hypercube."""
        cube = np.atleast_2d(np.asarray(cube))
        n_active = len(self._active_priors)
        result = np.zeros((cube.shape[0], n_active))
        for i, prior in enumerate(self._active_priors):
            result[:, i] = prior.transform(cube[:, i])
        return result

    def _build_params_dict(self, active_params):
        """Insert fixed parameters back into full parameter dict."""
        params = dict(self._fixed_params)
        for i, name in enumerate(self._active_prior_names):
            params[name] = active_params[i]
        return params

    def loglike(self, params):
        """Vectorized log-likelihood for UltraNest.
        
        params: (N, n_active) — UltraNest passes batched parameters.
        """
        params = np.atleast_2d(np.asarray(params))
        N = len(params)
        
        if N == 0:
            return np.array([])
        
        # Build full parameter arrays including fixed params
        n_total = len(self._active_prior_names) + len(self._fixed_params)
        full_params = np.zeros((N, n_total))
        param_index = {}
        idx = 0
        for name in self._all_param_names:
            if name in self._fixed_params:
                full_params[:, idx] = self._fixed_params[name]
            else:
                active_idx = self._active_prior_names.index(name)
                full_params[:, idx] = params[:, active_idx]
            param_index[name] = idx
            idx += 1
        
        # Extract logZsun and SFH params
        logZsun_idx = param_index.get('logZsun', 0)
        logZsun_arr = full_params[:, logZsun_idx]
        
        # SFH param indices (all non-logZsun params)
        sfh_indices = [i for name, i in param_index.items() if name != 'logZsun']
        sfh_params_2d = full_params[:, sfh_indices]
        
        # Batch likelihood evaluation
        chi2 = self.like.call_batch(logZsun_arr, self.sfh_class, sfh_params_2d)
        return -0.5 * chi2

    def run(self, min_live_points=400, max_ncalls=None, frac_remain=0.5, **kwargs):
        """Run UltraNest sampling."""
        import ultranest
        self.sampler = ultranest.ReactiveNestedSampler(
            self.param_names, self.loglike, self.prior_transform,
            log_dir=self.out_dir, resume="overwrite", vectorized=True)
        self.result = self.sampler.run(
            min_num_live_points=min_live_points,
            max_ncalls=max_ncalls,
            frac_remain=frac_remain, **kwargs)
        self.sampler.print_results()
        return self.result

    def get_bestfit(self):
        """Return best-fit active parameter values."""
        return self.result["maximum_likelihood"]["point"]

    def get_posterior(self):
        """Return posterior samples."""
        return self.result["samples"]
