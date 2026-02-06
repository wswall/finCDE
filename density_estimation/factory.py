from __future__ import annotations
from collections.abc import Sequence
from multiprocessing import Pool

import numpy as np
from scipy.optimize import OptimizeResult

from density_estimation.core import Model


class ModelFactory:
    """Factory for building models from a specification class.

    Handles spec initialization defaults and supports building many
    models in parallel from configuration dictionaries.

    Attributes:
        model_class (type): Model specification class to instantiate.
        defaults (dict): Default keyword arguments for spec construction.
    """

    def __init__(self, model_class, **kwargs):
        """Initialize the factory.

        Args:
            model_class (type): Model specification class to instantiate.
            **kwargs: Default keyword arguments for spec construction.
        """
        self.model_class = model_class
        self.defaults = kwargs or {}

    def _init_spec(self, **kwargs):
        kwargs.update(self.defaults)
        return self.model_class(**kwargs)

    @staticmethod
    def _extract_kw_args(kw_list, **kwargs):
        """Split kwargs into spec args and fit args.

        Args:
            kw_list (Sequence[str]): Keys to extract into a separate dict.
            **kwargs: Input keyword arguments.

        Returns:
            tuple[dict, dict]: Remaining kwargs and extracted kwargs.
        """
        extracted = {}
        for key in kw_list:
            if key in kwargs:
                extracted[key] = kwargs.pop(key)
        return kwargs, extracted

    def build(self, data, **kwargs):
        """Build and fit a model from data and configuration kwargs.

        Args:
            data (np.ndarray): Input data to fit the model to.
            **kwargs: Spec configuration plus optional fit args
                (display, ftol, maxiter).

        Returns:
            Model | OptimizeResult: Fitted model or raw optimizer result.
        """
        spec_args, fit_args = self._extract_kw_args(
            ["display", "ftol", "maxiter", "compute_derivs"], **kwargs
        )
        model_spec = self._init_spec(**spec_args)
        return Model.fit(data, model_spec, **fit_args)

    def _worker(self, config):
        data = config.pop("data")
        return self.build(data, **config)

    def build_many(self, config_list, proc_count=1):
        """Build and fit multiple models from a list of configs.

        Args:
            config_list (Sequence[dict]): Each dict must include ``data`` and
                any spec/fit args for ``build``.
            proc_count (int, optional): Number of processes to use. Defaults
                to 1 (no multiprocessing).

        Returns:
            list[Model | OptimizeResult]: List of fitted models or optimizer
                results corresponding to the input configs.
        """
        # Method to build multiple models in parallel
        if proc_count > 1:
            with Pool(proc_count) as p:
                results = p.map(self._worker, config_list)
        else:
            results = [self._worker(config) for config in config_list]
        return results
