from __future__ import annotations
from abc import ABCMeta, abstractmethod

import numpy as np

from density_estimation.common import LLH_SCALING, Array1D
from density_estimation.core.dist import Distribution, FitData


class ModelSpec(metaclass=ABCMeta):
    """Abstract base class for model specifications.

    Implementations define the conditional mean/volatility dynamics and
    how to construct fit-time data for a given error distribution.

    Attributes:
        error_dist (type): Distribution class used to model standardized errors.
        bounds (Sequence[tuple[float, float]] | None): Optional parameter bounds
            used by optimizers.
        constraints (Sequence | None): Optional optimizer constraints.
    """

    def __init__(self, dist_class: Distribution):
        """Initialize the model specification.

        Args:
            dist_class (Distribution): Distribution class that will be
                instantiated with a ``FitData`` object during
                fitting/scoring.
        """
        self.error_dist = dist_class
        self.bounds = None
        self.constraints = None

    @abstractmethod
    def __call__(self, data: np.ndarray, x: Array1D) -> np.ndarray:
        """Compute the model-implied series for the given parameters.

        Args:
            data (np.ndarray): Input data used by the model.
            x (Array1D): Parameter vector.

        Returns:
            np.ndarray: Model-implied series (e.g., mean/volatility output).
        """
        raise NotImplementedError

    @abstractmethod
    def make_initial_guess(self, data) -> np.ndarray:
        """Provide a default parameter initialization.

        Returns:
            np.ndarray: Initial parameter vector.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def base_step(self) -> np.ndarray:
        """Provide base step sizes for numerical differentiation.

        Returns:
            np.ndarray: Step sizes aligned with the parameter vector.
        """
        raise NotImplementedError

    @abstractmethod
    def make_fit_data(self, data: np.ndarray, x: Array1D) -> FitData:
        """Construct fit data required by the error distribution.

        Args:
            data (np.ndarray): Input data used by the model.
            x (Array1D): Parameter vector.

        Returns:
            FitData: Residuals, volatility, and optional shape parameters.
        """
        raise NotImplementedError

    def score(self, x: Array1D, *args) -> Array1D:
        """Compute per-observation negative log-density scores.

        Args:
            x (Array1D): Parameter vector.
            *args: Positional arguments where the first item is the data
                array used by ``make_fit_data``.

        Returns:
            Array1D: Negative log-density contributions for each observation.
        """
        data = args[0]
        fit_data = self.make_fit_data(data, x)
        error_dist = self.error_dist(fit_data)
        return -(np.log(error_dist.pdf()) - np.log(fit_data.sigma))

    def fitness(self, x: Array1D, *args) -> float:
        """Compute the scaled negative log-likelihood objective.

        Args:
            x (Array1D): Parameter vector.
            *args: Positional arguments where the first item is the data
                array used by ``make_fit_data``.

        Returns:
            float: Scaled negative log-likelihood for optimization.
        """
        data = args[0]
        fit_data = self.make_fit_data(data, x)
        error_dist = self.error_dist(fit_data)
        llh = error_dist.llh() - np.sum(np.log(fit_data.sigma))
        return -llh * LLH_SCALING
