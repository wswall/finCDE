from abc import ABCMeta, abstractmethod

from numba import njit, f8, prange
import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import tanhsinh
from scipy.optimize import Bounds
from scipy.special import gamma, gammaln
from scipy.stats import laplace, norm, t, rv_continuous

from density_estimation.common import sumjit


@njit(f8(f8, f8[:]), fastmath=True)
def _norm_llh(c, z_std):
    """Numba-accelerated log-likelihood for normal distribution."""
    acc = 0.0
    N = z_std.shape[0]
    for i in prange(N):
        acc += z_std[i] ** 2
    return -0.5 * (c + acc)


@njit(f8(f8, f8[:]), fastmath=True)
def _laplace_llh(c, z_std):
    """Numba-accelerated log-likelihood for Laplace distribution."""
    acc = 0.0
    N = z_std.shape[0]
    for i in prange(N):
        acc += np.abs(z_std[i])
    return -c - acc


@njit(f8(f8, f8, f8[:]), fastmath=True)
def _t_llh(c, nu, z_std):
    """Numba-accelerated log-likelihood for Student's t distribution."""
    acc = 0.0
    div = nu - 2.0
    N = z_std.shape[0]
    for i in prange(N):
        acc += np.log(1.0 + (z_std[i] ** 2) / div)
    return c - 0.5 * (nu + 1.0) * acc


class ConditionalDistribution(metaclass=ABCMeta):
    """Base class for conditional distributions"""

    def __init__(self, params: dict[str, float] | None):
        self.params = params

    @abstractmethod
    def ppf(self, alpha):
        raise NotImplementedError

    @abstractmethod
    def pdf(self, z_std):
        raise NotImplementedError

    @abstractmethod
    def llh(self, z_std):
        raise NotImplementedError

    def _q_score(self, alpha, y, mu, sigma):
        quantile = mu + sigma * self.ppf(alpha)
        return ((y <= quantile) - alpha) * (quantile - y)

    def crps(
        self,
        y: np.ndarray[float],
        residuals: np.ndarray[float],
        sigma: np.ndarray[float],
        tol: float = 1e-10,
        maxlevel: int = 12,
    ) -> np.ndarray[float]:
        """Compute the Continuous Ranked Probability Score

        Given arrays of observations and corresponding residuals and 
        standard deviations, computes the Continuous Ranked Probability
        Score (CRPS). This method uses the tanh-sinh quadrature method
        for numerical integration of the quantile score over the
        interval [0, 1].
        
        Args:
            y (np.ndarray[float]): Observed values at each time t.
            residuals (np.ndarray[float]): Residuals at each time t.
            sigma (np.ndarray[float]): Standard deviations at each time t.
            tol (float, optional): Absolute tolerance for numerical 
                integration. Default is 1e-10.
            maxlevel (int, optional): Maximum level of refinement for
                numerical integration. Default is 12.
        
        Returns:
            np.ndarray[float]: Array of CRPS values for each observation.
        """
        t = len(y)
        assert t == len(residuals) == len(sigma)
        mu = y + residuals
        result = tanhsinh(
            self._q_score,
            np.zeros(t),
            np.ones(t),
            args=(y, mu, sigma),
            atol=tol,
            maxlevel=maxlevel,
        )
        return 2.0 * result.integral


class Generic(ConditionalDistribution):
    """Base class for symmetric distributions"""

    def __init__(self, params: dict[str, float] | None, base_dist: rv_continuous):
        super().__init__(params)
        self.base_dist = base_dist

    def ppf(self, alpha: ArrayLike, scale: ArrayLike = 1.0) -> ArrayLike:
        return self.base_dist.ppf(alpha, scale=scale)

    def pdf(self, z_std: ArrayLike) -> ArrayLike:
        return self.base_dist.pdf(z_std)


class Normal(Generic):
    """Normal distribution for use with standardized residuals"""

    name = "normal"
    n_params = 0
    bounds = Bounds(lb=[], ub=[])

    def __init__(self):
        super().__init__(None, norm)

    def llh(self, z_std: np.ndarray[float]) -> float:
        c = z_std.size * np.log(2.0 * np.pi)
        return _norm_llh(c, z_std)


class Laplace(Generic):
    """Laplace distribution for use with standardized residuals"""

    name = "laplace"
    n_params = 0
    bounds = Bounds(lb=[], ub=[])

    def __init__(self):
        super().__init__(None, laplace)

    def llh(self, z_std: np.ndarray[float]) -> float:
        c = z_std.size * np.log(2.0)
        return _laplace_llh(c, z_std)


class StudentT(Generic):
    """Student's t distribution for use with standardized residuals"""

    name = "t"
    n_params = 1
    bounds = Bounds(lb=[2.0 + 1e-8], ub=[60.0])

    def __init__(self, nu):
        params = {"tailweight": np.float64(nu)}
        super().__init__(params, t)

    def ppf(self, alpha: ArrayLike, scale: float = 1.0) -> ArrayLike:
        return t.ppf(alpha, df=self.params["tailweight"], scale=scale)

    def pdf(self, z_std: ArrayLike, scale: ArrayLike = 1.0) -> ArrayLike:
        return t.pdf(z_std, df=self.params["tailweight"], scale=scale)

    def llh(self, z_std: np.ndarray[float]) -> float:
        nu = self.params["tailweight"]
        c = z_std.size * (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - np.log(np.pi * (nu - 2.0)) / 2.0
        )
        return _t_llh(c, nu, z_std)


class GenericSkewed(ConditionalDistribution):
    """Base class for skewed distributions

    Uses the Wurtz et al. reparametrization of Fernandez and Steel to
    skew a symmetric distribution. The skewness parameter must be > 0,
    with a value of 1 yielding the symmetric distribution. Density, 
    quantile and likelihood are calculated as transformations of their
    outputs from the underlying symmetric distribution.
    """

    def __init__(self, params: dict[str, float] | None, base_dist: Generic, m1: float):
        super().__init__(params)
        self.base_dist = base_dist
        self.mu_xi = m1 * (self.params["skew"] - 1.0 / self.params["skew"])
        xi_sq = self.params["skew"] ** 2.0
        m1_sq = m1**2.0
        self.sigma_xi = np.sqrt(
            (1.0 - m1_sq) * (xi_sq + 1.0 / xi_sq) + 2.0 * m1_sq - 1.0
        )
        self.g = 2.0 / (self.params["skew"] + 1.0 / self.params["skew"])

    def _skew_z_std(self, z_std: ArrayLike) -> ArrayLike:
        z_xi = z_std * self.sigma_xi + self.mu_xi
        return z_xi / self.params["skew"] ** np.sign(z_xi)

    def ppf(self, alpha: ArrayLike) -> ArrayLike:
        z = alpha - (1.0 / (1.0 + self.params["skew"] ** 2.0))
        Xi = self.params["skew"] ** np.sign(z)
        alpha_skew = (np.heaviside(z, 0.0) - np.sign(z) * alpha) / (self.g * Xi)
        ppf = self.base_dist.ppf(alpha_skew, scale=Xi)
        return (-np.sign(z) * ppf - self.mu_xi) / self.sigma_xi

    def pdf(self, z_std: ArrayLike) -> ArrayLike:
        z_skew = self._skew_z_std(z_std)
        density = self.base_dist.pdf(z_skew)
        return self.g * density * self.sigma_xi

    def llh(self, z_std: np.ndarray[float]) -> float:
        z_skew = self._skew_z_std(z_std)
        return z_std.size * np.log(self.g * self.sigma_xi) + self.base_dist.llh(z_skew)


class CondSNorm(GenericSkewed):
    """Skewed normal distribution for use with standardized residuals"""

    name = "skewnorm"
    n_params = 1
    bounds = Bounds(lb=[1e-8], ub=[np.inf])

    def __init__(self, xi: float):
        params = {"skew": xi}
        m1 = 2.0 / np.sqrt(2.0 * np.pi)
        super().__init__(params, Normal(), m1)


class CondST(GenericSkewed):
    """Skewed t distribution for use with standardized residuals"""

    name = "skewt"
    n_params = 2
    bounds = Bounds(lb=[1e-8, 2 + 1e-8], ub=[np.inf, 60.0])

    def __init__(self, xi: float, nu: float):
        params = {"skew": xi, "tailweight": nu}
        betafn = (gamma(0.5) / gamma(0.5 + nu / 2.0)) * gamma(nu / 2.0)
        m1 = 2.0 * np.sqrt(nu - 2.0) / (nu - 1.0) / betafn
        super().__init__(params, StudentT(nu), m1)


class CondSLap(GenericSkewed):
    """Skewed Laplace distribution for use with standardized residuals"""

    name = "skewlap"
    n_params = 1
    bounds = Bounds(lb=[1e-8], ub=[np.inf])

    def __init__(self, xi: float):
        params = {"skew": xi}
        m1 = np.sqrt(0.5)
        super().__init__(params, Laplace(), m1)


class CondJsu(ConditionalDistribution):
    """Johnson's SU distribution for use with standardized residuals"""

    name = "jsu"
    n_params = 2
    bounds = Bounds(lb=[-20, 0.1], ub=[20, 10])

    def __init__(self, skew: float, tailweight: float):
        params = {"skew": skew, "tailweight": tailweight}
        super().__init__(params)
        w = np.exp(1.0 / tailweight**2.0)
        W = skew / tailweight
        self.scale = np.sqrt(2.0 / ((w - 1.0) * (w * np.cosh(2.0 * W) + 1.0)))
        self.loc = self.scale * np.sqrt(w) * np.sinh(W)

    def pdf(self, z_std: ArrayLike) -> ArrayLike:
        z = (z_std - self.loc) / self.scale
        Z = self.params["skew"] + self.params["tailweight"] * np.log(
            z + np.sqrt(z**2.0 + 1.0)
        )
        num = np.exp(-0.5 * Z**2.0) * self.params["tailweight"]
        denom = self.scale * np.sqrt(2.0 * np.pi * (1.0 + z**2.0))
        return num / denom

    def ppf(self, alpha: ArrayLike) -> ArrayLike:
        z = np.sinh(
            (1.0 / self.params["tailweight"]) 
            * (norm.ppf(alpha) + self.params["skew"])
        )
        return self.loc + self.scale * z

    def llh(self, z_std: np.ndarray[float]) -> float:
        return sumjit(np.log(self.pdf(z_std)))
