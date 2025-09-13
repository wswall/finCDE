from abc import ABCMeta, abstractmethod

from numba import njit, f8
import numpy as np
from scipy.integrate import tanhsinh
from scipy.optimize import Bounds
from scipy.special import gamma, gammaln
from scipy.stats import laplace, norm, t

from density_estimation.common import sumjit


@njit(f8(f8, f8[::1]))
def norm_llh(c, z_std):
    """Calculate log-likelihood for standard normal distribution"""
    acc = 0.0
    for x in z_std:
        acc += x * x
    return -0.5 * (c + acc)


@njit(f8(f8, f8[::1]))
def laplace_llh(c, z_std):
    """Calculate log-likelihood for laplace distribution"""
    acc = 0.0
    for x in z_std:
        acc += np.abs(x)
    return -c - acc


@njit(f8(f8, f8, f8[::1]))
def t_llh(c, nu, z_std):
    """Calculate log-likelihood for t distribution"""
    acc = 0.0
    for x in z_std:
        acc += np.log(1.0 + (x * x) / (nu - 2.0))
    return c - 0.5 * (nu + 1.0) * acc


class ConditionalDistribution(metaclass=ABCMeta):
    """Base class for conditional distributions"""

    def __init__(self, params):
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

    def _crps_func(self, alpha, y, mu, sigma):
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
        t = len(y)
        assert t == len(residuals) == len(sigma)
        mu = y + residuals
        result = tanhsinh(
            self._crps_func,
            np.zeros(t),
            np.ones(t),
            args=(y, mu, sigma),
            atol=tol,
            maxlevel=maxlevel,
        )
        return 2.0 * result.integral


class Generic(ConditionalDistribution):
    """Base class for symmetric distributions"""

    def __init__(self, params, base_dist):
        super().__init__(params)
        self.base_dist = base_dist

    def ppf(self, alpha, scale=1):
        return self.base_dist.ppf(alpha, scale=scale)

    def pdf(self, z_std):
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
        return norm_llh(c, z_std)


class Laplace(Generic):
    """Laplace distribution for use with standardized residuals"""

    name = "laplace"
    n_params = 0
    bounds = Bounds(lb=[], ub=[])

    def __init__(self):
        super().__init__(None, laplace)

    def llh(self, z_std: np.ndarray[float]) -> float:
        c = z_std.size * np.log(2.0)
        return laplace_llh(c, z_std)


class StudentT(Generic):
    """Student's t distribution for use with standardized residuals"""

    name = "t"
    n_params = 1
    bounds = Bounds(lb=[2.0 + 1e-8], ub=[60.0])

    def __init__(self, nu):
        params = {"tailweight": np.float64(nu)}
        super().__init__(params, t)

    def ppf(self, alpha, scale=1):
        return t.ppf(alpha, df=self.params["tailweight"], scale=scale)

    def pdf(self, z_std, scale=1):
        return t.pdf(z_std, df=self.params["tailweight"])

    def llh(self, z_std: np.ndarray[float]) -> float:
        nu = self.params["tailweight"]
        c = z_std.size * (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - np.log(np.pi * (nu - 2.0)) / 2.0
        )
        return t_llh(c, nu, z_std)


class GenericSkewed(ConditionalDistribution):
    """Base class for skewed distributions"""

    def __init__(self, params, base_dist, m1):
        super().__init__(params)
        self.base_dist = base_dist
        self.mu_xi = m1 * (self.params["skew"] - 1.0 / self.params["skew"])
        xi_sq = self.params["skew"] ** 2.0
        m1_sq = m1**2.0
        self.sigma_xi = np.sqrt(
            (1.0 - m1_sq) * (xi_sq + 1.0 / xi_sq) + 2.0 * m1_sq - 1.0
        )
        self.g = 2.0 / (self.params["skew"] + 1.0 / self.params["skew"])

    def _skew_z_std(self, z_std):
        z_xi = z_std * self.sigma_xi + self.mu_xi
        return z_xi / self.params["skew"] ** np.sign(z_xi)

    def ppf(self, alpha):
        z = alpha - (1.0 / (1.0 + self.params["skew"] ** 2.0))
        Xi = self.params["skew"] ** np.sign(z)
        alpha_skew = (np.heaviside(z, 0.0) - np.sign(z) * alpha) / (self.g * Xi)
        ppf = self.base_dist.ppf(alpha_skew, scale=Xi)
        return (-np.sign(z) * ppf - self.mu_xi) / self.sigma_xi

    def pdf(self, z_std):
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

    def __init__(self, xi):
        params = {"skew": np.float64(xi)}
        m1 = 2.0 / np.sqrt(2.0 * np.pi)
        super().__init__(params, Normal(), m1)


class CondST(GenericSkewed):
    """Skewed t distribution for use with standardized residuals"""

    name = "skewt"
    n_params = 2
    bounds = Bounds(lb=[1e-8, 2 + 1e-8], ub=[np.inf, 60.0])

    def __init__(self, xi, nu):
        params = {"skew": np.float64(xi), "tailweight": np.float64(nu)}
        betafn = (gamma(0.5) / gamma(0.5 + nu / 2.0)) * gamma(nu / 2.0)
        m1 = 2.0 * np.sqrt(nu - 2.0) / (nu - 1.0) / betafn
        super().__init__(params, StudentT(nu), m1)


class CondSLap(GenericSkewed):
    """Skewed Laplace distribution for use with standardized residuals"""

    name = "skewlap"
    n_params = 1
    bounds = Bounds(lb=[1e-8], ub=[np.inf])

    def __init__(self, xi):
        params = {"skew": np.float64(xi)}
        m1 = np.sqrt(0.5)
        super().__init__(params, Laplace(), m1)


class CondJsu(ConditionalDistribution):
    """Johnson's SU distribution for use with standardized residuals"""

    name = "jsu"
    n_params = 2
    bounds = Bounds(lb=[-20, 0.1], ub=[20, 10])

    def __init__(self, skew, tailweight):
        params = {"skew": np.float64(skew), "tailweight": np.float64(tailweight)}
        super().__init__(params)
        w = np.exp(1.0 / tailweight**2.0)
        W = skew / tailweight
        self.scale = np.sqrt(2.0 / ((w - 1.0) * (w * np.cosh(2.0 * W) + 1.0)))
        self.loc = self.scale * np.sqrt(w) * np.sinh(W)

    def pdf(self, z_std):
        z = (z_std - self.loc) / self.scale
        Z = self.params["skew"] + self.params["tailweight"] * np.log(
            z + np.sqrt(z**2.0 + 1.0)
        )
        num = np.exp(-0.5 * Z**2.0) * self.params["tailweight"]
        denom = self.scale * np.sqrt(2.0 * np.pi * (1.0 + z**2.0))
        return num / denom

    def ppf(self, alpha):
        z = np.sinh(
            (1.0 / self.params["tailweight"]) * (norm.ppf(alpha) + self.params["skew"])
        )
        return self.loc + self.scale * z

    def llh(self, z_std: np.ndarray[float]) -> float:
        return sumjit(np.log(self.pdf(z_std)))
