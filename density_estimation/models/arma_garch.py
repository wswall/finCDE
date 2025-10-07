from collections.abc import Sequence

from numba import njit, f8, prange
import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import Bounds, minimize, OptimizeResult

from density_estimation.common import OFFSET, LLH_SCALING, sumjit
from density_estimation.dist import ConditionalDistribution


@njit(f8[::1](f8[:], f8, f8[::1], f8[::1]), fastmath=True)
def calc_arma(returns, mu, phi, theta):
    """Compute ARMA residuals for a given time series.

    Args:
        returns (np.ndarray): Input time series.
        mu (float): Mean parameter.
        phi (np.ndarray): AR coefficients.
        theta (np.ndarray): MA coefficients.

    Returns:
        np.ndarray: Residuals from ARMA model.
    """
    y_init = mu / (1.0 - sumjit(phi) - sumjit(theta))
    residuals = np.empty(returns.shape[0], dtype=np.float64)
    residuals[0] = returns[0] - mu - sumjit(phi) * y_init
    p, q = phi.shape[0], theta.shape[0]
    for i in range(1, max(p, q)):
        ar = 0.0
        ma = 0.0
        for t in prange(p):
            if t < i:
                ar += phi[t] * returns[i - 1 - t]
            else:
                ar += phi[t] * y_init
        for t in prange(q):
            if t < i:
                ma += theta[t] * residuals[i - 1 - t]
        residuals[i] = returns[i] - mu - ar - ma
    for i in range(max(p, q), returns.shape[0]):
        ar = 0.0
        ma = 0.0
        for t in prange(p):
            ar += phi[t] * returns[i - 1 - t]
        for t in prange(q):
            ma += theta[t] * residuals[i - 1 - t]
        residuals[i] = returns[i] - mu - ar - ma
    return residuals


@njit(f8[::1](f8[:], f8, f8[::1], f8[::1]), fastmath=True)
def calc_garch(residuals, omega, alpha, beta):
    """Compute GARCH conditional variances for a given residual series.

    Args:
        residuals (np.ndarray): Residuals from ARMA or similar model.
        omega (float): Constant term.
        alpha (np.ndarray): ARCH coefficients.
        beta (np.ndarray): GARCH coefficients.

    Returns:
        np.ndarray: Conditional variances from GARCH model.
    """
    variance = np.empty(residuals.shape[0], dtype=np.float64)
    p, q = alpha.shape[0], beta.shape[0]
    variance[: max(p, q)] = omega / (1.0 - sumjit(alpha) - sumjit(beta))
    for i in range(max(p, q), variance.shape[0]):
        e = 0.0
        h = 0.0
        for t in prange(p):
            e += alpha[t] * residuals[i - 1 - t] ** 2
        for t in prange(q):
            h += beta[t] * variance[i - 1 - t]
        variance[i] = omega + e + h
    return variance


class ARMA:
    """Autoregressive Moving Average (ARMA) model.

    Args:
        m (int): Order of the autoregressive part.
        n (int): Order of the moving average part.

    Attributes:
        params (dict): Model parameters (phi, theta, mu).
        bounds (Bounds): Parameter bounds for optimization.
    """

    def __init__(self, m: int, n: int):
        self.params = {
            "phi": np.zeros(m, dtype=np.float64),
            "theta": np.zeros(n, dtype=np.float64),
            "mu": 0.0,
        }
        self.bounds = self._make_bounds()

    @property
    def m(self) -> int:
        """Return the order of the autoregressive part."""
        return self.params["phi"].size

    @property
    def n(self) -> int:
        """Return the order of the moving average part."""
        return self.params["theta"].size

    def _make_bounds(self) -> Bounds:
        n_params = self.m + self.n + 1
        high = np.ones(n_params) - OFFSET
        high[0] = np.inf
        return Bounds(lb=-high, ub=high)

    def set_params(self, params: dict[str, float]) -> None:
        """Update model parameters.

        Args:
            params (dict[str, float]): Dictionary of parameter names and values.
                Valid keys are 'phi', 'theta', and 'mu'.
        """
        self.params.update(params)

    def __call__(self, returns: np.ndarray) -> np.ndarray:
        """Apply the ARMA model to compute residuals.

        Args:
            returns (np.ndarray): Input time series of returns.

        Returns:
            np.ndarray: Residuals from the ARMA model.
        """
        return calc_arma(
            returns, self.params["mu"], self.params["phi"], self.params["theta"]
        )

    def _check_stationary(self) -> bool:
        char_poly = poly.Polynomial(-np.array([-1, *self.params["phi"]]))
        return np.all(np.abs(char_poly.roots()) > 1.0)


class GARCH:
    """Generalized Autoregressive Conditional Heteroskedasticity (GARCH) model.

    Args:
        p (int): Number of lagged residual terms (alpha).
        q (int): Number of lagged volatility terms (beta).

    Attributes:
        params (dict): Model parameters (alpha, beta, omega).
        bounds (Bounds): Parameter bounds for optimization.
    """

    def __init__(self, p: int, q: int):
        self.params = {
            "alpha": np.zeros(p, dtype=np.float64),
            "beta": np.zeros(q, dtype=np.float64),
            "omega": 0.0,
        }
        self.bounds = self._make_bounds()

    @property
    def p(self) -> int:
        """Return the number of lagged residual terms."""
        return self.params["alpha"].size

    @property
    def q(self) -> int:
        """Return the number of lagged volatility terms."""
        return self.params["beta"].size

    def _make_bounds(self) -> Bounds:
        n_params = self.params["alpha"].size + self.params["beta"].size + 1
        low, high = np.zeros(n_params), np.ones(n_params)
        low[0], high[0] = OFFSET, np.inf
        return Bounds(lb=low, ub=high)

    def __call__(self, residuals: np.ndarray) -> np.ndarray:
        """Apply the GARCH model to compute conditional variances.

        Args:
            residuals (np.ndarray): Residuals from ARMA or similar model.

        Returns:
            np.ndarray: Conditional variances from the GARCH model.
        """
        return calc_garch(
            residuals, self.params["omega"], self.params["alpha"], self.params["beta"]
        )

    def set_params(self, params: dict[str, float]) -> None:
        """Update model parameters.

        Args:
            params (dict): Dictionary of parameter names and values.
                Valid keys are 'alpha', 'beta', and 'omega'.
        """
        self.params.update(params)

    def _check_stationary(self) -> bool:
        alpha_poly = poly.Polynomial(-np.array([-1, *self.params["alpha"]]))
        beta_poly = poly.Polynomial(np.array([0, *self.params["beta"]]))
        char_poly = alpha_poly - beta_poly
        return np.all(np.abs(char_poly.roots()) > 1.0)

    def _check_m4(self):
        raise NotImplementedError(
            "Constraint for existence of 4th moment not implemented for higher order \
            GARCH models. Use simple.ArGarch or implement the constraint manually."
        )


class ArmaGarch:
    """ARMA(m,n)-GARCH(p,q) model with various error distribution

    Args:
        arma_mn (tuple[int, int]): The orders (m,n) of the ARMA model.
        garch_pq (tuple[int, int]): The orders (p,q) of the GARCH model.
        error_dist (ConditionalDistribution): The distribution of
            standardized residuals.

    Attributes:
        arma (ARMA): The ARMA model.
        garch (GARCH): The GARCH model.
        n_model_params (int): The number of parameters in the ARMA and
            GARCH models.
        error_dist (ConditionalDistribution): The distribution of
            standardized residuals.
    """

    def __init__(
        self,
        arma_mn: tuple[int, int],
        garch_pq: tuple[int, int],
        error_dist: ConditionalDistribution,
    ):
        self.arma = ARMA(*arma_mn)
        self.garch = GARCH(*garch_pq)
        self.n_model_params = 2 + sum(arma_mn) + sum(garch_pq)
        self.error_dist = error_dist

    @property
    def bounds(self) -> Bounds:
        """Get parameter bounds object for optimization.

        Returns:
            Bounds: Combined bounds for ARMA, GARCH, and error distribution parameters.
        """
        lbs = [self.arma.bounds.lb, self.garch.bounds.lb, self.error_dist.bounds.lb]
        ubs = [self.arma.bounds.ub, self.garch.bounds.ub, self.error_dist.bounds.ub]
        return Bounds(lb=np.concatenate(lbs), ub=np.concatenate(ubs))

    def _params_from_vec(self, x: Sequence[float]) -> dict[str, np.ndarray | float]:
        phi_bound = 1 + self.arma.m
        theta_bound = phi_bound + self.arma.n
        alpha_bound = 1 + theta_bound + self.garch.p
        return {
            "mu": x[0],
            "phi": x[1:phi_bound],
            "theta": x[phi_bound:theta_bound],
            "omega": x[theta_bound],
            "alpha": x[1 + theta_bound : alpha_bound],
            "beta": x[alpha_bound:],
        }

    def set_params(self, params: dict[str, np.ndarray | float]) -> None:
        """Update ARMA and GARCH model parameters.

        Args:
            params (dict): Dictionary of parameter names and values.
                Valid keys include ARMA parameters ('phi', 'theta', 'mu')
                and GARCH parameters ('alpha', 'beta', 'omega').
        """
        arma_params = {k: v for k, v in params.items() if k in self.arma.params}
        garch_params = {k: v for k, v in params.items() if k in self.garch.params}
        self.arma.set_params(arma_params)
        self.garch.set_params(garch_params)

    def _check_constraints(self) -> bool:
        if self.garch._check_stationary():
            return self.arma._check_stationary()
        return False

    def _init_dist(self, x: Sequence[float]) -> ConditionalDistribution:
        if self.error_dist.n_params:
            return self.error_dist(*x[self.n_model_params :])
        return self.error_dist()

    def _fitness(self, x: Sequence[float], *args: Any) -> float:
        model_params = self._params_from_vec(x[: self.n_model_params])
        self.set_params(model_params)
        if not self._check_constraints():
            return 1.0
        dist = self._init_dist(x)
        residuals = self.arma(args[0])
        sigma = np.sqrt(self.garch(residuals))
        sigma = np.maximum(sigma, np.finfo(np.float64).eps)
        llh = dist.llh(residuals / sigma) - sumjit(np.log(sigma))
        return -llh * LLH_SCALING

    def fit(
        self,
        returns: np.ndarray,
        initial_guess: np.ndarray,
        display: bool = True,
        ftol: float = OFFSET,
        maxiter: int = 100,
    ) -> OptimizeResult:
        """Fit the ARMA-GARCH model to financial returns data.

        Args:
            returns (np.ndarray): Time series of financial returns.
            initial_guess (np.ndarray): Initial parameter values for optimization.
            display (bool, optional): Whether to display optimization progress. Defaults to True.
            ftol (float, optional): Function tolerance for optimization convergence.
                Defaults to OFFSET.
            maxiter (int, optional): Maximum number of optimization iterations.
                Defaults to 100.

        Returns:
            OptimizeResult: Scipy optimization result object containing fitted parameters
                and optimization diagnostics.
        """
        res = minimize(
            self._fitness,
            initial_guess,
            args=(returns,),
            method="SLSQP",
            bounds=self.bounds,
            options={"disp": display, "ftol": ftol, "maxiter": maxiter},
        )
        if res.success:
            best_params = self._params_from_vec(res.x[: self.n_model_params])
            self.set_params(best_params)
            self.error_dist = self._init_dist(res.x)
        return res
