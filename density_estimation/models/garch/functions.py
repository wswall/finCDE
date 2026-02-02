from numba import njit, f8
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def calc_ar_1(returns, y_init, mu, phi, *args):
    output = np.zeros(returns.shape[0], dtype=np.float64)
    output[0] = returns[0] - mu - phi * y_init
    output[1:] = returns[1:] - mu - phi * returns[:-1]
    return output


def calc_ar_m(returns, y_init, mu, phi, *args):
    output = np.zeros(returns.shape[0], dtype=np.float64)
    m = phi.size
    output[:m] = returns[:m] - mu - phi.sum() * y_init
    windowed = sliding_window_view(returns[:-1], m)[:, ::-1]
    output[m:] = returns[m:] - mu - np.sum(windowed * phi, axis=1)
    return output


@njit(f8[:](f8[:], f8, f8, f8[:], f8[:]), fastmath=True)
def calc_arma_11(returns, y_init, mu, phi, theta):
    phi, theta = phi[0], theta[0]
    output = np.zeros(returns.shape[0], dtype=np.float64)
    output[0] = returns[0] - mu - phi * y_init
    for i in range(1, returns.shape[0]):
        output[i] = returns[i] - mu - phi * returns[i - 1] - theta * output[i - 1]
    return output


@njit(f8[:](f8[:], f8, f8, f8[:], f8[:]), fastmath=True)
def calc_arma_mn(returns, y_init, mu, phi, theta):
    max_lag = max(phi.size, theta.size)
    output = np.zeros(returns.shape[0], dtype=np.float64)
    output[:max_lag] = returns[:max_lag] - mu - phi.sum() * y_init
    for i in range(max_lag, returns.shape[0]):
        y_pred = mu
        for n in range(phi.size):
            y_pred += phi[n] * returns[i - 1 - n]
        for n in range(theta.size):
            y_pred += theta[n] * output[i - 1 - n]
        output[i] = returns[i] - y_pred
    return output


@njit(f8[:](f8[:], f8, f8, f8[:], f8[:]), fastmath=True)
def calc_garch_11(residuals, var_init, omega, alpha, beta):
    alpha, beta = alpha[0], beta[0]
    output = np.zeros(residuals.shape[0], dtype=np.float64)
    output[0] = var_init
    for i in range(1, residuals.shape[0]):
        output[i] = omega + alpha * residuals[i - 1] ** 2 + beta * output[i - 1]
    return output


@njit(f8[:](f8[:], f8, f8, f8[:], f8[:]), fastmath=True)
def calc_garch_pq(residuals, var_init, omega, alpha, beta):
    max_lag = max(alpha.size, beta.size)
    output = np.zeros(residuals.shape[0], dtype=np.float64)
    output[:max_lag] = var_init
    for i in range(max_lag, residuals.shape[0]):
        var_pred = omega
        for n in range(alpha.size):
            var_pred += alpha[n] * residuals[i - 1 - n] ** 2
        for n in range(beta.size):
            var_pred += beta[n] * output[i - 1 - n]
        output[i] = var_pred
    return output
