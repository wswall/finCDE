from math import log, sqrt

from numba import njit, f8, i8, prange
import numpy as np
import pandas as pd


OFFSET = 1e-8
LLH_SCALING = 1e-5


Array1D = np.ndarray[tuple[int,], np.dtype[np.number]]


@njit(f8(f8[:]), fastmath=True)
def sumjit(array):
    """Numba-accelerated sum for 1D arrays."""
    out = 0.0
    for x in array:
        out += x
    return out


@njit(f8(f8[:], f8), fastmath=True)
def qu_r(array, mean):
    """Numba-accelerated computation of 4th central moment."""
    out = 0.0
    for x in array:
        out += (x - mean) ** 4
    return out


def kurtosis(values: np.ndarray[float], mean: float, variance: float) -> float:
    """Compute excess kurtosis"""
    return qu_r(values, mean) / (len(values) * variance**2) - 3


@njit(f8[:](f8[:], i8), fastmath=True, parallel=True)
def calc_rv_d(array, M):
    """Numba-accelerated calculation of daily realized volatility."""
    T = array.shape[0] // M
    out = np.empty(T, dtype=np.float64)
    for t in prange(T):
        d_idx = t * M
        acc = 0.0
        for i in prange(1, M):
            acc += log(array[d_idx + i] / array[d_idx + i - 1]) ** 2
        out[t] = sqrt(acc)
    return out


@njit(f8[:, :](f8[:]), fastmath=True, parallel=True)
def calc_rv_mw(rv_d):
    """Numba-accelerated calculation of weekly/monthly realized volatility."""
    T = rv_d.shape[0]
    out = np.empty((T, 2), dtype=np.float64)
    for t in prange(22, T):
        acc = 0.0
        for m in prange(5):
            acc += rv_d[t - m]
        out[t, 0] = acc / 5.0
        for m in prange(5, 22):
            acc += rv_d[t - m]
        out[t, 1] = acc / 22.0
    return out


def get_data(taq_data_path: str, M: int = 79) -> np.ndarray[float]:
    """Load and preprocess TAQ data from CSV.

    Args:
        taq_data_path (str): Path to TAQ CSV file.
        M (int, optional): Number of intervals per day. Default is 79.

    Returns:
        np.ndarray: Preprocessed data array.
    """
    df = pd.read_csv(
        taq_data_path, engine="pyarrow", header=None, skiprows=1, usecols=(0, 2)
    )
    T = (df[0].diff() > 0.0).sum() + 1
    data = np.zeros((T, 4), dtype=np.float64)
    prices = df[1].values
    data[:, 0] = np.log(prices[M - 1 :: M] / prices[::M])
    data[:, 1] = calc_rv_d(prices, M)
    data[:, 2:] = calc_rv_mw(data[:, 1])
    return data[22:]
