from numba import njit, f8, i8, prange
import numpy as np
import pandas as pd


OFFSET = 1e-8
LLH_SCALING = 1e-5


@njit(f8(f8[:]))
def sumjit(array):
    out = 0.0
    for x in array:
        out += x
    return out


@njit(f8(f8[:], f8), fastmath=True)
def qu_r(array, mean):
    out = 0.0
    for x in array:
        out += (x - mean) ** 4
    return out


def kurtosis(values: np.ndarray[float], mean: float, variance: float) -> float:
    return qu_r(values, mean) / (len(values) * variance**2) - 3


@njit(f8[:](f8[:], f8[:], i8), fastmath=True)
def calc_rv_d(array, out, M):
    for t in prange(out.shape[0]):
        for i in prange(1, M):
            out[t] += (array[t * M + i] - array[t * M + i - 1]) ** 2
        out[t] = np.sqrt(out[t])
    return out


@njit(f8[:, :](f8[:], f8[:, :]), fastmath=True)
def calc_rv_mw(rv_d, out):
    T = rv_d.shape[0]
    for t in prange(22, T):
        array = rv_d[t - 22 : t]
        rv_w = sumjit(array[17:22])
        out[t, 0] = rv_w / 5.0
        out[t, 1] = (rv_w + sumjit(array[:17])) / 22.0
    return out


def get_data(taq_data_path: str, M: int = 78) -> np.ndarray[float]:
    df = pd.read_csv(
        taq_data_path, engine="pyarrow", header=None, skiprows=1, usecols=(0, 2)
    )
    T = (df[0].diff() > 0.0).sum() + 1
    data = np.zeros((T, 4), dtype=np.float64)
    log_prices = np.log(df[1].values)
    data[:, 0] = log_prices[M :: M + 1] - log_prices[: -M : M + 1]
    data[:, 1] = calc_rv_d(log_prices, np.zeros(T, dtype=np.float64), M + 1)
    data[:, 2:] = calc_rv_mw(data[:, 1], np.zeros((T, 2), dtype=np.float64))
    return data[22:]
