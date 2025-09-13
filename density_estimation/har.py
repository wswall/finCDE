from collections.abc import Sequence

import numpy as np

from density_estimation.common import sumjit, LLH_SCALING
from density_estimation.dist import ConditionalDistribution


def calc_har(rv_dmw: np.ndarray[float], params: Sequence[float]) -> np.ndarray[float]:
    c, beta_d, beta_w, beta_m = params
    return c + beta_d * rv_dmw[:, 0] + beta_w * rv_dmw[:, 1] + beta_m * rv_dmw[:, 2]


def har_fitness(
    x: Sequence[float], *args: np.ndarray[float] | ConditionalDistribution
) -> float:
    data = args[0]
    dist = args[1]
    returns, rv_dmw = data[1:, 0], data[:-1, 1:]
    sigma = calc_har(rv_dmw, x[:4])
    if dist.n_params:
        dist = dist(*x[4:])
    else:
        dist = dist()
    llh = dist.llh(returns / sigma) - sumjit(np.log(sigma))
    return -llh * LLH_SCALING
