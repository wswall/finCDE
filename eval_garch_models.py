from itertools import product
from pathlib import Path
import platform
import sys

import pandas as pd

from density_estimation import ModelFactory, Model
from density_estimation.models.garch import ArmaGarch
from density_estimation.common import get_data
from density_estimation.distributions import *


# Get count of CPUs available to this process
if platform.system() == "Windows":
    from psutil import Process
    PROC_COUNT = len(Process().cpu_affinity())
else:
    from os import sched_getaffinity
    PROC_COUNT = len(sched_getaffinity(0))


DATA_DIR = Path("data/TAQ")
OUTPUT_DIR = Path("trials/arma_garch/model_selection")


def make_garch_configs(data, arma_mn, garch_pq, dist_list, display=False):
    combos = product(arma_mn, garch_pq, dist_list)
    configs = []
    for combo in combos:
        configs.append({
            'data': data,
            'arma_order': combo[0],
            'garch_order': combo[1],
            'error_dist': combo[2],
            'display': display,
            'ftol': 1e-8,
            'maxiter': 500
        })
    return configs


if __name__ == "__main__":

    company = sys.argv[1]
    distributions = [Normal, Laplace, StudentT, SkewNorm, SkewLaplace, SkewT, JohnsonSU]
    arma_orders = list(product(range(1, 6), range(6)))
    garch_orders = list(product(range(1, 6), range(1, 6)))
    garch_factory = ModelFactory(ArmaGarch)
    history = get_data(DATA_DIR / f'{company}_300_cts.csv')
    T = round(history.shape[0] * .8)
    garch_configs = make_garch_configs(
        history[:T, 0],
        arma_orders,
        garch_orders,
        distributions
    )
    garch_models = garch_factory.build_many(garch_configs, proc_count=PROC_COUNT)
    results = []
    for model in garch_models:
        if not isinstance(model, Model):
            continue
        results.append([
            model.spec.arma_order,
            model.spec.garch_order,
            model.spec.error_dist.__name__,
            model.evaluate.log_likelihood,
            model.evaluate.aic(),
            model.evaluate.bic()
        ])

    cols = ['arma_order', 'garch_order', 'error_dist', 'llh', 'aic', 'bic']
    results_df = pd.DataFrame(results, columns=cols)
    results_df.to_csv(OUTPUT_DIR / f'{company}_garch_results.csv', index=False)
