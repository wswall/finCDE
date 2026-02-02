from itertools import product
from pathlib import Path
import platform
import pickle
import time

from density_estimation import ModelFactory
from density_estimation.models.garch import ArmaGarch
from density_estimation.models.har import HarRV
from density_estimation.common import get_data
from density_estimation.dist import *


# Get count of CPUs available to this process
if platform.system() == "Windows":
    from psutil import Process
    PROC_COUNT = len(Process().cpu_affinity())
else:
    from os import sched_getaffinity
    PROC_COUNT = len(sched_getaffinity(0))


DATA_DIR = Path("data/TAQ")
OUTPUT_DIR = Path("trials")


def make_garch_configs(data, arma_mn, garch_pq, dist_list, display=False):
    combos = product(arma_mn, garch_pq, dist_list)
    for combo in combos:
        yield {
            'data': data,
            'arma_order': combo[0],
            'garch_order': combo[1],
            'error_dist': combo[2],
            'display': display
        }


def make_har_configs(data, dist_list, display=False):
    for dist in dist_list:
        yield {
            'data': data,
            'error_dist': dist,
            'display': display
        }


if __name__ == "__main__":

    print("Building configs\n")
    distributions = [Normal, Laplace, StudentT, SkewNorm, SkewLaplace, SkewT, JohnsonSU]
    arma_orders = list(product(range(1, 4), range(4)))
    garch_orders = list(product(range(1, 4), range(1, 4)))

    garch_factory = ModelFactory(ArmaGarch)
    har_factory = ModelFactory(HarRV)
    for fp in DATA_DIR.glob('*_300_cts.csv'):
        company = fp.name.split('_')[0]
        print(f"Getting data for {company}")
        history = get_data(fp)
        garch_configs = make_garch_configs(
            history[:, 0],
            arma_orders,
            garch_orders,
            distributions
        )
        print(f"Starting GARCH builds with {PROC_COUNT} processes")
        s = time.time()
        garch_models = garch_factory.build_many(
            list(garch_configs),
            proc_count=PROC_COUNT
        )
        e = time.time()
        print(f"Completed in {e - s:.2f} seconds.")
        print(f"Saving models")
        trial = OUTPUT_DIR / 'arma_garch'
        trial.mkdir(parents=True, exist_ok=True)
        with open(trial / f'{company}_models.pkl', 'wb') as pfile:
            pickle.dump(garch_models, pfile)

        har_configs = make_har_configs(history, distributions)
        print(f"Starting HAR builds with {PROC_COUNT} processes")
        s = time.time()
        har_models = har_factory.build_many(
            list(har_configs),
            proc_count=PROC_COUNT
        )
        e = time.time()
        print(f"Completed in {e - s:.2f} seconds.")
        print(f"Saving models")
        trial = OUTPUT_DIR / 'har'
        trial.mkdir(parents=True, exist_ok=True)
        with open(trial / f'{company}_models.pkl', 'wb') as pfile:
            pickle.dump(har_models, pfile)
