import json
from pathlib import Path
import platform
import pickle

from density_estimation import ModelFactory
from density_estimation.models import ArmaGarch, HarRV
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
OUTPUT_DIR = Path("trials")
DISTRIBUTIONS = [Normal, Laplace, StudentT, SkewNorm, SkewLaplace, SkewT, JohnsonSU]


def make_har_configs(data, display=False):
    configs = []
    for dist in DISTRIBUTIONS:
        configs.append({
            'data': data,
            'error_dist': dist,
            'display': display,
            'compute_derivs': True,
            'ftol': 1e-8,
            'maxiter': 500
        })
    return configs


def make_garch_configs(data, specs):
    configs = []
    dist_dict = {dist.__name__: dist for dist in DISTRIBUTIONS}

    for spec in specs:
        configs.append({
            'data': data,
            'arma_order': tuple(int(x) for x in spec['arma_order'].strip('()').split(',')),
            'garch_order': tuple(int(x) for x in spec['garch_order'].strip('()').split(',')),
            'error_dist': dist_dict[spec['error_dist']],
            'display': False,
            'compute_derivs': True,
            'ftol': 1e-8,
            'maxiter': 500
        })

    for dist in DISTRIBUTIONS:
        configs.append({
            'data': data,
            'arma_order': (1, 0),
            'garch_order': (1, 1),
            'error_dist': dist,
            'display': False,
            'compute_derivs': True,
            'ftol': 1e-8,
            'maxiter': 500
        })
        configs.append({
            'data': data,
            'arma_order': (1, 1),
            'garch_order': (1, 1),
            'error_dist': dist,
            'display': False,
            'compute_derivs': True,
            'ftol': 1e-8,
            'maxiter': 500
        })

    return configs


if __name__ == "__main__":

    with open('garch_specs.json', 'r') as jfile:
        garch_specs = json.load(jfile)

    garch_factory = ModelFactory(ArmaGarch)
    har_factory = ModelFactory(HarRV)

    for fp in DATA_DIR.glob('*_300_cts.csv'):
        company = fp.name.split('_')[0]
        history = get_data(fp)
        T = round(history.shape[0] * .8)

        garch_configs = make_garch_configs(history[:T, 0], garch_specs[company])
        garch_models = garch_factory.build_many(garch_configs,proc_count=PROC_COUNT)
        trial = OUTPUT_DIR / 'arma_garch'
        trial.mkdir(parents=True, exist_ok=True)
        with open(trial / f'{company}_models.pkl', 'wb') as pfile:
            pickle.dump(garch_models, pfile)

        har_configs = make_har_configs(history[:T])
        har_models = har_factory.build_many(har_configs,proc_count=PROC_COUNT)
        trial = OUTPUT_DIR / 'har'
        trial.mkdir(parents=True, exist_ok=True)
        with open(trial / f'{company}_models.pkl', 'wb') as pfile:
            pickle.dump(har_models, pfile)
