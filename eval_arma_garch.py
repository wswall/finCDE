from multiprocessing import Pool
from pathlib import Path
import platform
import time
from itertools import product

import numpy as np
import pandas as pd

from density_estimation.dist import (
    Normal,
    StudentT,
    Laplace,
    CondSNorm,
    CondST,
    CondSLap,
    CondJsu,
)
from density_estimation.models.arma_garch import ArmaGarch
from density_estimation.common import get_data, kurtosis


# Get count of CPUs available to this process
if platform.system() == "Windows":
    from psutil import Process
    PROC_COUNT = len(Process().cpu_affinity())
else:
    from os import sched_getaffinity
    PROC_COUNT = len(sched_getaffinity(0))

DATA_DIR = "data/TAQ"
PHI_RANGE = range(1, 3)
THETA_RANGE = range(0, 3)
ALPHA_RANGE = range(1, 3)
BETA_RANGE = range(1, 3)


# Global variable to hold data used by worker processes
company_data = {}


def worker(spec: dict) -> list:
    phis, thetas, alphas, betas = spec["params"]
    model = ArmaGarch((phis, thetas), (alphas, betas), spec["dist"])
    data = company_data[spec["company"]]
    res = model.fit(data, spec["initial"], display=False)
    return [spec["company"], spec["dist"].name, str(spec["params"]), res.x, res.fun]


if __name__ == "__main__":

    param_combos = list(product(PHI_RANGE, THETA_RANGE, ALPHA_RANGE, BETA_RANGE))
    distributions = {
        "normal": {"cls": Normal, "shape_guess": []},
        "t": {"cls": StudentT, "shape_guess": []},
        "laplace": {"cls": Laplace, "shape_guess": []},
        "skewnorm": {"cls": CondSNorm, "shape_guess": [1.0]},
        "skewt": {"cls": CondST, "shape_guess": []},
        "skewlap": {"cls": CondSLap, "shape_guess": [1.0]},
        "jsu": {"cls": CondJsu, "shape_guess": [0.0, 1.0]},
    }

    sample_T = {}
    model_specs = []
    for csv_path in Path(DATA_DIR).iterdir():
        company = csv_path.stem.split("_")[0]
        data = get_data(csv_path)
        sample_T = data.shape[0] // 10 * 8

        sample = data[: sample_T, 0]
        company_data[company] = sample
        # Calculate initial guesses for parameters
        lr_mean = sample.mean()
        lr_var = sample.var(ddof=1)
        lr_kurt = kurtosis(sample, lr_mean, lr_var)
        distributions["t"]["shape_guess"] = [lr_kurt]
        distributions["skewt"]["shape_guess"] = [1.0, lr_kurt]

        for distribution, ddict in distributions.items():
            dist = ddict["cls"]
            for combo in param_combos:
                m, n, p, q = combo
                initial = np.array(
                    [
                        lr_mean,
                        *np.zeros(m + n),
                        lr_var,
                        *np.repeat(0.05 / p, p),
                        *np.repeat(0.9 / q, q),
                        *ddict["shape_guess"],
                    ]
                )
                model_specs.append(
                    {
                        "company": company,
                        "params": combo,
                        "dist": dist,
                        "initial": initial
                    }
                )

    print(f"Fitting {len(model_specs)} models on {PROC_COUNT} cores.")
    s = time.time()
    with Pool(PROC_COUNT) as p:
        results = p.map(worker, model_specs)
    e = time.time()
    print(f"Completed in {e - s:.2f} seconds.")

    columns = ["company", "dist", "order", "fit_params", "llh"]
    df = pd.DataFrame(results, columns=columns)
    k = df.fit_params.apply(len)
    T = df.company.apply(lambda x: company_data[x].shape[0])

    df["llh"] = -df.llh * 1e5
    df["aic"] = (2 * k - 2 * df.llh) / T
    df["bic"] = (np.log(T) * k - 2 * df.llh) / T
    df["fit_params"] = df.fit_params.apply(lambda x: np.array2string(x, precision=8))
    df[["llh", "aic", "bic"]] = df[["llh", "aic", "bic"]].round(3)

    write_order = ["company", "dist", "order", "llh", "aic", "bic", "fit_params"]
    df[write_order].to_csv("arma_garch_results.csv", index=False)
