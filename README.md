## Setup

```bash
pip install -r requirements.txt
```

To use `keras_tuner` with `tensorflow_probability` and `tf_keras`, copy
`assets/config.py` to `keras_tuner/src/backend` in your environment's
`python3.*/dist-packages` directory.

---

## Project Structure

```
density_estimation/          # Core package
├── __init__.py              # Public API re-exports
├── common.py                # Shared utilities, Numba helpers, data loading
├── distributions.py         # Concrete distribution classes
├── factory.py               # ModelFactory for batch model construction
├── core/
│   ├── dist.py              # Base distribution classes and FitData container
│   ├── model.py             # Model fitting via MLE (SLSQP)
│   ├── model_fit.py         # Post-estimation diagnostics (AIC, BIC, tests)
│   └── spec.py              # Abstract ModelSpec base class
└── models/
    ├── har.py               # HAR-RV model specification
    ├── garch/
    │   ├── arma_garch.py    # ARMA(m,n)-GARCH(p,q) model specification
    │   ├── functions.py     # Numba-accelerated ARMA/GARCH recursions
    └── ddnn/
        └── hypermodel.py    # Distributional deep neural network builder

build_parametric.py          # Script: build best parametric models per company
eval_garch_models.py         # Script: grid search over ARMA-GARCH specifications
DDNN.ipynb                   # Notebook: DDNN hyperparameter tuning & evaluation
```

---

## Quick Start

```python
import numpy as np
from density_estimation import Model, ArmaGarch, StudentT
from density_estimation.common import get_data

# Load TAQ data → (T, 4) array of [log_returns, rv_d, rv_w, rv_m]
data = get_data("...")
split = round(len(data) * 0.8)
train, test = data[:split], data[split:]

# --- Fit a single ARMA(1,1)-GARCH(1,1) with Student-t errors ---
spec = ArmaGarch(arma_order=(1, 1), garch_order=(1, 1), error_dist=StudentT)
model = Model.fit(train[:, 0], spec)

print("Log-likelihood:", model.evaluate.log_likelihood)
print("AIC:", model.evaluate.aic())
print("BIC:", model.evaluate.bic())

# Log score (negative log-density per observation)
log_score = model.calc_log_score(test[:, 0])
print("Log Score:", log_score.mean())

# CRPS via tanh-sinh quadrature
crps = model.calc_crps(test[:, 0])
print("CRPS:", crps.mean())
```

---

## The `density_estimation` Package

### Core Module

The core module (`density_estimation.core`) provides the foundational abstractions:

| Class | Description |
|---|---|
| **`ModelSpec`** | Abstract base class for model specifications. Subclasses define the conditional mean/variance dynamics, parameter bounds, constraints, and how to construct `FitData`. |
| **`Model`** | Fitted model container. Created by `Model.fit(data, spec)` using SLSQP optimization. Holds the specification, fitted parameters, derived `FitData`, and a `ModelFit` diagnostics object. |
| **`ModelFit`** | Post-estimation diagnostics: log-likelihood, AIC, BIC, standard errors (sandwich estimator), significance tests, Ljung-Box, ARCH-LM, and Jarque-Bera tests. |
| **`Distribution`** | Abstract base for error distributions. Subclasses implement `.pdf()`, `.ppf()`, and `.llh()`. |

#### Fitting Pipeline

1. A `ModelSpec` subclass is instantiated with distribution and order parameters.
2. `Model.fit(data, spec)` minimizes the scaled negative log-likelihood via SLSQP.
3. On success, a `Model` object is returned containing:
   - `spec` — the model specification
   - `data` — `FitData` with residuals, volatility, and standardized residuals
   - `parameters` — the MLE parameter vector
   - `evaluate` — a `ModelFit` with diagnostic methods


### Model Factory

`ModelFactory` provides a convenient interface for fitting multiple models of the same
class:

```python
from density_estimation import ModelFactory, ArmaGarch
from density_estimation.distributions import Normal, Laplace, StudentT

factory = ModelFactory(ArmaGarch, garch_order=(1, 1))
configs = []
for distribution in [Normal, Laplace, StudentT]:    
    configs.extend([
        {"data": returns, "arma_order": (1, 0), "error_dist": distribution},
        {"data": returns, "arma_order": (1, 1), "error_dist": distribution},
        {"data": returns, "arma_order": (2, 1), "error_dist": distribution},
    ])


# Fit models sequentially
models = factory.build_many(configs)

# Fit models in parallel (uses multiprocessing.Pool)
models = factory.build_many(configs, proc_count=3)
```

Each config dict must include a `"data"` key. All other keys are passed to the
model specification constructor. Optional keys `display`, `ftol`, `maxiter`, and
`compute_derivs` control the fitting behavior.

### Data Utilities

`density_estimation.common.get_data(taq_data_path, M=79)` loads a TAQ CSV file and 
computes log returns, daily realized volatility, weekly realized volatility, and 
monthly realized volatility.
---

## Workflows

### GARCH Model Selection

Performs a grid search over ARMA-GARCH specifications for a single company. Uses 
`ModelFactory.build_many()` with multiprocessing for parallel fitting. Results
(log-likelihood, AIC, BIC) are saved to CSV in `trials/arma_garch/model_selection/`.

```bash
python eval_garch_models.py <company>
# e.g., python eval_garch_models.py ibm
```


### Building Parametric Models

Fits the best ARMA-GARCH specifications (selected via prior grid search) and all
HAR-RV distribution variants for each company in `data/TAQ/`:

1. Reads pre-selected GARCH specifications from
   `trials/arma_garch/model_selection/garch_specs.json`.
2. Fits each specification plus ARMA(1,0)-GARCH(1,1) and ARMA(1,1)-GARCH(1,1)
   baselines across all seven distributions.
3. Fits HAR-RV models for all seven distributions.
4. Computes Jacobian and Hessian matrices for standard errors.
5. Serializes fitted models to `trials/arma_garch/` and `trials/har/`.

```bash
python build_parametric.py
```

### Distributional Deep Neural Network

**Notebook:** `DDNN.ipynb`

Tunes and evaluates DDNN models for each company and distribution:

1. **Data split:** 64% train / 16% validation / 20% test (the 80% in-sample split is
   further split 80/20 for train/validation).
2. **Hyperparameter search:** Uses `keras_tuner.RandomSearch` (30 trials) with early
   stopping (patience=5) over 50 epochs per trial.
3. **Model evaluation:** The best model per company/distribution is evaluated on the
   held-out test set using:
   - **Log score** (negative log-probability)
   - **CRPS** (continuous ranked probability score via tanh-sinh quadrature)

Best hyperparameters are saved to `results/nn_model_hp.json` and evaluation results
are aggregated into `results/nn_trials.csv`.
