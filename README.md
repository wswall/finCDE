## Setup

Install requirements

```
pip install -r requirements.txt
```

To use keras_tuner with tensorflow_probability and tf_keras, copy `assets/config.py` 
to `keras_tuner/src/backend` in your environment's `python3.*/dist-packages`

## Models

Parametric models are available for estimating conditional mean and variance of log 
returns. Additionally, a "distributional neural network" is available for estimating 
location, scale, and shape parameters of a conditional distribution.

### Parametric Models
A `ModelSpec` contains a model specification and the methods needed for fitting its 
parameters. The `ModelSpec` is subclassed for each parametric model class. The subclassed
model specifications are located in `models`. Currently, only ARMA(m,n)-GARCH(p,q) and 
HAR-RV models are implemented in.

The model parameters are fit by calling `Model.fit()` with the data to fit the model to
and a `ModelSpec`. The `Model` instance initialized with `.fit()` contains the data, the 
`ModelSpec`, the fitted parameters, and a class `ModelFit` which contains methods for 
evaluating the fitted model. 

The `ModelFactory` class provides a convenient way to fit multiple models of a single 
class. A `ModelFactory` is initialized with a model class and arguments to be used
as defaults for the model class. Fitted models can then be constructed by calling 
`ModelFactory.build()` with the data and any other arguments to be used for the 
specification. Multiple models can be constructed in parallel by passing a list of
dictionaries containing data and specification arguments to `ModelFactory.build_many()`,
with the keyword argument `cpu_count` set to the desired number of parallel processes. 

### Probabilistic Neural Network

`models.nn` adapts code from [Marcjasz et al. (2022)](
https://doi.org/10.48550/arXiv.2207.02832) to implement a neural network for density estimation. Input data is optionally passed to batch normalization and dropout layers then two fully connected hidden layers. Outputs from the second hidden layer are passed to a separate hidden layer for each parameter in the specified distribution. Outputs from the distribution parameter hidden layers are concatenated and passed to the distribution layer which outputs a TensorFlow distribution. See the [TensorFlow documentation](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions) for details on usage of distribution classes.

![](assets/ddnn.png)

The model is constructed by passing a Keras Hyperparameters object and a distribution type to the function `build_model()`. Hyperparameter values can be selected using keras_tuner. See example in `nn.ipynb`

## Distributions

Distributions for parametric models are implemented as classes in `dist.py`. The inputs 
to these distributions are assumed to be standardized residuals, so the above described
parametric models must apply the transformation:  

$$
f(r_t;\mu_t, \sigma_t, \eta) = \frac{1}{\sigma_t}\cdot f\left (\frac{r_t - \mu_t}{\sigma_t}; \eta \right)
$$  

Each distribution defines a density function `.pdf()`, quantile function `.ppf()`, and 
log-likelihood function `.llh()`

### Symmetric distributions

`dist.Normal`, `dist.Laplace`, `dist.StudentT` use density and quantile functions from 
distributions imported from `scipy.stats`. The `.llh()` methods call jit-compiled 
log-likelihood functions for faster MLE estimation.

### Skewed distributions

`dist.CondSNorm`, `dist.CondSLap`, `dist.CondST` add skewness to the corresponding 
symmetric distribution using [Wurtz et al. (2006)](https://api.semanticscholar.org/CorpusID:17916711) reparametrization of 
[Fernandez and Steel (1998)](https://doi.org/10.2307/2669632).

$$
\begin{aligned}
& f(z_t;\xi,\eta) = \frac{2\sigma_\xi}{\xi + \frac{1}{\xi}} \cdot f(z_{\xi t};\eta) 
\\\,\\
& z_{\xi t} = (\sigma_{\xi}z_t + \mu_{\xi})\xi^{sgn(\sigma_{\xi}z_t + \mu_{\xi})}
\\
& \mu_{\xi} = \text{M}_1(\xi - \frac{1}{\xi})
\\
&\sigma_{\xi} = (1 - \text{M}_1^2)(\xi^2 + \frac{1}{\xi^2}) + 2\text{M}_1^2 - 1
\end{aligned}
$$

`dist.CondJsu` uses density and quantile functions adapted from the R package `rugarch`
rather than equivalent methods from `scipy.stats`. A jit-compiled log-likelihood function
has not been implemented for this distribution, but is intended to be added in the 
future. Instead, `.llh()` sums the log of the values returned by its `.pdf()` method.  

$$
f(z_t; \xi, \lambda, \gamma, \delta) = 
    \frac{\delta}{
        \lambda\sqrt{1 + \Big(\frac{z_t - \xi}{\lambda}\Big)^2}
        } 
    \cdot 
    \phi \left[
        \gamma + \delta \sinh^{-1} \Big(\frac{z_t - \xi}{\lambda}\Big)
    \right]
$$  

$$
\begin{aligned}
    & \omega = \exp(\delta^{-2})\\
    &\Omega = \frac{\gamma}{\delta}
\end{aligned}
\quad
\begin{aligned}
    & \xi = -\lambda\omega^{\frac{1}{2}}\sinh\Omega \\
    & \lambda = \left[ \frac{1}{2}(\omega - 1)(\omega\cosh2\Omega + 1) \right]^{-\frac{1}{2}}
\end{aligned}
$$

<br>

## Future Development

Implement VaR and expected shortfall from estimated densities

Implement additional models  
- ARCD model of Hansen (1994)
- GARCHS model of Harvey and Siddique (1999)
- GARCHSK model of Leon et al. (2005)
- HEAVY model of Shephard and Shephard (2009)
- Other GARCH/HAR family models
- Other Neural network architectures (beneath param/distribution layers)
    - Kim and Won 2018
    - Benitez et al 2021
    - Barunik et al 2024
