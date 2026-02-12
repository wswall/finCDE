from keras_tuner.engine.hyperparameters import HyperParameters, HyperParameter
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as keras
from tf_keras.regularizers import L1, Regularizer

from density_estimation.models.pnn.dist_functions import *


ACTIVATIONS = ["sigmoid", "relu", "elu", "tanh", "softplus", "softmax"]
DISTRIBUTIONS = {
    "normal": {"params": ["loc", "scale"], "fun": make_normal_tfd},
    "laplace": {"params": ["loc", "scale"], "fun": make_laplace_tfd},
    "t": {"params": ["loc", "scale", "df"], "fun": make_t_tfd},
    "skewnorm": {"params": ["loc", "scale", "skewness"], "fun": make_skewnorm_tfd},
    "skewt": {"params": ["loc", "scale", "skewness", "df"], "fun": make_skewt_tfd},
    "jsu": {"params": ["loc", "scale", "skewness", "tailweight"], "fun": make_jsu_tfd}
}


def _choose_regularization(hp: HyperParameter, hp_name: str) -> Regularizer | None:
    """Select L1 regularization if enabled in hyperparameters.

    Args:
        hp (HyperParameter): Hyperparameter object.
        hp_name (str): Name of the regularization parameter.

    Returns:
        Regularizer or None: L1 regularizer if enabled, else None.
    """
    if hp.Boolean(f"{hp_name}_bool"):
        reg_val = hp.Float(hp_name, min_value=1e-5, max_value=1e1, sampling="log")
        return L1(reg_val)
    return None


def _make_dense(
    n: int,
    act: str = "linear",
    kreg: Regularizer | None = None,
    areg: Regularizer | None = None,
):
    """Create a Keras dense layer with optional regularization and activation.

    Args:
        n (int): Number of units.
        act (str): Activation function.
        kreg (Regularizer, optional): Kernel regularizer.
        areg (Regularizer, optional): Activity regularizer.

    Returns:
        keras.layers.Dense: Configured dense layer.
    """
    return keras.layers.Dense(
        n, activation=act, kernel_regularizer=kreg, activity_regularizer=areg
    )


def build_prob_nn(hp: HyperParameters, dist_name: str = "normal", input_dim: int = 3):
    """Builds a probabilistic neural network model using Keras and TensorFlow Probability.

    Args:
        hp: An instance of HyperParameters for managing hyperparameter tuning.
        dist_name (str): The name of the probability distribution to model the output.
            Must be a key in the DISTRIBUTIONS dictionary. Default is 'normal'.
        input_dim (int): The dimensionality of the input features. Default is 3.

    Returns:
        keras.Model: A compiled Keras model with a probabilistic output layer.
    """
    if not dist_name in DISTRIBUTIONS:
        raise ValueError(f"Distribution '{dist_name}' is not supported.")

    input_layer = keras.layers.Input(shape=(input_dim,))

    if hp.Boolean("batch_norm"):
        x = keras.layers.BatchNormalization()(input_layer)
    else:
        x = input_layer

    if hp.Boolean("dropout"):
        d = hp.Float("dropout_rate", min_value=0.1, max_value=0.9, step=0.1)
        x = keras.layers.Dropout(d)(x)

    n1 = hp.Int("neurons1", min_value=8, max_value=128, step=8)
    k1 = _choose_regularization(hp, "kreg1")
    a1 = _choose_regularization(hp, "areg1")
    act1 = hp.Choice("activation1", ACTIVATIONS)
    x = _make_dense(n1, act=act1, kreg=k1, areg=a1)(x)

    n2 = hp.Int("neurons2", min_value=8, max_value=128, step=8)
    k2 = _choose_regularization(hp, "kreg2")
    a2 = _choose_regularization(hp, "areg2")
    act2 = hp.Choice("activation2", ACTIVATIONS)
    x = _make_dense(n2, act=act2, kreg=k2, areg=a2)(x)

    params = DISTRIBUTIONS[dist_name]["params"]
    param_reg = _choose_regularization(hp, "param_reg")
    param_layers = [_make_dense(1, kreg=param_reg)(x) for _ in params]
    concat = keras.layers.Concatenate()(param_layers)

    dist_func = DISTRIBUTIONS[dist_name]["fun"]
    output = tfp.layers.DistributionLambda(dist_func)(concat)
    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss=lambda y, rv_y: -rv_y.log_prob(y),
    )
    return model
