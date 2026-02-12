import tensorflow as tf
from tensorflow_probability import distributions as tfd


__all__ = [
    "make_normal_tfd",
    "make_laplace_tfd",
    "make_t_tfd",
    "make_skewnorm_tfd",
    "make_skewt_tfd",
    "make_jsu_tfd"
]


def make_normal_tfd(tensor):
    return tfd.Laplace(
        loc=tensor[..., 0],
        scale=1e-3 + 3 * tf.math.softplus(tensor[..., 1])
    )

def make_laplace_tfd(tensor):
    return tfd.Laplace(
        loc=tensor[..., 0],
        scale=1e-3 + 3 * tf.math.softplus(tensor[..., 1])
    )

def make_t_tfd(tensor):
    return tfd.Laplace(
        loc=tensor[..., 0],
        scale=1e-3 + 3 * tf.math.softplus(tensor[..., 1]),
        df = 1 + 3 * tf.math.softplus(tensor[..., 2])
    )

def make_skewnorm_tfd(tensor):
    # tfd two piece normal is Fernández-Steel Skew Normal
    return tfd.TwoPieceNormal(
        loc=tensor[..., 0],
        scale=1e-3 + 3 * tf.math.softplus(tensor[..., 1]),
        skewness=1e-3 + 3 * tf.math.softplus(tensor[..., 2]),
    )

def make_skewt_tfd(tensor):
    # tfd two piece student t is Fernández-Steel Skew Student T
    return tfd.TwoPieceStudentT(
        loc=tensor[..., 0],
        scale=1e-3 + 3 * tf.math.softplus(tensor[..., 1]),
        skewness=1e-3 + 3 * tf.math.softplus(tensor[..., 2]),
        df=1 + 3 * tf.math.softplus(tensor[..., 3]),
    )

def make_jsu_tfd(tensor):
    return tfd.JohnsonSU(
            loc=tensor[..., 0],
            scale=1e-3 + 3 * tf.math.softplus(tensor[..., 1]),
            skewness=tensor[..., 2],
            tailweight=1 + 3 * tf.math.softplus(tensor[..., 3]),
        )