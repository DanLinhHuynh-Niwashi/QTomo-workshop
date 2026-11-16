import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from qtomo_skeleton import gradient
from qtomo_skeleton.optimizer import AdamOptimizer, Optimizer, SGDOptimizer


def distance_to_2i(x: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.math.square(tf.math.abs(x - 2 * tf.eye(2, dtype=x.dtype))))


@pytest.fixture
def ones() -> tf.Tensor:
    return tf.constant(np.ones((2, 2), dtype=np.complex128))


# ------------------------------------------------------------------ gradient


def test_euclidean_gradient_matches_analytic(ones):
    grad, loss = gradient.euclidean(distance_to_2i, ones)
    assert float(loss) == pytest.approx(4)  # |1-2|^2 * 2 + 1^2 * 2
    # d/d(conj x) |x - a|^2 = (x - a); TensorFlow returns 2 * conj-derivative.
    assert_allclose(grad, 2 * (ones.numpy() - 2 * np.eye(2)))


def test_gradient_raises_when_loss_ignores_param(ones):
    with pytest.raises(ValueError):
        gradient.euclidean(lambda x: tf.constant(1.0, dtype=tf.complex128), ones)


def test_riemann_gradient_is_tangent_at_unitary(rng):
    u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
    u = tf.constant(u)
    g, _ = gradient.riemann(distance_to_2i, u)
    # Tangent space of the unitary group at U: U†G is anti-Hermitian.
    a = (tf.linalg.adjoint(u) @ g).numpy()
    assert_allclose(a, -a.conj().T, atol=1e-10)


def test_riemann_handles_stacked_parameters(ones):
    stacked = tf.stack([ones, 2 * ones])
    g, _ = gradient.riemann(lambda p: distance_to_2i(p[0]) + distance_to_2i(p[1]), stacked)
    assert g.shape == (2, 2, 2)


def test_riemann_descent_decreases_loss(ones):
    param = tf.Variable(ones)
    losses = []
    for _ in range(30):
        g, loss = gradient.riemann(distance_to_2i, param)
        param.assign(param - 0.03 * g)
        losses.append(float(tf.math.real(loss)))
    assert all(a >= b for a, b in zip(losses, losses[1:]))


# ----------------------------------------------------------------- optimizer


def test_optimizer_is_abstract():
    with pytest.raises(TypeError):
        Optimizer()


@pytest.mark.parametrize("alpha", [0, -1])
def test_optimizer_rejects_bad_learning_rate(alpha):
    with pytest.raises(ValueError):
        SGDOptimizer(alpha=alpha)


def test_sgd_step(ones):
    updated, loss = SGDOptimizer(alpha=0.1).step_and_cost(ones, distance_to_2i, gradient.euclidean)
    assert float(tf.math.real(loss)) == pytest.approx(4)
    assert_allclose(updated, ones.numpy() - 0.1 * 2 * (ones.numpy() - 2 * np.eye(2)))


def test_adam_steps_and_state(ones):
    opt = AdamOptimizer(alpha=0.1)
    param, losses = ones, []
    for step in range(1, 6):
        param, loss = opt.step_and_cost(param, distance_to_2i, gradient.riemann)
        losses.append(float(tf.math.real(loss)))
        assert opt.t == step
    assert param.shape == ones.shape
    assert opt.m is not None and opt.v is not None
    assert losses[1] <= losses[0] + 1e-3


def test_adam_reset(ones):
    opt = AdamOptimizer(alpha=0.1)
    first, _ = opt.step_and_cost(ones, distance_to_2i, gradient.riemann)
    opt.reset()
    assert (opt.m, opt.v, opt.t) == (None, None, 0)
    again, _ = opt.step_and_cost(ones, distance_to_2i, gradient.riemann)
    assert_allclose(again, first)
