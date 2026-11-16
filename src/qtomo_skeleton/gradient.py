"""Gradient rules used by the optimizers."""

from __future__ import annotations

from collections.abc import Callable

import tensorflow as tf

__all__ = ["euclidean", "riemann"]


def euclidean(loss_fn: Callable[[tf.Tensor], tf.Tensor], param: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Plain gradient of ``loss_fn`` at ``param``.

    Returns:
        ``(gradient, loss)`` where ``loss`` is evaluated at ``param``.
    """
    with tf.GradientTape() as tape:
        tape.watch(param)
        loss = loss_fn(param)
    grad = tape.gradient(loss, param)
    if grad is None:
        raise ValueError("Gradient is None: the loss does not depend on the parameter.")
    return grad, loss


def riemann(loss_fn: Callable[[tf.Tensor], tf.Tensor], param: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Riemannian gradient on the unitary (Stiefel) manifold.

    Projects the Euclidean gradient ``G`` onto the tangent space at ``X``::

        G - X (G† X + X† G) / 2

    ``param`` may be a single ``(d, d)`` matrix or a stacked ``(k, d, d)`` tensor;
    in the latter case each matrix is projected independently.

    Returns:
        ``(projected_gradient, loss)`` where ``loss`` is evaluated at ``param``.
    """
    grad, loss = euclidean(loss_fn, param)
    adj = tf.linalg.adjoint
    projected = grad - tf.matmul(param, tf.matmul(adj(grad), param) + tf.matmul(adj(param), grad)) / 2
    return projected, loss
