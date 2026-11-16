"""First-order optimizers for complex matrix parameters.

An optimizer only knows its own hyper-parameters. Each call to
:meth:`Optimizer.step_and_cost` receives the current parameter, the loss as a
function of the parameter, and a gradient rule (e.g. :func:`qtomo_skeleton.gradient.riemann`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import tensorflow as tf

from qtomo_skeleton._types import GradientFn

__all__ = ["Optimizer", "SGDOptimizer", "AdamOptimizer"]

LossOfParam = Callable[[tf.Tensor], tf.Tensor]


class Optimizer(ABC):
    """Base class: subclasses implement :meth:`step_and_cost`."""

    def __init__(self, alpha: float = 0.001) -> None:
        if alpha <= 0:
            raise ValueError(f"Learning rate alpha must be positive, got {alpha}.")
        self.alpha = alpha

    @abstractmethod
    def step_and_cost(
        self, param: tf.Tensor, loss_fn: LossOfParam, grad_fn: GradientFn
    ) -> tuple[tf.Tensor, tf.Tensor]:
        """Take one optimization step.

        Returns:
            ``(updated_param, loss)`` where ``loss`` is evaluated *before* the update.
        """

    def reset(self) -> None:
        """Clear internal state so the optimizer can be reused for a fresh run."""


class SGDOptimizer(Optimizer):
    """Plain gradient descent: ``x ← x - α g``."""

    def step_and_cost(
        self, param: tf.Tensor, loss_fn: LossOfParam, grad_fn: GradientFn
    ) -> tuple[tf.Tensor, tf.Tensor]:
        grad, loss = grad_fn(loss_fn, param)
        return param - self.alpha * grad, loss


class AdamOptimizer(Optimizer):
    """Adam (Kingma & Ba, 2014) applied element-wise to complex parameters.

    The moment estimates are kept in complex arithmetic (``v`` accumulates ``g**2``,
    not ``|g|**2``), matching the behaviour of the original research code.
    """

    def __init__(
        self,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(alpha)
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("beta1 and beta2 must be in [0, 1).")
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: tf.Tensor | None = None
        self.v: tf.Tensor | None = None
        self.t = 0

    def reset(self) -> None:
        self.m = None
        self.v = None
        self.t = 0

    def step_and_cost(
        self, param: tf.Tensor, loss_fn: LossOfParam, grad_fn: GradientFn
    ) -> tuple[tf.Tensor, tf.Tensor]:
        grad, loss = grad_fn(loss_fn, param)
        if self.m is None or self.v is None:
            self.m = tf.zeros_like(grad)
            self.v = tf.zeros_like(grad)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * tf.math.square(grad)

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return param - self.alpha * m_hat / (tf.math.sqrt(v_hat) + self.epsilon), loss
