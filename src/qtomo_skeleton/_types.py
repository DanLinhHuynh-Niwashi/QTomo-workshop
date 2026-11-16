"""Type aliases shared across modules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Union

import numpy as np
import tensorflow as tf

#: A matrix given either as a NumPy array or a TensorFlow tensor/variable.
ArrayLike = Union[np.ndarray, tf.Tensor, tf.Variable]

#: A density matrix ``ρ`` of shape ``(d, d)``.
DensityMatrix = ArrayLike

#: A list of density matrices.
DensityMatrixList = Sequence[DensityMatrix]

#: ``loss_fn(predicted_states, target_states) -> scalar tensor``.
LossFn = Callable[[DensityMatrixList, DensityMatrixList], tf.Tensor]

#: ``gradient_fn(loss_of_param, param) -> (gradient, loss)``.
GradientFn = Callable[[Callable[[tf.Tensor], tf.Tensor], tf.Tensor], tuple[tf.Tensor, tf.Tensor]]
