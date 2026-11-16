"""Distances and fidelities between density matrices.

All functions accept NumPy arrays or TensorFlow tensors and are differentiable
with respect to TensorFlow inputs, so the list-valued ones can be used directly
as ``loss_fn`` in :meth:`qtomo.tomography.Tomography.fit`.

Fidelity follows the (squared) Uhlmann convention
``F(ρ, σ) = (Tr sqrt(sqrt(ρ) σ sqrt(ρ)))**2``. It is returned as a ``complex128``
scalar tensor because TensorFlow's ``sqrtm`` works in complex arithmetic;
take ``tf.math.real`` (or ``float(...)``) for reporting.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from qtomo_skeleton import linalg
from qtomo_skeleton._types import ArrayLike, DensityMatrix, DensityMatrixList

__all__ = [
    "compilation_trace_fidelity",
    "infidelity",
    "mean_infidelity",
    "mean_squared_error",
    "frobenius_norm",
    "pauli_expectation",
]


def _check_same_length(predicted: DensityMatrixList, target: DensityMatrixList) -> None:
    if len(predicted) != len(target):
        raise ValueError(f"Got {len(predicted)} predicted states but {len(target)} targets.")
    if not target:
        raise ValueError("State lists must not be empty.")


# --------------------------------------------------------------------------- #
# Pairwise metrics
# --------------------------------------------------------------------------- #


def compilation_trace_fidelity(rho: DensityMatrix, sigma: DensityMatrix) -> tf.Tensor:
    """Uhlmann fidelity ``F(ρ, σ)`` in ``[0, 1]`` (as a complex scalar tensor)."""
    sqrt_rho = tf.linalg.sqrtm(linalg.as_complex_tensor(rho))
    inner = sqrt_rho @ linalg.as_complex_tensor(sigma) @ sqrt_rho
    return tf.math.square(tf.linalg.trace(tf.linalg.sqrtm(inner)))


def infidelity(rho: DensityMatrix, sigma: DensityMatrix) -> tf.Tensor:
    """``1 - F(ρ, σ)``."""
    return 1 - compilation_trace_fidelity(rho, sigma)


def frobenius_norm(rho: ArrayLike, sigma: ArrayLike) -> tf.Tensor:
    """Frobenius distance ``||ρ - σ||_F = sqrt(Σ |ρ_ij - σ_ij|²)`` as a real tensor."""
    diff = linalg.as_complex_tensor(rho) - linalg.as_complex_tensor(sigma)
    return tf.sqrt(tf.reduce_sum(tf.math.square(tf.math.abs(diff))))


# --------------------------------------------------------------------------- #
# List metrics (usable as ``loss_fn``)
# --------------------------------------------------------------------------- #


def mean_infidelity(predicted: DensityMatrixList, target: DensityMatrixList) -> tf.Tensor:
    """Average ``1 - F(target_i, predicted_i)`` over paired lists of states."""
    _check_same_length(predicted, target)
    fidelities = [compilation_trace_fidelity(t, p) for p, t in zip(predicted, target)]
    return 1 - tf.add_n(fidelities) / len(fidelities)


def mean_squared_error(predicted: DensityMatrixList, target: DensityMatrixList) -> tf.Tensor:
    """Average squared Frobenius distance ``Σ_ij |ρ_ij - σ_ij|²`` over paired lists."""
    _check_same_length(predicted, target)
    errors = [tf.math.square(frobenius_norm(t, p)) for p, t in zip(predicted, target)]
    return tf.add_n(errors) / len(errors)


# --------------------------------------------------------------------------- #
# Observables
# --------------------------------------------------------------------------- #


def pauli_expectation(rho: ArrayLike, qubit_index: int, pauli: np.ndarray) -> complex:
    """``Tr(P_k ρ)`` where ``P_k`` is the single-qubit operator ``pauli`` on qubit ``qubit_index``."""
    rho = np.asarray(rho, dtype=np.complex128)
    operator = linalg.kron_insert(linalg.num_qubits_of(rho), qubit_index, pauli)
    return complex(np.trace(operator @ rho))
