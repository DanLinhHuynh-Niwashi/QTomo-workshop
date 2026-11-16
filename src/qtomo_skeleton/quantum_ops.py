"""Quantum operations on density matrices.

Two families live here:

* **Differentiable maps** (:func:`apply_unitary`, :func:`apply_unitary_dagger`,
  :func:`apply_kraus_operators`) built on TensorFlow, so gradients flow back to the
  learned unitary / Kraus parameters. They accept NumPy arrays or tensors and
  return ``tf.complex128`` tensors.
* **Fixed noise channels** (:func:`dephasing_channel`, :func:`amplitude_damping_channel`)
  that act on data, not parameters. They accept NumPy arrays or tensors and return
  NumPy arrays.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from qtomo_skeleton import linalg
from qtomo_skeleton._types import ArrayLike

__all__ = [
    "apply_unitary",
    "apply_unitary_dagger",
    "apply_kraus_operators",
    "dephasing_channel",
    "amplitude_damping_channel",
]

_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _check_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")


# --------------------------------------------------------------------------- #
# Differentiable maps
# --------------------------------------------------------------------------- #


def apply_unitary(rho: ArrayLike, unitary: ArrayLike) -> tf.Tensor:
    """Return ``U ρ U†``."""
    unitary = linalg.as_complex_tensor(unitary)
    return unitary @ linalg.as_complex_tensor(rho) @ tf.linalg.adjoint(unitary)


def apply_unitary_dagger(rho: ArrayLike, unitary: ArrayLike) -> tf.Tensor:
    """Return ``U† ρ U`` (the inverse of :func:`apply_unitary`)."""
    unitary = linalg.as_complex_tensor(unitary)
    return tf.linalg.adjoint(unitary) @ linalg.as_complex_tensor(rho) @ unitary


def apply_kraus_operators(rho: ArrayLike, kraus_ops: Sequence[ArrayLike] | tf.Tensor) -> tf.Tensor:
    """Return ``Σ_i K_i ρ K_i†`` for a list (or stacked tensor) of Kraus operators."""
    rho = linalg.as_complex_tensor(rho)
    return tf.add_n([apply_unitary(rho, k) for k in kraus_ops])


# --------------------------------------------------------------------------- #
# Fixed noise channels
# --------------------------------------------------------------------------- #


def dephasing_channel(rho: ArrayLike, gamma: float) -> np.ndarray:
    """Apply independent dephasing of strength ``gamma`` to every qubit.

    Per qubit: ``ρ → α ρ + β Z ρ Z`` with ``α, β = (1 ± sqrt(1 - γ)) / 2``.
    """
    _check_probability("gamma", gamma)
    rho = np.array(rho, dtype=np.complex128)
    num_qubits = linalg.num_qubits_of(rho)
    alpha = 0.5 * (1 + np.sqrt(1 - gamma))
    beta = 0.5 * (1 - np.sqrt(1 - gamma))

    for qubit in range(num_qubits):
        z = linalg.kron_insert(num_qubits, qubit, _SIGMA_Z)
        rho = alpha * rho + beta * (z @ rho @ z)
    return rho


def amplitude_damping_channel(rho: ArrayLike, gamma: float) -> np.ndarray:
    """Apply independent amplitude damping of strength ``gamma`` to every qubit."""
    _check_probability("gamma", gamma)
    rho = np.array(rho, dtype=np.complex128)
    num_qubits = linalg.num_qubits_of(rho)
    k0_single = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=np.complex128)
    k1_single = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128)

    for qubit in range(num_qubits):
        k0 = linalg.kron_insert(num_qubits, qubit, k0_single)
        k1 = linalg.kron_insert(num_qubits, qubit, k1_single)
        rho = k0 @ rho @ k0.conj().T + k1 @ rho @ k1.conj().T
    return rho
