"""Small linear-algebra helpers shared by the rest of the package."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import tensorflow as tf

from qtomo_skeleton._types import ArrayLike

#: dtype used for every TensorFlow computation in the package.
COMPLEX_DTYPE = tf.complex128


def as_complex_tensor(x: ArrayLike) -> tf.Tensor:
    """Convert ``x`` to a ``tf.complex128`` tensor (no-op if it already is one).

    Real inputs (``float64`` arrays/tensors) are cast instead of rejected, so the
    public functions accept NumPy arrays and TensorFlow tensors interchangeably.
    """
    if isinstance(x, (tf.Tensor, tf.Variable)):
        return x if x.dtype == COMPLEX_DTYPE else tf.cast(x, COMPLEX_DTYPE)
    return tf.convert_to_tensor(np.asarray(x, dtype=np.complex128))


def dagger(matrix: ArrayLike) -> tf.Tensor:
    """Conjugate transpose (works on batched ``(..., d, d)`` tensors too)."""
    return tf.linalg.adjoint(as_complex_tensor(matrix))


def num_qubits_of(matrix: ArrayLike) -> int:
    """Return ``n`` for a ``2**n x 2**n`` operator, raising if the shape is not valid."""
    dim = int(np.shape(matrix)[-1])
    n = dim.bit_length() - 1
    if dim < 1 or 2**n != dim:
        raise ValueError(f"Operator dimension must be a power of two, got {dim}.")
    return n


def kron_all(matrices: Sequence[np.ndarray]) -> np.ndarray:
    """Kronecker product ``M_0 ⊗ M_1 ⊗ ... ⊗ M_{k-1}``."""
    if not matrices:
        raise ValueError("kron_all needs at least one matrix.")
    result = np.asarray(matrices[0])
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result


def kron_insert(num_qubits: int, index: int, matrix: np.ndarray) -> np.ndarray:
    """Return ``I ⊗ ... ⊗ matrix ⊗ ... ⊗ I`` with ``matrix`` acting on qubit ``index``."""
    if not 0 <= index < num_qubits:
        raise ValueError(f"index must be in [0, {num_qubits}), got {index}.")
    factors = [np.eye(2)] * num_qubits
    factors[index] = matrix
    return kron_all(factors)


def normalize_kraus(kraus_ops: Sequence[ArrayLike] | tf.Tensor) -> list[tf.Tensor]:
    """Rescale Kraus operators so that ``Σ_i K_i† K_i = I`` (trace preservation).

    Accepts a list of ``(d, d)`` matrices or a stacked ``(k, d, d)`` tensor and
    always returns a list of ``(d, d)`` tensors.
    """
    if isinstance(kraus_ops, (tf.Tensor, tf.Variable)):
        stacked = as_complex_tensor(kraus_ops)
    else:
        stacked = tf.stack([as_complex_tensor(k) for k in kraus_ops])
    gram = tf.reduce_sum(tf.matmul(tf.linalg.adjoint(stacked), stacked), axis=0)
    inv_sqrt = tf.linalg.inv(tf.linalg.sqrtm(gram))
    return tf.unstack(tf.matmul(stacked, inv_sqrt))


def normalize_unitary(matrix: ArrayLike) -> tf.Tensor:
    """Project a square matrix back onto the unitary group (Q factor of a QR decomposition)."""
    q, _ = tf.linalg.qr(as_complex_tensor(matrix))
    return q


def diagonalize_matrices(matrices: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Eigen-decompose Hermitian matrices, returning ``(eigenvalues, eigenvectors)`` pairs."""
    return [np.linalg.eigh(mat) for mat in matrices]
