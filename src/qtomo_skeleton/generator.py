"""Generators for bases, measurements, probe states, unitaries and Kraus operators.

Every random generator takes an optional ``rng`` (:class:`numpy.random.Generator`).
Pass ``np.random.default_rng(seed)`` for reproducible results; if omitted, a fresh
unseeded generator is used. Nothing in this module touches global random state.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import qr

from qtomo_skeleton import linalg

if TYPE_CHECKING:
    import tensorflow as tf

__all__ = [
    "gellmann_matrices",
    "measurement_projectors",
    "merge_projectors_into_povm",
    "measurement_operators",
    "probe_states_gellmann",
    "random_unitary",
    "haar",
    "haar_probe_state",
    "haar_probe_states",
    "kraus_operators",
]


def _ensure_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _complex_gaussian(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


# --------------------------------------------------------------------------- #
# Bases and measurements
# --------------------------------------------------------------------------- #


def gellmann_matrices(dim: int) -> list[np.ndarray]:
    """Return the ``dim**2 - 1`` generalized Gell-Mann matrices spanning ``su(dim)``."""
    matrices: list[np.ndarray] = []

    # Off-diagonal symmetric and antisymmetric generators.
    for i in range(dim):
        for j in range(i + 1, dim):
            sym = np.zeros((dim, dim), dtype=complex)
            sym[i, j] = sym[j, i] = 1
            matrices.append(sym)

            antisym = np.zeros((dim, dim), dtype=complex)
            antisym[i, j] = -1j
            antisym[j, i] = 1j
            matrices.append(antisym)

    # Diagonal traceless generators.
    for k in range(1, dim):
        diag = np.zeros((dim, dim), dtype=complex)
        diag[np.arange(k), np.arange(k)] = 1
        diag[k, k] = -k
        matrices.append(diag / np.sqrt(k * (k + 1)))

    return matrices


def measurement_projectors(num_qubits: int = 1) -> list[np.ndarray]:
    """Rank-1 projectors onto the eigenvectors of every Gell-Mann matrix of ``d = 2**num_qubits``.

    The projectors are rescaled so that they sum to the identity (i.e. form a POVM).
    """
    dim = 2**num_qubits
    projectors = [
        np.outer(vec, vec.conj())
        for _, eigvecs in linalg.diagonalize_matrices(gellmann_matrices(dim))
        for vec in eigvecs.T
    ]
    scale = np.trace(sum(projectors)) / dim
    return [p / scale for p in projectors]


def merge_projectors_into_povm(projectors: Sequence[np.ndarray], num_elements: int) -> list[np.ndarray]:
    """Group ``projectors`` round-robin into ``num_elements`` POVM elements.

    The sum of the elements equals the sum of the input projectors, so a
    normalized projector set yields a normalized POVM.
    """
    if not 1 <= num_elements <= len(projectors):
        raise ValueError(
            f"num_elements must be in [1, {len(projectors)}], got {num_elements}."
        )
    groups: list[list[np.ndarray]] = [[] for _ in range(num_elements)]
    for i, projector in enumerate(projectors):
        groups[i % num_elements].append(projector)
    return [sum(group) for group in groups]


def measurement_operators(projectors: Sequence[np.ndarray], num_qubits: int) -> list[np.ndarray]:
    """All ``num_qubits``-fold tensor products of single-qubit ``projectors``."""
    return [linalg.kron_all(combo) for combo in itertools.product(projectors, repeat=num_qubits)]


# --------------------------------------------------------------------------- #
# Probe states
# --------------------------------------------------------------------------- #


def probe_states_gellmann(
    eigen_gellmann: Sequence[tuple[np.ndarray, np.ndarray]],
    num_qubits: int,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Build ``6**num_qubits`` probe states from Gell-Mann eigen-decompositions.

    Each state uses the eigenvector basis of the first decomposition and the
    (absolute, normalized) spectrum of a randomly chosen decomposition.
    """
    rng = _ensure_rng(rng)
    eigvals_list = [np.abs(ev) / np.sum(np.abs(ev)) for ev, _ in eigen_gellmann]
    basis = eigen_gellmann[0][1]

    states = []
    for _ in range(6**num_qubits):
        eigvals = eigvals_list[rng.integers(len(eigvals_list))]
        rho = basis @ np.diag(eigvals) @ basis.conj().T
        states.append(rho / np.trace(rho))
    return states


def haar_probe_state(num_qubits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random pure state ``ρ = U|ψ⟩⟨ψ|U†`` with Haar-random ``U``."""
    rng = _ensure_rng(rng)
    psi = _complex_gaussian(rng, (2**num_qubits,))
    psi /= np.linalg.norm(psi)
    unitary = haar(num_qubits, rng)
    return unitary @ np.outer(psi, psi.conj()) @ unitary.conj().T


def haar_probe_states(
    num_qubits: int,
    num_states: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """``num_states`` Haar-random probe states (default: ``6**num_qubits``)."""
    rng = _ensure_rng(rng)
    if num_states is None:
        num_states = 6**num_qubits
    if num_states < 1:
        raise ValueError(f"num_states must be positive, got {num_states}.")
    return [haar_probe_state(num_qubits, rng) for _ in range(num_states)]


# --------------------------------------------------------------------------- #
# Unitaries and Kraus operators
# --------------------------------------------------------------------------- #


def random_unitary(num_qubits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Random unitary from the Q factor of a complex Gaussian matrix.

    Note: this is *not* Haar-distributed; use :func:`haar` when that matters.
    """
    rng = _ensure_rng(rng)
    dim = 2**num_qubits
    q, _ = qr(_complex_gaussian(rng, (dim, dim)))
    return q


def haar(num_qubits: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Haar-random unitary on ``num_qubits`` qubits (Mezzadri, arXiv:math-ph/0609050)."""
    rng = _ensure_rng(rng)
    dim = 2**num_qubits
    z = _complex_gaussian(rng, (dim, dim)) / np.sqrt(2)
    q, r = qr(z)
    diag = np.diag(r)
    return q * (diag / np.abs(diag))


def kraus_operators(
    num_qubits: int,
    num_operators: int | None = None,
    rng: np.random.Generator | None = None,
) -> list[tf.Tensor]:
    """Random set of Kraus operators normalized so that ``Σ K†K = I``.

    ``num_operators`` defaults to the maximum useful number, ``d**2``.
    Returns a list of ``tf.complex128`` tensors.
    """
    rng = _ensure_rng(rng)
    max_ops = 4**num_qubits
    if num_operators is None:
        num_operators = max_ops
    if not 1 <= num_operators <= max_ops:
        raise ValueError(f"num_operators must be in [1, {max_ops}], got {num_operators}.")
    return linalg.normalize_kraus([random_unitary(num_qubits, rng) for _ in range(num_operators)])
