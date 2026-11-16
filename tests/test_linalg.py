import numpy as np
import pytest
import tensorflow as tf
from conftest import PAULI_X, PAULI_Z
from numpy.testing import assert_allclose

from qtomo_skeleton import linalg


def test_as_complex_tensor_casts_real_inputs():
    for x in (np.eye(2), tf.eye(2, dtype=tf.float64)):
        t = linalg.as_complex_tensor(x)
        assert t.dtype == tf.complex128
        assert_allclose(t, np.eye(2))


def test_as_complex_tensor_keeps_variables():
    v = tf.Variable(np.eye(2, dtype=np.complex128))
    assert linalg.as_complex_tensor(v) is v


@pytest.mark.parametrize("dim, expected", [(2, 1), (4, 2), (8, 3)])
def test_num_qubits_of(dim, expected):
    assert linalg.num_qubits_of(np.zeros((dim, dim))) == expected


@pytest.mark.parametrize("dim", [3, 6])
def test_num_qubits_of_rejects_non_power_of_two(dim):
    with pytest.raises(ValueError):
        linalg.num_qubits_of(np.zeros((dim, dim)))


def test_kron_insert_matches_explicit_product():
    expected = np.kron(np.kron(np.eye(2), PAULI_X), np.eye(2))
    assert_allclose(linalg.kron_insert(3, 1, PAULI_X), expected)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_kron_insert_z_diagonal(index):
    diag = np.diagonal(linalg.kron_insert(3, index, PAULI_Z)).real
    bits = (np.arange(8) >> (2 - index)) & 1  # qubit 0 is the most significant bit
    assert_allclose(diag, 1 - 2 * bits)


def test_kron_insert_rejects_bad_index():
    with pytest.raises(ValueError):
        linalg.kron_insert(2, 2, PAULI_X)


@pytest.mark.parametrize("as_stack", [False, True])
def test_normalize_kraus(as_stack, rng):
    ops = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(3)]
    normalized = linalg.normalize_kraus(tf.constant(np.stack(ops)) if as_stack else ops)
    assert isinstance(normalized, list) and len(normalized) == 3
    total = sum(tf.linalg.adjoint(k) @ k for k in normalized)
    assert_allclose(total, np.eye(2), atol=1e-10)


def test_normalize_unitary(rng):
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    u = linalg.normalize_unitary(m).numpy()
    assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-10)


def test_diagonalize_matrices():
    (vals, vecs), = linalg.diagonalize_matrices([PAULI_Z])
    assert_allclose(vals, [-1, 1])
    assert_allclose(vecs @ np.diag(vals) @ vecs.conj().T, PAULI_Z, atol=1e-12)
