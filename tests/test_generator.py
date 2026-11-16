import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

from qtomo_skeleton import generator


def is_density_matrix(rho: np.ndarray, atol: float = 1e-10) -> bool:
    return (
        np.allclose(rho, rho.conj().T, atol=atol)
        and np.isclose(np.trace(rho), 1, atol=atol)
        and np.all(np.linalg.eigvalsh(rho) >= -atol)
    )


# ---------------------------------------------------------------- Gell-Mann


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_gellmann_count_hermitian_traceless(dim):
    gm = generator.gellmann_matrices(dim)
    assert len(gm) == dim**2 - 1
    for m in gm:
        assert m.shape == (dim, dim)
        assert np.allclose(m, m.conj().T)
        assert np.isclose(np.trace(m), 0)


def test_gellmann_orthogonal():
    gm = generator.gellmann_matrices(3)
    gram = np.array([[np.trace(a @ b) for b in gm] for a in gm])
    assert_allclose(gram - np.diag(np.diag(gram)), 0, atol=1e-12)


def test_gellmann_dim2_are_paulis():
    # Diagonal generators are scaled by 1/sqrt(k(k+1)) (kept from the research code),
    # so for d = 2 the third matrix is Z / sqrt(2).
    x, y, z = generator.gellmann_matrices(2)
    assert_allclose(x, [[0, 1], [1, 0]])
    assert_allclose(y, [[0, -1j], [1j, 0]])
    assert_allclose(z, np.array([[1, 0], [0, -1]]) / np.sqrt(2))


# ------------------------------------------------------------- measurements


@pytest.mark.parametrize("num_qubits", [1, 2])
def test_measurement_projectors_sum_to_identity(num_qubits):
    total = sum(generator.measurement_projectors(num_qubits))
    assert_allclose(total, np.eye(2**num_qubits), atol=1e-10)


def test_merge_projectors_into_povm():
    projectors = generator.measurement_projectors(1)
    povm = generator.merge_projectors_into_povm(projectors, 4)
    assert len(povm) == 4
    assert_allclose(sum(povm), np.eye(2), atol=1e-10)


@pytest.mark.parametrize("num_elements", [0, 100])
def test_merge_projectors_rejects_bad_count(num_elements):
    with pytest.raises(ValueError):
        generator.merge_projectors_into_povm(generator.measurement_projectors(1), num_elements)


def test_measurement_operators_single_qubit_is_identity_map():
    projectors = [np.diag([1, 0]), np.diag([0, 1])]
    result = generator.measurement_operators(projectors, 1)
    assert len(result) == 2
    for m, p in zip(result, projectors):
        assert_allclose(m, p)


def test_measurement_operators_two_qubits_orthonormal():
    projectors = [np.diag([1, 0]), np.diag([0, 1])]
    result = generator.measurement_operators(projectors, 2)
    assert len(result) == 4
    assert all(m.shape == (4, 4) for m in result)
    gram = np.array([[np.trace(a.conj().T @ b) for b in result] for a in result])
    assert_allclose(gram, np.eye(4))
    assert all(np.isrealobj(m) for m in result)


# --------------------------------------------------------------- probe states


def test_probe_states_gellmann_are_density_matrices(rng):
    eigen = []
    for _ in range(8):
        eigvals = rng.normal(size=3)
        eigvecs, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        eigen.append((eigvals, eigvecs))
    states = generator.probe_states_gellmann(eigen, 1, rng)
    assert len(states) == 6
    assert all(rho.shape == (3, 3) and is_density_matrix(rho) for rho in states)


def test_probe_states_gellmann_real_inputs_give_real_states():
    eigen = [(np.array([1, 0.5, 0.2]), np.eye(3))]
    states = generator.probe_states_gellmann(eigen, 1)
    assert all(np.isrealobj(rho) for rho in states)


def test_haar_probe_state_is_pure_density_matrix(rng):
    rho = generator.haar_probe_state(1, rng)
    assert rho.shape == (2, 2)
    assert is_density_matrix(rho)
    assert np.isclose(np.trace(rho @ rho), 1)  # pure


@pytest.mark.parametrize("num_qubits", [1, 2])
def test_haar_probe_states(num_qubits, rng):
    states = generator.haar_probe_states(num_qubits, 10, rng)
    assert len(states) == 10
    assert all(is_density_matrix(rho) for rho in states)


def test_haar_probe_states_default_count():
    assert len(generator.haar_probe_states(1)) == 6


def test_haar_probe_states_rejects_non_positive():
    with pytest.raises(ValueError):
        generator.haar_probe_states(1, 0)


def test_generators_are_reproducible_with_seed():
    a = generator.haar_probe_states(2, 3, np.random.default_rng(7))
    b = generator.haar_probe_states(2, 3, np.random.default_rng(7))
    for x, y in zip(a, b):
        assert_allclose(x, y)


# ---------------------------------------------------------------- unitaries


@pytest.mark.parametrize("make", [generator.random_unitary, generator.haar])
def test_unitaries_are_unitary(make, rng):
    u = make(2, rng)
    assert u.shape == (4, 4)
    assert_allclose(u.conj().T @ u, np.eye(4), atol=1e-10)
    assert np.isclose(abs(np.linalg.det(u)), 1)


# ----------------------------------------------------------- Kraus operators


@pytest.mark.parametrize("num_ops", [1, 2, 4])
def test_kraus_operators_satisfy_completeness(num_ops, rng):
    ops = generator.kraus_operators(1, num_ops, rng)
    assert len(ops) == num_ops
    total = sum(tf.linalg.adjoint(k) @ k for k in ops)
    assert_allclose(total, np.eye(2), atol=1e-10)


def test_kraus_operators_default_count():
    assert len(generator.kraus_operators(1)) == 4


@pytest.mark.parametrize("num_ops", [0, 5])
def test_kraus_operators_rejects_bad_count(num_ops):
    with pytest.raises(ValueError):
        generator.kraus_operators(1, num_ops)
