import numpy as np
import pytest
import tensorflow as tf
from conftest import PAULI_X, PAULI_Z, random_density_matrix

from qtomo_skeleton import metric


def real(x) -> float:
    return float(tf.math.real(x))


@pytest.fixture
def nearly_orthogonal():
    """Full-rank, nearly orthogonal states (keeps sqrtm numerically stable)."""
    eps = 1e-3
    rho = (1 - eps) * np.diag([1, 0]) + eps * np.eye(2) / 2
    sigma = (1 - eps) * np.diag([0, 1]) + eps * np.eye(2) / 2
    return rho.astype(np.complex128), sigma.astype(np.complex128)


# ----------------------------------------------------------------- fidelity


@pytest.mark.parametrize("dim", [2, 4])
def test_fidelity_with_itself_is_one(dim):
    rho = random_density_matrix(dim)
    assert real(metric.compilation_trace_fidelity(rho, rho)) == pytest.approx(1, abs=1e-8)


def test_fidelity_is_symmetric():
    rho, sigma = random_density_matrix(seed=1), random_density_matrix(seed=2)
    f1 = real(metric.compilation_trace_fidelity(rho, sigma))
    f2 = real(metric.compilation_trace_fidelity(sigma, rho))
    assert f1 == pytest.approx(f2, abs=1e-8)
    assert 0 < f1 < 1


def test_fidelity_of_commuting_states_matches_classical_formula():
    p, q = np.array([0.7, 0.3]), np.array([0.2, 0.8])
    expected = np.sum(np.sqrt(p * q)) ** 2
    assert real(metric.compilation_trace_fidelity(np.diag(p), np.diag(q))) == pytest.approx(expected)


def test_infidelity_complements_fidelity():
    rho, sigma = random_density_matrix(4, seed=1), random_density_matrix(4, seed=2)
    total = real(metric.infidelity(rho, sigma)) + real(metric.compilation_trace_fidelity(rho, sigma))
    assert total == pytest.approx(1)


# ------------------------------------------------------------ list metrics


def test_mean_infidelity_identical_is_zero():
    states = [random_density_matrix(seed=s) for s in range(3)]
    assert real(metric.mean_infidelity(states, states)) == pytest.approx(0, abs=1e-8)


def test_mean_infidelity_nearly_orthogonal_is_one(nearly_orthogonal):
    rho, sigma = nearly_orthogonal
    value = real(metric.mean_infidelity([rho], [sigma]))
    assert not np.isnan(value)
    assert value == pytest.approx(1, abs=1e-2)


def test_mean_infidelity_is_average_of_pairs():
    a = [random_density_matrix(seed=s) for s in range(3)]
    b = [random_density_matrix(seed=s + 10) for s in range(3)]
    expected = np.mean([real(metric.infidelity(t, p)) for p, t in zip(a, b)])
    assert real(metric.mean_infidelity(a, b)) == pytest.approx(expected)


def test_mean_squared_error_known_value():
    assert float(metric.mean_squared_error([np.eye(2)], [np.eye(2) / 2])) == pytest.approx(0.5)
    assert float(metric.mean_squared_error([np.eye(2)], [np.eye(2)])) == 0


@pytest.mark.parametrize("loss", [metric.mean_infidelity, metric.mean_squared_error])
def test_list_metrics_reject_mismatched_lengths(loss):
    with pytest.raises(ValueError):
        loss([np.eye(2) / 2], [np.eye(2) / 2, np.eye(2) / 2])
    with pytest.raises(ValueError):
        loss([], [])


def test_mean_infidelity_is_differentiable():
    target = [random_density_matrix(seed=s) for s in range(2)]
    u = tf.Variable(np.eye(2, dtype=np.complex128))
    with tf.GradientTape() as tape:
        predicted = [u @ t @ tf.linalg.adjoint(u) for t in target]
        loss = tf.math.real(metric.mean_infidelity(predicted, target))
    assert tape.gradient(loss, u) is not None


# ------------------------------------------------------------ Frobenius norm


def test_frobenius_norm_complex_entries():
    # ||diag(i, 0)||_F = 1; the original code summed diff**2 instead of |diff|**2 and returned 1j.
    value = metric.frobenius_norm(np.diag([1j, 0]), np.zeros((2, 2)))
    assert value.dtype == tf.float64
    assert float(value) == pytest.approx(1)


def test_frobenius_norm_matches_numpy():
    rho, sigma = random_density_matrix(4, seed=1), random_density_matrix(4, seed=2)
    assert float(metric.frobenius_norm(rho, sigma)) == pytest.approx(np.linalg.norm(rho - sigma))


# ------------------------------------------------------------ observables


def test_pauli_expectation_values():
    zero = np.diag([1, 0])
    plus = np.full((2, 2), 0.5)
    assert metric.pauli_expectation(zero, 0, PAULI_Z) == pytest.approx(1)
    assert metric.pauli_expectation(plus, 0, PAULI_X) == pytest.approx(1)
    assert metric.pauli_expectation(plus, 0, PAULI_Z) == pytest.approx(0)


def test_pauli_expectation_picks_the_right_qubit():
    rho = np.kron(np.diag([1, 0]), np.diag([0, 1]))  # |01⟩⟨01|
    assert metric.pauli_expectation(rho, 0, PAULI_Z) == pytest.approx(1)
    assert metric.pauli_expectation(rho, 1, PAULI_Z) == pytest.approx(-1)
