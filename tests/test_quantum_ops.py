import numpy as np
import pytest
import tensorflow as tf
from conftest import PAULI_X, random_density_matrix
from numpy.testing import assert_allclose

from qtomo_skeleton import quantum_ops


def rotation(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128)


# --------------------------------------------------------- differentiable maps


def test_apply_unitary_identity():
    rho = random_density_matrix()
    assert_allclose(quantum_ops.apply_unitary(rho, np.eye(2)), rho)


def test_apply_unitary_then_dagger_recovers_input():
    rho = random_density_matrix()
    u = rotation(np.pi / 4)
    recovered = quantum_ops.apply_unitary_dagger(quantum_ops.apply_unitary(rho, u), u)
    assert_allclose(recovered, rho, atol=1e-12)


def test_apply_unitary_preserves_trace_and_hermiticity():
    out = quantum_ops.apply_unitary(random_density_matrix(), PAULI_X).numpy()
    assert np.isclose(np.trace(out), 1)
    assert_allclose(out, out.conj().T)


def test_apply_unitary_accepts_mixed_inputs():
    rho = np.diag([1.0, 0.0])  # real float64 array
    out = quantum_ops.apply_unitary(rho, tf.constant(PAULI_X))
    assert out.dtype == tf.complex128
    assert_allclose(out, np.diag([0, 1]))


def test_apply_kraus_amplitude_damping_is_trace_preserving():
    gamma = 0.3
    kraus = [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]]),
        np.array([[0, np.sqrt(gamma)], [0, 0]]),
    ]
    out = quantum_ops.apply_kraus_operators(random_density_matrix(), kraus)
    assert np.isclose(np.trace(out), 1)


def test_apply_kraus_accepts_stacked_tensor():
    rho = random_density_matrix()
    stacked = tf.constant(np.stack([np.eye(2), PAULI_X]) / np.sqrt(2), dtype=tf.complex128)
    out = quantum_ops.apply_kraus_operators(rho, stacked)
    assert_allclose(out, 0.5 * (rho + PAULI_X @ rho @ PAULI_X))


def test_apply_kraus_is_differentiable():
    k = tf.Variable(np.eye(2, dtype=np.complex128))
    with tf.GradientTape() as tape:
        out = quantum_ops.apply_kraus_operators(random_density_matrix(), [k])
        loss = tf.math.real(tf.linalg.trace(out))
    assert tape.gradient(loss, k) is not None


# --------------------------------------------------------- amplitude damping


def test_amplitude_damping_gamma_0_is_identity():
    rho = np.array([[0.6, 0.3], [0.3, 0.4]], dtype=np.complex128)
    assert_allclose(quantum_ops.amplitude_damping_channel(rho, 0.0), rho)


def test_amplitude_damping_gamma_1_decays_to_ground():
    rho = np.diag([0.4, 0.6])
    assert_allclose(quantum_ops.amplitude_damping_channel(rho, 1.0), np.diag([1, 0]))


@pytest.mark.parametrize("dim", [2, 4])
def test_amplitude_damping_output_is_density_matrix(dim):
    out = quantum_ops.amplitude_damping_channel(random_density_matrix(dim), 0.4)
    assert_allclose(out, out.conj().T)
    assert np.isclose(np.trace(out), 1)
    assert np.all(np.linalg.eigvalsh(out) >= -1e-12)


# ---------------------------------------------------------------- dephasing


@pytest.mark.parametrize("gamma", [0.0, 0.2, 0.8, 1.0])
def test_dephasing_scales_coherence(gamma):
    rho = np.full((2, 2), 0.5, dtype=np.complex128)
    out = quantum_ops.dephasing_channel(rho, gamma)
    assert_allclose(out, out.conj().T)
    assert np.isclose(np.trace(out), 1)
    assert_allclose(np.diag(out), np.diag(rho))
    assert np.isclose(out[0, 1], 0.5 * np.sqrt(1 - gamma))


def test_dephasing_two_qubits_kills_all_coherences_at_gamma_1():
    rho = random_density_matrix(4)
    out = quantum_ops.dephasing_channel(rho, 1.0)
    assert_allclose(out, np.diag(np.diag(rho)), atol=1e-12)


def test_dephasing_accepts_tensor_input():
    rho = tf.constant(np.full((2, 2), 0.5), dtype=tf.complex128)
    out = quantum_ops.dephasing_channel(rho, 0.5)
    assert isinstance(out, np.ndarray)


@pytest.mark.parametrize("channel", [quantum_ops.dephasing_channel, quantum_ops.amplitude_damping_channel])
@pytest.mark.parametrize("gamma", [-0.1, 1.5])
def test_channels_reject_invalid_gamma(channel, gamma):
    with pytest.raises(ValueError):
        channel(np.eye(2) / 2, gamma)
