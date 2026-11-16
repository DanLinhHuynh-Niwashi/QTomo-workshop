"""Hợp đồng (contract) của ``KrausTomography`` mà sinh viên viết trong ``qtomo_skeleton.tomography``.

Mỗi test chạy trên hai bản cài đặt:

* ``solution``       – ``tomography-solution.py`` (chỉ có ở nhánh ``solution``; chứng minh hợp đồng đúng);
* ``qtomo_skeleton`` – bản của sinh viên. Test nào chạm vào method còn ``raise NotImplementedError``
  sẽ **skip** (kèm tên TODO); viết xong method nào thì test của method đó chạy thật.

Chạy riêng: ``python -m pytest tests/test_skeleton.py -v``.
"""

import functools
import importlib

import numpy as np
import pytest
import tensorflow as tf
from conftest import PAULI_X, has_solution, load_solution_tomography
from numpy.testing import assert_allclose

#: "solution" = src/qtomo_skeleton/tomography-solution.py (chỉ có ở nhánh solution); "qtomo_skeleton" = bài của bạn.
IMPLEMENTATIONS = (["solution"] if has_solution() else []) + ["qtomo_skeleton"]


def student_todo(test):
    """Đổi ``NotImplementedError`` (method sinh viên chưa viết) thành SKIPPED thay vì FAILED."""

    @functools.wraps(test)
    def wrapper(*args, **kwargs):
        try:
            return test(*args, **kwargs)
        except NotImplementedError as exc:
            pytest.skip(f"chưa viết: {exc}")

    return wrapper


@pytest.fixture(params=IMPLEMENTATIONS)
def pkg(request):
    """Namespace với ``KrausTomography`` và các module phụ trợ của một bản cài đặt."""
    name = request.param
    if name == "solution":  # load_solution_tomography() tự skip ở nhánh main
        tomography, name = load_solution_tomography(), "qtomo_skeleton"
    else:
        tomography = importlib.import_module(f"{name}.tomography")

    class Pkg:
        KrausTomography = tomography.KrausTomography
        quantum_ops = importlib.import_module(f"{name}.quantum_ops")
        optimizer = importlib.import_module(f"{name}.optimizer")
        writer = importlib.import_module(f"{name}.writer")

    return Pkg


def x_process(quantum_ops):
    """``E† K(ρ) E`` với ``E = X``."""

    def process(rho_set, kraus_ops):
        return [
            quantum_ops.apply_unitary_dagger(quantum_ops.apply_kraus_operators(r, kraus_ops), PAULI_X)
            for r in rho_set
        ]

    return process


# ------------------------------------------------------- skeleton packaging


def test_skeleton_imports_without_student_code():
    """Mọi module (kể cả tomography với các method còn TODO) phải import được, không side effect."""
    for module in ("generator", "linalg", "metric", "gradient", "optimizer", "quantum_ops", "writer", "plot"):
        importlib.import_module(f"qtomo_skeleton.{module}")
    importlib.import_module("qtomo_skeleton.tomography")


# --------------------------------------------------------------- __init__


@student_todo
def test_init_attributes(pkg):
    tomo = pkg.KrausTomography(num_qubits=1, num_rho=5, num_kraus=2, seed=0)
    assert tomo.num_qubits == 1
    assert len(tomo.target_rho_set) == 5
    assert all(rho.shape == (2, 2) and rho.dtype == np.complex128 for rho in tomo.target_rho_set)
    assert len(tomo.init_kraus) == 2
    assert tomo.init_kraus[0].dtype == tf.complex128
    assert tomo.kraus_operators is None
    assert tomo.loss_history == []
    assert tomo.fit_seconds is None


@student_todo
def test_not_fitted_after_init(pkg):
    assert not pkg.KrausTomography(1, 3, seed=0).is_fitted


@student_todo
def test_set_logging(pkg):
    tomo = pkg.KrausTomography(1, 3)
    tomo.set_logging(True)
    assert tomo.verbose is True
    tomo.set_logging(False)
    assert tomo.verbose is False


@student_todo
def test_default_number_of_training_states(pkg):
    assert len(pkg.KrausTomography(1).target_rho_set) == 6
    assert len(pkg.KrausTomography(2).target_rho_set) == 36


@student_todo
def test_seed_is_reproducible(pkg):
    a, b = pkg.KrausTomography(1, 3, seed=7), pkg.KrausTomography(1, 3, seed=7)
    for x, y in zip(a.target_rho_set, b.target_rho_set):
        assert_allclose(x, y)
    assert_allclose(a.init_kraus[0], b.init_kraus[0])


@student_todo
def test_explicit_training_states(pkg, fixed_probe_states):
    tomo = pkg.KrausTomography(1, training_states=fixed_probe_states)
    assert len(tomo.target_rho_set) == 3
    for got, expected in zip(tomo.target_rho_set, fixed_probe_states):
        assert_allclose(got, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_qubits": 0},
        {"num_qubits": 2, "training_states": [np.eye(2) / 2]},
        {"num_qubits": 1, "training_states": []},
    ],
    ids=["num_qubits=0", "wrong-shape", "empty"],
)
@student_todo
def test_invalid_input_raises_value_error(pkg, kwargs):
    with pytest.raises(ValueError):
        pkg.KrausTomography(**kwargs)


# -------------------------------------------------------------------- fit


@student_todo
def test_fit_contract(pkg):
    tomo = pkg.KrausTomography(1, 3, num_kraus=2, seed=0)
    kraus, history = tomo.fit(data_process_fn=x_process(pkg.quantum_ops), epochs=3)
    assert isinstance(kraus, list) and len(kraus) == 2
    assert len(history) == 3 and all(isinstance(x, float) for x in history)
    assert tomo.loss_history == history
    assert tomo.is_fitted and tomo.fit_seconds >= 0
    total = sum(tf.linalg.adjoint(k) @ k for k in kraus)
    assert_allclose(total, np.eye(2), atol=1e-10)  # projected back onto CPTP


@student_todo
def test_fit_rejects_non_positive_epochs(pkg):
    with pytest.raises(ValueError):
        pkg.KrausTomography(1).fit(data_process_fn=x_process(pkg.quantum_ops), epochs=0)


@student_todo
def test_repeated_fit_is_independent(pkg):
    """fit() restarts from init_kraus and resets the optimizer."""
    tomo = pkg.KrausTomography(1, 3, seed=0)
    opt = pkg.optimizer.AdamOptimizer(alpha=0.05)
    _, first = tomo.fit(data_process_fn=x_process(pkg.quantum_ops), epochs=5, optimizer=opt)
    _, second = tomo.fit(data_process_fn=x_process(pkg.quantum_ops), epochs=5, optimizer=opt)
    assert first == pytest.approx(second)


# --------------------------------------------------------------- evaluate


@student_todo
def test_evaluate_before_fit_raises(pkg):
    with pytest.raises(RuntimeError) as excinfo:
        pkg.KrausTomography(1).evaluate(np.eye(2) / 2, lambda r: r)
    if excinfo.type is NotImplementedError:  # lớp con của RuntimeError: chưa phải lời giải
        raise excinfo.value


@student_todo
def test_evaluate_and_writer_snapshot(pkg):
    tomo = pkg.KrausTomography(1, 3, seed=0)
    tomo.fit(data_process_fn=x_process(pkg.quantum_ops), epochs=2)
    rho = np.eye(2, dtype=np.complex128) / 2
    out, target, score = tomo.evaluate(rho, lambda r: r, metric_fn=lambda a, b: 0.5)
    assert out.shape == (2, 2) and score == 0.5
    assert_allclose(target, rho)

    result = pkg.writer.TomographyResult.from_model(tomo, seed=0)
    assert result.model == "kraus"
    assert result.params.shape == (1, 2, 2)


@pytest.mark.slow
@student_todo
def test_learns_pauli_x(pkg):
    tomo = pkg.KrausTomography(1, 3, num_kraus=1, seed=2)
    opt = pkg.optimizer.AdamOptimizer(alpha=0.05)
    _, history = tomo.fit(data_process_fn=x_process(pkg.quantum_ops), epochs=200, optimizer=opt)
    assert history[-1] < 0.1
