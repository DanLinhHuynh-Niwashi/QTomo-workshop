import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from conftest import kraus_tomography_class, skip_if_not_implemented

from qtomo_skeleton import generator, quantum_ops
from qtomo_skeleton.optimizer import AdamOptimizer
from qtomo_skeleton.writer import (
    Evaluation,
    ExperimentWriter,
    TomographyResult,
    get_value,
    group_by,
    load_result,
    load_results,
    save_result,
)

# Nhánh solution: lớp trong file đáp án; nhánh main: lớp của sinh viên (skip nếu chưa làm xong).
KrausTomography = kraus_tomography_class()

TARGET = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def kraus_process(rho_set, kraus_ops):
    return [
        quantum_ops.apply_unitary_dagger(quantum_ops.apply_kraus_operators(r, kraus_ops), TARGET) for r in rho_set
    ]


@pytest.fixture(scope="module")
def kraus_model():
    tomo = KrausTomography(1, 3, num_kraus=2, seed=0)
    skip_if_not_implemented(tomo.fit, data_process_fn=kraus_process, epochs=4,
                            optimizer=AdamOptimizer(0.05))
    return tomo


@pytest.fixture(scope="module")
def single_kraus_model():
    tomo = KrausTomography(1, 3, num_kraus=1, seed=1)
    skip_if_not_implemented(tomo.fit, data_process_fn=kraus_process, epochs=4,
                            optimizer=AdamOptimizer(0.05))
    return tomo


@pytest.fixture(scope="module")
def evaluation(kraus_model):
    states = generator.haar_probe_states(1, 4, np.random.default_rng(0))
    return Evaluation.from_model(kraus_model, states, lambda r: quantum_ops.apply_unitary(r, TARGET))


# --------------------------------------------------------------- records


def test_evaluation_shapes(evaluation):
    assert evaluation.inputs.shape == evaluation.outputs.shape == evaluation.targets.shape == (4, 2, 2)
    assert evaluation.fidelities.shape == (4,)
    assert evaluation.fidelities.dtype == np.float64
    assert evaluation.mean_infidelity == pytest.approx(1 - evaluation.fidelities.mean())


def test_evaluation_requires_states(kraus_model):
    with pytest.raises(ValueError):
        Evaluation.from_model(kraus_model, [], lambda r: r)


def test_result_from_kraus_model(kraus_model, evaluation):
    result = TomographyResult.from_model(kraus_model, evaluation, gamma=0.1)
    assert result.model == "kraus"
    assert result.num_qubits == 1
    assert result.params.shape == result.initial_params.shape == (2, 2, 2)
    assert result.training_states.shape == (3, 2, 2)
    assert result.final_loss == pytest.approx(kraus_model.loss_history[-1])
    assert result.test_infidelity == pytest.approx(evaluation.mean_infidelity)
    assert result.fit_seconds > 0
    assert result.metadata == {"gamma": 0.1}


def test_result_from_single_kraus_model(single_kraus_model):
    result = TomographyResult.from_model(single_kraus_model)
    assert result.model == "kraus"
    assert result.params.shape == (1, 2, 2)
    assert_allclose(result.params[0], single_kraus_model.kraus_operators[0])
    assert np.isnan(result.test_infidelity)


def test_result_requires_fitted_model():
    with pytest.raises(RuntimeError) as excinfo:
        TomographyResult.from_model(KrausTomography(1))
    if excinfo.type is NotImplementedError:  # lớp con của RuntimeError: is_fitted chưa viết
        pytest.skip(f"chưa viết: {excinfo.value}")


def test_get_value_and_group_by(kraus_model, single_kraus_model):
    results = [
        TomographyResult.from_model(single_kraus_model, gamma=0.5),
        TomographyResult.from_model(kraus_model, gamma=0.1),
        TomographyResult.from_model(single_kraus_model, gamma=0.1),
    ]
    assert get_value(results[0], "model") == "kraus"
    assert get_value(results[0], "gamma") == 0.5
    with pytest.raises(KeyError):
        get_value(results[0], "missing")
    groups = group_by(results, "gamma")
    assert list(groups) == [0.1, 0.5]
    assert [len(g) for g in groups.values()] == [2, 1]


# ------------------------------------------------------------------ files


def test_round_trip(tmp_path, kraus_model, evaluation):
    original = TomographyResult.from_model(kraus_model, evaluation, gamma=np.float64(0.2), seed=0)
    path = save_result(tmp_path / "run", original)
    assert path.suffix == ".npz"

    loaded = load_result(path)
    assert loaded.model == original.model
    assert loaded.num_qubits == original.num_qubits
    assert loaded.fit_seconds == pytest.approx(original.fit_seconds)
    assert loaded.metadata == {"gamma": 0.2, "seed": 0}
    for attr in ("loss_history", "training_states", "initial_params", "params"):
        assert_allclose(getattr(loaded, attr), getattr(original, attr))
    for attr in ("inputs", "outputs", "targets", "fidelities"):
        assert_allclose(getattr(loaded.evaluation, attr), getattr(original.evaluation, attr))


def test_file_opens_with_plain_numpy(tmp_path, single_kraus_model):
    path = save_result(tmp_path / "run.npz", TomographyResult.from_model(single_kraus_model, gamma=0.3))
    with np.load(path) as data:  # allow_pickle defaults to False
        assert str(data["model"]) == "kraus"
        assert data["loss_history"].shape == (4,)
        assert data["params"].dtype == np.complex128
        assert json.loads(str(data["metadata"])) == {"gamma": 0.3}
        assert "eval_fidelities" not in data


def test_save_refuses_to_overwrite(tmp_path, single_kraus_model):
    result = TomographyResult.from_model(single_kraus_model)
    save_result(tmp_path / "run", result)
    with pytest.raises(FileExistsError):
        save_result(tmp_path / "run", result)
    save_result(tmp_path / "run", result, overwrite=True)


def test_metadata_must_be_json_serializable(tmp_path, single_kraus_model):
    result = TomographyResult.from_model(single_kraus_model, bad=object())
    with pytest.raises(TypeError):
        save_result(tmp_path / "run", result)


def test_load_rejects_unknown_version(tmp_path, single_kraus_model):
    path = save_result(tmp_path / "run", TomographyResult.from_model(single_kraus_model))
    with np.load(path) as data:
        arrays = dict(data)
    arrays["format_version"] = np.array(999)
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        load_result(path)


def test_experiment_writer(tmp_path, kraus_model, single_kraus_model):
    writer = ExperimentWriter(tmp_path / "exp")
    first = writer.save(TomographyResult.from_model(kraus_model, run=0))
    second = writer.save(TomographyResult.from_model(single_kraus_model, run=1))
    named = writer.save(TomographyResult.from_model(single_kraus_model, run=2), name="custom")
    assert [p.name for p in (first, second, named)] == ["run_0000.npz", "run_0001.npz", "custom.npz"]
    assert [r.metadata["run"] for r in writer.load_all()] == [2, 0, 1]  # sorted by file name
    assert len(load_results(tmp_path / "exp", "run_*.npz")) == 2


def test_load_results_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "nope")


def test_group_by_several_keys(kraus_model, single_kraus_model):
    results = [
        TomographyResult.from_model(kraus_model, num_kraus=2, lr=0.1),
        TomographyResult.from_model(single_kraus_model, num_kraus=1, lr=0.1),
        TomographyResult.from_model(single_kraus_model, num_kraus=1, lr=0.2),
    ]
    groups = group_by(results, ("num_kraus", "lr"))
    assert list(groups) == [(1, 0.1), (1, 0.2), (2, 0.1)]
