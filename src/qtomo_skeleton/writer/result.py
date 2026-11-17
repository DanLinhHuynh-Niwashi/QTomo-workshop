"""In-memory records of a tomography run, saved / loaded as NumPy ``.npz`` files.

``TomographyResult.to_npz`` writes a plain NumPy archive (no pickling), so the files
open directly with ``np.load(path)`` in any NumPy-only environment::

    data = np.load("run_0000.npz")
    data["loss_history"], data["params"], json.loads(str(data["metadata"]))

Keys: ``format_version``, ``model``, ``num_qubits``, ``loss_history``,
``training_states``, ``initial_params``, ``params``, ``fit_seconds``, ``metadata``
(JSON string) and, when an evaluation was recorded, ``eval_inputs``,
``eval_outputs``, ``eval_targets``, ``eval_fidelities``.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

from qtomo_skeleton import metric
from qtomo_skeleton._types import ArrayLike

if TYPE_CHECKING:  # chỉ dùng cho type hint: writer không phụ thuộc vào tomography.py (file của sinh viên)
    from qtomo_skeleton.tomography import KrausTomography

__all__ = ["FORMAT_VERSION", "Evaluation", "TomographyResult", "get_value", "group_by"]


def _stack_complex(matrices: Iterable[ArrayLike]) -> np.ndarray:
    return np.stack([np.asarray(m, dtype=np.complex128) for m in matrices])


#: Version of the ``.npz`` layout written by :meth:`TomographyResult.to_npz`.
FORMAT_VERSION = 1

_EVAL_KEYS = {
    "eval_inputs": "inputs",
    "eval_outputs": "outputs",
    "eval_targets": "targets",
    "eval_fidelities": "fidelities",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Metadata value {value!r} of type {type(value).__name__} is not JSON-serializable.")


def _with_suffix(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path.with_name(path.name + ".npz")


@dataclass(frozen=True, eq=False)
class Evaluation:
    """Learned vs. reference outputs on a set of test states.

    Attributes:
        inputs: Test states ``ρ_j``, shape ``(M, d, d)``.
        outputs: Learned process applied to each input, shape ``(M, d, d)``.
        targets: Reference process applied to each input, shape ``(M, d, d)``.
        fidelities: ``F(output_j, target_j)``, shape ``(M,)``.
    """

    inputs: np.ndarray
    outputs: np.ndarray
    targets: np.ndarray
    fidelities: np.ndarray

    @classmethod
    def from_model(
        cls,
        tomo: KrausTomography,
        states: Sequence[ArrayLike],
        target_process_fn: Callable[[ArrayLike], ArrayLike],
        metric_fn: Callable[[ArrayLike, ArrayLike], Any] = metric.compilation_trace_fidelity,
    ) -> Evaluation:
        """Run ``tomo.evaluate`` on every state and collect the results."""
        if not states:
            raise ValueError("states must not be empty.")
        outputs, targets, scores = [], [], []
        for rho in states:
            out, tgt, score = tomo.evaluate(rho, target_process_fn, metric_fn)
            outputs.append(out)
            targets.append(tgt)
            scores.append(float(tf.math.real(score)))
        return cls(
            inputs=_stack_complex(states),
            outputs=_stack_complex(outputs),
            targets=_stack_complex(targets),
            fidelities=np.asarray(scores, dtype=np.float64),
        )

    @property
    def mean_infidelity(self) -> float:
        return float(1 - np.mean(self.fidelities))


@dataclass(frozen=True, eq=False)
class TomographyResult:
    """Everything needed to reproduce the figures of one training run.

    ``initial_params`` and ``params`` always have shape ``(k, d, d)``: the ``k`` Kraus
    operators of a ``model == "kraus"`` run.
    ``metadata`` holds JSON-serializable run settings (``seed``, ``lr``, ``run``, ...).
    """

    model: str
    num_qubits: int
    loss_history: np.ndarray
    training_states: np.ndarray
    initial_params: np.ndarray
    params: np.ndarray
    fit_seconds: float = float("nan")
    evaluation: Evaluation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model(
        cls, tomo: KrausTomography, evaluation: Evaluation | None = None, **metadata: Any
    ) -> TomographyResult:
        """Snapshot a fitted model (plus optional evaluation and run settings).

        ``tomo`` chỉ cần có các thuộc tính mà ``KrausTomography`` trong TODO yêu cầu:
        ``is_fitted``, ``num_qubits``, ``target_rho_set``, ``init_kraus``, ``kraus_operators``,
        ``loss_history``, ``fit_seconds``.
        """
        if not tomo.is_fitted:
            raise RuntimeError("Call fit() before creating a TomographyResult.")
        initial, learned = tomo.init_kraus, tomo.kraus_operators
        return cls(
            model="kraus",
            num_qubits=tomo.num_qubits,
            loss_history=np.asarray(tomo.loss_history, dtype=np.float64),
            training_states=_stack_complex(tomo.target_rho_set),
            initial_params=_stack_complex(initial),
            params=_stack_complex(learned),
            fit_seconds=float(tomo.fit_seconds) if tomo.fit_seconds is not None else float("nan"),
            evaluation=evaluation,
            metadata=dict(metadata),
        )

    @property
    def final_loss(self) -> float:
        """Training loss (mean infidelity on the training states) at the last epoch."""
        return float(self.loss_history[-1])

    @property
    def test_infidelity(self) -> float:
        """``1 - mean fidelity`` on the evaluation states (``nan`` if not evaluated)."""
        return self.evaluation.mean_infidelity if self.evaluation is not None else float("nan")

    def to_npz(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Write this result to ``path`` (``.npz`` is appended if missing). Returns the final path.

        The file is a plain NumPy archive (no pickling), so it also opens with ``np.load(path)``.
        """
        path = _with_suffix(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it).")
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays: dict[str, np.ndarray] = {
            "format_version": np.array(FORMAT_VERSION),
            "model": np.array(self.model),
            "num_qubits": np.array(self.num_qubits),
            "loss_history": np.asarray(self.loss_history, dtype=np.float64),
            "training_states": np.asarray(self.training_states, dtype=np.complex128),
            "initial_params": np.asarray(self.initial_params, dtype=np.complex128),
            "params": np.asarray(self.params, dtype=np.complex128),
            "fit_seconds": np.array(self.fit_seconds, dtype=np.float64),
            "metadata": np.array(json.dumps(self.metadata, default=_json_default, sort_keys=True)),
        }
        if self.evaluation is not None:
            for key, attr in _EVAL_KEYS.items():
                arrays[key] = np.asarray(getattr(self.evaluation, attr))

        with path.open("wb") as fh:
            np.savez_compressed(fh, **arrays)
        return path

    @classmethod
    def from_npz(cls, path: str | os.PathLike[str]) -> TomographyResult:
        """Read a file written by :meth:`to_npz`."""
        with np.load(_with_suffix(path), allow_pickle=False) as data:
            version = int(data["format_version"])
            if version != FORMAT_VERSION:
                raise ValueError(f"Unsupported result format version {version} (expected {FORMAT_VERSION}).")
            evaluation = None
            if all(key in data for key in _EVAL_KEYS):
                evaluation = Evaluation(**{attr: data[key] for key, attr in _EVAL_KEYS.items()})
            return cls(
                model=str(data["model"]),
                num_qubits=int(data["num_qubits"]),
                loss_history=data["loss_history"],
                training_states=data["training_states"],
                initial_params=data["initial_params"],
                params=data["params"],
                fit_seconds=float(data["fit_seconds"]),
                evaluation=evaluation,
                metadata=json.loads(str(data["metadata"])),
            )


def get_value(result: TomographyResult, key: str) -> Any:
    """Look ``key`` up as a result attribute/property first, then in ``result.metadata``."""
    if hasattr(result, key):
        return getattr(result, key)
    try:
        return result.metadata[key]
    except KeyError:
        raise KeyError(f"{key!r} is neither a TomographyResult attribute nor a metadata key.") from None


def group_by(
    results: Iterable[TomographyResult], key: str | Sequence[str]
) -> dict[Any, list[TomographyResult]]:
    """Group results by ``get_value(result, key)``, with groups sorted by key.

    ``key`` may be a sequence of keys, in which case groups are keyed by tuples,
    e.g. ``group_by(results, ("model", "num_qubits"))``.
    """
    keys = (key,) if isinstance(key, str) else tuple(key)
    groups: dict[Any, list[TomographyResult]] = defaultdict(list)
    for result in results:
        value = tuple(get_value(result, k) for k in keys)
        groups[value[0] if len(keys) == 1 else value].append(result)
    return dict(sorted(groups.items()))
