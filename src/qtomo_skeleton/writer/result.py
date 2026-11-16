"""In-memory records of a tomography run."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import tensorflow as tf

from qtomo_skeleton import metric
from qtomo_skeleton._types import ArrayLike

if TYPE_CHECKING:  # chỉ dùng cho type hint: writer không phụ thuộc vào tomography.py (file của sinh viên)
    from qtomo_skeleton.tomography import KrausTomography

__all__ = ["Evaluation", "TomographyResult", "get_value", "group_by"]


def _stack_complex(matrices: Iterable[ArrayLike]) -> np.ndarray:
    return np.stack([np.asarray(m, dtype=np.complex128) for m in matrices])


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
