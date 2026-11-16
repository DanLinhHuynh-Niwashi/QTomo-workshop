"""Save / load :class:`TomographyResult` as ``.npz`` files.

The files are plain NumPy archives: no pickling, so they open directly with
``np.load(path)`` in any NumPy-only environment::

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
from pathlib import Path
from typing import Any

import numpy as np

from qtomo_skeleton.writer.result import Evaluation, TomographyResult

__all__ = ["FORMAT_VERSION", "ExperimentWriter", "load_result", "load_results", "save_result"]

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


def save_result(
    path: str | os.PathLike[str], result: TomographyResult, *, overwrite: bool = False
) -> Path:
    """Write ``result`` to ``path`` (``.npz`` is appended if missing). Returns the final path."""
    path = _with_suffix(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists (pass overwrite=True to replace it).")
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "format_version": np.array(FORMAT_VERSION),
        "model": np.array(result.model),
        "num_qubits": np.array(result.num_qubits),
        "loss_history": np.asarray(result.loss_history, dtype=np.float64),
        "training_states": np.asarray(result.training_states, dtype=np.complex128),
        "initial_params": np.asarray(result.initial_params, dtype=np.complex128),
        "params": np.asarray(result.params, dtype=np.complex128),
        "fit_seconds": np.array(result.fit_seconds, dtype=np.float64),
        "metadata": np.array(json.dumps(result.metadata, default=_json_default, sort_keys=True)),
    }
    if result.evaluation is not None:
        for key, attr in _EVAL_KEYS.items():
            arrays[key] = np.asarray(getattr(result.evaluation, attr))

    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)
    return path


def load_result(path: str | os.PathLike[str]) -> TomographyResult:
    """Read a file written by :func:`save_result`."""
    with np.load(_with_suffix(path), allow_pickle=False) as data:
        version = int(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"Unsupported result format version {version} (expected {FORMAT_VERSION}).")
        evaluation = None
        if all(key in data for key in _EVAL_KEYS):
            evaluation = Evaluation(**{attr: data[key] for key, attr in _EVAL_KEYS.items()})
        return TomographyResult(
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


def load_results(directory: str | os.PathLike[str], pattern: str = "*.npz") -> list[TomographyResult]:
    """Load every result file in ``directory`` matching ``pattern``, sorted by file name."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory.")
    return [load_result(p) for p in sorted(directory.glob(pattern))]


class ExperimentWriter:
    """Store the runs of one experiment as ``<directory>/<name>.npz`` files.

    Example::

        writer = ExperimentWriter("output/random_process")
        for run in range(num_runs):
            ...
            writer.save(TomographyResult.from_model(tomo, run=run))
        results = writer.load_all()
    """

    def __init__(self, directory: str | os.PathLike[str], *, overwrite: bool = False) -> None:
        self.directory = Path(directory)
        self.overwrite = overwrite
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, result: TomographyResult, name: str | None = None) -> Path:
        """Save ``result`` as ``name.npz`` (default: the next free ``run_XXXX``)."""
        if name is None:
            name = self._next_run_name()
        return save_result(self.directory / name, result, overwrite=self.overwrite)

    def load_all(self) -> list[TomographyResult]:
        return load_results(self.directory)

    def _next_run_name(self) -> str:
        index = 0
        while (self.directory / f"run_{index:04d}.npz").exists():
            index += 1
        return f"run_{index:04d}"
