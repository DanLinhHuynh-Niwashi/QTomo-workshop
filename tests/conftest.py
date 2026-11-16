import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence TensorFlow C++ start-up logs

import numpy as np  # noqa: E402
import pytest  # noqa: E402

PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


def random_density_matrix(dim: int = 2, seed: int = 42) -> np.ndarray:
    """Full-rank random density matrix."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


@pytest.fixture
def fixed_probe_states() -> list[np.ndarray]:
    """|0⟩⟨0|, |+⟩⟨+|, |+i⟩⟨+i| — an informationally complete single-qubit set."""
    return [
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),
        np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128),
    ]


SOLUTION_FILE = Path(__file__).resolve().parents[1] / "src" / "qtomo_skeleton" / "tomography-solution.py"


def has_solution() -> bool:
    """File đáp án chỉ có ở nhánh ``solution``."""
    return SOLUTION_FILE.is_file()


def load_solution_tomography():
    """Nạp ``tomography-solution.py`` (tên có dấu ``-`` nên không import thường được) thành một module."""
    if not has_solution():
        pytest.skip("tomography-solution.py chỉ có ở nhánh solution")
    name = "qtomo_skeleton.tomography_solution"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, SOLUTION_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def kraus_tomography_class():
    """``KrausTomography`` dùng cho test của các module khác (writer, …).

    Nhánh ``solution``: lớp trong file đáp án. Nhánh ``main``: lớp của sinh viên trong
    ``qtomo_skeleton.tomography`` — test nào gọi tới method còn ``NotImplementedError`` thì skip.
    """
    if has_solution():
        return load_solution_tomography().KrausTomography
    from qtomo_skeleton.tomography import KrausTomography

    return KrausTomography


def skip_if_not_implemented(fn, *args, **kwargs):
    """Gọi ``fn``; nếu chạm vào TODO chưa làm (``NotImplementedError``) thì skip test."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        pytest.skip(f"chưa viết: {exc}")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
