"""Chạy ``experiment_run.ipynb`` từ đầu đến cuối với cấu hình nhỏ.

Ngoài việc notebook chạy được và ra hình, test còn gắn thêm một ô cuối đối chiếu Phần 1 (unpack, không class)
với ``KrausTomography`` trong ``tomography-solution.py`` cùng seed: trạng thái train, Kraus khởi tạo và loss
từng epoch phải trùng.
Đây là bảo đảm cho sinh viên: copy đúng các khối [B1]–[B8] vào class là đủ.
"""

import sys
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
from conftest import kraus_tomography_class  # noqa: E402
from jupyter_client.manager import KernelManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "experiment_run.ipynb"

SMALL_CONFIG = """
NUM_QUBITS = 1
SEED = 100
QUBITS = [1]
RUNS = 1
NUM_RHO = 3
NUM_KRAUS = 1
NUM_TEST = 3
EPOCHS = 150
LR = 0.05
LOG_EVERY = 50
OUTPUT_DIR = Path({output!r})
FIGURE_DIR = OUTPUT_DIR / "figures"
"""

COMPARE_WITH_REFERENCE = """
import numpy as np
import importlib.util

_solution_file = Path("src") / "qtomo_skeleton" / "tomography-solution.py"
if _solution_file.is_file():  # nhánh solution: so với đáp án
    _spec = importlib.util.spec_from_file_location("tomography_solution", _solution_file)
    _solution = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_solution)
    RefKraus = _solution.KrausTomography
else:  # nhánh main: so với lớp của sinh viên
    from qtomo_skeleton import KrausTomography as RefKraus
from qtomo_skeleton import quantum_ops as ref_ops
from qtomo_skeleton.optimizer import AdamOptimizer as RefAdam

ref = RefKraus(NUM_QUBITS, NUM_RHO, NUM_KRAUS, seed=SEED)
for a, b in zip(target_rho_set, ref.target_rho_set):
    np.testing.assert_allclose(a, b)
for a, b in zip(init_kraus, ref.init_kraus):
    np.testing.assert_allclose(a, b)
_, ref_history = ref.fit(
    data_process_fn=lambda rhos, ks: [
        ref_ops.apply_unitary_dagger(ref_ops.apply_kraus_operators(r, ks), target) for r in rhos
    ],
    epochs=EPOCHS,
    optimizer=RefAdam(alpha=LR),
)
np.testing.assert_allclose(loss_history, ref_history, atol=1e-10)
for a, b in zip(kraus_operators, ref.kraus_operators):
    np.testing.assert_allclose(a, b, atol=1e-8)
print("UNPACKED == KrausTomography")
"""


def _cell_with_tag(nb, tag):
    matches = [c for c in nb.cells if tag in c.metadata.get("tags", [])]
    assert len(matches) == 1, f"notebook phải có đúng một ô gắn tag {tag!r}"
    return matches[0]


@pytest.mark.slow
def test_experiment_notebook_runs_and_matches_reference(tmp_path):
    nb = nbformat.read(NOTEBOOK, as_version=4)
    _cell_with_tag(nb, "config").source = SMALL_CONFIG.format(output=str(tmp_path / "out"))
    _cell_with_tag(nb, "final-check")  # ô kiểm tra cuối phải còn và phải chạy qua
    try:  # có KrausTomography hoàn chỉnh (đáp án hoặc bài của sinh viên) thì đối chiếu thêm
        kraus_tomography_class()(1, 1, seed=0).is_fitted
        compare = True
    except NotImplementedError:
        compare = False
    if compare:
        nb.cells.append(nbformat.v4.new_code_cell(COMPARE_WITH_REFERENCE))

    km = KernelManager(kernel_name="python3")
    km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = nbclient.NotebookClient(nb, km=km, timeout=600, resources={"metadata": {"path": str(ROOT)}})
    client.execute()

    figures = tmp_path / "out" / "figures"
    for name in ("fig2.png", "fig2.eps", "fig3.png", "fig3.eps"):
        assert (figures / name).is_file(), name
    assert (tmp_path / "out" / "1q_run0.npz").is_file()

    if compare:
        last_output = "".join(o.get("text", "") for o in nb.cells[-1].outputs)
        assert "UNPACKED == KrausTomography" in last_output
    images = [o for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    assert len(images) >= 2, "Fig. 2 và Fig. 3 phải hiển thị inline"


# ----------------------------------------------------------------- packed notebooks (TODO + solution)

PACKED = ROOT / "experiment_run_packed.ipynb"
PACKED_SOLUTION = ROOT / "experiment_run_packed_solution.ipynb"


needs_solution = pytest.mark.skipif(not PACKED_SOLUTION.is_file(), reason="lời giải chỉ có ở nhánh solution")


@needs_solution
def test_packed_notebooks_stay_in_sync():
    """Bản TODO và bản lời giải chỉ được khác nhau ở ô ``setup`` và các ô ``todo``."""
    todo_nb = nbformat.read(PACKED, as_version=4)
    sol_nb = nbformat.read(PACKED_SOLUTION, as_version=4)
    assert len(todo_nb.cells) == len(sol_nb.cells)
    num_todo = 0
    for i, (a, b) in enumerate(zip(todo_nb.cells, sol_nb.cells)):
        tags = a.metadata.get("tags", [])
        assert a.cell_type == b.cell_type and tags == b.metadata.get("tags", []), f"ô {i}"
        if "todo" in tags:
            num_todo += 1
            assert "TODO" in a.source and "TODO" not in b.source, f"ô {i}"
        elif "setup" not in tags:
            assert a.source == b.source, f"ô {i} khác nhau giữa bản TODO và lời giải"
    assert num_todo == 5  # P1–P5 (quét + Phần 3 cho sẵn)
    assert not any(c.get("outputs") for c in todo_nb.cells if c.cell_type == "code"), "bản TODO không lưu output"


@pytest.mark.slow
@needs_solution
def test_packed_solution_notebook_runs(tmp_path):
    nb = nbformat.read(PACKED_SOLUTION, as_version=4)
    config = _cell_with_tag(nb, "config")
    config.source = SMALL_CONFIG.format(output=str(tmp_path / "out")).replace("LOG_EVERY = 50\n", "")
    _cell_with_tag(nb, "final-check")

    km = KernelManager(kernel_name="python3")
    km.kernel_cmd = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    nbclient.NotebookClient(nb, km=km, timeout=600, resources={"metadata": {"path": str(ROOT)}}).execute()

    for name in ("fig2.png", "fig3.png", "fig2.eps", "fig3.eps"):
        assert (tmp_path / "out" / "figures" / name).is_file(), name
    images = [o for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    assert len(images) >= 2
