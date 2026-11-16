"""qtomo_skeleton — bản khung của :mod:`qtomo` cho sinh viên tự hoàn thiện ``KrausTomography``.

Mọi submodule giống hệt :mod:`qtomo`, **trừ** :mod:`qtomo_skeleton.tomography`: ở đó ``__init__`` đã viết
sẵn, các method khác chỉ có chữ ký + docstring + TODO (thân hàm ``raise NotImplementedError``).

* :mod:`qtomo_skeleton.generator`   – probe states, Haar unitaries, Kraus operators, measurements
* :mod:`qtomo_skeleton.quantum_ops` – unitary / Kraus maps and noise channels
* :mod:`qtomo_skeleton.metric`      – fidelities and distances (loss functions)
* :mod:`qtomo_skeleton.gradient`    – gradient rules (Euclidean, Riemannian)
* :mod:`qtomo_skeleton.optimizer`   – SGD and Adam for complex matrix parameters
* :mod:`qtomo_skeleton.tomography`  – **sinh viên điền**: :class:`KrausTomography`
* :mod:`qtomo_skeleton.linalg`      – small linear-algebra helpers
* :mod:`qtomo_skeleton.writer`, :mod:`qtomo_skeleton.plot` – lưu ``.npz`` và vẽ hình

Paper: "Advancing quantum process tomography through universal compilation",
https://arxiv.org/abs/2504.14958
"""

from qtomo_skeleton.optimizer import AdamOptimizer, Optimizer, SGDOptimizer
from qtomo_skeleton.tomography import KrausTomography

__version__ = "0.2.0"

__all__ = [
    "AdamOptimizer",
    "KrausTomography",
    "Optimizer",
    "SGDOptimizer",
    "__version__",
]
