"""Quantum process tomography với mô hình Kraus — PHẦN SINH VIÊN ĐIỀN.

``__init__`` đã viết sẵn. Các method khác đã có tên, tham số, type hint và docstring; thân hàm chỉ là
``raise NotImplementedError``. Việc của bạn: **thay mỗi ``raise NotImplementedError`` bằng code thật**.

Nguyên liệu nằm trong notebook ``experiment_run.ipynb`` (ở ``workshop/``): cùng thí nghiệm nhưng viết "unpack",
không dùng class, chia thành các khối ``[B1]`` … ``[B9]``. Mỗi TODO bên dưới chỉ ra khối cần copy và
những gì phải đổi khi đưa vào class (biến cục bộ -> ``self.<tên>``, ``print`` -> ``logger.info``, …).

Kiểm tra: ``python -m pytest tests/test_skeleton.py -v`` — test của method chưa viết sẽ báo SKIPPED,
viết xong method nào thì test của method đó chuyển sang PASSED (hoặc FAILED nếu sai).
"""

from __future__ import annotations

import logging
import time  # noqa: F401  (dùng trong fit — TODO 3)
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

# linalg, quantum_ops: dùng trong TODO 3, 4
from qtomo_skeleton import generator, gradient, linalg, metric, quantum_ops  # noqa: F401
from qtomo_skeleton._types import ArrayLike, DensityMatrixList, GradientFn, LossFn
from qtomo_skeleton.optimizer import AdamOptimizer, Optimizer  # noqa: F401  (AdamOptimizer: TODO 3)

__all__ = ["KrausTomography"]

logger = logging.getLogger(__name__)

#: ``data_process_fn(training_states, kraus_operators) -> processed_states`` (khối [B4]).
DataProcessFn = Callable[[DensityMatrixList, list[tf.Tensor]], DensityMatrixList]
#: ``target_process_fn(rho) -> rho_out`` – process tham chiếu dùng trong :meth:`KrausTomography.evaluate`.
TargetProcessFn = Callable[[ArrayLike], ArrayLike]
#: ``metric_fn(rho_learned, rho_target) -> score``.
MetricFn = Callable[[ArrayLike, ArrayLike], Any]


class KrausTomography:
    """Học một process CPTP dưới dạng ``num_kraus`` toán tử Kraus ``{K_i}``.

    Training cực tiểu hoá ``loss_fn(data_process_fn(ρ_set, K), ρ_set)``: người dùng mô tả cách ghép
    process đã học với process cần tìm qua ``data_process_fn``, sao cho fit hoàn hảo đưa mỗi ``ρ_i``
    về chính nó.

    Ví dụ (học unitary ``E`` bằng một toán tử Kraus)::

        tomo = KrausTomography(num_qubits=1, num_rho=3, num_kraus=1, seed=0)
        kraus, history = tomo.fit(data_process_fn=lambda rhos, ks: [
            apply_unitary_dagger(apply_kraus_operators(r, ks), E) for r in rhos
        ])
        rho_out, rho_ref, fid = tomo.evaluate(rho, lambda r: apply_unitary(r, E))

    Args:
        num_qubits: Số qubit mà process tác động.
        num_rho: Số trạng thái train Haar-random (mặc định ``6**num_qubits``).
            Bị bỏ qua khi có ``training_states``.
        num_kraus: Số toán tử Kraus cần học.
        training_states: Tập trạng thái train cho trước (ví dụ để test tất định).
        seed: Seed cho trạng thái train và tham số khởi tạo.
        verbose: Log loss sau mỗi epoch (qua :mod:`logging`, mức ``INFO``).
    """

    # ------------------------------------------------- ĐÃ VIẾT SẴN: [B1] [B2] [B3]

    def __init__(
        self,
        num_qubits: int,
        num_rho: int | None = None,
        num_kraus: int = 1,
        *,
        training_states: Sequence[ArrayLike] | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        # [B1] kiểm tra input + MỘT Generator cho mọi thứ ngẫu nhiên của model.
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}.")
        self.num_qubits = num_qubits
        self.verbose = verbose
        self._rng = np.random.default_rng(seed)

        # [B2] tập trạng thái train (dùng RNG TRƯỚC [B3] để cùng seed cho cùng kết quả).
        self.target_rho_set: list[np.ndarray] = self._build_training_set(num_rho, training_states)

        # [B3] toán tử Kraus khởi tạo, đã chuẩn hoá Σ K†K = I.
        self.init_kraus: list[tf.Tensor] = generator.kraus_operators(num_qubits, num_kraus, self._rng)

        # Trạng thái "chưa train" — fit() sẽ ghi đè các giá trị này ([B7]).
        self.kraus_operators: list[tf.Tensor] | None = None
        self.loss_history: list[float] = []
        #: Thời gian (giây) của lần fit() gần nhất.
        self.fit_seconds: float | None = None

    def _build_training_set(
        self, num_rho: int | None, training_states: Sequence[ArrayLike] | None
    ) -> list[np.ndarray]:
        """[B2]: sinh trạng thái Haar, hoặc kiểm tra và copy ``training_states`` của người dùng."""
        if training_states is None:
            return generator.haar_probe_states(self.num_qubits, num_rho, self._rng)

        dim = 2**self.num_qubits
        states = [np.array(rho, dtype=np.complex128) for rho in training_states]
        if not states:
            raise ValueError("training_states must not be empty.")
        for rho in states:
            if rho.shape != (dim, dim):
                raise ValueError(f"Training state has shape {rho.shape}, expected {(dim, dim)}.")
        return states

    # ------------------------------------------------------ SINH VIÊN ĐIỀN TỪ ĐÂY

    @property
    def is_fitted(self) -> bool:
        """``True`` khi :meth:`fit` đã chạy (đã có ``kraus_operators``)."""
        # TODO 1: một dòng — xem chú thích "is_fitted" ở khối [B7] trong experiment_run.ipynb.
        raise NotImplementedError("TODO 1: KrausTomography.is_fitted")

    def set_logging(self, is_logging: bool) -> None:
        """Bật / tắt log loss theo từng epoch."""
        # TODO 2: một dòng — gán cho thuộc tính đã tạo trong __init__.
        raise NotImplementedError("TODO 2: KrausTomography.set_logging")

    def fit(
        self,
        *,
        data_process_fn: DataProcessFn,
        epochs: int = 100,
        gradient_fn: GradientFn = gradient.riemann,
        loss_fn: LossFn = metric.mean_infidelity,
        optimizer: Optimizer | None = None,
    ) -> tuple[list[tf.Tensor], list[float]]:
        """Train các toán tử Kraus, luôn bắt đầu lại từ ``init_kraus``.

        Mỗi lần gọi đều reset optimizer, nên gọi lặp lại (ví dụ quét tham số) là độc lập nhau.

        Args:
            data_process_fn: ``(training_states, kraus_operators) -> processed_states`` (khối [B4]).
            epochs: Số bước tối ưu (``>= 1``).
            gradient_fn: Quy tắc gradient, xem :mod:`qtomo_skeleton.gradient`.
            loss_fn: ``(processed_states, training_states) -> scalar``.
            optimizer: Optimizer (mặc định ``AdamOptimizer(alpha=0.01)``).

        Returns:
            ``(kraus_operators, loss_history)``; ``loss_history[i]`` là phần thực của loss trước bước ``i``.

        Raises:
            ValueError: nếu ``epochs < 1``.
        """
        # TODO 3: copy khối [B5], [B6], [B7] của experiment_run.ipynb rồi sửa:
        #   a) Kiểm tra ``epochs`` (ValueError). ``optimizer is None`` -> ``AdamOptimizer(alpha=0.01)``.
        #      Giữ ``optimizer.reset()``. Câu hỏi: nếu bỏ dòng này thì gọi fit lần 2 sai ở đâu?
        #   b) [B5]: closure ``loss_of_param`` dùng tham số ``data_process_fn``/``loss_fn`` của hàm này
        #      và ``self.target_rho_set``.
        #   c) [B6]: bắt đầu từ ``self.init_kraus``; ``print`` -> ``logger.info`` khi ``self.verbose``.
        #   d) [B7]: ghi ``self.kraus_operators``, ``self.loss_history``, ``self.fit_seconds``;
        #      ``return self.kraus_operators, history``.
        raise NotImplementedError("TODO 3: KrausTomography.fit")

    def evaluate(
        self,
        rho_input: ArrayLike,
        target_process_fn: TargetProcessFn,
        metric_fn: MetricFn = metric.compilation_trace_fidelity,
    ) -> tuple[tf.Tensor, ArrayLike, Any]:
        """So sánh process đã học với process tham chiếu trên MỘT trạng thái đầu vào.

        Returns:
            ``(rho_learned, rho_target, metric_fn(rho_learned, rho_target))``.

        Raises:
            RuntimeError: nếu gọi trước :meth:`fit`.
        """
        # TODO 4: thân vòng for của khối [B8] (chỉ cho một ``rho_input``, không vòng for):
        #   a) Chưa fit -> ``raise RuntimeError("Call fit() before evaluate().")`` (dùng ``is_fitted``).
        #   b) ``rho_output`` từ ``self.kraus_operators``; ``rho_target`` từ ``target_process_fn``.
        #   c) Trả ``(rho_output, rho_target, metric_fn(rho_output, rho_target))`` — để score nguyên
        #      dạng, người dùng tự lấy phần thực.
        raise NotImplementedError("TODO 4: KrausTomography.evaluate")


# TODO 5 (khi đã xong 1–4):
#   * ``python -m pytest tests/test_skeleton.py -v`` — không còn SKIPPED.
#   * Viết lại thí nghiệm bằng class: [B1]–[B3] -> ``KrausTomography(...)``, [B5]–[B7] -> ``tomo.fit(...)``,
#     [B8] -> vòng for gọi ``tomo.evaluate(...)``, [B9] -> ``TomographyResult.from_model(...)``.
#     Cùng seed thì ``loss_history`` phải TRÙNG với ``loss_history`` ở Phần 1 của notebook.
