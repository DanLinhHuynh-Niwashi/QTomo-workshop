"""LỜI GIẢI của ``qtomo_skeleton/tomography.py`` — ``KrausTomography`` hoàn chỉnh.

Dành cho giảng viên / để sinh viên so sánh **sau khi** đã tự làm. Tên file có dấu ``-`` nên không import
trực tiếp được — cố ý, để không vô tình dùng thay bản của sinh viên. Cách dùng:

* copy đè lên ``tomography.py``, hoặc
* nạp bằng ``importlib`` như trong ``tests/test_skeleton.py`` và ``experiment_run_packed_solution.ipynb``.

Mỗi method ghi chú khối ``[B?]`` tương ứng trong ``experiment_run.ipynb``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import tensorflow as tf

from qtomo_skeleton import generator, gradient, linalg, metric, quantum_ops
from qtomo_skeleton._types import ArrayLike, DensityMatrixList, GradientFn, LossFn
from qtomo_skeleton.optimizer import AdamOptimizer, Optimizer

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

    # ------------------------------------------------------------- __init__: [B1] [B2] [B3]

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

    # ----------------------------------------------- lời giải: is_fitted, set_logging, fit, evaluate

    @property
    def is_fitted(self) -> bool:
        """``True`` khi :meth:`fit` đã chạy (đã có ``kraus_operators``)."""
        return self.kraus_operators is not None

    def set_logging(self, is_logging: bool) -> None:
        """Bật / tắt log loss theo từng epoch."""
        self.verbose = is_logging

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
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}.")
        optimizer = optimizer if optimizer is not None else AdamOptimizer(alpha=0.01)
        optimizer.reset()  # optimizer có thể được dùng lại giữa các lần fit (vd. quét tham số)

        # [B5] loss theo MỘT tensor tham số shape (k, d, d)
        def loss_of_param(param: tf.Tensor) -> tf.Tensor:
            processed = data_process_fn(self.target_rho_set, tf.unstack(param))
            return loss_fn(processed, self.target_rho_set)

        # [B6] vòng lặp tối ưu, luôn bắt đầu lại từ init_kraus
        param = tf.stack(self.init_kraus)
        history: list[float] = []
        start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            param, loss = optimizer.step_and_cost(param, loss_of_param, gradient_fn)
            param = tf.stack(linalg.normalize_kraus(param))  # chiếu lại về CPTP
            history.append(float(tf.math.real(loss)))
            if self.verbose:
                logger.info("Epoch %d/%d, loss: %.6f", epoch, epochs, history[-1])

        # [B7] lưu kết quả lên instance
        self.fit_seconds = time.perf_counter() - start
        self.kraus_operators = tf.unstack(param)
        self.loss_history = history
        return self.kraus_operators, history

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
        # [B8] cho một trạng thái
        if not self.is_fitted:
            raise RuntimeError("Call fit() before evaluate().")
        rho_output = quantum_ops.apply_kraus_operators(rho_input, self.kraus_operators)
        rho_target = target_process_fn(rho_input)
        return rho_output, rho_target, metric_fn(rho_output, rho_target)

