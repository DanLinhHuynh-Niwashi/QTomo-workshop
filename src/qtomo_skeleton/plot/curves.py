"""Line plots built from :class:`~qtomo.writer.TomographyResult` lists.

Each function draws on ``ax`` if given (so panels can be composed) or on a new
standalone figure, and returns the :class:`~matplotlib.figure.Figure`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from qtomo_skeleton.plot._common import (
    for_log,
    format_label,
    is_integral,
    new_axes,
    reduce_by,
    style_axes,
    values,
)
from qtomo_skeleton.writer import TomographyResult, group_by

__all__ = ["plot_curves_with_inset", "plot_fidelity_curves", "plot_metric_vs"]

#: Human-readable axis labels for common keys.
LABELS: dict[str, str] = {
    "final_loss": r"Infidelity $\rho_i$ vs $\rho_f$ (train)",
    "test_infidelity": r"Infidelity $\rho_\mathcal{E}$ vs $\rho_k$ (test)",
    "num_qubits": "Number of qubits",
    "gamma": r"$\gamma$",
}

MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def _label(key: str) -> str:
    return LABELS.get(key, key)


def plot_fidelity_curves(
    results: Sequence[TomographyResult],
    *,
    group: str | Sequence[str] | None = "num_qubits",
    label_fmt: str = "{} qubit(s)",
    ax: Axes | None = None,
) -> Figure:
    """Training fidelity ``1 - loss`` vs. iteration (paper fig. 2).

    Assumes the default ``loss_fn=metric.mean_infidelity``, so ``1 - loss`` is the
    mean fidelity between the processed and the original training states.

    With ``group`` set, runs sharing that key are averaged into one curve
    (shaded band = min..max over runs); with ``group=None`` every run is drawn.
    """
    if not results:
        raise ValueError("results must not be empty.")
    fig, ax = new_axes(ax)
    if group is None:
        for i, result in enumerate(results):
            fidelity = 1 - result.loss_history
            ax.plot(np.arange(1, len(fidelity) + 1), fidelity, label=f"run {i}")
    else:
        for key, runs in group_by(results, group).items():
            curves = 1 - _stack_equal_length([r.loss_history for r in runs])
            iterations = np.arange(1, curves.shape[1] + 1)
            (line,) = ax.plot(iterations, curves.mean(axis=0), label=format_label(label_fmt, key))
            if len(runs) > 1:
                ax.fill_between(
                    iterations, curves.min(axis=0), curves.max(axis=0), color=line.get_color(), alpha=0.2
                )
    style_axes(ax, "Iteration", "Fidelity")
    return fig


def plot_metric_vs(
    results: Sequence[TomographyResult],
    x: str,
    y: str = "test_infidelity",
    *,
    group: str | Sequence[str] | None = None,
    label_fmt: str = "{}",
    reduce: str = "mean",
    logx: bool = False,
    logy: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Plot result key ``y`` against key ``x`` (attributes or metadata), one line per ``group``.

    Runs with the same ``x`` inside a group are combined with ``reduce``
    (``"mean"``, ``"min"``, ``"max"`` or ``"median"``); ``reduce="min"`` on a loss
    keeps the best of several restarts. On a log axis, values at or below zero
    (rounding noise of converged runs) are drawn at ``LOG_FLOOR``. Example::

        plot_metric_vs(results, "gamma", "final_loss", group="num_qubits", logx=True)  # paper fig. 4
    """
    if not results:
        raise ValueError("results must not be empty.")
    fig, ax = new_axes(ax)
    groups = group_by(results, group) if group is not None else {None: list(results)}
    for i, (key, runs) in enumerate(groups.items()):
        xs, ys = reduce_by(values(runs, x), values(runs, y), reduce)
        label = format_label(label_fmt, key) if key is not None else None
        ax.plot(xs, for_log(ys, logy), marker=MARKERS[i % len(MARKERS)], label=label)
    integer_x = is_integral(values(results, x)) and not logx
    style_axes(ax, _label(x), _label(y), logx=logx, logy=logy, integer_x=integer_x)
    return fig


def plot_curves_with_inset(
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    inset_xlim: tuple[float, float] | None = None,
    styles: Mapping[str, Mapping[str, Any]] | None = None,
    xlabel: str = "t",
    ylabel: str = "Expectation value",
    ax: Axes | None = None,
) -> Figure:
    """Several ``label -> (x, y)`` curves, optionally with a zoomed inset (paper fig. 6).

    ``styles`` maps a label to extra ``Axes.plot`` keyword arguments, e.g.
    ``{"learned": {"linestyle": ":", "marker": "o"}}``.
    """
    if not curves:
        raise ValueError("curves must not be empty.")
    fig, ax = new_axes(ax, figsize=(7.5, 5))
    styles = styles or {}
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=label, **styles.get(label, {}))
    style_axes(ax, xlabel, ylabel)
    ax.legend(loc="best")

    if inset_xlim is not None:
        inset = ax.inset_axes((0.55, 0.55, 0.42, 0.4))
        for label, (xs, ys) in curves.items():
            xs, ys = np.asarray(xs), np.asarray(ys)
            mask = (xs >= inset_xlim[0]) & (xs <= inset_xlim[1])
            inset.plot(xs[mask], ys[mask], **styles.get(label, {}))
        inset.set_xlim(inset_xlim)
        inset.grid(True, linestyle="--", alpha=0.5)
        inset.tick_params(axis="both", labelsize=9)
    return fig


def _stack_equal_length(histories: Sequence[np.ndarray]) -> np.ndarray:
    lengths = {len(h) for h in histories}
    if len(lengths) != 1:
        raise ValueError(f"Cannot average curves of different lengths {sorted(lengths)}.")
    return np.stack(histories)
