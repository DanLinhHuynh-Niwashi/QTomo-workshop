"""Two-panel figures matching the layouts used in the paper."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.figure import Figure

from qtomo_skeleton.plot.curves import plot_metric_vs
from qtomo_skeleton.writer import TomographyResult

__all__ = ["figure_noise_sweep"]

FIGSIZE = (10, 4)


def figure_noise_sweep(
    results: Sequence[TomographyResult], x: str = "gamma", reduce: str = "mean"
) -> Figure:
    """Paper fig. 4: (a) training and (b) test infidelity vs. noise strength, per qubit count.

    With several runs per point, ``reduce`` combines them (``"min"`` = best restart).
    """
    fig = Figure(figsize=FIGSIZE, layout="tight")
    ax_train, ax_test = fig.subplots(1, 2, sharey=True)
    for ax, y in ((ax_train, "final_loss"), (ax_test, "test_infidelity")):
        plot_metric_vs(
            results, x, y, group="num_qubits", label_fmt="{} qubit(s)", reduce=reduce, logx=True, ax=ax
        )
    return fig
