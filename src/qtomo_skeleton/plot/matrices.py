"""3D bar charts of density matrices (paper figs. 3 and 5)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from matplotlib.figure import Figure

from qtomo_skeleton import linalg
from qtomo_skeleton._types import ArrayLike

__all__ = ["plot_density_matrices"]

POSITIVE_COLOR = "teal"
NEGATIVE_COLOR = "red"
BAR_WIDTH = 0.5


def _basis_labels(dim: int) -> list[str]:
    n = linalg.num_qubits_of(np.zeros((dim, dim)))
    return [rf"$|{i:0{n}b}\rangle$" for i in range(dim)]


def plot_density_matrices(matrices: Mapping[str, ArrayLike], *, zlim: float | None = None) -> Figure:
    """One row per matrix, columns = real and imaginary part, as 3D bar charts.

    Args:
        matrices: ``label -> (d, d)`` matrix, e.g. ``{r"$\\rho_\\mathcal{E}$": rho_target,
            r"$\\rho_k$": rho_learned}``. All matrices must have the same shape.
        zlim: Symmetric z-range ``[-zlim, zlim]``; defaults to the largest absolute entry
            over all matrices so the panels are directly comparable.
    """
    if not matrices:
        raise ValueError("matrices must not be empty.")
    arrays = {label: np.asarray(m, dtype=np.complex128) for label, m in matrices.items()}
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"All matrices must have the same shape, got {sorted(shapes)}.")
    dim = next(iter(shapes))[0]
    if zlim is None:
        zlim = max(max(np.abs(a.real).max(), np.abs(a.imag).max()) for a in arrays.values()) or 1.0

    xs, ys = (g.ravel() for g in np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij"))
    ticks = np.arange(dim) + BAR_WIDTH / 2
    labels = _basis_labels(dim)

    fig = Figure(figsize=(10, 5 * len(arrays)), layout="constrained")
    for row, (label, array) in enumerate(arrays.items()):
        for col, (part_name, part) in enumerate((("Real part", array.real), ("Imaginary part", array.imag))):
            ax = fig.add_subplot(len(arrays), 2, 2 * row + col + 1, projection="3d")
            heights = part.ravel()
            colors = [POSITIVE_COLOR if h >= 0 else NEGATIVE_COLOR for h in heights]
            ax.bar3d(xs, ys, np.zeros_like(heights), BAR_WIDTH, BAR_WIDTH, heights, color=colors, alpha=0.7)
            ax.set_title(f"{label} — {part_name}", fontsize=14)
            ax.set_xticks(ticks, labels)
            ax.set_yticks(ticks, labels)
            ax.set_zlim(-zlim, zlim)
    return fig
