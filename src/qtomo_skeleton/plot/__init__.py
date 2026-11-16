"""Figures for tomography results (requires ``matplotlib``: ``pip install qtomo[plot]``).

Every function returns a :class:`matplotlib.figure.Figure` and never calls
``plt.show()`` or writes files on its own; use :func:`save_figure` to export.
Inputs are :class:`~qtomo.writer.TomographyResult` objects (see :mod:`qtomo_skeleton.writer`),
so no text parsing or folder-name conventions are involved.

==============================  ===================================================
Paper figure                    Function
==============================  ===================================================
Fig. 2 (fidelity vs iteration)  :func:`plot_fidelity_curves`
Fig. 3 / 5 (density matrices)   :func:`plot_density_matrices`
Fig. 4 (noise sweep)            :func:`figure_noise_sweep`
Fig. 6 (expectation + inset)    :func:`plot_curves_with_inset`
==============================  ===================================================
"""

from qtomo_skeleton.plot._common import save_figure
from qtomo_skeleton.plot.curves import plot_curves_with_inset, plot_fidelity_curves, plot_metric_vs
from qtomo_skeleton.plot.figures import figure_noise_sweep
from qtomo_skeleton.plot.matrices import plot_density_matrices

__all__ = [
    "figure_noise_sweep",
    "plot_curves_with_inset",
    "plot_density_matrices",
    "plot_fidelity_curves",
    "plot_metric_vs",
    "save_figure",
]
