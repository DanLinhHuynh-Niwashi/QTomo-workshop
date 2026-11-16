"""Helpers shared by the plotting modules."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from qtomo_skeleton.writer import TomographyResult, get_value

__all__ = ["save_figure"]

GRID_STYLE: dict[str, Any] = {"which": "both", "linestyle": "--", "linewidth": 0.5}
TICK_LABEL_SIZE = 12

#: Values at or below zero cannot be shown on a log axis. Infidelities of a
#: converged run come out as +-1e-9 rounding noise, so they are drawn at this floor.
LOG_FLOOR = 1e-12

REDUCERS = {"mean": np.mean, "min": np.min, "max": np.max, "median": np.median}


def new_axes(ax: Axes | None, figsize: tuple[float, float] = (5.0, 4.0)) -> tuple[Figure, Axes]:
    """Return ``(figure, ax)``, creating a standalone figure when ``ax`` is ``None``."""
    if ax is not None:
        return ax.figure, ax
    fig = Figure(figsize=figsize, layout="tight")
    return fig, fig.add_subplot()


def style_axes(
    ax: Axes, xlabel: str, ylabel: str, *, logx: bool = False, logy: bool = False, integer_x: bool = False
) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    if integer_x:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, **GRID_STYLE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()


def values(results: Iterable[TomographyResult], key: str) -> np.ndarray:
    return np.array([get_value(r, key) for r in results], dtype=float)


def for_log(ys: np.ndarray, log: bool) -> np.ndarray:
    """Clip ``ys`` to :data:`LOG_FLOOR` when drawn on a log axis."""
    return np.maximum(ys, LOG_FLOOR) if log else ys


def is_integral(xs: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(xs)) and np.all(xs == np.round(xs)))


def format_label(fmt: str, key: Any) -> str:
    return fmt.format(*key) if isinstance(key, tuple) else fmt.format(key)


def reduce_by(xs: np.ndarray, ys: np.ndarray, how: str = "mean") -> tuple[np.ndarray, np.ndarray]:
    """Reduce ``ys`` over equal ``xs`` (``how`` in ``REDUCERS``); returns sorted unique ``xs``."""
    if how not in REDUCERS:
        raise ValueError(f"reduce must be one of {sorted(REDUCERS)}, got {how!r}.")
    unique = np.unique(xs)
    return unique, np.array([REDUCERS[how](ys[xs == x]) for x in unique])


def save_figure(
    fig: Figure, path: str | os.PathLike[str], formats: Sequence[str] = ("png",), dpi: int = 200
) -> list[Path]:
    """Save ``fig`` as ``path.<fmt>`` for every format (e.g. ``("png", "eps")``); returns the paths."""
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    stem = base.with_suffix("") if base.suffix.lstrip(".") in formats else base
    written = []
    for fmt in formats:
        target = stem.with_name(f"{stem.name}.{fmt}")
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches="tight")
        written.append(target)
    return written
