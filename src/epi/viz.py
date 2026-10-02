"""Estilo comum dos gráficos do projeto, sem depender do pyplot nem de backend gráfico."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

SERIES: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def new_figure(width: float = 7.0, height: float = 4.0) -> tuple[Figure, Axes]:
    figure = Figure(figsize=(width, height), dpi=150, facecolor=SURFACE, layout="constrained")
    ax = figure.add_subplot()
    style_axes(ax)
    return figure, ax


def set_title(ax: Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=30 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.06, subtitle, transform=ax.transAxes, color=INK_SECONDARY, fontsize=9)


def thousands(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def style_legend(ax: Axes, **kwargs: object) -> None:
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, **kwargs)
    for line in legend.get_lines():
        line.set_linewidth(2)
