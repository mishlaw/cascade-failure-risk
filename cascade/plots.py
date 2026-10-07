"""Рисунки: схема сети и распределение шага отказа."""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch

from .stats import decile_boundaries, survival_curve

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
EDGE = "#a9a8a1"

LAYER_COLORS = {"A": "#2a78d6", "B": "#eb6834"}
LAYER_LABEL_INK = {"A": "#ffffff", "B": INK}
BLUE = "#2a78d6"
# один оттенок от светлого к тёмному: чем больше значение, тем темнее
BLUE_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BLUE_SCALE = LinearSegmentedColormap.from_list("blue_scale", ["#cde2fb", "#3987e5", "#0d366b"])


def new_figure(width, height, nrows=1, ncols=1, **kwargs):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), facecolor=SURFACE, **kwargs)
    return fig, axes


def style_axes(ax, grid_axis="y"):
    """Спокойное оформление: тонкая сетка, без рамки, приглушённые подписи осей."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def read_layout(file_path):
    """Координаты узлов из CSV с колонками node,x,y."""
    with open(file_path, newline="", encoding="utf-8") as f:
        return {row["node"]: (float(row["x"]), float(row["y"])) for row in csv.DictReader(f)}


def two_column_layout(graph):
    """Слой A в левом столбце, слой B в правом; номера узлов растут сверху вниз."""
    pos = {}
    for column, nodes in ((0.0, graph.a_nodes), (1.0, graph.b_nodes)):
        for name in nodes:
            pos[name] = (column, -float(name[1:]))
    return pos


def draw_network(graph, pos=None, ax=None, title=None, node_size=420, font_size=9, legend=True):
    """Рисует сеть: слой A синим, слой B оранжевым.

    Без координат узлы ставятся в два столбца, а связи между несоседними
    узлами одного столбца рисуются дугами, чтобы не накладываться друг на друга.
    """
    columns = pos is None
    if columns:
        pos = two_column_layout(graph)
    if ax is None:
        _, ax = new_figure(5, 5)

    ax.set_facecolor(SURFACE)
    straight = []
    for node1, node2 in graph.edges():
        (x1, y1), (x2, y2) = pos[node1], pos[node2]
        span = abs(y1 - y2)
        if columns and x1 == x2 and span > 1:
            # дуга выгибается наружу: у слоя A влево, у слоя B вправо
            outward = 1 if x1 == 0 else -1
            start, end = ((x1, y1), (x2, y2)) if y1 > y2 else ((x2, y2), (x1, y1))
            rad = outward * (0.25 + 0.07 * span)
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-", color=EDGE, linewidth=1.2,
                                         connectionstyle=f"arc3,rad={rad}", zorder=1))
        else:
            straight.append([(x1, y1), (x2, y2)])
    ax.add_collection(LineCollection(straight, colors=EDGE, linewidths=1.2, zorder=1))

    for layer, nodes in (("A", graph.a_nodes), ("B", graph.b_nodes)):
        xs = [pos[name][0] for name in nodes]
        ys = [pos[name][1] for name in nodes]
        ax.scatter(xs, ys, s=node_size, color=LAYER_COLORS[layer], edgecolors=SURFACE,
                   linewidths=2, zorder=2, label=f"слой {layer}")
        for name, x, y in zip(nodes, xs, ys):
            ax.text(x, y, name, ha="center", va="center_baseline", fontsize=font_size,
                    color=LAYER_LABEL_INK[layer], zorder=3)

    if columns:
        ax.set_xlim(-0.75, 1.75)
        ax.set_ylim(-max(graph.n_a, graph.n_b) - 0.6, -0.4)
    else:
        ax.margins(0.04)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    if legend:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right",
                  markerscale=0.6, handletextpad=0.3)
    return ax


def plot_decile_histogram(fail_steps, ax=None, title=None):
    """Децильная гистограмма шага отказа.

    Каждый столбец содержит 10% опытов, поэтому ширина столбца равна длине
    децильного интервала, а высота (плотность) обратно пропорциональна ширине.
    """
    if ax is None:
        _, ax = new_figure(8, 4.2)

    bounds = decile_boundaries(fail_steps)
    widths = np.diff(bounds)
    density = 0.1 / widths

    ax.bar(bounds[:-1], density, width=widths, align="edge", color=BLUE,
           edgecolor=SURFACE, linewidth=2)
    style_axes(ax)
    ax.set_xlabel("Шаг отказа сети")
    ax.set_ylabel("Плотность (доля опытов на один шаг)")
    ax.set_xlim(left=0)
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    return ax


def plot_survival(series, n_steps, ax=None, title=None, x_max=None, legend_title=None):
    """Доля опытов, в которых сеть ещё работает к данному шагу.

    series: словарь {подпись: массив шагов отказа}, не более четырёх рядов;
    порядок рядов задаёт цвет от светлого к тёмному.
    """
    if len(series) > len(BLUE_RAMP):
        raise ValueError(f"На одном графике не более {len(BLUE_RAMP)} рядов")
    if ax is None:
        _, ax = new_figure(8, 4.2)

    for (label, fail_steps), color in zip(series.items(), BLUE_RAMP):
        steps, alive_share = survival_curve(fail_steps, n_steps)
        ax.plot(steps, alive_share, color=color, linewidth=2, label=label,
                solid_capstyle="round", solid_joinstyle="round")

    style_axes(ax)
    ax.set_xlabel("Шаг")
    ax.set_ylabel("Доля работающих сетей")
    ax.set_xlim(0, x_max or n_steps)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, title=legend_title,
              title_fontsize=9, alignment="left")
    if title:
        ax.set_title(title, fontsize=11, loc="left")
    return ax


def plot_heatmap(values, row_labels, col_labels, ax=None, title=None, xlabel=None, fmt="{:.0f}"):
    """Таблица значений с заливкой: чем больше значение, тем темнее ячейка."""
    values = np.asarray(values, dtype=float)
    if ax is None:
        _, ax = new_figure(1.6 + 1.0 * values.shape[1], 0.9 + 0.5 * values.shape[0])

    ax.set_facecolor(SURFACE)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    mesh = ax.pcolormesh(values, cmap=BLUE_SCALE, vmin=vmin, vmax=vmax,
                         edgecolors=SURFACE, linewidth=2)
    mesh.set_antialiased(True)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            share = (values[i, j] - vmin) / (vmax - vmin) if vmax > vmin else 0
            ax.text(j + 0.5, i + 0.5, fmt.format(values[i, j]), ha="center", va="center",
                    fontsize=9, color="#ffffff" if share > 0.45 else INK)

    ax.set_xticks(np.arange(values.shape[1]) + 0.5, col_labels)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5, row_labels)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(colors=INK_SECONDARY, length=0, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9, labelpad=8)
    if title:
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=14)
    return ax


def save_figure(fig, file_path):
    fig.savefig(file_path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
