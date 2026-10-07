"""Серия вычислительных экспериментов: как топология сети и корреляция нагрузок
влияют на время до отказа.

Запуск:
    python experiments.py

Таблицы сохраняются в results/, рисунки в figures/. Все сценарии считаются
с одним и тем же зерном генератора, то есть на одних и тех же случайных числах.
Полный расчёт занимает несколько минут.
"""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from cascade import (NO_FORCED_NODE, failure_statistics, graph_from_adjacency,
                     graph_from_edges, read_edge_list, run_monte_carlo)
from cascade.excel_io import load_all_data
from cascade.plots import (draw_network, new_figure, plot_decile_histogram, plot_heatmap,
                           plot_survival, read_layout, save_figure)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED = 42
CORRELATIONS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]
SURVIVAL_CORRELATIONS = [0.0, 0.5, 0.9, 0.99]

TOPOLOGIES = {
    "01_article": "1. Граф из статьи",
    "02_two_rings": "2. Два кольца",
    "03_cubic": "3. По три связи в слое",
    "04_cubic_dense_links": "4. По три связи, частые межслойные",
    "05_chords_a": "5. Кольца, хорды в слое A",
    "06_hub_a4": "6. Узел A4 связан с четырьмя узлами B",
    "07_chords_both": "7. Кольца, хорды в обоих слоях",
}

COLUMNS = ["scenario", "correlation", "forced_node", "n_runs", "n_failed", "n_no_fail",
           "mean", "sem", "std", "median", "lower", "upper"]


def constant_correlation(rho, size):
    """Матрица корреляций, в которой нагрузки любых двух узлов связаны одинаково."""
    corr = np.full((size, size), float(rho))
    np.fill_diagonal(corr, 1.0)
    return corr


def result_row(scenario, rho, params, fail_steps):
    stats = failure_statistics(fail_steps)
    stats["sem"] = stats["std"] / np.sqrt(stats["n_failed"]) if stats["n_failed"] > 1 else np.nan
    row = {"scenario": scenario, "correlation": rho, "forced_node": params.forced_node, **stats}
    return {column: row[column] for column in COLUMNS}


def save_table(rows, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:.2f}" if isinstance(value, float) else value)
                             for key, value in row.items()})


def base_scenario(params, adj_matrix, corr_matrix):
    """Сценарий из data/params.xlsx без изменений."""
    graph = graph_from_adjacency(adj_matrix, params.n_a, params.n_b, params.level_a, params.level_b)
    fail_steps = run_monte_carlo(graph, corr_matrix, params, seed=SEED)
    save_table([result_row("base", corr_matrix.iloc[0, 1], params, fail_steps)], RESULTS / "base_scenario.csv")

    fig, ax = new_figure(8, 4.2)
    plot_decile_histogram(fail_steps, ax=ax)
    save_figure(fig, FIGURES / "base_decile_histogram.png")


def topology_and_correlation(params):
    """Семь топологий на 12 узлах при разной корреляции нагрузок."""
    graphs = {
        name: graph_from_edges(read_edge_list(DATA / "topologies" / f"{name}.csv"),
                               params.level_a, params.level_b)
        for name in TOPOLOGIES
    }

    rows = []
    fail_steps = {}
    for forced_node in (NO_FORCED_NODE, "A1"):
        scenario_params = replace(params, forced_node=forced_node)
        for name, graph in graphs.items():
            for rho in CORRELATIONS:
                steps = run_monte_carlo(graph, constant_correlation(rho, graph.size),
                                        scenario_params, seed=SEED)
                fail_steps[forced_node, name, rho] = steps
                rows.append(result_row(name, rho, scenario_params, steps))
            print(f"  {TOPOLOGIES[name]}, принудительный отказ: {forced_node}")
    save_table(rows, RESULTS / "topology_correlation.csv")

    # схемы топологий
    fig, axes = new_figure(13, 7.4, nrows=2, ncols=4)
    for ax, (name, title) in zip(axes.flat, TOPOLOGIES.items()):
        draw_network(graphs[name], ax=ax, title=title, node_size=300, font_size=8, legend=False)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, frameon=False, loc="center", fontsize=10, markerscale=0.8)
    fig.tight_layout()
    save_figure(fig, FIGURES / "topologies.png")

    # среднее время до отказа: топология x корреляция
    means = [[failure_statistics(fail_steps[NO_FORCED_NODE, name, rho])["mean"]
              for rho in CORRELATIONS] for name in TOPOLOGIES]
    fig, ax = new_figure(9.5, 4.4)
    plot_heatmap(means, list(TOPOLOGIES.values()), [f"{rho:g}" for rho in CORRELATIONS], ax=ax,
                 xlabel="Корреляция нагрузок")
    save_figure(fig, FIGURES / "mean_steps_heatmap.png")

    # кривые выживания для топологии «два кольца»
    fig, ax = new_figure(8, 4.2)
    series = {f"{rho:g}": fail_steps[NO_FORCED_NODE, "02_two_rings", rho] for rho in SURVIVAL_CORRELATIONS}
    plot_survival(series, params.n_steps, ax=ax, x_max=4000, legend_title="Корреляция нагрузок")
    save_figure(fig, FIGURES / "survival_by_correlation.png")


def large_network(params):
    """Сеть из 67 узлов слоя A и 39 узлов слоя B."""
    folder = DATA / "network_106"
    graph = graph_from_edges(read_edge_list(folder / "edges.csv"), params.level_a, params.level_b)
    # узел B35 не имеет связей и в расчёте не участвует
    graph.remove_isolated_nodes()
    # как и в сети на 12 узлов, слой считается отказавшим, когда работает меньше половины узлов
    net_params = replace(params, n_a=graph.n_a, n_b=graph.n_b, forced_node=NO_FORCED_NODE,
                         min_alive_a=(graph.n_a + 1) // 2, min_alive_b=(graph.n_b + 1) // 2)

    rows = []
    for rho in CORRELATIONS:
        steps = run_monte_carlo(graph, constant_correlation(rho, graph.size), net_params, seed=SEED)
        rows.append(result_row("network_106", rho, net_params, steps))
    save_table(rows, RESULTS / "network_106.csv")

    fig, ax = new_figure(9, 10)
    draw_network(graph, pos=read_layout(folder / "layout.csv"), ax=ax, node_size=230, font_size=6.5)
    save_figure(fig, FIGURES / "network_106.png")


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    params, adj_matrix, corr_matrix = load_all_data(DATA / "params.xlsx")

    print("Базовый сценарий")
    base_scenario(params, adj_matrix, corr_matrix)
    print("Топология и корреляция")
    topology_and_correlation(params)
    print("Сеть 67 + 39 узлов")
    large_network(params)
    print(f"Готово: таблицы в {RESULTS.name}/, рисунки в {FIGURES.name}/")


if __name__ == "__main__":
    main()
