"""Один расчёт по файлу параметров: Монте-Карло и статистика шага отказа сети.

Примеры:
    python main.py
    python main.py --params data/params.xlsx --seed 42 --figures figures
"""

import argparse
from pathlib import Path

from cascade import failure_statistics, graph_from_adjacency, run_monte_carlo
from cascade.excel_io import load_all_data

DEFAULT_PARAMS = Path(__file__).parent / "data" / "params.xlsx"


def print_statistics(stats):
    print(f"Опытов всего:              {stats['n_runs']}")
    print(f"Опытов с отказом сети:     {stats['n_failed']}")
    print(f"Опытов без отказа:         {stats['n_no_fail']}")
    if stats["n_failed"] == 0:
        print("Отказов не было ни в одном опыте")
        return
    print(f"Мат. ожидание шага отказа: {stats['mean']:.1f}")
    print(f"Медиана:                   {stats['median']:.1f}")
    print(f"Дисперсия:                 {stats['variance']:.1f}")
    print(f"СКО:                       {stats['std']:.1f}")
    print(f"Границы 95%-го интервала:  {stats['lower']:.0f} .. {stats['upper']:.0f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", default=DEFAULT_PARAMS, help="файл Excel с параметрами и матрицами")
    parser.add_argument("--seed", type=int, default=None, help="зерно генератора случайных чисел")
    parser.add_argument("--figures", default=None, help="папка, куда сохранить рисунки")
    args = parser.parse_args()

    params, adj_matrix, corr_matrix = load_all_data(args.params)
    graph = graph_from_adjacency(adj_matrix, params.n_a, params.n_b, params.level_a, params.level_b)

    print(f"Сеть: узлов {params.n_a} + {params.n_b}, связей {len(graph.edges())}")
    print(f"Опытов: {params.n_runs}, шагов в опыте: до {params.n_steps}, "
          f"режим отказа: {params.fail_mode}, принудительный отказ узла: {params.forced_node}")

    fail_steps = run_monte_carlo(graph, corr_matrix, params, seed=args.seed)
    print_statistics(failure_statistics(fail_steps))

    if args.figures:
        # рисунки нужны не всегда, поэтому matplotlib подключается только здесь
        from cascade.plots import draw_network, new_figure, plot_decile_histogram, save_figure

        folder = Path(args.figures)
        folder.mkdir(parents=True, exist_ok=True)

        fig, ax = new_figure(4.5, 5)
        draw_network(graph, ax=ax)
        save_figure(fig, folder / "network.png")

        fig, ax = new_figure(8, 4.2)
        plot_decile_histogram(fail_steps, ax=ax)
        save_figure(fig, folder / "decile_histogram.png")
        print(f"Рисунки сохранены в папку {folder}")


if __name__ == "__main__":
    main()
