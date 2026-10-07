"""Имитационная модель каскадных отказов в двухуровневой сети (метод Монте-Карло).

Один опыт устроен так:
  1. На каждом шаге генерируется вектор коррелированных нормальных нагрузок x.
  2. Узел отказывает, если |x| превышает его текущий порог.
  3. Отказ узла тянет за собой отказ всех связанных с ним узлов другого слоя,
     те в свою очередь выводят из строя своих соседей из первого слоя, и так далее.
  4. У уцелевших узлов пороги снижаются: чем меньше связей, тем ниже порог.
  5. Опыт заканчивается, когда выполняется критерий отказа сети.
Результат опыта: номер шага, на котором сеть отказала.
"""

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np

NO_FORCED_NODE = "0"   # значение параметра «принудительный узел», если отказ не задаётся
NO_FAILURE = -1        # сеть не отказала за отведённое число шагов

FAIL_TOO_FEW_NODES = 1   # в обоих слоях осталось меньше минимального числа узлов
FAIL_FRAGMENTED = 2      # сеть распалась на две и более части
FAIL_ANY = 3             # любое из двух условий


@dataclass
class Params:
    """Параметры модели (в скобках названия из файла data/params.xlsx)."""

    n_a: int              # число узлов слоя A (n_A)
    n_b: int              # число узлов слоя B (n_B)
    n_steps: int          # максимальная длительность одного опыта в шагах (N)
    n_runs: int           # число опытов Монте-Карло (M)
    level_a: float        # базовый порог узлов слоя A (Alevel)
    level_b: float        # базовый порог узлов слоя B (Blevel)
    d_a: float            # прибавка к порогу узла A за каждую связь внутри слоя (d(A))
    d_b: float            # прибавка к порогу узла B за каждую связь внутри слоя (d(B))
    d_ab: float           # прибавка к порогу за каждую межслойную связь (d(AB))
    min_alive_a: int      # минимальное число работающих узлов слоя A (KlFunc_A)
    min_alive_b: int      # минимальное число работающих узлов слоя B (KlFunc_B)
    fail_mode: int        # критерий отказа сети: 1, 2 или 3 (Режим отказа)
    forced_node: str = NO_FORCED_NODE   # узел, отказывающий в начале опыта (Num_Node)

    def __post_init__(self):
        if self.fail_mode not in (FAIL_TOO_FEW_NODES, FAIL_FRAGMENTED, FAIL_ANY):
            raise ValueError(f"Режим отказа должен быть 1, 2 или 3, получено: {self.fail_mode}")


def recalc_thresholds(graph, d_a, d_b, d_ab):
    """Пересчитывает пороги всех работающих узлов по числу их связей.

    порог = базовый уровень + d_own * (связи внутри слоя) + d_ab * (межслойные связи)
    """
    for data in graph.nodes.values():
        deg_own = 0
        deg_other = 0
        for nb in data["neighbors"]:
            if graph.nodes[nb]["type"] == data["type"]:
                deg_own += 1
            else:
                deg_other += 1

        d_own = d_a if data["type"] == "A" else d_b
        data["level"] = data["base_level"] + d_own * deg_own + d_ab * deg_other


def cascade_remove(graph, start_node):
    """Удаляет узел и всех, кто зависит от него через межслойные связи.

    Сосед из другого слоя отказывает вместе с узлом, сосед из того же слоя
    остаётся работать, но теряет связь. Возвращает список удалённых узлов.
    """
    removed = []
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in graph.nodes:
            continue

        node_type = graph.nodes[node]["type"]
        neighbors = list(graph.nodes[node]["neighbors"])
        graph.remove_node(node)
        removed.append(node)

        for nb in neighbors:
            if nb in graph.nodes and graph.nodes[nb]["type"] != node_type:
                stack.append(nb)

    return removed


def count_components(graph):
    """Число компонент связности (обход в ширину)."""
    visited = set()
    components = 0

    for start in graph.nodes:
        if start in visited:
            continue
        components += 1
        visited.add(start)
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nb in graph.nodes[current]["neighbors"]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)

    return components


def network_failed(graph, params):
    """Проверяет критерий отказа сети."""
    if params.fail_mode in (FAIL_TOO_FEW_NODES, FAIL_ANY):
        too_few = (len(graph.a_nodes) < params.min_alive_a
                   and len(graph.b_nodes) < params.min_alive_b)
        if too_few:
            return True
        if params.fail_mode == FAIL_TOO_FEW_NODES:
            return False

    # Сеть без единого узла тоже считается отказавшей
    return count_components(graph) != 1


def correlated_loads(chol, rng):
    """Вектор нагрузок x ~ N(0, C), где C = chol @ chol.T — матрица корреляций."""
    return chol @ rng.standard_normal(chol.shape[0])


def _alive_arrays(graph):
    """Имена, номера и пороги работающих узлов в виде массивов для быстрых сравнений."""
    names = list(graph.nodes)
    ids = np.array([graph.nodes[name]["id"] for name in names], dtype=int)
    levels = np.array([graph.nodes[name]["level"] for name in names], dtype=float)
    return names, ids, levels


def run_single(base_graph, chol, params, rng):
    """Один опыт. Возвращает шаг отказа сети или NO_FAILURE.

    Шаг 0 означает, что сеть отказала сразу после принудительного отказа узла.
    """
    graph = copy.deepcopy(base_graph)

    if params.forced_node != NO_FORCED_NODE:
        cascade_remove(graph, params.forced_node)
        if network_failed(graph, params):
            return 0

    recalc_thresholds(graph, params.d_a, params.d_b, params.d_ab)
    names, ids, levels = _alive_arrays(graph)

    for step in range(1, params.n_steps + 1):
        x = correlated_loads(chol, rng)
        exceeded = np.abs(x[ids]) > levels
        if not exceeded.any():
            continue   # сеть не изменилась, проверять критерий отказа незачем

        for i in np.flatnonzero(exceeded):
            # узел мог уже отказать в каскаде, запущенном другим узлом на этом шаге
            if names[i] in graph.nodes:
                cascade_remove(graph, names[i])

        if network_failed(graph, params):
            return step

        recalc_thresholds(graph, params.d_a, params.d_b, params.d_ab)
        names, ids, levels = _alive_arrays(graph)

    return NO_FAILURE


def run_monte_carlo(graph, corr_matrix, params, seed=None):
    """Проводит params.n_runs независимых опытов.

    graph        исходная сеть (не изменяется);
    corr_matrix  матрица корреляций нагрузок размера graph.size x graph.size;
    seed         зерно генератора случайных чисел для воспроизводимости.

    Возвращает массив длины n_runs: шаг отказа сети в каждом опыте
    или NO_FAILURE, если за n_steps шагов сеть не отказала.
    """
    graph = copy.deepcopy(graph)
    corr = np.asarray(corr_matrix, dtype=float)

    if corr.shape != (graph.size, graph.size):
        raise ValueError(
            f"Матрица корреляций имеет размер {corr.shape}, "
            f"а в сети {graph.size} узлов ({graph.n_a} + {graph.n_b})"
        )
    if not np.allclose(corr, corr.T):
        raise ValueError("Матрица корреляций должна быть симметричной")

    isolated = graph.remove_isolated_nodes()
    if isolated:
        print(f"Узлы без связей исключены из расчёта: {', '.join(isolated)}")

    if network_failed(graph, params):
        raise ValueError(
            "Сеть находится в состоянии отказа ещё до начала моделирования: "
            "проверьте связность графа и минимальные числа узлов"
        )
    if params.forced_node != NO_FORCED_NODE and params.forced_node not in graph.nodes:
        raise ValueError(f"Принудительно отказывающего узла {params.forced_node} нет в сети")

    # Разложение Холецкого C = L @ L.T: из независимых нормальных величин u
    # получаем нагрузки x = L @ u с заданной матрицей корреляций
    chol = np.linalg.cholesky(corr)
    rng = np.random.default_rng(seed)

    fail_steps = np.empty(params.n_runs, dtype=int)
    for run in range(params.n_runs):
        fail_steps[run] = run_single(graph, chol, params, rng)

    return fail_steps
