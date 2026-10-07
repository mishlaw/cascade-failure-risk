"""Двухуровневая сеть: узлы слоёв A и B, связи внутри слоёв и между ними."""

import csv
import re

NODE_NAME = re.compile(r"^([AB])(\d+)$")


class Graph:
    """Граф из двух слоёв.

    Узлы называются A1..A<n_a> и B1..B<n_b>. Для каждого узла хранится:
      type        слой, 'A' или 'B';
      neighbors   множество имён соседей;
      base_level  базовый порог без учёта связей;
      level       текущий порог (пересчитывается по мере отказов);
      id          номер узла в векторе нагрузок и в матрице корреляций.
    """

    def __init__(self, n_a, n_b, level_a, level_b):
        self.n_a = n_a
        self.n_b = n_b
        self.nodes = {}
        self.a_nodes = []
        self.b_nodes = []

        for i in range(n_a):
            self._add_node(f"A{i + 1}", "A", level_a, i)
        for i in range(n_b):
            self._add_node(f"B{i + 1}", "B", level_b, n_a + i)

    def _add_node(self, name, node_type, base_level, node_id):
        self.nodes[name] = {
            "type": node_type,
            "neighbors": set(),
            "base_level": base_level,
            "level": base_level,
            "id": node_id,
        }
        (self.a_nodes if node_type == "A" else self.b_nodes).append(name)

    @property
    def size(self):
        """Размер вектора нагрузок: исходное число узлов в обоих слоях."""
        return self.n_a + self.n_b

    def add_edge(self, node1, node2):
        for name in (node1, node2):
            if name not in self.nodes:
                raise KeyError(f"Узла {name} нет в сети")
        self.nodes[node1]["neighbors"].add(node2)
        self.nodes[node2]["neighbors"].add(node1)

    def remove_node(self, node):
        """Удаляет узел вместе со всеми его связями."""
        for nb in self.nodes[node]["neighbors"]:
            self.nodes[nb]["neighbors"].discard(node)

        if self.nodes[node]["type"] == "A":
            self.a_nodes.remove(node)
        else:
            self.b_nodes.remove(node)

        del self.nodes[node]

    def remove_isolated_nodes(self):
        """Удаляет узлы без связей и возвращает их имена.

        Узел без связей сам по себе является отдельной компонентой, поэтому
        сеть с таким узлом считалась бы распавшейся ещё до начала расчёта.
        """
        isolated = [name for name, data in self.nodes.items() if not data["neighbors"]]
        for name in isolated:
            self.remove_node(name)
        return isolated

    def edges(self):
        """Список связей без повторов: каждая пара узлов встречается один раз."""
        result = []
        for name, data in self.nodes.items():
            for nb in data["neighbors"]:
                if self.nodes[name]["id"] < self.nodes[nb]["id"]:
                    result.append((name, nb))
        return result


def graph_from_adjacency(adj_matrix, n_a, n_b, level_a, level_b):
    """Строит сеть по матрице смежности (pandas.DataFrame или numpy-массив).

    Строки и столбцы идут в порядке A1..A<n_a>, B1..B<n_b>.
    """
    graph = Graph(n_a, n_b, level_a, level_b)
    names = graph.a_nodes + graph.b_nodes
    values = getattr(adj_matrix, "values", adj_matrix)

    for i in range(graph.size):
        for j in range(i + 1, graph.size):
            if values[i][j] == 1:
                graph.add_edge(names[i], names[j])

    return graph


def read_edge_list(file_path):
    """Читает список связей из CSV с колонками source,target."""
    with open(file_path, newline="", encoding="utf-8") as f:
        return [(row["source"], row["target"]) for row in csv.DictReader(f)]


def graph_from_edges(edges, level_a, level_b, n_a=None, n_b=None):
    """Строит сеть по списку связей вида ('A1', 'B3').

    Если число узлов в слоях не задано, оно берётся по наибольшему номеру узла.
    """
    numbers = {"A": [0], "B": [0]}
    for edge in edges:
        for name in edge:
            match = NODE_NAME.match(name)
            if not match:
                raise ValueError(f"Имя узла должно иметь вид A1 или B12, получено: {name!r}")
            numbers[match.group(1)].append(int(match.group(2)))

    graph = Graph(n_a or max(numbers["A"]), n_b or max(numbers["B"]), level_a, level_b)
    for node1, node2 in edges:
        graph.add_edge(node1, node2)
    return graph
