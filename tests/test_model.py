from pathlib import Path

import numpy as np
import pytest

from cascade import (NO_FAILURE, Params, failure_statistics, graph_from_adjacency,
                     graph_from_edges, read_edge_list, run_monte_carlo)
from cascade.excel_io import load_all_data
from cascade.simulation import (cascade_remove, correlated_loads, count_components,
                                network_failed, recalc_thresholds)

DATA = Path(__file__).parent.parent / "data"

# Лестница: цепочка A1-A2-A3, цепочка B1-B2-B3 и перемычки A1-B1, A2-B2, A3-B3
LADDER = [("A1", "A2"), ("A2", "A3"), ("B1", "B2"), ("B2", "B3"),
          ("A1", "B1"), ("A2", "B2"), ("A3", "B3")]


def make_params(**changes):
    values = dict(n_a=3, n_b=3, n_steps=200, n_runs=50, level_a=3.0, level_b=3.0,
                  d_a=0.15, d_b=0.2, d_ab=0.1, min_alive_a=2, min_alive_b=2, fail_mode=3)
    values.update(changes)
    return Params(**values)


def constant_correlation(rho, size):
    corr = np.full((size, size), float(rho))
    np.fill_diagonal(corr, 1.0)
    return corr


def test_threshold_grows_with_each_link():
    graph = graph_from_edges(LADDER, level_a=3.0, level_b=3.0)
    recalc_thresholds(graph, d_a=0.15, d_b=0.2, d_ab=0.1)

    # A2: две связи в своём слое и одна межслойная
    assert graph.nodes["A2"]["level"] == pytest.approx(3.0 + 2 * 0.15 + 0.1)
    # B1: одна связь в своём слое и одна межслойная
    assert graph.nodes["B1"]["level"] == pytest.approx(3.0 + 0.2 + 0.1)


def test_failure_spreads_only_through_interlayer_links():
    graph = graph_from_edges(LADDER, level_a=3.0, level_b=3.0)

    removed = cascade_remove(graph, "A1")

    assert sorted(removed) == ["A1", "B1"]
    assert "A2" in graph.nodes and "B2" in graph.nodes
    assert graph.nodes["A2"]["neighbors"] == {"A3", "B2"}


def test_dense_interlayer_links_destroy_the_whole_network():
    edges = LADDER + [("A1", "B2"), ("A2", "B3")]
    graph = graph_from_edges(edges, level_a=3.0, level_b=3.0)

    cascade_remove(graph, "A3")

    assert graph.nodes == {}


def test_network_failure_criteria():
    graph = graph_from_edges(LADDER, level_a=3.0, level_b=3.0)
    assert count_components(graph) == 1
    assert not network_failed(graph, make_params(fail_mode=3))

    # без A2 и B2 сеть распадается на две части, но узлов в слоях ещё достаточно
    cascade_remove(graph, "A2")
    assert count_components(graph) == 2
    assert network_failed(graph, make_params(fail_mode=2))
    assert not network_failed(graph, make_params(fail_mode=1))

    cascade_remove(graph, "A1")
    assert network_failed(graph, make_params(fail_mode=1))


def test_empty_network_counts_as_failed():
    graph = graph_from_edges(LADDER, level_a=3.0, level_b=3.0)
    for node in ("A1", "A2", "A3"):
        cascade_remove(graph, node)

    assert graph.nodes == {}
    assert network_failed(graph, make_params(fail_mode=2))


def test_loads_have_requested_correlation():
    corr = constant_correlation(0.6, 4)
    chol = np.linalg.cholesky(corr)
    rng = np.random.default_rng(0)

    sample = np.array([correlated_loads(chol, rng) for _ in range(20000)])

    assert np.allclose(np.corrcoef(sample, rowvar=False), corr, atol=0.03)
    assert np.allclose(sample.std(axis=0), 1.0, atol=0.03)


def test_same_seed_gives_same_result():
    graph = graph_from_edges(LADDER, level_a=2.0, level_b=2.0)
    corr = constant_correlation(0.5, graph.size)
    params = make_params()

    first = run_monte_carlo(graph, corr, params, seed=1)
    second = run_monte_carlo(graph, corr, params, seed=1)

    assert np.array_equal(first, second)
    assert len(first) == params.n_runs
    assert graph.nodes["A1"]["neighbors"] == {"A2", "B1"}   # исходная сеть не изменилась


def test_high_thresholds_mean_no_failure():
    graph = graph_from_edges(LADDER, level_a=50.0, level_b=50.0)

    fail_steps = run_monte_carlo(graph, constant_correlation(0.0, graph.size), make_params(), seed=1)

    assert np.all(fail_steps == NO_FAILURE)


def test_forced_failure_can_stop_the_network_at_step_zero():
    edges = read_edge_list(DATA / "topologies" / "04_cubic_dense_links.csv")
    graph = graph_from_edges(edges, level_a=3.0, level_b=3.0)
    params = make_params(n_a=6, n_b=6, min_alive_a=3, min_alive_b=3, forced_node="A1")

    fail_steps = run_monte_carlo(graph, constant_correlation(0.5, graph.size), params, seed=1)

    assert np.all(fail_steps == 0)


def test_invalid_input_is_rejected():
    graph = graph_from_edges(LADDER, level_a=3.0, level_b=3.0)

    with pytest.raises(ValueError):   # матрица корреляций не того размера
        run_monte_carlo(graph, constant_correlation(0.5, 4), make_params(), seed=1)
    with pytest.raises(ValueError):   # принудительно отказывающего узла нет в сети
        run_monte_carlo(graph, constant_correlation(0.5, 6), make_params(forced_node="A9"), seed=1)
    with pytest.raises(ValueError):   # несуществующий режим отказа
        make_params(fail_mode=5)


def test_statistics_use_only_runs_with_failure():
    fail_steps = np.array([10, 20, 30, 40, NO_FAILURE])

    stats = failure_statistics(fail_steps)

    assert stats["n_runs"] == 5
    assert stats["n_failed"] == 4
    assert stats["n_no_fail"] == 1
    assert stats["mean"] == pytest.approx(25.0)
    assert stats["variance"] == pytest.approx(np.var([10, 20, 30, 40], ddof=1))
    assert (stats["lower"], stats["upper"]) == (10, 40)


def test_interval_cuts_both_tails():
    stats = failure_statistics(np.arange(1, 1001))

    # при 1000 опытах с каждого края отбрасывается по 25 значений
    assert (stats["lower"], stats["upper"]) == (26, 975)


def test_parameters_file_loads():
    params, adj_matrix, corr_matrix = load_all_data(DATA / "params.xlsx")
    graph = graph_from_adjacency(adj_matrix, params.n_a, params.n_b, params.level_a, params.level_b)

    assert adj_matrix.shape == corr_matrix.shape == (12, 12)
    assert np.array_equal(adj_matrix.values, adj_matrix.values.T)
    assert count_components(graph) == 1
    assert params.forced_node in graph.nodes
