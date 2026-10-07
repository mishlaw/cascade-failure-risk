"""Статистическая обработка результатов Монте-Карло."""

import numpy as np

from .simulation import NO_FAILURE


def failure_statistics(fail_steps, probability=0.95):
    """Статистика шага отказа сети по опытам, в которых отказ произошёл.

    Возвращает словарь:
      n_runs      всего опытов;
      n_failed    опытов с отказом сети;
      n_no_fail   опытов без отказа за отведённое число шагов;
      mean, variance, std   выборочные среднее, дисперсия (несмещённая) и СКО;
      median      выборочная медиана;
      lower, upper   границы интервала, в который попадает доля probability опытов:
                     с каждого края вариационного ряда отбрасывается (1 - probability) / 2.
    """
    fail_steps = np.asarray(fail_steps)
    failed = np.sort(fail_steps[fail_steps != NO_FAILURE]).astype(float)
    n_failed = len(failed)

    stats = {
        "n_runs": len(fail_steps),
        "n_failed": n_failed,
        "n_no_fail": len(fail_steps) - n_failed,
        "mean": np.nan,
        "variance": np.nan,
        "std": np.nan,
        "median": np.nan,
        "lower": np.nan,
        "upper": np.nan,
    }
    if n_failed == 0:
        return stats

    cut = int(n_failed * (1 - probability) * 0.5)
    stats.update(
        mean=failed.mean(),
        median=np.median(failed),
        lower=failed[cut],
        upper=failed[n_failed - cut - 1],
    )
    if n_failed > 1:
        stats["variance"] = failed.var(ddof=1)
        stats["std"] = np.sqrt(stats["variance"])

    return stats


def decile_boundaries(fail_steps):
    """Границы децилей шага отказа: 11 чисел от минимума до максимума."""
    fail_steps = np.asarray(fail_steps)
    failed = fail_steps[fail_steps != NO_FAILURE]
    return np.quantile(failed, np.linspace(0, 1, 11))


def survival_curve(fail_steps, n_steps):
    """Доля опытов, в которых сеть ещё работает после шага 0, 1, ..., n_steps."""
    fail_steps = np.asarray(fail_steps)
    failed = np.sort(fail_steps[fail_steps != NO_FAILURE])
    steps = np.arange(n_steps + 1)
    n_failed_by_step = np.searchsorted(failed, steps, side="right")
    return steps, 1 - n_failed_by_step / len(fail_steps)
