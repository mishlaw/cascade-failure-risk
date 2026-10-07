"""Чтение параметров модели, матрицы смежности и матрицы корреляций из Excel.

Структура файла (см. data/params.xlsx):
  лист «Параметры»   значения во втором столбце, по одному параметру в строке;
  лист «Смежности»   матрица смежности с заголовками в первой строке и первом столбце;
  лист «Корреляции»  матрица корреляций нагрузок в том же формате.
"""

import pandas as pd

from .simulation import Params

SHEET_PARAMS = "Параметры"
SHEET_ADJACENCY = "Смежности"
SHEET_CORRELATIONS = "Корреляции"


def read_parameters(file_path):
    df = pd.read_excel(file_path, sheet_name=SHEET_PARAMS, header=None)

    return Params(
        n_a=int(df.iloc[0, 1]),
        n_b=int(df.iloc[1, 1]),
        n_steps=int(df.iloc[2, 1]),
        n_runs=int(df.iloc[3, 1]),
        level_a=float(df.iloc[4, 1]),
        level_b=float(df.iloc[5, 1]),
        d_a=float(df.iloc[6, 1]),
        d_b=float(df.iloc[7, 1]),
        min_alive_a=int(df.iloc[8, 1]),
        min_alive_b=int(df.iloc[8, 3]),
        fail_mode=int(df.iloc[9, 1]),
        forced_node=str(df.iloc[10, 1]).strip(),
        d_ab=float(df.iloc[11, 1]),
    )


def read_matrix(file_path, sheet_name, size):
    """Читает квадратную матрицу size x size, пропуская строку и столбец заголовков."""
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    matrix = df.iloc[1:size + 1, 1:size + 1]

    if matrix.shape != (size, size) or matrix.isna().any().any():
        raise ValueError(
            f"На листе «{sheet_name}» ожидается матрица {size}x{size} без пустых ячеек"
        )

    return matrix.reset_index(drop=True).astype(float)


def load_all_data(file_path):
    """Возвращает параметры, матрицу смежности и матрицу корреляций."""
    params = read_parameters(file_path)
    size = params.n_a + params.n_b

    adj_matrix = read_matrix(file_path, SHEET_ADJACENCY, size)
    corr_matrix = read_matrix(file_path, SHEET_CORRELATIONS, size)

    return params, adj_matrix, corr_matrix
