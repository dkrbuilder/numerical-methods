import csv
import numpy as np
from time import perf_counter


# 1. Формирование точного решения
def make_exact_solution(n):
    x = np.zeros(n, dtype=np.float64)
    x[::2] = 7.0
    return x


# 2. Фактическая полуширина ленты
def effective_bandwidth(n, k):
    if n <= 0:
        raise ValueError("Размер матрицы n должен быть положительным")

    if k < 0:
        raise ValueError("Полуширина ленты k не может быть отрицательной")

    return min(k, n - 1)


# 3. Формирование полной матрицы

def make_dense_matrix(n, k):
    k = effective_bandwidth(n, k)
    matrix_a = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(matrix_a, min(n, 10))

    for j in range(1, k + 1):
        side_element = 1.0 / j
        indices = np.arange(n - j)
        matrix_a[indices, indices + j] = side_element
        matrix_a[indices + j, indices] = side_element

    return matrix_a



# 4. Формирование правой части

def make_right_side(n, k, x):
    k = effective_bandwidth(n, k)
    right_b = min(n, 10) * x.copy()

    for j in range(1, k + 1):
        side_element = 1.0 / j
        right_b[:-j] += side_element * x[j:]
        right_b[j:] += side_element * x[:-j]

    return right_b


# 5. QR-разложение полной матрицы

def givens_qr_dense_gaussian(matrix_a, right_b, k, save_history=False):
    matrix_r = matrix_a.copy()
    y = right_b.copy()
    n = matrix_r.shape[0]
    k = effective_bandwidth(n, k)

    rotations = []
    matrices = []

    for column in range(n - 1):
        last_row = min(n, column + k + 1)
        last_column = min(n, column + 2 * k + 1)

        for row in range(column + 1, last_row):
            alpha = matrix_r[column, column]
            beta = matrix_r[row, column]

            if beta == 0.0:
                continue

            radius = np.hypot(alpha, beta)
            cos_phi = alpha / radius
            sin_phi = beta / radius

            support_row = matrix_r[column, column:last_column].copy()
            bottom_row = matrix_r[row, column:last_column].copy()

            matrix_r[column, column:last_column] = cos_phi * support_row + sin_phi * bottom_row
            matrix_r[row, column:last_column] = -sin_phi * support_row + cos_phi * bottom_row
            matrix_r[row, column] = 0.0

            old_y_support = y[column]
            old_y_bottom = y[row]

            y[column] = cos_phi * old_y_support + sin_phi * old_y_bottom
            y[row] = -sin_phi * old_y_support + cos_phi * old_y_bottom

            if save_history:
                rotations.append({"column": column, "row_support": column, "row_bottom": row, "cos_phi": cos_phi, "sin_phi": sin_phi})

        if save_history:
            matrices.append(matrix_r.copy())

    return matrix_r, y, rotations, matrices


# 6. Обратная подстановка для обычной матрицы
def back_substitution_dense(matrix_r, y, k):
    n = len(y)
    k = effective_bandwidth(n, k)
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        last_column = min(n, i + 2 * k + 1)
        known_part = np.dot(matrix_r[i, i + 1:last_column], x[i + 1:last_column])

        if abs(matrix_r[i, i]) < np.finfo(np.float64).eps:
            raise ArithmeticError(f"Нулевой диагональный элемент R[{i}, {i}]")

        x[i] = (y[i] - known_part) / matrix_r[i, i]

    return x


# 7. Компактное хранение ленточной матрицы

def make_compact_band_matrix(n, k):
    k = effective_bandwidth(n, k)
    compact_matrix = np.zeros((n, 3 * k + 1), dtype=np.float64)
    compact_matrix[:, k] = min(n, 10)

    for j in range(1, k + 1):
        side_element = 1.0 / j
        compact_matrix[:n - j, k + j] = side_element
        compact_matrix[j:, k - j] = side_element

    return compact_matrix


# 8. Перевод компактной матрицы в обычную (только для вывода)

def compact_to_dense(compact_matrix, n, k):
    k = effective_bandwidth(n, k)
    dense_matrix = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        first_column = max(0, i - k)
        last_column = min(n, i + 2 * k + 1)

        for j in range(first_column, last_column):
            compact_column = k + j - i
            dense_matrix[i, j] = compact_matrix[i, compact_column]

    return dense_matrix


# 9. QR-разложение ленточной матрицы

def givens_qr_band_gaussian(compact_matrix, right_b, k, save_history=False):
    matrix_r = compact_matrix.copy()
    y = right_b.copy()
    n = matrix_r.shape[0]
    k = effective_bandwidth(n, k)

    rotations = []
    matrices = []

    for column in range(n - 1):
        last_row = min(n, column + k + 1)
        last_column = min(n, column + 2 * k + 1)
        row_length = last_column - column

        for row in range(column + 1, last_row):
            support_start = k
            bottom_start = k + column - row

            alpha = matrix_r[column, support_start]
            beta = matrix_r[row, bottom_start]

            if beta == 0.0:
                continue

            radius = np.hypot(alpha, beta)
            cos_phi = alpha / radius
            sin_phi = beta / radius

            support_row = matrix_r[column, support_start:support_start + row_length].copy()
            bottom_row = matrix_r[row, bottom_start:bottom_start + row_length].copy()

            matrix_r[column, support_start:support_start + row_length] = cos_phi * support_row + sin_phi * bottom_row
            matrix_r[row, bottom_start:bottom_start + row_length] = -sin_phi * support_row + cos_phi * bottom_row
            matrix_r[row, bottom_start] = 0.0

            old_y_support = y[column]
            old_y_bottom = y[row]

            y[column] = cos_phi * old_y_support + sin_phi * old_y_bottom
            y[row] = -sin_phi * old_y_support + cos_phi * old_y_bottom

            if save_history:
                rotations.append({"column": column, "row_support": column, "row_bottom": row, "cos_phi": cos_phi, "sin_phi": sin_phi})

        if save_history:
            matrices.append(compact_to_dense(matrix_r, n, k))

    return matrix_r, y, rotations, matrices



# 10. Обратная подстановка для ленточной матрицы

def back_substitution_band(compact_r, y, k):
    n = len(y)
    k = effective_bandwidth(n, k)
    x = np.zeros(n, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        last_column = min(n, i + 2 * k + 1)
        number_of_upper_elements = last_column - i - 1
        known_part = np.dot(compact_r[i, k + 1:k + 1 + number_of_upper_elements], x[i + 1:last_column])
        diagonal_element = compact_r[i, k]

        if abs(diagonal_element) < np.finfo(np.float64).eps:
            raise ArithmeticError(f"Нулевой диагональный элемент R[{i}, {i}]")

        x[i] = (y[i] - known_part) / diagonal_element

    return x


# 11. Форматирование результатов

def format_array(array):
    """Выводит элементы с шестью значащими цифрами."""
    return np.array2string(np.asarray(array), formatter={"float_kind": lambda value: f"{value:.6g}"}, max_line_width=150)


# 12. Подсчёт условного объёма работы

def count_work(n, k):
    k = effective_bandwidth(n, k)
    work = 0

    for column in range(n - 1):
        number_of_rotations = min(k, n - column - 1)
        number_of_columns = min(n - column, 2 * k + 1)
        work += number_of_rotations * number_of_columns

    return work



# 13. Подробный вывод

def print_detailed_result(result):
    """Выводит подробную информацию о процессе решения."""
    n = result["n"]

    print("=" * 70)
    print(f"ПОДРОБНЫЙ РАСЧЁТ: {result['type'].upper()} МАТРИЦА, N = {n}, K = {result['k']}")
    print("=" * 70)

    print("\nИсходная матрица A:")
    print(format_array(result["A"]))

    print("\nТочное решение x*:")
    print(format_array(result["x_exact"]))

    print("\nПравая часть b = A*x*:")
    print(format_array(result["b"]))

    print("\nПараметры вращений Гивенса:")

    if len(result["rotations"]) == 0:
        print("Вращения не выполнялись.")

    for step, rotation in enumerate(result["rotations"], start=1):
        print(f"Шаг {step}: столбец {rotation['column'] + 1}, плоскость ({rotation['row_support'] + 1}, {rotation['row_bottom'] + 1}); cos(phi) = {rotation['cos_phi']:.6g}; sin(phi) = {rotation['sin_phi']:.6g}")

    print("\nПромежуточные матрицы:")

    for column, matrix in enumerate(result["matrices"], start=1):
        print(f"\nПосле обработки {column}-го столбца:")
        print(format_array(matrix))

    print("\nИтоговая верхняя треугольная матрица R:")
    print(format_array(result["R"]))

    print("\nПреобразованная правая часть y = Q^T*b:")
    print(format_array(result["y"]))

    print("\nПолученная треугольная система R*x = y:")

    for i in range(n):
        coefficients = "  ".join(f"{result['R'][i, j]:.6g}" for j in range(n))
        print(f"[{coefficients}]   x = {result['y'][i]:.6g}")

    print("\nТочное решение:")
    print(format_array(result["x_exact"]))

    print("\nНайденное решение:")
    print(format_array(result["x_calculated"]))

    print(f"\nАбсолютная погрешность ||x - x*||_2 = {result['error']:.6g}")
    print(f"Невязка ||A*x - b||_2 = {result['residual']:.6g}")
    print(f"Время решения = {result['time']:.6g} с")


# 14. Общая функция решения одной задачи

def solve_problem(n, matrix_type="full", detailed=False):
    """
    matrix_type:
        "full" — заполненная матрица;
        "band" — ленточная матрица

    detailed:
        True — вывести подробную информацию;
        False — вернуть только итоговые результаты
    """
    if matrix_type not in ("full", "band"):
        raise ValueError('matrix_type должен быть "full" или "band"')

    if matrix_type == "full":
        k = n
        matrix_name = "Заполненная"
    else:
        k = n // 10
        matrix_name = "Ленточная"

    x_exact = make_exact_solution(n)
    right_b = make_right_side(n, k, x_exact)

    if matrix_type == "full":
        matrix_a = make_dense_matrix(n, k)

        start = perf_counter()
        matrix_r, y, rotations, matrices = givens_qr_dense_gaussian(matrix_a, right_b, k, save_history=detailed)
        x_calculated = back_substitution_dense(matrix_r, y, k)
        elapsed = perf_counter() - start

        matrix_r_for_print = matrix_r

    else:
        compact_matrix = make_compact_band_matrix(n, k)
        matrix_a = make_dense_matrix(n, k) if detailed else None

        start = perf_counter()
        matrix_r, y, rotations, matrices = givens_qr_band_gaussian(compact_matrix, right_b, k, save_history=detailed)
        x_calculated = back_substitution_band(matrix_r, y, k)
        elapsed = perf_counter() - start

        matrix_r_for_print = compact_to_dense(matrix_r, n, k) if detailed else None

    absolute_error = np.linalg.norm(x_calculated - x_exact, ord=2)
    calculated_b = make_right_side(n, k, x_calculated)
    residual = np.linalg.norm(calculated_b - right_b, ord=2)

    result = {
        "type": matrix_name,
        "matrix_type": matrix_type,
        "n": n,
        "k": k,
        "time": elapsed,
        "error": absolute_error,
        "residual": residual,
        "work": count_work(n, k)
    }

    if detailed:
        result.update({
            "A": matrix_a,
            "b": right_b,
            "R": matrix_r_for_print,
            "y": y,
            "x_exact": x_exact,
            "x_calculated": x_calculated,
            "rotations": rotations,
            "matrices": matrices
        })

        print_detailed_result(result)

    return result


# 15. Расчёты для различных N

def run_series(sizes, matrix_type):
    """
    Решает систему для каждого N и вычисляет:

    p времени = ln(T2/T1) / ln(N2/N1);
    p работы  = ln(W2/W1) / ln(N2/N1)
    """
    results = []
    previous_result = None

    if matrix_type == "full":
        title = "Заполненная матрица, K = N"
    else:
        title = "Ленточная матрица, K = floor(0.1*N)"

    print("\n" + title)
    print(f"{'N':>8} {'K':>8} {'Время, с':>14} {'Погрешность':>16} {'p времени':>14} {'p работы':>14}")

    for n in sizes:
        result = solve_problem(n=n, matrix_type=matrix_type, detailed=False)

        if previous_result is None:
            result["p_time"] = np.nan
            result["p_work"] = np.nan
        else:
            logarithm_n = np.log(result["n"] / previous_result["n"])
            result["p_time"] = np.log(result["time"] / previous_result["time"]) / logarithm_n

            if previous_result["work"] > 0 and result["work"] > 0:
                result["p_work"] = np.log(result["work"] / previous_result["work"]) / logarithm_n
            else:
                result["p_work"] = np.nan

        p_time_text = "—" if np.isnan(result["p_time"]) else f"{result['p_time']:.6g}"
        p_work_text = "—" if np.isnan(result["p_work"]) else f"{result['p_work']:.6g}"

        print(f"{result['n']:>8d} {result['k']:>8d} {result['time']:>14.6g} {result['error']:>16.6g} {p_time_text:>14} {p_work_text:>14}")

        results.append(result)
        previous_result = result

    return results


# 16. Сохранение результатов

def save_results_to_csv(filename, results):
    """Сохраняет результаты в CSV-файл."""
    with open(filename, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        writer.writerow(["Тип матрицы", "N", "K", "Время, с", "Абсолютная погрешность", "Невязка", "p времени", "p работы"])

        for result in results:
            p_time = "" if np.isnan(result["p_time"]) else f"{result['p_time']:.6g}"
            p_work = "" if np.isnan(result["p_work"]) else f"{result['p_work']:.6g}"
            writer.writerow([result["type"], result["n"], result["k"], f"{result['time']:.6g}", f"{result['error']:.6g}", f"{result['residual']:.6g}", p_time, p_work])

    print(f"\nРезультаты сохранены в файл: {filename}")


# 17. Главная часть программы

def main():
    # Раздел А: подробный расчёт для заполненной матрицы
    solve_problem(n=4, matrix_type="full", detailed=True)

    #Для подробного расчёта ленточной матрицы с K > 0:
    
    #solve_problem(n=20, matrix_type="band", detailed=True)
    
    #При N = 20 получаем K = 2.

    sizes = [4, 10, 50, 100, 500, 1000, 2000]

    run_n10000 = False

    if run_n10000:
        sizes.append(10_000)

    print("\n" + "=" * 90)
    print("РАЗДЕЛ Б. ИЗМЕРЕНИЕ ВРЕМЕНИ И ПРОВЕРКА ТРУДОЁМКОСТИ")
    print("=" * 90)

    full_results = run_series(sizes, matrix_type="full")
    band_results = run_series(sizes, matrix_type="band")

    all_results = full_results + band_results
    save_results_to_csv("benchmark_results.csv", all_results)


if __name__ == "__main__":
    main()