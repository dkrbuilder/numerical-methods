from dataclasses import dataclass
from time import perf_counter
import numpy as np


np.set_printoptions(suppress=True, linewidth=180, formatter={"float_kind": lambda value: f"{value:.6g}"})

# хранение результатов QR-разложения 
@dataclass
class QRDecomposition:
    r: np.ndarray   # матрица R
    reflectors: list[tuple[int, np.ndarray]]    # отражение Хаусхолдера 
    permutation: np.ndarray     # перестановка столбцов 
    rank: int   # ранг матрицы 


def separator(title: str) -> None:  
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def fmt(value: float) -> str:
    """Представление числа с шестью значащими цифрами"""
    return f"{float(value):.6g}"


def make_exact_solution(n: int) -> np.ndarray:
    """формирование точного решения N=5 возвращает (1, 1, 8, 6, 2), год рождения Давида Гильберта — 1862"""
    if n == 5:
        return np.array([1.0, 1.0, 8.0, 6.0, 2.0])
    pattern = np.array([1.0, 8.0, 6.0, 2.0])
    x_exact = np.empty(n, dtype=np.float64)
    x_exact[0] = 1.0
    if n > 1:
        x_exact[1:] = np.resize(pattern, n - 1)
    return x_exact


def make_full_rank_matrix(n: int) -> np.ndarray:
    """формирование матрицы части 1 размера (N+2) x N"""
    if n <= 0:
        raise ValueError("N должно быть положительным")
    row_values = 10.0 / np.arange(1, n + 3, dtype=np.float64)
    a = np.repeat(row_values[:, None], n, axis=1)
    for i in range(n):
        a[i, i] = float(min(n, 10))
    return a


def make_rank_deficient_matrix(n: int) -> np.ndarray:
    """формирование матрицы части 2 для нечётного варианта: A[i,j]=i+j-2ij."""
    if n <= 0:
        raise ValueError("N должно быть положительным")
    i = np.arange(1, n + 3, dtype=np.float64)[:, None]
    j = np.arange(1, n + 1, dtype=np.float64)[None, :]
    return i + j - 2.0 * i * j


def make_part1_right_sides(a: np.ndarray, x_exact: np.ndarray, perturbation: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Правые части части 1"""
    b_test = a @ x_exact
    b_first_changed = b_test.copy()
    b_first_changed[0] += perturbation
    b_main = b_test.copy()
    b_main[-1] += perturbation
    return b_test, b_first_changed, b_main


def make_part2_right_sides(a: np.ndarray, x_exact: np.ndarray, perturbation: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Правые части части 2"""
    b_test = a @ x_exact
    b_main = b_test.copy()
    b_main[0] += perturbation
    return b_test, b_main

# нахождения ранга 
def numerical_rank(a: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(a))


def householder_qr(a: np.ndarray) -> QRDecomposition:
    """QR-разложение Хаусхолдера без явного формирования Q"""
    r = np.asarray(a, dtype=np.float64).copy()
    m, n = r.shape
    reflectors = []
    tolerance = np.finfo(np.float64).eps * max(m, n) * max(1.0, np.linalg.norm(r, ord=np.inf))
    for k in range(min(m, n)):
        x = r[k:, k].copy()
        norm_x = np.linalg.norm(x)
        if norm_x <= tolerance:
            raise np.linalg.LinAlgError("Матрица не имеет полного столбцового ранга")
        alpha = -np.copysign(norm_x, x[0])
        # формирования вектора отражения 
        v = x.copy()
        v[0] -= alpha
        v /= np.linalg.norm(v)
        r[k:, k:] -= 2.0 * np.outer(v, v @ r[k:, k:])
        r[k + 1:, k] = 0.0
        reflectors.append((k, v))
    r[np.abs(r) <= tolerance] = 0.0
    return QRDecomposition(r, reflectors, np.arange(n), n)


def pivoted_householder_qr(a: np.ndarray) -> QRDecomposition:
    """QR-разложение Хаусхолдера с выбором ведущего столбца: A Pi = QR"""
    r = np.asarray(a, dtype=np.float64).copy()
    m, n = r.shape
    permutation = np.arange(n)
    reflectors = []
    tolerance = np.finfo(np.float64).eps * max(m, n) * max(1.0, np.linalg.norm(r, ord=np.inf))
    for k in range(min(m, n)):
        remaining_norms = np.linalg.norm(r[k:, k:], axis=0)    
        leading_position = k + int(np.argmax(remaining_norms))      # вычисления нормы для оставшихся столбцов 
        if remaining_norms[leading_position - k] <= tolerance:
            break
        if leading_position != k:
            r[:, [k, leading_position]] = r[:, [leading_position, k]]       #если выбранный столбец не первый 
            permutation[[k, leading_position]] = permutation[[leading_position, k]]
        x = r[k:, k].copy()
        norm_x = np.linalg.norm(x)
        alpha = -np.copysign(norm_x, x[0])
        v = x.copy()
        v[0] -= alpha
        v /= np.linalg.norm(v)
        r[k:, k:] -= 2.0 * np.outer(v, v @ r[k:, k:])
        r[k + 1:, k] = 0.0
        reflectors.append((k, v))
    r[np.abs(r) <= tolerance] = 0.0
    return QRDecomposition(r, reflectors, permutation, len(reflectors))


def apply_q_transpose(reflectors: list[tuple[int, np.ndarray]], vector: np.ndarray) -> np.ndarray:
    """вычислить Q^T vector без формирования Q"""
    result = np.asarray(vector, dtype=np.float64).copy()
    for k, v in reflectors:
        result[k:] -= 2.0 * v * np.dot(v, result[k:])
    return result


def back_substitution(upper: np.ndarray, right: np.ndarray) -> np.ndarray:
    """обратный ход"""
    n = upper.shape[0]
    solution = np.zeros(n, dtype=np.float64)
    tolerance = np.finfo(np.float64).eps * max(upper.shape) * max(1.0, np.linalg.norm(upper, ord=np.inf))
    for i in range(n - 1, -1, -1):
        if abs(upper[i, i]) <= tolerance:
            raise np.linalg.LinAlgError("Матрица R11 вырождена")
        solution[i] = (right[i] - np.dot(upper[i, i + 1:], solution[i + 1:])) / upper[i, i]
    return solution


def solve_least_squares_qr(decomposition: QRDecomposition, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """найти одно МНК-решение; свободные переменные в переставленной системе равны нулю"""
    rank = decomposition.rank
    n = decomposition.r.shape[1]
    qtb = apply_q_transpose(decomposition.reflectors, b)
    z = np.zeros(n, dtype=np.float64)
    z[:rank] = back_substitution(decomposition.r[:rank, :rank], qtb[:rank])
    x = np.zeros(n, dtype=np.float64)
    x[decomposition.permutation] = z
    return x, z, qtb


def form_q_explicitly(m: int, reflectors: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """восстановить Q по столбцам без матричных перемножений и явных H_k"""
    q = np.zeros((m, m), dtype=np.float64)
    for column in range(m):
        vector = np.zeros(m, dtype=np.float64)
        vector[column] = 1.0
        for k, v in reversed(reflectors):
            vector[k:] -= 2.0 * v * np.dot(v, vector[k:])
        q[:, column] = vector
    return q

# матрица перестановки 
def make_permutation_matrix(permutation: np.ndarray) -> np.ndarray:
    return np.eye(len(permutation))[:, permutation]


def solve_least_squares_svd(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """SVD"""
    u, singular_values, vt = np.linalg.svd(a, full_matrices=True)
    tolerance = np.finfo(np.float64).eps * max(a.shape) * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    inverse_values = np.zeros_like(singular_values)
    inverse_values[:rank] = 1.0 / singular_values[:rank]
    x = vt.T @ (inverse_values * (u[:, :len(singular_values)].T @ b))
    return x, u, singular_values, vt, rank


def pseudoinverse_from_svd(u: np.ndarray, singular_values: np.ndarray, vt: np.ndarray, rank: int) -> np.ndarray:
    """псевдообратная матрица A^+=V Sigma^+ U^T"""
    inverse_values = np.zeros_like(singular_values)
    inverse_values[:rank] = 1.0 / singular_values[:rank]
    return (vt.T * inverse_values) @ u[:, :len(singular_values)].T


def make_sigma(shape: tuple[int, int], singular_values: np.ndarray) -> np.ndarray:
    """прямоугольная матрица Sigma размера m x n"""
    sigma = np.zeros(shape, dtype=np.float64)
    np.fill_diagonal(sigma, singular_values)
    return sigma

# невязка
def residual_norm(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a @ x - b))


def relative_residual_norm(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """относительная невязка ||Ax-b||_2 / ||b||_2"""
    denominator = np.linalg.norm(b)
    return float(np.linalg.norm(a @ x - b) / denominator) if denominator != 0.0 else float(np.linalg.norm(a @ x - b))


def empirical_order(previous_n: int | None, previous_time: float | None, n: int, current_time: float) -> str:
    """Оценка экспериментального порядка"""
    if previous_n is None or previous_time is None or previous_time <= 0.0 or current_time <= 0.0:
        return "-"
    return fmt(np.log(current_time / previous_time) / np.log(n / previous_n))


# ==================================================================================================
# ЧАСТЬ 1 ЛАБОРАТОРНОЙ РАБОТЫ
# Полноранговые системы
# ==================================================================================================


def run_part1() -> None:
    """Часть 1: полноранговая система, пункты 0-7"""
    separator("ЧАСТЬ 1. ПОЛНОРАНГОВЫЕ СИСТЕМЫ")
    n = 5
    a = make_full_rank_matrix(n)
    x_exact = make_exact_solution(n)
    b_test, b_first_changed, b_main = make_part1_right_sides(a, x_exact)
    rank_a = numerical_rank(a)
    rank_test = numerical_rank(np.column_stack((a, b_test)))
    rank_first_changed = numerical_rank(np.column_stack((a, b_first_changed)))
    rank_main = numerical_rank(np.column_stack((a, b_main)))

    separator("ЧАСТЬ 1. ПУНКТ 0")
    print("A =\n", a)
    print("x_* =", x_exact)
    print("b_test =", b_test)
    print("b после изменения первого элемента =", b_first_changed)
    print("b_main =", b_main)
    print("rank(A) =", rank_a)
    print("rank([A|b_test]) =", rank_test)
    print("rank([A|b с изменённым b[0]]) =", rank_first_changed)
    print("rank([A|b_main]) =", rank_main)
    print("Изменение b[0] сохраняет совместность:", rank_first_changed == rank_a)
    print("Основная задача несовместна:", rank_main > rank_a)

    decomposition = householder_qr(a)

    separator("ЧАСТЬ 1. РАЗДЕЛ А (пп. 1, 2). РЕШЕНИЕ И НЕВЯЗКА")
    x_qr_main, z_main, qtb_main = solve_least_squares_qr(decomposition, b_main)
    print("R =\n", decomposition.r)
    print("Q^T b_main =", qtb_main)
    print("x_QR =", x_qr_main)
    print("||A x_QR - b_main||_2 =", fmt(residual_norm(a, x_qr_main, b_main)))
    print("||A x_QR - b_main||_2 / ||b_main||_2 =", fmt(relative_residual_norm(a, x_qr_main, b_main)))

    separator("ЧАСТЬ 1. РАЗДЕЛ Б (п. 3). ТЕСТОВЫЙ ПРИМЕР")
    x_qr_test, z_test, qtb_test = solve_least_squares_qr(decomposition, b_test)
    print("Найденное решение =", x_qr_test)
    print("Точное решение =", x_exact)
    print("Вычисление правой части b_test = A @ x_* =", b_test)
    print("Полученная треугольная матрица R1 =\n", decomposition.r[:n, :])
    print("Правая часть треугольной системы c1 = (Q^T b_test)[0:N] =", qtb_test[:n])
    print("Треугольная система: R1 @ x = c1")
    print("||x_QR - x_*||_2 =", fmt(np.linalg.norm(x_qr_test - x_exact)))
    print("||A x_QR - b_test||_2 =", fmt(residual_norm(a, x_qr_test, b_test)))

    separator("ЧАСТЬ 1. РАЗДЕЛ В (пп. 4, 5). МАТРИЧНЫЕ РАЗЛОЖЕНИЯ")
    q = form_q_explicitly(a.shape[0], decomposition.reflectors)
    pi = np.eye(n)
    qr_product = q @ decomposition.r
    print("Q =\n", q)
    print("Pi = I, поскольку в полноранговой части перестановки столбцов не выполняются:\n", pi)
    print("R =\n", decomposition.r)
    print("A Pi =\n", a @ pi)
    print("Q R =\n", qr_product)
    print("||Q^T Q - I||_F =", fmt(np.linalg.norm(q.T @ q - np.eye(n + 2), ord="fro")))
    print("||A - QR||_F =", fmt(np.linalg.norm(a - qr_product, ord="fro")))
    print("Проверка A = QR:", np.allclose(a, qr_product))

    separator("ЧАСТЬ 1. РАЗДЕЛ Г (пп. 6, 7). SVD И ПСЕВДООБРАТНАЯ МАТРИЦА")
    x_svd_main, u, singular_values, vt, rank_svd = solve_least_squares_svd(a, b_main)
    x_svd_test, u_test, singular_values_test, vt_test, rank_svd_test = solve_least_squares_svd(a, b_test)
    v = vt.T
    sigma = make_sigma(a.shape, singular_values)
    print("U =\n", u)
    print("V =\n", v)
    print("Sigma =\n", sigma)
    print("V^T =\n", vt)
    print("Сингулярные числа =", singular_values)
    print("Ранг по SVD =", rank_svd)
    print("||U^T U - I||_F =", fmt(np.linalg.norm(u.T @ u - np.eye(u.shape[0]), ord="fro")))
    print("||V^T V - I||_F =", fmt(np.linalg.norm(v.T @ v - np.eye(v.shape[0]), ord="fro")))
    print("U^T A V =\n", u.T @ a @ v)
    print("||U^T A V - Sigma||_F =", fmt(np.linalg.norm(u.T @ a @ v - sigma, ord="fro")))
    print("||A - U Sigma V^T||_F =", fmt(np.linalg.norm(a - u @ sigma @ vt, ord="fro")))
    print("SVD-решение тестовой задачи =", x_svd_test)
    print("Ошибка тестового решения =", fmt(np.linalg.norm(x_svd_test - x_exact)))
    print("SVD-решение основной задачи =", x_svd_main)
    print("Невязка основной задачи =", fmt(residual_norm(a, x_svd_main, b_main)))
    print("||x_QR - x_SVD||_2 =", fmt(np.linalg.norm(x_qr_main - x_svd_main)))

    a_plus = pseudoinverse_from_svd(u, singular_values, vt, rank_svd)
    a_plus_a = a_plus @ a
    a_a_plus = a @ a_plus
    x_pinv = a_plus @ b_main
    print("Способ построения: A^+ = V Sigma^+ U^T; ненулевые сингулярные числа заменяются обратными.")
    print("A^+ =\n", a_plus)
    print("A^+ A =\n", a_plus_a)
    print("A A^+ =\n", a_a_plus)
    print("||A^+ A - I_N||_F =", fmt(np.linalg.norm(a_plus_a - np.eye(n), ord="fro")))
    print("||A A^+ - I_(N+2)||_F =", fmt(np.linalg.norm(a_a_plus - np.eye(n + 2), ord="fro")))
    print("||(A^+A)^2 - A^+A||_F =", fmt(np.linalg.norm(a_plus_a @ a_plus_a - a_plus_a, ord="fro")))
    print("||(AA^+)^2 - AA^+||_F =", fmt(np.linalg.norm(a_a_plus @ a_a_plus - a_a_plus, ord="fro")))
    print("||(A^+A)^T - A^+A||_F =", fmt(np.linalg.norm(a_plus_a.T - a_plus_a, ord="fro")))
    print("||(AA^+)^T - AA^+||_F =", fmt(np.linalg.norm(a_a_plus.T - a_a_plus, ord="fro")))
    print("x = A^+ b_main =", x_pinv)
    print("||x_pinv - x_SVD||_2 =", fmt(np.linalg.norm(x_pinv - x_svd_main)))


def benchmark_part1(n_values: list[int]) -> None:
    separator("ЧАСТЬ 1. РАЗДЕЛ Д (п. 8). БОЛЬШАЯ СИСТЕМА И АНАЛИЗ ТРУДОЁМКОСТИ")
    print(f"{'N':>7} {'QR, с':>12} {'p_QR':>8} {'SVD, с':>12} {'p_SVD':>8} {'невязка QR':>14} {'отн. невязка':>14} {'невязка SVD':>14} {'отн. невязка':>14}")
    previous_n = None
    previous_qr_time = None
    previous_svd_time = None
    for n in n_values:
        a = make_full_rank_matrix(n)
        x_exact = make_exact_solution(n)
        b_test, b_first_changed, b_main = make_part1_right_sides(a, x_exact)
        qr_times = []
        for repeat in range(TIMING_REPEATS):
            start = perf_counter()
            decomposition = householder_qr(a)
            x_qr, z_qr, qtb = solve_least_squares_qr(decomposition, b_main)
            qr_times.append(perf_counter() - start)
        qr_time = float(np.median(qr_times))
        svd_times = []
        for repeat in range(TIMING_REPEATS):
            start = perf_counter()
            x_svd, u, singular_values, vt, rank_svd = solve_least_squares_svd(a, b_main)
            svd_times.append(perf_counter() - start)
        svd_time = float(np.median(svd_times))
        qr_order = empirical_order(previous_n, previous_qr_time, n, qr_time)
        svd_order = empirical_order(previous_n, previous_svd_time, n, svd_time)
        print(f"{n:7d} {qr_time:12.6g} {qr_order:>8} {svd_time:12.6g} {svd_order:>8} {residual_norm(a, x_qr, b_main):14.6g} {relative_residual_norm(a, x_qr, b_main):14.6g} {residual_norm(a, x_svd, b_main):14.6g} {relative_residual_norm(a, x_svd, b_main):14.6g}")
        previous_n = n
        previous_qr_time = qr_time
        previous_svd_time = svd_time
        if max(qr_time, svd_time) >= MAX_TIME_SECONDS:
            print("Достигнуто ограничение в один час; дальнейшее увеличение N остановлено.")
            break
    print("Теоретически для плотной полноранговой матрицы QR и SVD имеют кубическую трудоёмкость O(N^3); эмпирические p оцениваются по измеренному времени.")


# ==================================================================================================
# НАЧАЛО ЧАСТИ 2 ЛАБОРАТОРНОЙ РАБОТЫ
# Системы неполного ранга, нечётный вариант
# ==================================================================================================


def run_part2() -> None:
    """Часть 2: система неполного ранга нечётного варианта, пункты 1-7."""
    separator("ЧАСТЬ 2. СИСТЕМЫ НЕПОЛНОГО РАНГА. НЕЧЁТНЫЙ ВАРИАНТ")
    n = 5
    a = make_rank_deficient_matrix(n)
    x_exact = make_exact_solution(n)
    b_test, b_main = make_part2_right_sides(a, x_exact)
    rank_a = numerical_rank(a)
    rank_test = numerical_rank(np.column_stack((a, b_test)))
    rank_main = numerical_rank(np.column_stack((a, b_main)))
    print("A =\n", a)
    print("x_* =", x_exact)
    print("b_test =", b_test)
    print("b_main =", b_main)
    print("rank(A) =", rank_a)
    print("rank([A|b_test]) =", rank_test)
    print("rank([A|b_main]) =", rank_main)
    print("Тестовая система совместна:", rank_test == rank_a)
    print("Основная система несовместна:", rank_main > rank_a)

    decomposition = pivoted_householder_qr(a)

    separator("ЧАСТЬ 2. РАЗДЕЛ А (пп. 1, 2). РЕШЕНИЕ И НЕВЯЗКА")
    x_qr_main, z_main, qtb_main = solve_least_squares_qr(decomposition, b_main)
    print("Перестановка столбцов (нумерация с 1) =", decomposition.permutation + 1)
    print("Ранг, найденный QR-разложением =", decomposition.rank)
    print("R =\n", decomposition.r)
    print("Q^T b_main =", qtb_main)
    print("z =", z_main)
    print("Одно из МНК-решений x_QR =", x_qr_main)
    print("||A x_QR - b_main||_2 =", fmt(residual_norm(a, x_qr_main, b_main)))
    print("||A x_QR - b_main||_2 / ||b_main||_2 =", fmt(relative_residual_norm(a, x_qr_main, b_main)))

    separator("ЧАСТЬ 2. РАЗДЕЛ Б (п. 3). ТЕСТОВЫЙ ПРИМЕР")
    x_qr_test, z_test, qtb_test = solve_least_squares_qr(decomposition, b_test)
    print("Заданное решение x_* =", x_exact)
    print("Найденное QR-решение =", x_qr_test)
    print("Вычисление правой части b_test = A @ x_* =", b_test)
    print("Главная треугольная матрица R11 =\n", decomposition.r[:decomposition.rank, :decomposition.rank])
    print("Правая часть треугольной системы c1 = (Q^T b_test)[0:r] =", qtb_test[:decomposition.rank])
    print("Треугольная система: R11 @ z1 = c1; свободные компоненты z2 приняты равными нулю.")
    print("||A x_QR - b_test||_2 =", fmt(residual_norm(a, x_qr_test, b_test)))
    print("||x_QR - x_*||_2 =", fmt(np.linalg.norm(x_qr_test - x_exact)))
    print("Различие решений допустимо: при rank(A) < N совместная система имеет бесконечно много решений.")

    separator("ЧАСТЬ 2. РАЗДЕЛ В (пп. 4, 5). МАТРИЧНЫЕ РАЗЛОЖЕНИЯ")
    q = form_q_explicitly(a.shape[0], decomposition.reflectors)
    pi = make_permutation_matrix(decomposition.permutation)
    a_pi = a @ pi
    qr_product = q @ decomposition.r
    print("Q =\n", q)
    print("Pi =\n", pi)
    print("A Pi =\n", a_pi)
    print("Q R =\n", qr_product)
    print("||Q^T Q - I||_F =", fmt(np.linalg.norm(q.T @ q - np.eye(n + 2), ord="fro")))
    print("||A Pi - QR||_F =", fmt(np.linalg.norm(a_pi - qr_product, ord="fro")))
    print("Проверка A Pi = QR:", np.allclose(a_pi, qr_product))

    separator("ЧАСТЬ 2. РАЗДЕЛ Г (пп. 6, 7). SVD И ПСЕВДООБРАТНАЯ МАТРИЦА")
    x_svd_main, u, singular_values, vt, rank_svd = solve_least_squares_svd(a, b_main)
    x_svd_test, u_test, singular_values_test, vt_test, rank_svd_test = solve_least_squares_svd(a, b_test)
    v = vt.T
    sigma = make_sigma(a.shape, singular_values)
    print("U =\n", u)
    print("V =\n", v)
    print("Sigma =\n", sigma)
    print("V^T =\n", vt)
    print("Сингулярные числа =", singular_values)
    print("Ранг по SVD =", rank_svd)
    print("||U^T U - I||_F =", fmt(np.linalg.norm(u.T @ u - np.eye(u.shape[0]), ord="fro")))
    print("||V^T V - I||_F =", fmt(np.linalg.norm(v.T @ v - np.eye(v.shape[0]), ord="fro")))
    print("U^T A V =\n", u.T @ a @ v)
    print("||U^T A V - Sigma||_F =", fmt(np.linalg.norm(u.T @ a @ v - sigma, ord="fro")))
    print("||A - U Sigma V^T||_F =", fmt(np.linalg.norm(a - u @ sigma @ vt, ord="fro")))
    print("SVD-решение тестовой задачи =", x_svd_test)
    print("Невязка тестовой задачи =", fmt(residual_norm(a, x_svd_test, b_test)))
    print("SVD-решение основной задачи =", x_svd_main)
    print("Невязка основной задачи =", fmt(residual_norm(a, x_svd_main, b_main)))
    print("||x_QR - x_SVD||_2 =", fmt(np.linalg.norm(x_qr_main - x_svd_main)))
    print("||x_QR||_2 =", fmt(np.linalg.norm(x_qr_main)))
    print("||x_SVD||_2 =", fmt(np.linalg.norm(x_svd_main)))

    a_plus = pseudoinverse_from_svd(u, singular_values, vt, rank_svd)
    a_plus_a = a_plus @ a
    a_a_plus = a @ a_plus
    x_pinv = a_plus @ b_main
    print("Способ построения: A^+ = V Sigma^+ U^T; только ненулевые сингулярные числа заменяются обратными.")
    print("A^+ =\n", a_plus)
    print("A^+ A =\n", a_plus_a)
    print("A A^+ =\n", a_a_plus)
    print("||A^+ A - I_N||_F =", fmt(np.linalg.norm(a_plus_a - np.eye(n), ord="fro")))
    print("||A A^+ - I_(N+2)||_F =", fmt(np.linalg.norm(a_a_plus - np.eye(n + 2), ord="fro")))
    print("||(A^+A)^2 - A^+A||_F =", fmt(np.linalg.norm(a_plus_a @ a_plus_a - a_plus_a, ord="fro")))
    print("||(AA^+)^2 - AA^+||_F =", fmt(np.linalg.norm(a_a_plus @ a_a_plus - a_a_plus, ord="fro")))
    print("||(A^+A)^T - A^+A||_F =", fmt(np.linalg.norm(a_plus_a.T - a_plus_a, ord="fro")))
    print("||(AA^+)^T - AA^+||_F =", fmt(np.linalg.norm(a_a_plus.T - a_a_plus, ord="fro")))
    print("x = A^+ b_main =", x_pinv)
    print("||x_pinv - x_SVD||_2 =", fmt(np.linalg.norm(x_pinv - x_svd_main)))


def benchmark_part2(n_values: list[int]) -> None:
    separator("ЧАСТЬ 2. РАЗДЕЛ Д (п. 8). БОЛЬШАЯ СИСТЕМА И АНАЛИЗ ТРУДОЁМКОСТИ")
    print(f"{'N':>7} {'rank':>6} {'QR, с':>12} {'p_QR':>8} {'SVD, с':>12} {'p_SVD':>8} {'невязка QR':>14} {'отн. невязка':>14} {'невязка SVD':>14} {'отн. невязка':>14}")
    previous_n = None
    previous_qr_time = None
    previous_svd_time = None
    for n in n_values:
        a = make_rank_deficient_matrix(n)
        x_exact = make_exact_solution(n)
        b_test, b_main = make_part2_right_sides(a, x_exact)
        qr_times = []
        for repeat in range(TIMING_REPEATS):
            start = perf_counter()
            decomposition = pivoted_householder_qr(a)
            x_qr, z_qr, qtb = solve_least_squares_qr(decomposition, b_main)
            qr_times.append(perf_counter() - start)
        qr_time = float(np.median(qr_times))
        svd_times = []
        for repeat in range(TIMING_REPEATS):
            start = perf_counter()
            x_svd, u, singular_values, vt, rank_svd = solve_least_squares_svd(a, b_main)
            svd_times.append(perf_counter() - start)
        svd_time = float(np.median(svd_times))
        qr_order = empirical_order(previous_n, previous_qr_time, n, qr_time)
        svd_order = empirical_order(previous_n, previous_svd_time, n, svd_time)
        print(f"{n:7d} {rank_svd:6d} {qr_time:12.6g} {qr_order:>8} {svd_time:12.6g} {svd_order:>8} {residual_norm(a, x_qr, b_main):14.6g} {relative_residual_norm(a, x_qr, b_main):14.6g} {residual_norm(a, x_svd, b_main):14.6g} {relative_residual_norm(a, x_svd, b_main):14.6g}")
        previous_n = n
        previous_qr_time = qr_time
        previous_svd_time = svd_time
        if max(qr_time, svd_time) >= MAX_TIME_SECONDS:
            print("Достигнуто ограничение в один час; дальнейшее увеличение N остановлено.")
            break
    print("Для этой матрицы rank(A)=2: QR с ранней остановкой обрабатывает два ведущих столбца, а полное SVD остаётся существенно более трудоёмким.")


# Требуемые размеры из раздела Д 
LARGE_N_VALUES = [100, 500, 1000]
MAX_TIME_SECONDS = 3600.0
TIMING_REPEATS = 3


# ЗАПУСК ЧАСТИ 1
run_part1()
benchmark_part1(LARGE_N_VALUES)

# ЗАПУСК ЧАСТИ 2
run_part2()
benchmark_part2(LARGE_N_VALUES)
