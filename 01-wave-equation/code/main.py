import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')


# МЕТОД ПРОГОНКИ

def thomas_algorithm(a, b, c, d):
    n = len(d)
    c_new = np.zeros(n)
    d_new = np.zeros(n)
    x = np.zeros(n)

    c_new[0] = c[0] / b[0]
    d_new[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_new[i-1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        if i < n - 1:
            c_new[i] = c[i] / denom
        d_new[i] = (d[i] - a[i] * d_new[i-1]) / denom

    x[n-1] = d_new[n-1]
    for i in range(n-2, -1, -1):
        x[i] = d_new[i] - c_new[i] * x[i+1]
    return x


# ТЕСТОВЫЕ ЗАДАЧИ

class TestTask:
    def __init__(self, name, sol_func, y_t, y_tt, y_x, y_xx, a_coef):
        self.name = name
        self.u = sol_func
        self.u_t = y_t
        self.u_x = y_x
        self.u_xx = y_xx
        self.f = lambda t, x: y_tt(t, x) - a_coef * y_xx(t, x)

def test_constant(a_coef):
    return TestTask("Константа: y = 5", 
                    lambda t,x: 5 + 0*x, lambda t,x: 0*x, lambda t,x: 0*x, 
                    lambda t,x: 0*x, lambda t,x: 0*x, a_coef)

def test_plane(a_coef):
    return TestTask("Плоскость: y = 2x + 3t + 1", 
                    lambda t,x: 2*x + 3*t + 1, lambda t,x: 3+0*x, lambda t,x: 0*x, 
                    lambda t,x: 2+0*x, lambda t,x: 0*x, a_coef)

def test_linear_quadratic(a_coef):
    return TestTask("Линейно-квадр: y = 2x^2 + 3t + 1", 
                    lambda t,x: 2*x**2 + 3*t + 1, lambda t,x: 3+0*x, lambda t,x: 0*x, 
                    lambda t,x: 4*x, lambda t,x: 4+0*x, a_coef)

def test_paraboloid(a_coef):
    return TestTask("Параболоид: y = 2x^2 + 3t^2 + 1", 
                    lambda t,x: 2*x**2 + 3*t**2 + 1, lambda t,x: 6*t+0*x, lambda t,x: 6+0*x, 
                    lambda t,x: 4*x, lambda t,x: 4+0*x, a_coef)

def test_exponential(a_coef):
    """
    Тест: y = exp(t / ln(2 + x))
    """
    def func(t, x):
        return np.exp(t / np.log(2 + x))
    
    def y_t(t, x):
        alpha = 1.0 / np.log(2 + x)
        return alpha * func(t, x)
    
    def y_tt(t, x):
        alpha = 1.0 / np.log(2 + x)
        return alpha**2 * func(t, x)
    
    def y_x(t, x):
        alpha = 1.0 / np.log(2 + x)
        return -t * alpha**2 / (2 + x) * func(t, x)
    
    def y_xx(t, x):
        alpha = 1.0 / np.log(2 + x)
        alpha_prime = -alpha**2 / (2 + x)
        
        term1 = 2 * alpha * alpha_prime / (2 + x)
        term2 = -alpha**2 / (2 + x)**2
        term3 = t * alpha**2 * alpha_prime / (2 + x)
        
        return -t * (term1 + term2 + term3) * func(t, x)
    
    return TestTask("Экспонента: y = exp(t/ln(2+x))", 
                   func, y_t, y_tt, y_x, y_xx, a_coef)

def get_tests(a_coef):
    return [
        test_constant(a_coef),
        test_plane(a_coef),
        test_linear_quadratic(a_coef),
        test_paraboloid(a_coef),
        test_exponential(a_coef)
    ]


# РС

def solve_wave(test, L, T, N, K, a_coef, sigma=0.5, bc_order=2):
    h = L / N
    tau = T / K
    gamma = a_coef * tau**2 / h**2
    x = np.linspace(0, L, N+1)
    t = np.linspace(0, T, K+1)
    y = np.zeros((K+1, N+1))
    
    # cлой n=0
    y[0, :] = test.u(0, x)
    
    # cлой n=1
    y_tt_0 = a_coef * test.u_xx(0, x) + test.f(0, x)
    y[1, :] = y[0, :] + tau * test.u_t(0, x) + 0.5 * tau**2 * y_tt_0
    
    # левая граница
    y[1, 0] = test.u(tau, 0)
    
    # правая граница для первого слоя
    gL_val = test.u_x(tau, L)
    if bc_order == 1:
        y[1, N] = y[1, N-1] + h * gL_val
    else:
        y[1, N] = (4*y[1, N-1] - y[1, N-2] + 2*h*gL_val) / 3
    
    # основной цикл
    for n in range(1, K):
        a_diag = np.zeros(N+1)
        b_diag = np.zeros(N+1)
        c_diag = np.zeros(N+1)
        d_vec = np.zeros(N+1)
        
        # левая граница
        b_diag[0] = 1.0
        d_vec[0] = test.u(t[n+1], 0)
        
        # внутренние узлы
        for i in range(1, N):
            a_diag[i] = -gamma * sigma
            b_diag[i] = 1.0 + 2.0 * gamma * sigma
            c_diag[i] = -gamma * sigma
            
            y_xx_n = (y[n, i+1] - 2*y[n, i] + y[n, i-1]) / h**2
            y_xx_nm1 = (y[n-1, i+1] - 2*y[n-1, i] + y[n-1, i-1]) / h**2
            
            d_vec[i] = 2*y[n, i] - y[n-1, i] + \
                       a_coef * tau**2 * (1 - 2*sigma) * y_xx_n + \
                       a_coef * tau**2 * sigma * y_xx_nm1 + \
                       tau**2 * test.f(t[n], x[i])

        # правая граница
        gL_next = test.u_x(t[n+1], L)
        if bc_order == 1:
            a_diag[N] = -1.0
            b_diag[N] = 1.0
            d_vec[N] = h * gL_next
        else:
            a_diag[N] = -2.0 * gamma * sigma
            b_diag[N] = 1.0 + 2.0 * gamma * sigma
            
            y_xx_n_N = (2.0/h**2) * (y[n, N-1] - y[n, N] + h * test.u_x(t[n], L))
            y_xx_nm1_N = (2.0/h**2) * (y[n-1, N-1] - y[n-1, N] + h * test.u_x(t[n-1], L))
            
            d_vec[N] = 2*y[n, N] - y[n-1, N] + \
                       a_coef * tau**2 * (1 - 2*sigma) * y_xx_n_N + \
                       a_coef * tau**2 * sigma * y_xx_nm1_N + \
                       tau**2 * test.f(t[n], x[N]) + \
                       2.0 * gamma * sigma * h * gL_next
            
        y[n+1, :] = thomas_algorithm(a_diag, b_diag, c_diag, d_vec)
        
    return x, t, y


# ТАБЛИЦЫ ПОГРЕШНОСТЕЙ (для заданного порядка ГУ)

def run_error_matrix_experiment(test_task, L_val, a_val, T_multiplier=10.0, bc_order=2):
    """
    Построение таблицы погрешностей
    bc_order: 1 - первый порядок, 2 - второй порядок
    """
    T_val = T_multiplier * L_val
    
    hs = [0.1, 0.05, 0.025, 0.0125]
    taus = [0.05, 0.025, 0.0125, 0.00625]
    
    order_text = "1-й порядок ГУ" if bc_order == 1 else "2-й порядок ГУ"
    
    print(f"\n{'='*90}")
    print(f"ТАБЛИЦА ПОГРЕШНОСТЕЙ (T = {T_multiplier}L = {T_val:.1f})")
    print(f"Тест: {test_task.name}, {order_text}")
    print("="*90)
    
    # заголовок таблицы
    header = f"{'tau \\ h':<12} | " + " | ".join([f"h = {h:<8.4f}" for h in hs])
    print(header)
    print("-" * len(header))
    
    for i, tau_curr in enumerate(taus):
        row = f"tau = {tau_curr:<6.4f} | "
        for j, h_curr in enumerate(hs):
            N_steps = int(round(L_val / h_curr))
            K_steps = int(round(T_val / tau_curr))
            
            try:
                x_m, t_m, y_m = solve_wave(test_task, L_val, T_val, N_steps, K_steps, a_val, sigma=0.5, bc_order=bc_order)
                u_exact = np.array([test_task.u(ti, x_m) for ti in t_m])
                error_max = np.max(np.abs(y_m - u_exact))
            except Exception:
                error_max = np.nan
            
            row += f"{error_max:12.2e} | "
        print(row)
    
    print("="*90)


# ПОСТРОЕНИЕ ГРАФИКОВ 

def plot_solution(test, L, T, a_coef, N_plot=50, K_plot=100):
    """
    Построение графиков решения при разных t
    """
    x_plot, t_plot, y1 = solve_wave(test, L, T, N_plot, K_plot, a_coef, bc_order=1)
    _, _, y2 = solve_wave(test, L, T, N_plot, K_plot, a_coef, bc_order=2)
    
    u_exact = np.array([test.u(ti, x_plot) for ti in t_plot])
    
    # несколько моментов времени
    time_indices = [0, K_plot//4, K_plot//2, 3*K_plot//4, K_plot-1]
    time_labels = [f"t = {t_plot[idx]:.3f}" for idx in time_indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # точное решение при разных t
    ax = axes[0]
    for idx, label in zip(time_indices, time_labels):
        ax.plot(x_plot, u_exact[idx, :], linewidth=2, label=label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Точное решение\n{test.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # численное решение (1-й порядок ГУ) при разных t
    ax = axes[1]
    for idx, label in zip(time_indices, time_labels):
        ax.plot(x_plot, y1[idx, :], linewidth=2, label=label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Численное решение\n(sigma=0.5, 1-й порядок ГУ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # численное решение (2-й порядок ГУ) при разных t
    ax = axes[2]
    for idx, label in zip(time_indices, time_labels):
        ax.plot(x_plot, y2[idx, :], linewidth=2, label=label)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Численное решение\n(sigma=0.5, 2-й порядок ГУ)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Сравнение решений для теста: {test.name}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # сравнение 1-го и 2-го порядка при t = T/2
    fig2, ax = plt.subplots(figsize=(10, 6))
    mid_time = K_plot // 2
    
    ax.plot(x_plot, u_exact[mid_time, :], 'k-', linewidth=2, label='Точное решение')
    ax.plot(x_plot, y1[mid_time, :], 'r--', linewidth=2, label='1-й порядок ГУ')
    ax.plot(x_plot, y2[mid_time, :], 'b-.', linewidth=2, label='2-й порядок ГУ')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Сравнение решений при t = {T/2:.3f} для теста: {test.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()



# ЗАПУСК

L, T_short, a_coef = 1.0, 0.5, 1.0
N, K = 40, 80
h = L / N
tau = T_short / K

print("="*80)
print("ИССЛЕДОВАНИЕ СХЕМЫ С ВЕСОМ sigma = 0.5 ДЛЯ ВОЛНОВОГО УРАВНЕНИЯ")
print("="*80)
print(f"  L = {L}, T_short = {T_short}, a = {a_coef}")
print(f"  N = {N}, K = {K}")
print(f"  h = {h:.6f}, tau = {tau:.6f}")
print(f"  gamma = a·tau^2/h^2 = {a_coef * tau**2 / h**2:.6f}")
print(f"  tua/h = {tau/h:.4f} (условие Куранта: tau <= h -> {tau <= h})")
print("="*80)

# все тесты
tests = get_tests(a_coef)


# СВОДНАЯ ТАБЛИЦА ПОГРЕШНОСТЕЙ 

print("\n" + "="*80)
print("СВОДНАЯ ТАБЛИЦА ПОГРЕШНОСТЕЙ (T = 0.5, N=40, K=80)")
print("="*80)
print(f"{'Тест':<42} | {'Погр. 1-й пор.':<15} | {'Погр. 2-й пор.':<15} | {'Улучшение':<12}")
print("-" * 95)

for test in tests:
    try:
        _, _, y1 = solve_wave(test, L, T_short, N, K, a_coef, bc_order=1)
        x, t, y2 = solve_wave(test, L, T_short, N, K, a_coef, bc_order=2)
        
        u_exact = np.array([test.u(ti, x) for ti in t])
        err1 = np.max(np.abs(y1 - u_exact))
        err2 = np.max(np.abs(y2 - u_exact))
        
        improvement = err1 / err2 if err2 > 0 else 0
        
        print(f"{test.name:<42} | {err1:15.2e} | {err2:15.2e} | {improvement:12.2f}x")
    except Exception as e:
        print(f"{test.name:<42} | {'ОШИБКА':>15} | {'ОШИБКА':>15} | {'---':>12}")

print("="*95)



# ТАБЛИЦЫ ПОГРЕШНОСТЕЙ ДЛЯ 1-ГО И 2-ГО ПОРЯДКА

print("\n" + "="*80)
print("ЧИСЛЕННЫЙ ЭКСПЕРИМЕНТ: ТАБЛИЦЫ ПОГРЕШНОСТЕЙ (T = 10L = 10.0)")


print("\n")
print("ТАБЛИЦЫ ДЛЯ 1-ГО ПОРЯДКА АППРОКСИМАЦИИ КРАЕВОГО УСЛОВИЯ")


for test in tests:
    run_error_matrix_experiment(test, L, a_coef, T_multiplier=10.0, bc_order=1)

print("\n" + "="*80 + "\n")
print("ТАБЛИЦЫ ДЛЯ 2-ГО ПОРЯДКА АППРОКСИМАЦИИ КРАЕВОГО УСЛОВИЯ")


for test in tests:
    run_error_matrix_experiment(test, L, a_coef, T_multiplier=10.0, bc_order=2)



# ИССЛЕДОВАНИЕ СХОДИМОСТИ ДЛЯ ЭКСПОНЕНТЫ

print("\n" + "="*80)
print("ИССЛЕДОВАНИЕ СХОДИМОСТИ (экспонента, 2-й порядок ГУ)")
print("="*80)
print(f"{'N':>6} | {'h':>10} | {'Погрешность':>15}")
print("-" * 50)

test_exp = tests[-1]
hs_conv = []
errors_conv = []

for step in [10, 20, 40, 80]:
    cur_x, cur_t, cur_y = solve_wave(test_exp, L, T_short, step, step*2, a_coef, bc_order=2)
    u_ex = np.array([test_exp.u(ti, cur_x) for ti in cur_t])
    err = np.max(np.abs(cur_y - u_ex))
    hs_conv.append(L / step)
    errors_conv.append(err)
    
    print(f"{step:>6} | {hs_conv[-1]:10.6f} | {err:15.2e}")

print("="*70)


# ГРАФИКИ РЕШЕНИЯ ДЛЯ ВСЕХ ТЕСТОВ 

print("\n" + "="*80)
print("ПОСТРОЕНИЕ ГРАФИКОВ РЕШЕНИЯ ДЛЯ ВСЕХ ТЕСТОВ (при разных t)")
print("="*80)

for test in tests:
    print(f"\nПостроение графиков для теста: {test.name}")
    plot_solution(test, L, T_short, a_coef, N_plot=50, K_plot=100)
