import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')



# 1. МЕТОД ПРОГОНКИ 
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



# 2. ТЕСТОВЫЕ ЗАДАЧИ
class TestTask:
    def __init__(self, name, sol_func, u_t, u_x, u_xx):
        self.name = name
        self.u = sol_func
        self.u_t = u_t
        self.u_x = u_x
        self.u_xx = u_xx

    def f(self, t, x, a_func):
        eps = 1e-5
        def flux(t_val, x_val):
            u_val = self.u(t_val, x_val)
            return a_func(t_val, x_val, u_val) * self.u_x(t_val, x_val)
        div_flux = (flux(t, x + eps) - flux(t, x - eps)) / (2 * eps)
        return self.u_t(t, x) - div_flux


def test_constant():
    return TestTask("Константа: y = 5", 
                    lambda t,x: 5 + 0*x, lambda t,x: 0*x, 
                    lambda t,x: 0*x, lambda t,x: 0*x)

def test_plane():
    return TestTask("Плоскость: y = 2x + 3t + 1", 
                    lambda t,x: 2*x + 3*t + 1, lambda t,x: 3+0*x, 
                    lambda t,x: 2+0*x, lambda t,x: 0*x)

def test_linear_quadratic():
    return TestTask("Линейно-квадр: y = 2x² + 3t + 1", 
                    lambda t,x: 2*x**2 + 3*t + 1, lambda t,x: 3+0*x, 
                    lambda t,x: 4*x, lambda t,x: 4+0*x)

def test_paraboloid():
    return TestTask("Параболоид: y = 2x² + 3t² + 1", 
                    lambda t,x: 2*x**2 + 3*t**2 + 1, lambda t,x: 6*t+0*x, 
                    lambda t,x: 4*x, lambda t,x: 4+0*x)

def test_exponential():
    def func(t, x):
        return np.exp(t / np.log(2 + x))
    def u_t(t, x):
        alpha = 1.0 / np.log(2 + x)
        return alpha * func(t, x)
    def u_x(t, x):
        alpha = 1.0 / np.log(2 + x)
        return -t * alpha**2 / (2 + x) * func(t, x)
    def u_xx(t, x):
        alpha = 1.0 / np.log(2 + x)
        alpha_prime = -alpha**2 / (2 + x)
        term1 = 2 * alpha * alpha_prime / (2 + x)
        term2 = -alpha**2 / (2 + x)**2
        term3 = t * alpha**2 * alpha_prime / (2 + x)
        return -t * (term1 + term2 + term3) * func(t, x)
    return TestTask("Экспонента: y = exp(t/ln(2+x))", 
                   func, u_t, u_x, u_xx)


def test_new():
    """тест: u(x,t) = 1 + t² + (1+x)²"""
    def func(t, x):
        return 1 + t**2 + (1 + x)**2
    
    def u_t(t, x):
        return 2 * t
    
    def u_x(t, x):
        return 2 * (1 + x)
    
    def u_xx(t, x):
        return 2 * np.ones_like(x)
    
    return TestTask("тест: y = 1 + t² + (1+x)²", 
                   func, u_t, u_x, u_xx)


def get_tests():
    return [test_constant(), test_plane(), test_linear_quadratic(), 
            test_paraboloid(), test_exponential(), test_new()]



# 3. КОЭФФИЦИЕНТЫ
def coeff_const(t, x, u):
    return 1.0

def coeff_tx(t, x, u):
    return 1.0 + 0.5 * np.sin(t) * np.cos(np.pi * x)

def coeff_txu(t, x, u):
    return 1.0 + 1.0 / (1.0 + u**2)

def coeff_new(t, x, u):
    """k(u) = 1 + u^{1/3}"""
    return 1.0 + np.abs(u)**(1/3)



# 4. НЕЯВНАЯ СХЕМА
def implicit_scheme(test, L, T, N, K, a_func, sigma=0.5, max_iter=50, tol=1e-8):
    h = L / N
    tau = T / K
    x = np.linspace(0, L, N+1)
    t = np.linspace(0, T, K+1)
    y = np.zeros((K+1, N+1))
    y[0, :] = test.u(0, x)
    
    def g0(t_val): return test.u(t_val, 0)
    def gL(t_val): return test.u(t_val, L)
    
    for n in range(K):
        y_n = y[n, :].copy()
        y_new = y_n.copy()
        for s in range(max_iter):
            y_old = y_new.copy()
            a_half = np.zeros(N)
            for i in range(N):
                x_half = (x[i] + x[i+1]) / 2
                u_half = (y_old[i] + y_old[i+1]) / 2
                a_half[i] = a_func(t[n+1], x_half, u_half)
            
            a_diag = np.zeros(N+1); b_diag = np.zeros(N+1); c_diag = np.zeros(N+1); d_vec = np.zeros(N+1)
            b_diag[0] = 1.0; d_vec[0] = g0(t[n+1])
            for i in range(1, N):
                a_diag[i] = - (tau * sigma / h**2) * a_half[i-1]
                b_diag[i] = 1 + (tau * sigma / h**2) * (a_half[i-1] + a_half[i])
                c_diag[i] = - (tau * sigma / h**2) * a_half[i]
                explicit_op = (a_half[i] * (y_n[i+1] - y_n[i]) - a_half[i-1] * (y_n[i] - y_n[i-1])) / h**2
                f_val = test.f(t[n] + tau/2, x[i], a_func)
                d_vec[i] = y_n[i] + tau * (1 - sigma) * explicit_op + tau * f_val
            b_diag[N] = 1.0; d_vec[N] = gL(t[n+1])
            y_new = thomas_algorithm(a_diag, b_diag, c_diag, d_vec)
            if np.max(np.abs(y_new - y_old)) < tol:
                break
        y[n+1, :] = y_new
    return x, t, y



# 5. МЕТОД ПРЯМЫХ + РУНГЕ-КУТТА 4
def mol_rhs(t, u, a_func, g0, gL, h, f_func, test):
    N = len(u) - 1
    du_dt = np.zeros(N + 1)
    u_t = u.copy()
    u_t[0] = g0(t)
    u_t[N] = gL(t)
    du_dt[0] = 0.0
    du_dt[N] = 0.0
    for i in range(1, N):
        x_half_right = (i*h + (i+1)*h) / 2
        x_half_left = ((i-1)*h + i*h) / 2
        a_right = a_func(t, x_half_right, (u_t[i] + u_t[i+1])/2)
        a_left = a_func(t, x_half_left, (u_t[i-1] + u_t[i])/2)
        du_dt[i] = (a_right * (u_t[i+1] - u_t[i]) - a_left * (u_t[i] - u_t[i-1])) / h**2 + f_func(t, i*h, a_func)
    return du_dt

def runge_kutta_4_mol(test, L, T, N, K, a_func):
    h = L / N
    tau = T / K
    x = np.linspace(0, L, N+1)
    t = np.linspace(0, T, K+1)
    stability_limit = 0.69
    if tau / h**2 > stability_limit:
        print(f"ВНИМАНИЕ: τ/h² = {tau/h**2:.4f} > {stability_limit}, возможна неустойчивость!")
        print(f"    Рекомендуемое τ/h² должно быть < {stability_limit}")
    u0 = test.u(0, x)
    def g0(t_val): return test.u(t_val, 0)
    def gL(t_val): return test.u(t_val, L)
    def rhs_wrapper(t_val, u_val):
        return mol_rhs(t_val, u_val, a_func, g0, gL, h, test.f, test)
    y = np.zeros((K+1, N+1))
    y[0, :] = u0
    for n in range(K):
        tn = t[n]
        un = y[n, :]
        k1 = rhs_wrapper(tn, un)
        k2 = rhs_wrapper(tn + tau/2, un + tau/2 * k1)
        k3 = rhs_wrapper(tn + tau/2, un + tau/2 * k2)
        k4 = rhs_wrapper(tn + tau, un + tau * k3)
        y[n+1, :] = un + tau/6 * (k1 + 2*k2 + 2*k3 + k4)
        y[n+1, 0] = g0(t[n+1])
        y[n+1, N] = gL(t[n+1])
        if np.any(np.isnan(y[n+1, :])) or np.any(np.abs(y[n+1, :]) > 1e100):
            print(f"НЕУСТОЙЧИВОСТЬ! Решение взорвалось на шаге n={n}")
            y_truncated = y[:n+2, :]
            t_truncated = t[:n+2]
            return x, t_truncated, y_truncated
    return x, t, y



# 6. ВЫЧИСЛЕНИЕ ПОГРЕШНОСТИ
def compute_error(y, test, x, t):
    t_truncated = t[:y.shape[0]]
    u_exact = np.zeros_like(y)
    for n in range(len(t_truncated)):
        u_exact[n, :] = test.u(t_truncated[n], x)
    return np.max(np.abs(y - u_exact))



# 7. ИССЛЕДОВАНИЕ СХОДИМОСТИ
def convergence_study(test, L, T, N_list, K_list, a_func, method='implicit'):
    print(f"\n{'='*90}")
    print(f"ТЕСТ: {test.name}")
    print(f"МЕТОД: {'Неявная схема (σ=0.5)' if method == 'implicit' else 'Метод прямых + Рунге-Кутта 4'}")
    print(f"{'='*90}")
    print(f"{'N':>6} | {'K':>6} | {'h':>10} | {'τ':>10} | {'τ/h²':>10} | {'Погрешность':>15} | {'Время, с':>12}")
    print("-" * 90)
    
    results = []
    for N, K in zip(N_list, K_list):
        h = L / N
        tau = T / K
        stability_param = tau / h**2
        
        start_time = time.time()
        try:
            if method == 'implicit':
                x, t, y = implicit_scheme(test, L, T, N, K, a_func)
            else:
                x, t, y = runge_kutta_4_mol(test, L, T, N, K, a_func)
            elapsed_time = time.time() - start_time
            error = compute_error(y, test, x, t)
            if method == 'mol_rk4' and (np.isnan(error) or error > 1e10):
                error_str = "НЕУСТОЙЧИВО"
            else:
                error_str = f"{error:15.2e}"
            print(f"{N:>6} | {K:>6} | {h:10.6f} | {tau:10.6f} | {stability_param:10.4f} | {error_str:>15} | {elapsed_time:12.4f}")
            results.append({'N':N, 'K':K, 'h':h, 'tau':tau, 'stability_param':stability_param, 'error':error, 'time':elapsed_time})
        except Exception as e:
            print(f"{N:>6} | {K:>6} | {h:10.6f} | {tau:10.6f} | {stability_param:10.4f} | {'ОШИБКА':>15} | {'---':>12}")
            results.append({'N':N, 'K':K, 'h':h, 'tau':tau, 'stability_param':stability_param, 'error':np.nan, 'time':np.nan})
    return results


# 8. СРАВНЕНИЕ МЕТОДОВ ПРИ ФИКСИРОВАННЫХ ПАРАМЕТРАХ
def comparison_table(test, L, T, N, K_imp, K_rk4, a_func):
    print(f"\n{'='*80}")
    print(f"СРАВНЕНИЕ МЕТОДОВ ДЛЯ ТЕСТА: {test.name}")
    h = L / N
    print(f"Параметры: N={N}, h={h:.6f}")
    print(f"  Неявная схема: K={K_imp}, τ={T/K_imp:.6f}, τ/h²={(T/K_imp)/h**2:.4f}")
    print(f"  Метод РК4:      K={K_rk4}, τ={T/K_rk4:.6f}, τ/h²={(T/K_rk4)/h**2:.4f}")
    print(f"{'='*80}")
    
    start = time.time()
    x1, t1, y1 = implicit_scheme(test, L, T, N, K_imp, a_func)
    time_imp = time.time() - start
    error_imp = compute_error(y1, test, x1, t1)
    
    start = time.time()
    x2, t2, y2 = runge_kutta_4_mol(test, L, T, N, K_rk4, a_func)
    time_mol = time.time() - start
    error_mol = compute_error(y2, test, x2, t2)
    
    print(f"\n{'Метод':<35} | {'Погрешность':<15} | {'Время, с':<12}")
    print("-" * 65)
    print(f"{'Неявная схема (σ=0.5)':<35} | {error_imp:15.2e} | {time_imp:12.4f}")
    print(f"{'Метод прямых + РК4':<35} | {error_mol:15.2e} | {time_mol:12.4f}")
    return {'implicit': {'error': error_imp, 'time': time_imp},
            'mol_rk4': {'error': error_mol, 'time': time_mol}}



# 9. ВИЗУАЛИЗАЦИЯ
def plot_solution(test, L, T, a_func, N_plot=40, K_plot_imp=80, K_plot_mol=1600):
    """Построение графиков решения"""
    x_imp, t_imp, y_imp = implicit_scheme(test, L, T, N_plot, K_plot_imp, a_func)
    x_mol, t_mol, y_mol = runge_kutta_4_mol(test, L, T, N_plot, K_plot_mol, a_func)
    
    u_exact = np.zeros((K_plot_imp+1, N_plot+1))
    for n in range(K_plot_imp+1):
        u_exact[n, :] = test.u(t_imp[n], x_imp)
    
    X_mesh, T_mesh = np.meshgrid(x_imp, t_imp)
    
    fig = plt.figure(figsize=(18, 5))
    
    ax = fig.add_subplot(131, projection='3d')
    ax.plot_surface(X_mesh, T_mesh, u_exact, cmap=cm.viridis, alpha=0.9)
    ax.set_title(f'Точное решение\n{test.name}')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    
    ax = fig.add_subplot(132, projection='3d')
    ax.plot_surface(X_mesh, T_mesh, y_imp, cmap=cm.viridis, alpha=0.9)
    ax.set_title(f'Неявная схема (σ=0.5)\n{test.name}')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    
    ax = fig.add_subplot(133, projection='3d')
    # Для РК4 может быть разная сетка
    step = max(1, y_mol.shape[0] // (K_plot_imp+1))
    y_mol_downsampled = y_mol[::step, :]
    t_mol_downsampled = t_mol[::step]
    if len(t_mol_downsampled) > K_plot_imp+1:
        t_mol_downsampled = t_mol_downsampled[:K_plot_imp+1]
        y_mol_downsampled = y_mol_downsampled[:K_plot_imp+1, :]
    X_mesh_mol, T_mesh_mol = np.meshgrid(x_mol, t_mol_downsampled)
    ax.plot_surface(X_mesh_mol, T_mesh_mol, y_mol_downsampled, cmap=cm.viridis, alpha=0.9)
    ax.set_title(f'Метод прямых + РК4\n{test.name}')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    
    plt.tight_layout()
    plt.show()
    
    # Сравнение при t = T/2
    fig2, ax = plt.subplots(figsize=(10, 6))
    mid_time_imp = K_plot_imp // 2
    ax.plot(x_imp, u_exact[mid_time_imp, :], 'k-', linewidth=2, label='Точное решение')
    ax.plot(x_imp, y_imp[mid_time_imp, :], 'r--', linewidth=2, label='Неявная схема')
    
    mid_time_mol = min(len(t_mol)-1, int(len(t_mol) // 2))
    ax.plot(x_mol, y_mol[mid_time_mol, :], 'b-.', linewidth=2, label='Метод прямых + РК4')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Сравнение решений при t = {T/2:.3f} для теста: {test.name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()



# 10. МЕНЮ ДЛЯ ВЫБОРА ТЕСТА И КОЭФФИЦИЕНТА
def main():
    tests = get_tests()
    coeffs = [
        ('k = const (1)', coeff_const),
        ('k = k(t,x) = 1 + 0.5*sin(t)*cos(πx)', coeff_tx),
        ('k = k(t,x,u) = 1 + 1/(1+u²)', coeff_txu),
        ('k = k(u) = 1 + u^{1/3}', coeff_new)
    ]
    
    # Уменьшенные сетки, чтобы программа работала быстро
    N_list = [20, 40, 80]
    # K_list_imp = [40, 80, 160]  
    K_list_imp = [1600, 6400, 25600]        
    K_list_rk4_std = [400, 1600, 6400]
    
    while True:
        print("\n" + "="*60)
        print("ЛАБОРАТОРНАЯ РАБОТА №2")
        print("НЕЛИНЕЙНОЕ УРАВНЕНИЕ ТЕПЛОПРОВОДНОСТИ")
        print("="*60)
        print("Выберите тестовую функцию:")
        for i, t in enumerate(tests, 1):
            print(f"  {i}. {t.name}")
        print("  0. Выход")
        try:
            test_choice = int(input("\nВведите номер теста: "))
            if test_choice == 0:
                break
            if test_choice < 1 or test_choice > len(tests):
                print("Неверный номер. Попробуйте снова.")
                continue
            test = tests[test_choice-1]
        except:
            print("Введите число.")
            continue
        
        print("\nВыберите тип коэффициента:")
        for i, (name, _) in enumerate(coeffs, 1):
            print(f"  {i}. {name}")
        try:
            coeff_choice = int(input("\nВведите номер коэффициента: "))
            if coeff_choice < 1 or coeff_choice > len(coeffs):
                print("Неверный номер.")
                continue
            coeff_name, a_func = coeffs[coeff_choice-1]
        except:
            print("Введите число.")
            continue
        
        # Параметры по умолчанию
        print("\nПараметры расчёта (можно оставить по умолчанию):")
        print(f"  N_list = {N_list}")
        print(f"  K_list_imp = {K_list_imp}")
        # Для новых нелинейных коэффициентов (3 и 4) используем более мелкий шаг для РК4
        if coeff_choice == 3 or coeff_choice == 4:
            print(f"  K_list_rk4 = {K_list_rk4_std} (увеличено для нелинейного коэффициента)")
            print("  Внимание: для нелинейного коэффициента может потребоваться более мелкий шаг.")
        else:
            print(f"  K_list_rk4 = {K_list_rk4_std}")
        
        use_default = input("\nИспользовать параметры по умолчанию? (y/n): ").lower()
        if use_default == 'y' or use_default == '':
            N_list_curr = N_list
            K_list_imp_curr = K_list_imp
            K_list_rk4_curr = K_list_rk4_std
            K_imp_comp = 80
            if coeff_choice == 3 or coeff_choice == 4:
                K_rk4_comp = 6400
            else:
                K_rk4_comp = 1600

            

            
            if coeff_choice == 4:
                K_list_rk4_curr = [1600, 6400, 25600]   # τ/h² = 0.125
            else:
                K_list_rk4_curr = [400, 1600, 6400]     # τ/h² = 0.5

            
        else:
            try:
                N_list_curr = list(map(int, input("Введите N_list через пробел: ").split()))
                K_list_imp_curr = list(map(int, input("Введите K_list_imp через пробел: ").split()))
                K_list_rk4_curr = list(map(int, input("Введите K_list_rk4 через пробел: ").split()))
                K_imp_comp = int(input("Введите K_imp для сравнения (N=40): "))
                K_rk4_comp = int(input("Введите K_rk4 для сравнения (N=40): "))
            except:
                print("Ошибка ввода. Использую параметры по умолчанию.")
                N_list_curr = N_list
                K_list_imp_curr = K_list_imp
                K_list_rk4_curr = K_list_rk4_std
                K_imp_comp = 80
                K_rk4_comp = 1600 if (coeff_choice != 3 and coeff_choice != 4) else 6400
        
        # Вывод результатов
        print(f"\n{'#'*80}")
        print(f"КОЭФФИЦИЕНТ: {coeff_name}")
        print(f"{'#'*80}")
        
        # Исследование сходимости неявной схемы
        convergence_study(test, L=1.0, T=0.5, N_list=N_list_curr, K_list=K_list_imp_curr, a_func=a_func, method='implicit')
        
        # Исследование сходимости метода РК4
        convergence_study(test, L=1.0, T=0.5, N_list=N_list_curr, K_list=K_list_rk4_curr, a_func=a_func, method='mol_rk4')
        
        # Сравнение при N=40
        comparison_table(test, L=1.0, T=0.5, N=40, K_imp=K_imp_comp, K_rk4=K_rk4_comp, a_func=a_func)
        
        # Визуализация
        show_plots = input("\nПоказать графики решения? (y/n): ").lower()
        if show_plots == 'y' or show_plots == '':
            plot_solution(test, 1.0, 0.5, a_func, N_plot=40, K_plot_imp=80, K_plot_mol=K_rk4_comp)
        
        input("\nНажмите Enter для продолжения...")


if __name__ == "__main__":
    main()