from scipy.integrate import odeint
from scipy.signal import tf2ss
import numpy as np
import sympy
import scipy
import math
import matplotlib.pyplot as plt
import control as ct
import pygame


# Параметры системы
l = 0.25  # длина маятника (м)
m = 0.2  # масса маятника (кг)
M = 0.4  # масса каретки (кг)
g = 9.8  # ускорение свободного падения (м/с²)
f = 0  # сила воздействия (Н)


# Функция, определяющая систему диф уравнений для перевернутого маятника
def ode(y, t):
    theta, dtheta, x, dx = y

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    # Общий знаменатель
    D = l * (M + m) - m * l * math.pow(cos_theta, 2)

    # Уравнения
    d2theta = (
        (M + m) * g * sin_theta
        - m * l * dtheta**2 * sin_theta * cos_theta
        - f * cos_theta
    ) / D
    d2x = (
        m * l**2 * dtheta**2 * sin_theta + l * f - m * g * l * sin_theta * cos_theta
    ) / D

    return [dtheta, d2theta, dx, d2x]


# Функция решения системы диф уравнений
def calcODE(theta0, dtheta0, x0=0, dx0=0, ts=10, nt=101):
    args = [theta0, dtheta0, x0, dx0]
    t = np.linspace(0, ts, nt)
    sol = odeint(ode, args, t)
    return sol


# Функция отрисовки фазового портрета
def drawPhasePortrait(
    deltaX=0.5,
    deltaDX=0.5,
    startX=-np.pi,
    stopX=np.pi,
    startDX=-5,
    stopDX=5,
    ts=10,
    nt=101,
    xlim=None,
    ylim=None,
):
    plt.figure(figsize=(10, 6))
    startX = np.radians(startX)
    stopX = np.radians(stopX)
    deltaX = np.radians(deltaX)
    for theta0 in np.arange(startX, stopX, deltaX):
        for dtheta0 in np.arange(startDX, stopDX, deltaDX):
            sol = calcODE(theta0, dtheta0, ts, nt)
            plt.plot(sol[:, 0], sol[:, 1], "b", alpha=0.5)

    plt.xlabel("x")
    plt.ylabel("dx/dе")
    plt.title("dx/dt")
    plt.grid()

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
        plt.show()


# Отрисовка фазового портрета
drawPhasePortrait(
    15,
    0.5,
    -180,
    180,
    -3,
    3,
    ts=5,
    nt=1000,
    xlim=[-3 * np.pi, 3 * np.pi],
    ylim=[-20, 20],
)


# Матрицы пространства состояний линеаризованной системы
A = np.array(
    [
        [0, 1, 0, 0],
        [(M + m) * g / (l * M), 0, 0, 0],
        [0, 0, 0, 1],
        [-m * g / M, 0, 0, 0],
    ]
)

B = np.array([[0], [-1 / (l * M)], [0], [1 / M]])
C = np.array([[0, 0, 1, 0]])
D = np.array([[0]])
C_theta = np.array([[1, 0, 0, 0]])

# Создание объекта системы
sys = ct.ss(A, B, C, D)

# Проверка при theta
Q = ct.obsv(A, C_theta)
W = ct.ctrb(A, B)
print("Матрица наблюдаемости\n", Q)
print("Матрица управляемости\n", W)
print("Ранг матрицы Q: ", np.linalg.matrix_rank(Q))
print("Ранг матрицы W: ", np.linalg.matrix_rank(W))

# Проверка при x
Q = ct.obsv(A, C)
W = ct.ctrb(A, B)
print("Матрица наблюдаемости\n", Q)
print("Матрица управляемости\n", W)
print("Ранг матрицы Q: ", np.linalg.matrix_rank(Q))
print("Ранг матрицы W: ", np.linalg.matrix_rank(W))

# Передаточная функция
W = ct.ss2tf(sys)
print("W(s):", W)

num = np.poly1d(W.num[0][0])
den = np.poly1d(W.den[0][0])
print("Числитель передаточной функции\n", num)
print("Знаменатель передаточной функции\n", den)

# Построение карты полюсов и нулей
poles = ct.poles(sys)
zeros = ct.zeros(sys)

pzmap_object = ct.pzmap(sys, title="Полюса и Нули")
plt.show()

# Компьютерное моделирование
stats = calcODE(1, 0, 0, 0)
plt.figure(figsize=(8, 6))
plt.plot(stats[:, 0], label="Угол")
plt.plot(stats[:, 2], label="Каретка")
plt.xlim(0, 30)
plt.legend(loc="upper right")
plt.title("Компьютерное моделирование")
plt.show()

# Синтез
s = sympy.symbols("s")
W_o = 1 / M * (s**2 - g / l) / (s**2 * (s**2 - (M + m) / (l * M) * g))
a_1, a_2, a_0 = sympy.symbols("a_1, a_2, a_0")
b_1, b_2, b_3, b_0 = sympy.symbols("b_1, b_2, b_3, b_0")
W_p = (b_3 * s**3 + b_2 * s**2 + b_1 * s + b_0) / (s**3 + a_2 * s**2 + a_1 * s + a_0)
print("Регулятор ", W_p)

B_o = 1 / M * (s**2 - g / l)
A_o = s**2 * (s**2 - (M + m) / (l * M) * g)
W_o = B_o / A_o
print(W_o)
B_p = b_3 * s**3 + b_2 * s**2 + b_1 * s + b_0
A_p = s**3 + a_2 * s**2 + a_1 * s + a_0
W_p = B_p / A_p
print(W_p)
B_s = B_o * B_p
A_s = A_o * A_p + B_o * B_p
W_s = sympy.together(B_s / A_s)
print("Передаточная функция с обратной связью ", W_s)

eq = sympy.Eq(
    A_s, (s + 1) * (s + 2) * (s + 4) * (s + 7) * (s + 10) * (s + 20) * (s + 35)
)
solv = sympy.solve(eq, [a_1, a_2, a_0, b_1, b_2, b_3, b_0])
print(solv)

# Моделирование
w_r_num = np.array(sympy.Poly(B_p.subs(solv), s).all_coeffs()).astype(np.float32)
w_r_den = np.array(sympy.Poly(A_p.subs(solv), s).all_coeffs()).astype(np.float32)
A_, B_, C_, D_ = tf2ss(w_r_num, w_r_den)
reg = ct.ss(A_, B_, C_, D_)
n_regs = A_.shape[0]
sys_reg = ct.feedback(sys, reg)


# Алгоритм для стабилизации маятника
def control(y, t):
    theta, dtheta, x, dx = y[:4]
    reg_state = y[4 : 4 + n_regs]
    # Ошибка
    e = 0 - x
    v = float(C_ @ reg_state + D_ * e)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    D = (M + m) * l - m * l * cos_theta**2
    d2theta = (
        (M + m) * g * sin_theta
        - m * l * dtheta**2 * sin_theta * cos_theta
        - v * cos_theta
    ) / D
    d2x = (
        m * l**2 * dtheta**2 * sin_theta + l * v - m * g * l * sin_theta * cos_theta
    ) / D
    dreg_state = A_ @ reg_state + B_.flatten() * e
    return np.concatenate([np.array([dtheta, d2theta, dx, d2x]), dreg_state])


# Построение графиков
def simulate(theta0=1.5, dtheta0=0, x0=0, dx0=0, t_max=100, dt=0.1):
    y0 = np.concatenate([np.array([theta0, dtheta0, x0, dx0]), np.zeros(n_regs)])
    t = np.arange(0, t_max, dt)
    solution = odeint(control, y0, t)
    plt.plot(t, solution[:, 0], label="Угол")
    plt.plot(t, solution[:, 2], label="Положение каретки")
    plt.xlim(0, t_max)
    plt.xlabel("Время")
    plt.ylabel("Состояние")
    plt.title("Моделирование системы с регулятором")
    plt.legend()
    plt.grid(True)
    plt.show()
    return t, solution


t, results = simulate(theta0=0.08, t_max=100)
for th0 in [0.02, 0.05, 0.5]:
    t, sol = simulate(theta0=th0, t_max=10, dt=0.01)

# Синтез регулятора
p = np.array([-1, -2, -4, -7])
W = ct.ctrb(A, B)
print("Ранг матрицы управляемости ", np.linalg.matrix_rank(W))
K = ct.place(A, B, p)
print("Матрица обратной связи ", K)
A_closed = A - B @ K
print("Матрица замкнутой системы\n", A_closed)
p_closed = np.linalg.eigvals(A_closed)
print("Полюса замкнутой системы ", p_closed)

# Синтез наблюдателя
p_L = 5 * p
Q = ct.obsv(A, C)
print("Ранг матрицы наблюдаемости:", np.linalg.matrix_rank(Q))
L = ct.place(A.T, C.T, p_L).T
print("Матрица наблюдателя\n", L)


# Динамический регулятор
def dynamic_reg(A, B, C, K, L):
    Ar = A - B @ K - L @ C
    Br = L
    Cr = -K
    Dr = np.zeros((K.shape[0], C.shape[0]))
    return Ar, Br, Cr, Dr


Ar, Br, Cr, Dr = dynamic_reg(A, B, C, K, L)
regulator = ct.ss(Ar, Br, Cr, Dr)


print("Матрицы динамического регулятора:")
print("Ar =\n", Ar)
print("Br =\n", Br)
print("Cr =\n", Cr)
print("Dr =\n", Dr)

regulator_poles = np.linalg.eigvals(Ar)
print("Собственные значения матрицы регулятора Ar\n", regulator_poles)
plt.figure(figsize=(8, 6))
plt.scatter(
    np.real(regulator_poles), np.imag(regulator_poles), marker="x", color="b", s=100
)
plt.axhline(0, color="k", linestyle="-", linewidth=0.5)
plt.axvline(0, color="k", linestyle="-", linewidth=0.5)
plt.title("Карта полюсов динамического регулятора")
plt.xlabel("Re")
plt.ylabel("Im")
plt.grid(True)
plt.show()


regulator_tf = ct.ss2tf(regulator)
print("Передаточная функция регулятора", regulator_tf)

# Анализ системы
closed_sys = ct.feedback(ct.ss(A, B, C, D), regulator_tf, sign=1)
poles = ct.poles(closed_sys)
print(poles)

pzmap_object = ct.pzmap(closed_sys, title="Полюса и Нули")
plt.show()


def closed_system(y, t):
    theta, dtheta, x, dx = y[:4]
    theta_est, dtheta_est, x_est, dx_est = y[4:8]
    y_meas = x
    f = -K @ np.array([theta_est, dtheta_est, x_est, dx_est])
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    D = (M + m) * l - m * l * cos_theta**2
    d2theta = (
        (M + m) * g * sin_theta
        - m * l * dtheta**2 * sin_theta * cos_theta
        - f * cos_theta
    ) / D
    d2x = (
        m * l**2 * dtheta**2 * sin_theta + l * f - m * g * l * sin_theta * cos_theta
    ) / D
    dz = (
        A @ np.array([theta_est, dtheta_est, x_est, dx_est])
        + B.flatten() * f
        + L.flatten() * (y_meas - x_est)
    )

    return [dtheta, d2theta[0], dx, d2x[0], dz[0], dz[1], dz[2], dz[3]]


def draw_graph(theta0=0.02, ts=10, nt=0.01):
    x0 = np.array([theta0, 0, 0, 0])
    z0 = np.zeros(4)
    y0 = np.concatenate([x0, z0])
    t = np.arange(0, ts, nt)
    sol = odeint(closed_system, y0, t)
    plt.figure(figsize=(8, 6))
    plt.plot(t, sol[:, 0], label="Угол")
    plt.plot(t, sol[:, 2], label="Положение каретки")
    plt.xlim(0, 10)
    plt.xlabel("Время")
    plt.ylabel("Состояние")
    plt.title("Система с регулятором")
    plt.legend()
    plt.show()
    return t, sol


initial_angles = [0.02, 0.05, 0.3]
results = []

for th0 in initial_angles:
    t, sol = draw_graph(theta0=th0, ts=10)
    results.append((t, sol))


for i, (t, sol) in enumerate(results):
    plt.plot(t, sol[:, 0], label=f'Начальный угол: {np.degrees(initial_angles[i]):.1f}°')


plt.title('Сравнение углов при разных начальных отклонениях')
plt.xlabel('Время, с')
plt.ylabel('Угол, рад')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

pygame.init()

# Параметры анимации
WIDTH, HEIGHT = 1000, 600
SCALE = 200 
CART_WIDTH, CART_HEIGHT = 60, 30
PENDULUM_LENGTH = l * SCALE
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Перевернутый маятник на каретке")
clock = pygame.time.Clock()

# Начальные условия
initial_state = [0.1, 0, 0, 0]  # небольшой начальный угол
t = np.linspace(0, 10, 1000)  # время моделирования

# Решение системы
solution = odeint(ode, initial_state, t)

# Основной цикл анимации
current_frame = 0
paused = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_r:
                current_frame = 0  # сброс анимации

    if not paused and current_frame < len(solution) - 1:
        current_frame += 1

    screen.fill(WHITE)

    # Получение текущего состояния
    theta, _, x, _ = solution[current_frame]

    # Расчет координат
    cart_x = WIDTH // 2 + x * SCALE
    cart_y = HEIGHT // 2

    pendulum_x = cart_x + PENDULUM_LENGTH * np.sin(theta)
    pendulum_y = cart_y - PENDULUM_LENGTH * np.cos(theta)

    # Отрисовка каретки
    pygame.draw.rect(
        screen,
        BLUE,
        (cart_x - CART_WIDTH // 2, cart_y - CART_HEIGHT // 2, CART_WIDTH, CART_HEIGHT),
    )

    # Отрисовка маятника
    pygame.draw.line(screen, BLACK, (cart_x, cart_y), (pendulum_x, pendulum_y), 3)
    pygame.draw.circle(screen, RED, (int(pendulum_x), int(pendulum_y)), 10)

    # Отрисовка текста
    font = pygame.font.SysFont("Arial", 20)
    time_text = font.render(f"Время: {t[current_frame]:.2f} с", True, BLACK)
    angle_text = font.render(f"Угол: {np.degrees(theta):.1f}°", True, BLACK)
    pos_text = font.render(f"Позиция: {x:.2f} м", True, BLACK)

    screen.blit(time_text, (10, 10))
    screen.blit(angle_text, (10, 40))
    screen.blit(pos_text, (10, 70))

    # Управляющие подсказки
    controls = font.render("Пробел: пауза, R: сброс", True, BLACK)
    screen.blit(controls, (WIDTH - 250, 10))

    pygame.display.flip()
    clock.tick(60)  # 60 фпс

pygame.quit()
