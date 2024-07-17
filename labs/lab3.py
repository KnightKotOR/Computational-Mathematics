import math

import numpy as np
from matplotlib import pyplot
from scipy import integrate
from prettytable import PrettyTable


# Выбор шага и получение точного решения
def pick_step(func, a, b, step):
    X = np.arange(a, b, step)
    Y = func(X)
    return (X, Y)


def rkf45(F, T, X0):
    r = (integrate.ode(F).set_integrator('dopri5').set_initial_value(X0, T[0]))
    X = np.zeros((len(T), len(X0)))
    X[0] = X0
    for i in range(1, len(T)):
        X[i] = r.integrate(T[i])
        if not r.successful():
            raise RuntimeError('Couldn\'t integrate')
    return X[:, 0]


def rk4(F, T, X0):
    X = np.zeros((len(T), len(X0)))
    X[0] = X0
    dt = T[1] - T[0]

    for i in range(0, len(T) - 1):
        f1 = F(T[i], X[i, :])
        f2 = F(T[i] + dt / 2.0, X[i, :] + dt * f1 / 2.0)
        f3 = F(T[i] + dt / 2.0, X[i, :] + dt * f2 / 2.0)
        f4 = F(T[i] + dt, X[i, :] + dt * f3)

        X[i + 1, :] = X[i, :] + dt * (f1 + 2.0 * f2 + 2.0 * f3 + f4) / 6.0
    return X[:, 0]


# Точное решение
def g(T):
    return math.e ** T - 1


def f(t, X):
    dX = np.zeros(X.shape)
    dX[0] = X[1]
    dX[1] = (2 * dX[0] + math.e ** t * X[0]) / (math.e ** t + 1)
    return dX

def solve(h=0.1):
    # Начальные условия
    a = 0
    b = 1
    X0 = np.array([0, 1])

    # Решение
    T, X_exact = pick_step(g, a, b + h, h)
    X_rkf45 = rkf45(f, T, X0)
    X_rk4 = rk4(f, T, X0)

    # Построение графиков
    pyplot.title('Green - exact; Red - RKF45; Blue - RK4')
    pyplot.plot(T, X_exact, 'g-')
    pyplot.plot(T, X_rkf45, 'r--')
    pyplot.plot(T, X_rk4, 'b--')
    pyplot.show()

    # Нахождение погрешностей
    LE_RKF45 = np.abs(X_exact - X_rkf45)
    LE_RK4 = np.abs(X_exact - X_rk4)

    print('Погрешность первого шага RKF45:', LE_RKF45[1])
    print('Погрешность первого шага RK4:', LE_RK4[1])

    print('Глобальная погрешность RKF45:', LE_RKF45.sum())
    print('Глобальная погрешность RK4:', LE_RK4.sum())

    # Таблица для сравнения результатов
    table = PrettyTable()
    table.add_column("T", T)
    table.add_column("Exact", X_exact)
    table.add_column("RKF45", X_rkf45)
    table.add_column("RK4", X_rk4)
    print(table)

    return LE_RK4[1]


def localErr():
    H5 = [0, 0, 0]
    LE = [0, 0, 0]
    ratio = [0, 0, 0]

    for i in range(0, len(H) - 1):
        H5[i] = (H[0] / H[i + 1]) ** 5
        LE[i] = LocErrors[0] / LocErrors[i + 1]
        ratio[i] = H5[i] / LE[i]

    errtable = PrettyTable()
    errtable.add_column("h[0]/h[i+1]^5 / ", H5)
    errtable.add_column("LE_RK4[0]/LE_RK4[i+1]", LE)
    errtable.add_column("Ratio", ratio)
    print(errtable)


if __name__ == "__main__":
    H = [0.1, 0.05, 0.025, 0.0125]
    LocErrors = [0, 0, 0, 0]
    for i in range (0, len(H)):
        LocErrors[i] = solve(H[i])
    localErr()

