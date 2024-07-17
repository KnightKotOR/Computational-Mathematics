import numpy as np
from numpy import cos, sqrt

from scipy import integrate
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.linalg import solve

from prettytable import PrettyTable
from matplotlib import pyplot


# Нахождение и печать параметров электрической цепи

def L_func(x):
    return cos(x) / x


def E1_fun(x):
    return x - 0.6 ** x


def findParams():
    global R, R2, E1, E2, L
    A = np.array([[53, 46, 20], [46, 50, 26], [20, 26, 17]])
    B = np.array([3060, 2866, 1337])
    Res = solve(A, B)

    R = round(Res[0], 5)
    R2 = round(Res[1], 5)
    E2 = round(Res[2], 5)

    L_res = quad(lambda x: L_func(x), 1, 2)
    L = round(L_res[0] * 0.4674158, 5)

    E1 = brentq(E1_fun, -1000, 1000, (), 0.00000001)
    E1 = round(E1 * 5.718088, 5)

    print('R = R1 = R3: ', R)
    print('R2: ', R2)
    print('E1: ', E1)
    print('E2: ', E2)
    print('L = L1 = L3: ', L)


# Решение СДУ на основе метода RKF45
def rkf45(F, T, X0):
    r = (integrate.ode(F).set_integrator('dopri5').set_initial_value(X0, T[0]))
    X = np.zeros((len(T), len(X0)))
    X[0] = X0
    for i in range(1, len(T)):
        X[i] = r.integrate(T[i])
        if not r.successful():
            raise RuntimeError('Couldn\'t integrate')
    return X[:, 2]


# Система дифференциальных уравнений
def fun(t, X):
    dX = np.zeros(X.shape)
    dX[0] = (E1 - E2 - X[2] + (X[1] * R2) - (X[0] * (R + R2))) / L
    dX[1] = (E2 + X[2] + (X[0] * R2) - (X[1] * (R + R2))) / L
    dX[2] = (X[0] - X[1]) / C
    return dX


# Получение погрешности для заданного значения C
def getDelta(x, print_flag):
    global C
    T = np.array([0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001])
    U_exp = np.array([-1.0, 7.777, 12.017, 10.701, 5.407, -0.843, -5.159, -6.015, -3.668, 0.283, 3.829])
    D = np.zeros(11)
    X0 = [0.1, 0, -1]
    C = x
    X_rkf = rkf45(fun, T, X0)

    for i in range(0, 11):
        D[i] = (U_exp[i] - X_rkf[i]) ** 2

    if print_flag:
        table = PrettyTable()
        table.add_column("T", T)
        table.add_column("U_exp", U_exp)
        table.add_column("U_RKF45", X_rkf)
        table.add_column("Diff", D)
        print(table)

        T_plot = np.arange(0, 0.00101, 0.00001)
        X_plot = rkf45(fun, T_plot, X0)

        pyplot.title('Green - exp; Red - RKF45')
        pyplot.plot(T_plot, X_plot, 'r--')
        pyplot.plot(T, U_exp, 'g-')
        pyplot.show()

        print('D: ', sqrt(sum(D) / 11))

    return sqrt(sum(D) / 11)


# Минимизация и получение результата
def getC():
    Cs = np.arange(5e-7, 2e-6 + 1e-7, 1e-7)
    Diff = []
    for c in Cs:
        Diff.append(getDelta(c, False))
    C = Cs[Diff.index(min(Diff))]
    getDelta(C, True)
    print('C: ', C)


if __name__ == "__main__":
    findParams()
    getC()
