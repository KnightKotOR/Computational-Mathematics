from math import inf
import numpy as np
import scipy.linalg as linalg


# Создание исследуемой матрицы
def create_A(p):
    A = [[p + 13, 2, 8, -7, 7, 5, -7, -7],
         [7, 2, -4, 2, 3, 3, -1, -2],
         [-7, 2, 1, 3, 6, -6, -3, -4],
         [-2, -8, -6, -1, 6, 2, 1, -4],
         [0, 4, -7, 1, 22, 0, -6, -6],
         [0, -3, -6, 6, 4, 13, 0, 6],
         [-8, -6, -4, 7, -5, -5, -2, 1],
         [5, 5, -2, -2, -3, 0, -7, 14]]
    return A


# Создание единичной матрицы
def create_E(N):
    E = np.zeros((N, N))
    for i in range(N):
        E[i][i] = 1
    return E


# Вычисление нормы
def norm(r):
    return linalg.norm(r, ord=inf)


# Вычисление числа обусловленности
def cond(a):
    return norm(a) * norm(linalg.inv(a))


# Построение матрицы, обратной A, при помощи LU-разложения
def invert(a, N):
    P, L, U = linalg.lu(a)
    inv = np.zeros((N, N))
    for i in range(N):
        e = np.zeros((N, 1))
        e[i] = 1
        x = linalg.solve(L, e)
        y = linalg.solve(U, x)
        for j in range(N):
            inv[j][i] = y[j][0]
    return inv

