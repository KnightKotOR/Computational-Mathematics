import math

from prettytable import PrettyTable
from scipy.interpolate import CubicSpline
from scipy.optimize import bisect
from scipy.optimize import brentq


def print_table():
    table = PrettyTable()
    table.add_column("e", error)
    table.add_column("bisection", res_bisect)
    table.add_column("brentq", res_brentq)
    table.add_column("bisection iterations", itr)
    print(table)

# Исследуемая функция
def func(x):
    return spline(x) + 5 * x - 3


if __name__ == '__main__':

    # Диапазон
    a = 0
    b = 2

    # Таблично заданная функция
    x = [0.0, 0.2, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0]
    f = [1.0, 1.1487, 1.4142, 1.6245, 2.0, 2.4623, 3.249, 4.0]

    # Исследуемые погрешности
    error = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

    # Объявление массивов для результатов и кол-ва итераций
    l = len(error)
    res_bisect = [0] * l
    res_brentq = [0] * l
    itr = [0] * l

    # Построение сплайна
    spline = CubicSpline(x, f)

    # Поиск корня при заданной погрешности error[i]
    for j in range(0, l):
        e = error[j]
        res_bisect[j] = bisect(func, a, b, (), e)
        res_brentq[j] = brentq(func, a, b, (), e)
        itr[j] = math.ceil(math.log2(1/e)+1)

    # Построение графика и таблицы
print_table()
