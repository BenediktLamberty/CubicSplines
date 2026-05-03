from typing import Callable
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from abc import ABC, abstractmethod
import time


def sort_points_by_x(unsorted_x: list[float], unsorted_y: list[float]) -> tuple[list[float], list[float]]:
    sorted_x, sorted_y = zip(*sorted(zip(unsorted_x, unsorted_y)))
    return list(sorted_x), list(sorted_y)

def remove_unsorted_points_by_x(dirty_x: list[float], dirty_y: list[float]) -> tuple[list[float], list[float]]:
    clean_x: list[float] = []
    clean_y: list[float] = []
    for x, y in zip(dirty_x, dirty_y):
        if len(clean_x) == 0 or x > clean_x[-1]:
            clean_x.append(x)
            clean_y.append(y)
    return clean_x, clean_y

def sort_unequal_pair(a: float, b: float, min_dist: float = 0.1) -> tuple[float, float]:
    if a > b:
        a, b = b, a
    if b - a < min_dist:
        mid = (a + b) * 0.5
        a = mid - min_dist * 0.5
        b = mid + min_dist * 0.5
    return a, b



class PointSet1D():
    n: int
    a: float
    b: float
    x: list[float]
    h: list[float]
    y: list[float]
    
    def __init__(self, partition: list[float], values: list[float] | Callable[[float], float]):
        self.n = len(partition) - 1
        self.a = partition[0]
        self.b = partition[-1]
        self.x = partition.copy()
        self.h = [0] * (self.n + 1)
        for j in range(self.n):
            self.h[j+1] = self.x[j+1] - self.x[j]
            if self.h[j+1] <= 0:
                raise ValueError("Die Partition ist nicht geordnet!")
        if isinstance(values, list):
            self.y = values.copy()
        else:
            self.y = [values(t) for t in self.x]

    def extract(self) -> tuple[int, list[float], list[float], list[float]]:
        return (self.n, self.x, self.h, self.y)


class InterpolatingCurve1D(ABC):
    nodes: PointSet1D

    def __init__(self, nodes: PointSet1D):
        self.nodes = nodes
        self.calculate()

    @abstractmethod
    def calculate(self):
        pass

    @abstractmethod
    def eval(self, t: float, deriv: int = 0) -> float:
        pass

    @abstractmethod
    def eval_array(self, t: np.ndarray, deriv: int = 0) -> np.ndarray:
        pass
    
    def get_nodes(self) -> PointSet1D:
        return self.nodes


class InterpolatingPolynomial1D(InterpolatingCurve1D):
    poly: Polynomial

    def calculate(self):
        n, x, _, y = self.nodes.extract()
        self.poly = Polynomial(np.polyfit(x, y, n)[::-1])

    def eval(self, t: float, deriv: int = 0) -> float:
        return self.poly.deriv(deriv)(t)
    
    def eval_array(self, t: np.ndarray, deriv: int = 0) -> np.ndarray:
        return self.poly.deriv(deriv)(t)


class PolynomialSpline1D(InterpolatingCurve1D, ABC):
    polys: list[Polynomial]

    def eval(self, t: float, deriv: int = 0, section: int | None = None) -> float:
        n, x, _, _ = self.nodes.extract()
        if section is None:
            for j in range(n):
                if x[j] <= t <= x[j+1]:
                    section = j
        if section is None:
            raise ValueError("t ist nicht im Intervall [a; b]")
        if not 0 <= section <= n-1:
            raise ValueError("Dieser Abschnitt existiert nicht!")
        return self.polys[section].deriv(deriv)(t - x[section])
    
    def eval_array(self, t: np.ndarray, deriv: int = 0, section: int | None = None) -> np.ndarray:
        n, x, _, _ = self.nodes.extract()
        if section is None:
            conds = [(x[j] <= t) & (t <= x[j+1]) for j in range(n)]
            functions: list[Callable[[float], float]] = [lambda z, j=j: self.polys[j].deriv(deriv)(z - x[j]) for j in range(n)]
            rv = np.piecewise(t, conds, functions)
            return rv
        else:
            return self.polys[section].deriv(deriv)(t - x[section])
        

class LinearSpline1D(PolynomialSpline1D):

    def calculate(self):
        n, _, h, y = self.nodes.extract()

        alpha = [y[j] for j in range(n)]
        beta  = [(y[j+1] - y[j]) / h[j+1] for j in range(n)]

        self.polys = [Polynomial([alpha[j], beta[j]]) for j in range(n)]


class QuadraticSpline1D(PolynomialSpline1D):
    slope_a: float

    def __init__(self, nodes: PointSet1D, slope_a: float):
        self.slope_a = slope_a
        super().__init__(nodes)

    def calculate(self):
        n, _, h, y = self.nodes.extract()
        left_slope = self.slope_a

        alpha: list[float] = [0] * n
        beta:  list[float] = [0] * n
        gamma: list[float] = [0] * n

        for j in range(n):
            alpha[j]   = y[j]
            beta[j]    = left_slope
            gamma[j]   = ((y[j+1] - y[j]) / h[j+1] - left_slope) / h[j+1]
            left_slope = beta[j] + 2 * gamma[j] * h[j+1]

        self.polys = [Polynomial([alpha[j], beta[j], gamma[j]]) for j in range(n)]


class CubicSpline1D(PolynomialSpline1D, ABC):

    @abstractmethod
    def calculate_moments(self) -> list[float]:
        pass
    
    def calculate(self):
        n, _, h, y = self.nodes.extract()
        m = self.calculate_moments()

        alpha = [y[j] for j in range(n)]
        beta  = [(y[j+1] - y[j]) / h[j+1] - (2 * m[j] + m[j+1]) * h[j+1] / 6 for j in range(n)]
        gamma = [m[j] / 2 for j in range(n)]
        delta = [(m[j+1] - m[j]) / (6 * h[j+1]) for j in range(n)]

        self.polys = [Polynomial([alpha[j], beta[j], gamma[j], delta[j]]) for j in range(n)]
        

class NaturalCubicSpline1D(CubicSpline1D):

    def calculate_moments(self) -> list[float]:
        n, _, h, y = self.nodes.extract()

        lam: list[float] = [0] * (n + 1)
        mu:  list[float] = [0] * (n + 1)
        d:   list[float] = [0] * (n + 1)

        for j in range(1, n):
            lam[j] = h[j+1] / (h[j] + h[j+1])
            mu[j]  = h[j]   / (h[j] + h[j+1])
            d[j]   = (6 / (h[j] + h[j+1])) * ((y[j+1] - y[j]) / h[j+1] - (y[j] - y[j-1]) / h[j])

        matrix = np.zeros(((n+1), (n+1)), dtype=np.float64)

        for i in range(n+1):
            matrix[i, i] = 2
            matrix[i, (i+1) % (n+1)] = lam[i]
            matrix[i, (i-1) % (n+1)] = mu[i]

        return list(np.linalg.solve(matrix, d))
    

class PeriodicCubicSpline1D(CubicSpline1D):

    def __init__(self, nodes: PointSet1D):
        if abs(nodes.y[0] - nodes.y[-1]) >= 1e-6:
            raise ValueError(f"Bei einer PeriodicCubicSpline muss y[0] = y[n] gelten! Es ist aber {nodes.y[0]} != {nodes.y[-1]}.")
        super().__init__(nodes)
    
    def calculate_moments(self) -> list[float]:
        n, _, h, y = self.nodes.extract()

        lam: list[float] = [0] * (n + 1)
        mu:  list[float] = [0] * (n + 1)
        d:   list[float] = [0] * (n + 1)

        lam[n] = h[1] / (h[n] + h[1])
        mu[n]  = h[n] / (h[n] + h[1])
        d[n]   = (6 / (h[n] + h[1])) * ((y[1] - y[n]) / h[1] - (y[n] - y[n-1]) / h[n])

        for j in range(1, n):
            lam[j] = h[j+1] / (h[j] + h[j+1])
            mu[j]  = h[j]   / (h[j] + h[j+1])
            d[j]   = (6 / (h[j] + h[j+1])) * ((y[j+1] - y[j]) / h[j+1] - (y[j] - y[j-1]) / h[j])

        matrix = np.zeros((n, n), dtype=float)

        for i in range(n):
            matrix[i, i] = 2
            matrix[i, (i+1) % n] = lam[i+1]
            matrix[i, (i-1) % n] = mu[i+1]

        moments = list(np.linalg.solve(matrix, d[1:]))
        return [moments[-1]] + moments
    

class ClampedCubicSpline1D(CubicSpline1D):
    slope_a: float
    slope_b: float
    
    def __init__(self, nodes: PointSet1D, clamp: tuple[float, float] | Callable[[float], float]):
        if isinstance(clamp, tuple):
            self.slope_a = clamp[0]
            self.slope_b = clamp[1]
        else: 
            h = 1e-6
            self.slope_a = (clamp(nodes.a + h) - clamp(nodes.a - h)) / (2 * h)
            self.slope_b = (clamp(nodes.b + h) - clamp(nodes.b - h)) / (2 * h)
        super().__init__(nodes)

    def calculate_moments(self) -> list[float]:
        n, _, h, y = self.nodes.extract()

        lam: list[float] = [0] * (n + 1)
        mu:  list[float] = [0] * (n + 1)
        d:   list[float] = [0] * (n + 1)

        lam[0] = 1
        mu[n]  = 1
        d[0]   = 6 * ((y[1] - y[0]) / h[1] - self.slope_a) / h[1]
        d[n]   = 6 * (self.slope_b - (y[n]-y[n-1]) / h[n]) / h[n]

        for j in range(1, n):
            lam[j] = h[j+1] / (h[j] + h[j+1])
            mu[j]  = h[j]   / (h[j] + h[j+1])
            d[j]   = (6 / (h[j] + h[j+1])) * ((y[j+1] - y[j]) / h[j+1] - (y[j] - y[j-1]) / h[j])

        matrix = np.zeros(((n+1), (n+1)), dtype=float)

        for i in range(n + 1):
            matrix[i, i] = 2
            matrix[i, (i+1) % (n+1)] = lam[i]
            matrix[i, (i-1) % (n+1)] = mu[i]

        return list(np.linalg.solve(matrix, d))
    

class CatmullRomSpline1D(PolynomialSpline1D):
    periodic: bool

    def __init__(self, nodes: PointSet1D, periodic: bool = False):
        self.periodic = periodic
        super().__init__(nodes)

    def calculate(self): 
        n, _, h, y = self.nodes.extract()

        alpha: list[float] = [0] * n
        beta:  list[float] = [0] * n
        gamma: list[float] = [0] * n
        delta: list[float] = [0] * n

        v: list[float] = [0] * (n + 1)
        v[0] = (y[1] - y[n-1]) / (h[n] + h[1]) if self.periodic else (y[1] - y[0]) / h[1]
        v[n] = (y[1] - y[n-1]) / (h[n] + h[1]) if self.periodic else (y[n]-y[n-1]) / h[n]
        for j in range(1, n):
            v[j] = (y[j+1] - y[j-1]) / (h[j] + h[j+1])

        for j in range(n):
            matrix = np.array([
                [1, 0,      0,         0          ],
                [1, h[j+1], h[j+1]**2, h[j+1]**3  ],
                [0, 1,      0,         0          ],
                [0, 1,      2*h[j+1],  3*h[j+1]**2]
            ])
            d = np.array([y[j], y[j+1], v[j], v[j+1]])
            solution = np.linalg.solve(matrix, d)
            alpha[j], beta[j], gamma[j], delta[j] = tuple(solution)

        self.polys = [Polynomial([alpha[j], beta[j], gamma[j], delta[j]]) for j in range(n)]

