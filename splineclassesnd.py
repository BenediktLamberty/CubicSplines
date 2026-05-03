from splineclasses import *
from typing import Callable, Generic, Sequence, TypeVar
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from abc import ABC, abstractmethod
import time


class PointSetND():
    dim: int
    n: int
    a: float
    b: float
    t: np.ndarray
    h: np.ndarray
    node_matrix: np.ndarray  # [axis, point_index]
    point_sets: list[PointSet1D]

    def __init__(self, dim: int, partition: np.ndarray, values: np.ndarray):
        self.n = len(partition) - 1
        self.a = partition[0]
        self.b = partition[-1]
        self.t = partition
        self.h = np.zeros(self.n + 1)
        for j in range(self.n):
            self.h[j+1] = self.t[j+1] - self.t[j]
            if self.h[j+1] <= 0:
                raise ValueError("Die Partition ist nicht geordnet!")
        self.dim = dim
        self.node_matrix = values
        partition_list = list(partition)
        self.point_sets = [PointSet1D(partition_list, list(values[k, :])) for k in range(self.dim)]
        
    def extract(self) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, list[PointSet1D]]:
        return self.dim, self.n, self.t, self.h, self.node_matrix, self.point_sets
    

class CurveND(ABC):
    dim: int

    @abstractmethod
    def eval(self, t: float, deriv: int = 0) -> np.ndarray:
        pass

    @abstractmethod
    def eval_array(self, t: np.ndarray, deriv: int = 0) -> np.ndarray:
        pass
    

T = TypeVar("T", bound=InterpolatingCurve1D)
class InterpolatingCurveND(CurveND, Generic[T], ABC):
    nodes: PointSetND
    components: list[T]

    def __init__(self, dim: int, nodes: PointSetND):
        self.dim = dim
        self.nodes = nodes
        self.calculate()

    @abstractmethod
    def calculate(self):
        pass

    def eval(self, t: float, deriv: int = 0) -> np.ndarray:
        return np.array([self.components[k].eval(t, deriv=deriv) for k in range(self.dim)])
    
    def eval_array(self, t: np.ndarray, deriv: int = 0) -> np.ndarray:
        return np.array([self.components[k].eval_array(t, deriv=deriv) for k in range(self.dim)])
    
    def get_nodes(self) -> PointSetND:
        return self.nodes
    

T = TypeVar("T", bound=PolynomialSpline1D)
class PolynomialSplineND(Generic[T], InterpolatingCurveND[T], ABC):
    
    def eval(self, t: float, deriv: int = 0, section: int | None = None) -> np.ndarray:
        return np.array([self.components[k].eval(t, deriv=deriv, section=section) for k in range(self.dim)])
    
    def eval_array(self, t: np.ndarray, deriv: int = 0, section: int | None = None) -> np.ndarray:
        return np.array([self.components[k].eval_array(t, deriv=deriv, section=section) for k in range(self.dim)])
    
    def get_nodes(self) -> PointSetND:
        return self.nodes
    

class LinearSplineND(PolynomialSplineND[LinearSpline1D]):

    def calculate(self):
        self.components = []
        for k in range(self.dim):
            self.components.append(LinearSpline1D(self.nodes.point_sets[k]))


class QuadraticSplineND(PolynomialSplineND[QuadraticSpline1D]):
    slope_a: np.ndarray

    def __init__(self, dim: int, nodes: PointSetND, slope_a: np.ndarray):
        self.slope_a = slope_a
        super().__init__(dim, nodes)

    def calculate(self):
        self.components = []
        for k in range(self.dim):
            self.components.append(QuadraticSpline1D(self.nodes.point_sets[k], self.slope_a[k]))


T = TypeVar("T", bound=CubicSpline1D)
class CubicSplineND(Generic[T], PolynomialSplineND[T], ABC):
    pass


class NaturalCubicSplineND(CubicSplineND[NaturalCubicSpline1D]):

    def calculate(self):
        self.components = []
        for k in range(self.dim):
            self.components.append(NaturalCubicSpline1D(self.nodes.point_sets[k]))


class PeriodicCubicSplineND(CubicSplineND[PeriodicCubicSpline1D]):

    def __init__(self, dim: int, nodes: PointSetND):
        for k in range(dim):
            if abs(nodes.node_matrix[k, 0] - nodes.node_matrix[k, -1]) >= 1e-6:
                raise ValueError(f"Bei einer PeriodicCubicSpline müssen die die Endpunkte gleich sein!")
        super().__init__(dim, nodes)

    def calculate(self):
        self.components = []
        for k in range(self.dim):
            self.components.append(PeriodicCubicSpline1D(self.nodes.point_sets[k]))


class ClampedCubicSplineND(CubicSplineND[ClampedCubicSpline1D]):
    slope_a: np.ndarray
    slope_b: np.ndarray

    def __init__(self, dim: int, nodes: PointSetND, clamp: tuple[np.ndarray, np.ndarray]):
        self.slope_a = clamp[0]
        self.slope_b = clamp[1]
        super().__init__(dim, nodes)

    def calculate(self):
        self.components = []
        for k in range(self.dim):
            self.components.append(ClampedCubicSpline1D(self.nodes.point_sets[k], (self.slope_a[k], self.slope_b[k])))


class CatmullRomSplineND(PolynomialSplineND[CatmullRomSpline1D]):
    periodic: bool

    def __init__(self, dim: int, nodes: PointSetND, periodic: bool = False):
        self.periodic = periodic
        super().__init__(dim, nodes)

    def calculate(self):
        self.components = []
        for k in range(self.dim):
            self.components.append(CatmullRomSpline1D(self.nodes.point_sets[k], periodic=self.periodic))


class ClampedBSplineND(CurveND):
    n: int  # t[-k] = ... = t[0] < ... < t[n] = ... = t[n+k]
    k: int  # degree
    t: dict[int, float]  # knot vector t[-k], ..., t[n+k]
    w: dict[int, np.ndarray]  # weights     w[-k], ..., w[n-1]

    def __init__(self, dim: int, n: int, k: int, t: dict[int, float], w: dict[int, np.ndarray]):
        self.dim = dim
        self.n = n
        self.k = k
        self.t = t
        self.w = w

    def b_spline(self, i: int, r: int, x: float) -> float:
        n, t = self.n, self.t

        if i >= n: return 0
        if r == 1:
            if i == n-1: return 1 if t[i] <= x <= t[i+1] else 0
            else: return 1 if t[i] <= x < t[i+1] else 0
        
        if t[i+r-1] - t[i] == 0: left_coeff = 0
        else: left_coeff = (x - t[i]) / (t[i+r-1] - t[i])

        if t[i+r] - t[i+1] == 0: right_coeff = 0
        else: right_coeff = (t[i+r] - x) / (t[i+r] - t[i+1])

        return left_coeff * self.b_spline(i, r-1, x) + right_coeff * self.b_spline(i+1, r-1, x)
    
    def eval_velocity(self, x: float, h: float = 1e-5) -> np.ndarray:
        n, t = self.n, self.t
        if t[0]+h <= x <= t[n]-h:
            return (self.eval(x+h) - self.eval(x-h)) / (2 * h)
        elif x <= (t[0]+t[n])/2: return self.eval_velocity(t[0]+h)
        elif x >  (t[0]+t[n])/2: return self.eval_velocity(t[n]-h)
        else: raise AssertionError()
    
    def eval_acceleration(self, x: float, h: float = 1e-5) -> np.ndarray:
        n, t = self.n, self.t
        if t[0]+h <= x <= t[n]-h:
            return (self.eval(x-h) - 2 * self.eval(x) + self.eval(x+h)) / h**2
        elif x <= (t[0]+t[n])/2: return self.eval_acceleration(t[0]+h)
        elif x >  (t[0]+t[n])/2: return self.eval_acceleration(t[n]-h)
        else: raise AssertionError()

    def eval(self, t: float, deriv: int = 0) -> np.ndarray:
        if deriv == 0: 
            return np.sum([self.w[i] * self.b_spline(i, self.k+1, t) for i in range(-self.k, self.n)], axis=0)
        elif deriv == 1:
            return self.eval_velocity(t)
        elif deriv == 2:
            return self.eval_acceleration(t)
        else: raise ValueError("Nur deriv = 0, 1, 2 unterstützt")

    def eval_array(self, t: np.ndarray, deriv: int = 0) -> np.ndarray:
        return np.transpose(np.array([self.eval(x, deriv) for x in t]))
    

class PeriodicBSplineND(CurveND):
    n: int  # Periodizität
    k: int  # degree
    period: float
    t: Callable[[int], float]
    w: Callable[[int], np.ndarray]

    def __init__(self, dim: int, n: int, k: int, t: np.ndarray, w: np.ndarray):
        # t[0, ..., n] muss aufsteigend sein
        # w[0, ..., n] oder # w[0, ..., n-1] muss periodisch sein.
        self.dim = dim
        self.n = n
        self.k = k
        self.period = t[n] - t[0]
        self.t = lambda i: (i // n) * self.period + t[i % n]
        self.w = lambda i: w[i % n]

    def b_spline(self, i: int, r: int, t: Callable[[int], float], x: float) -> float:
        if r == 1: return 1 if t(i) <= x < t(i+1) else 0

        left_coeff  = (x  -  t(i)) / (t(i+r-1) - t(i))
        right_coeff = (t(i+r) - x) / (t(i+r) - t(i+1))

        return left_coeff * self.b_spline(i, r-1, t, x) + right_coeff * self.b_spline(i+1, r-1, t, x)
    
    def t_inverse(self, t: float) -> int:
        m = int((t - self.t(0)) / self.period) - 1
        for i in range(m*self.n, (m+2)*self.n):
            if self.t(i) <= t < self.t(i+1): return i
        raise AssertionError(f"t_inverse failed at t={t}")

    def eval(self, t: float, deriv: int = 0) -> np.ndarray:
        if deriv == 0:
            j = self.t_inverse(t) 
            return np.sum([self.w(i) * self.b_spline(i, self.k+1, self.t, t) for i in range(j-self.k, j+1)], axis=0)
        elif deriv == 1:
            return self.eval_velocity(t)
        elif deriv == 2:
            return self.eval_acceleration(t)
        else: raise ValueError("Nur deriv = 0, 1, 2 unterstützt")

    def eval_acceleration(self, x: float, h: float = 1e-5) -> np.ndarray:
        return (self.eval(x-h) - 2 * self.eval(x) + self.eval(x+h)) / h**2
    
    def eval_velocity(self, x: float, h: float = 1e-5) -> np.ndarray:
        return (self.eval(x+h) - self.eval(x-h)) / (2 * h)

    def eval_array(self, t: np.ndarray, deriv: int = 0) -> np.ndarray:
        return np.transpose(np.array([self.eval(x, deriv) for x in t]))



def test():
    n = 4
    k = 3
    t = np.array([0, 0.5, 0.7, 0.9, 1])
    w = np.array([1, 1, 1, 1, 1])

    spline = PeriodicBSplineND(2, n, k, t, w)

    import matplotlib.pyplot as plt

    t_domain = np.linspace(-2, 2, 1000)
    y_domain = spline.eval_array(t_domain, 0)

    plt.plot(t_domain, y_domain)
    plt.show()

if __name__ == '__main__':
    test()


