import numpy as np
from scipy.special import comb
from scipy.optimize import bisect
import matplotlib.pyplot as plt

class BezierCurve2D:
    def __init__(self, control_points: list[tuple[float, float]]):
        self.control_points = control_points
        self.degree = len(control_points) - 1
        self.curve_points = None
    
    def evaluate_at_t(self, t: np.ndarray) -> np.ndarray:
        t_arr = np.array(t) if is_array_like(t) else np.array([t]) 
        n = self.degree
        expression_value = np.zeros((len(t_arr), 2))
        for i in range(n + 1):
            expression_value += np.multiply(bernstein_polynomial(n, i, t_arr).reshape(-1, 1), np.array(self.control_points[i]))
        return expression_value if is_array_like(t) else expression_value[0] 

    def evaluate_at_x(self, x: np.ndarray) -> np.ndarray:
        if self.evaluate_at_t(0)[0] > x:
            t = 0
        elif self.evaluate_at_t(1)[0] < x:
            t = 1
        
        t = bisect(lambda t: self.evaluate_at_t(t)[0] - x, 0, 1)
        return self.evaluate_at_t(t)

def is_array_like(variable):
    return isinstance(variable, np.ndarray) or isinstance(variable, list)

def normalize_to_array(variable):
    if isinstance(variable, np.ndarray):
        return variable

    return np.array(variable) if isinstance(variable, list) else np.array([variable])

def bernstein_polynomial(n, i, t):
    return comb(n, i) * np.multiply(np.power(t, i), np.power((1 - t),(n - i)))

if __name__ == '__main__':
    # Example
    control_points = [(0, 0), (0.75, -1), (0.5, 1), (1, 0)]
    bezier_curve = BezierCurve2D(control_points)
    t = np.linspace(0, 1, 100)
    points = bezier_curve.evaluate_at_t(t)

    x = 0.8
    point_at_x = bezier_curve.evaluate_at_x(x)

    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1])
    ax.scatter([control_point[0] for control_point in control_points], [control_point[1] for control_point in control_points], ec='r', fc='none')
    ax.plot([control_point[0] for control_point in control_points], [control_point[1] for control_point in control_points], c='r', linestyle='--', alpha=0.5)
    ax.scatter(point_at_x[0], point_at_x[1], ec='g', fc='none')
    plt.show()