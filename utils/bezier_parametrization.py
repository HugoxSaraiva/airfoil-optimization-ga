import matplotlib.pyplot as plt
import numpy as np
from bezier_curve import BezierCurve2D

if __name__ == '__main__':
    upper_control_points=[0,0,0.028830387858905057, 0.1411396509632579, 0.31520572542813763, 0.34363932259611313, 0.39109318267809434, 0.0424359669691409, 0.5709148615746626, 0.07410454920970207, 1, 0]
    lower_control_points=[0,0,0.019408711541432644, -0.06040774190578092, 0.4464414301191849, -0.09693677277639359, 0.47325435804708416, -0.11485557171942154, 0.363805246678832, 0.014626604247552747, 1, 0]
    upper_control_points = np.array(upper_control_points).reshape((-1, 2))
    lower_control_points = np.array(lower_control_points).reshape((-1, 2))

    upper_surface = BezierCurve2D(upper_control_points)
    lower_surface = BezierCurve2D(lower_control_points)

    t = np.linspace(0, 1, 100)
    upper_points = upper_surface.evaluate_at_t(t)
    lower_points = lower_surface.evaluate_at_t(t)

    fig, ax = plt.subplots()
    ax.plot(upper_points[:, 0], upper_points[:, 1], color='b', label='Upper surface')
    ax.plot(lower_points[:, 0], lower_points[:, 1], color='r', label='Lower surface')
    ax.scatter(upper_control_points[:, 0], upper_control_points[:, 1], ec='b', fc='none', label='Upper control points')
    ax.scatter(lower_control_points[:, 0], lower_control_points[:, 1], ec='r', fc='none', label='Lower control points')
    ax.plot(upper_control_points[:, 0], upper_control_points[:, 1], c='b', linestyle='--', alpha=0.5)
    ax.plot(lower_control_points[:, 0], lower_control_points[:, 1], c='r', linestyle='--', alpha=0.5)
    ax.set_title('Airfoil with Bézier parametrization')
    ax.legend()
    plt.show()
