import matplotlib.pyplot as plt
import numpy as np
import logging
from utils.bezier_curve import BezierCurve2D

class BezierAirfoil:
    def __init__(self, upper_points, lower_points, dz_te):
        self.upper_points = upper_points
        self.lower_points = lower_points
        self.dz_te = dz_te
        # Adding (0,0) to first upper control points to assure continuity at (0,0)
        # The last control point is (1, dz_te)
        upper_control_points_as_list = [0, 0] + upper_points + [1, dz_te]
        self.upper_control_points = np.array(upper_control_points_as_list).reshape((-1, 2))
        # Adding (0,0) to first lower control points to assure continuity at (0,0)
        # The last control point is (1, -dz_te)
        lower_control_points_as_list = [0, 0] + lower_points + [1, -dz_te]
        self.lower_control_points = np.array(lower_control_points_as_list).reshape((-1, 2))

        self.upper_surface = BezierCurve2D(self.upper_control_points)
        self.lower_surface = BezierCurve2D(self.lower_control_points)

        if not check_airfoil(self.upper_surface, self.lower_surface):
            raise Exception("Invalid airfoil surface")


    @staticmethod
    def random(sigma=0.07, mu=0.15, free_variables_shape=(4,4), try_count=1000):
        shape_condition, shape_condition_desc = lambda x: x % 2 == 0, 'even number'
        
        if(free_variables_shape[0] < 2 or free_variables_shape[1] < 2):
            raise Exception("Invalid control points shape")
        if(not shape_condition(free_variables_shape[0]) or not shape_condition(free_variables_shape[1])):
            raise Exception(f"Control points shape must satisfy shape_condition: {shape_condition_desc}")
        
        upper_points_x = np.sort(np.random.uniform(0.0, 1.0, free_variables_shape[0]))
        upper_points_y = sigma*np.random.randn(free_variables_shape[0])+mu

        lower_points_x = np.sort(np.random.uniform(0.0, 1.0, free_variables_shape[1]))
        lower_points_y = sigma*np.random.randn(free_variables_shape[1])-mu

        dz_te = np.random.uniform(0.0, 0.001)

        # Setting point to 0 to assure leading edge is smooth at (0,0)
        upper_points_x[0] = 0
        lower_points_x[0] = 0

        upper_surface_points = list(zip(upper_points_x, upper_points_y))
        lower_surface_points = list(zip(lower_points_x, lower_points_y))

        upper_points = np.array(upper_surface_points).reshape(-1).tolist()
        lower_points = np.array(lower_surface_points).reshape(-1).tolist()
        
        airfoil = None
        for i in range(try_count):
            try:
                airfoil = BezierAirfoil(upper_points, lower_points, dz_te)
            except Exception as e:
                logging.warning(f"Could not generate valid airfoil: {e}")
                continue
            break
        if airfoil is not None:
            return airfoil
        raise Exception("Could not generate valid airfoil")
    
    @staticmethod
    def get_random(sigma=0.07, mu=0.15, free_variables_shape=(4,4), try_count=1000):
        shape_condition, shape_condition_desc = lambda x: x % 2 == 0, 'even number'
        
        if(free_variables_shape[0] < 2 or free_variables_shape[1] < 2):
            raise Exception("Invalid control points shape")
        if(not shape_condition(free_variables_shape[0]) or not shape_condition(free_variables_shape[1])):
            raise Exception(f"Control points shape must satisfy shape_condition: {shape_condition_desc}")
        
        upper_points_x = np.sort(np.random.uniform(0.0, 1.0, free_variables_shape[0]))
        upper_points_y = sigma*np.random.randn(free_variables_shape[0])+mu

        lower_points_x = np.sort(np.random.uniform(0.0, 1.0, free_variables_shape[1]))
        lower_points_y = sigma*np.random.randn(free_variables_shape[1])-mu

        dz_te = np.random.uniform(0.0, 0.001)

        # Setting point to 0 to assure leading edge is smooth at (0,0)
        upper_points_x[0] = 0
        lower_points_x[0] = 0

        upper_surface_points = list(zip(upper_points_x, upper_points_y))
        lower_surface_points = list(zip(lower_points_x, lower_points_y))

        upper_points = np.array(upper_surface_points).reshape(-1).tolist()
        lower_points = np.array(lower_surface_points).reshape(-1).tolist()
        
        airfoil = None
        for i in range(try_count):
            try:
                airfoil = BezierAirfoil(upper_points, lower_points, dz_te)
            except Exception as e:
                logging.warning(f"Could not generate valid airfoil: {e}")
                continue
            break
        if airfoil is not None:
            return np.concatenate((upper_points, lower_points, [dz_te]))


    def plot(self, ax):
        t = np.linspace(0, 1, 100)
        upper_points = self.upper_surface.evaluate_at_t(t)
        lower_points = self.lower_surface.evaluate_at_t(t)
        upper_control_points = self.upper_surface.control_points
        lower_control_points = self.lower_surface.control_points

        ax.plot(upper_points[:, 0], upper_points[:, 1], color='b', label='Upper surface')
        ax.plot(lower_points[:, 0], lower_points[:, 1], color='r', label='Lower surface')
        ax.scatter(upper_control_points[:,0], upper_control_points[:,1], ec='b', fc='none', label='Upper control points')
        ax.scatter(lower_control_points[:,0], lower_control_points[:,1], ec='r', fc='none', label='Lower control points')
        ax.plot(upper_control_points[:,0], upper_control_points[:,1], c='b', linestyle='--', alpha=0.5)
        ax.plot(lower_control_points[:,0], lower_control_points[:,1], c='r', linestyle='--', alpha=0.5)
        ax.set_title('Airfoil with Bézier parametrization')
        ax.legend()
        plt.show()

    def get_coordinates(self):
        """
        This function returns the coordinates of an airfoil in the format required by XFoil.
        """
        coordinates = []

        t = np.linspace(0, 1, 100)
        t_reversed = np.linspace(1, 0, 100)
        upper_points = self.upper_surface.evaluate_at_t(t_reversed)
        lower_points = self.lower_surface.evaluate_at_t(t)

        for coordinate in upper_points:
            coordinates.append([coordinate[0], coordinate[1]])
        for coordinate in lower_points:
            coordinates.append([coordinate[0], coordinate[1]])
        
        return coordinates

def ccw(A, B, C):
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def check_airfoil(upper_surface: BezierCurve2D, lower_surface: BezierCurve2D):
    """
    Checks for intersections and incorrect slopes at trailing edge
    :param name:
    :return: True if airfoil is valid
    """
    min_angle = 60 * np.pi / 180

    n = 150
    upper_points = upper_surface.evaluate_at_t(np.linspace(0, 1, n))
    lower_points = lower_surface.evaluate_at_t(np.linspace(0, 1, n))
    x_upper = upper_points[:, 0]
    x_lower = lower_points[:, 0]

    y_upper = upper_points[:, 1]
    y_lower = lower_points[:, 1]

    # Check angle at leading edge
    if (y_upper[1] - y_upper[0]) < np.tan(min_angle) * (x_upper[1] - x_upper[0]):
        logging.debug("Upper surface slope at leading edge is too small")
        return False
    if (y_lower[1] - y_lower[0]) > -np.tan(min_angle) * (x_lower[1] - x_lower[0]):
        logging.debug("Lower surface slope at leading edge is too small")
        return False

    # Checks intersection between curves
    for i in range(n-1):
        for j in range(n-1):
            a = (x_upper[i], y_upper[i])
            b = (x_upper[i + 1], y_upper[i + 1])
            c = (x_lower[j], y_lower[j])
            d = (x_lower[j + 1], y_lower[j + 1])
            if intersect(a, b, c, d):
                logging.debug("Upper and lower surfaces intersect")
                return False

    # Checks intersection with itself
    for i in range(n-1):
        for j in range(n-1):
            if j - i > 1:
                a = (x_upper[i], y_upper[i])
                a1 = (x_lower[i], y_lower[i])
                b = (x_upper[i + 1], y_upper[i + 1])
                b1 = (x_lower[i + 1], y_lower[i + 1])
                c = (x_upper[j], y_lower[j])
                c1 = (x_lower[j], y_lower[j])
                d = (y_upper[j + 1], y_lower[j + 1])
                d1 = (x_lower[j + 1], y_lower[j + 1])
                if intersect(a, b, c, d) or intersect(a1, b1, c1, d1):
                    logging.debug("Upper or lower surface intersects with itself")
                    return False

    # # Using y = ax*b for endpoints to find camber line
    # a = (y_upper[-1] - y_upper[0]) / (x_upper[-1] - x_upper[0])
    # b = (y_upper[0] * x_upper[-1] - y_upper[-1] * x_upper[0]) / (x_upper[-1] - x_upper[0])
    # camber_line = lambda x: a * x + b
    # print(f'camber line = {a}x + {b}')
    # # Checks if upper points are above camber line and if lower points are below camber line
    # for i, x in enumerate(x_upper):
    #     if y_upper[i] < camber_line(x):
    #         logging.debug("Upper surface is below camber line")
    #         return False
    # for i, x in enumerate(x_lower):
    #     print(f'x={x}, y={y_lower[i]}, camber={camber_line(x)}')
    #     if y_lower[i] > camber_line(x):
    #         logging.debug("Lower surface is above camber line")
    #         return False
    return True