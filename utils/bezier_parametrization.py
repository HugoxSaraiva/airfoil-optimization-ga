import matplotlib.pyplot as plt
import numpy as np
import math
import logging
from bezier.curve import Curve as BezierCurve

class BezierAirfoil:
    def __init__(self, parameters: list, shape=tuple[int,int]):
        for parameter in parameters:
            if math.isnan(parameter):
                raise Exception("Invalid parameter, contains NaN")
        self.parameters = parameters
        self.shape = shape
        upper_control_points, lower_control_points = BezierAirfoil._map_parameters_to_control_points(parameters, shape)
        self.upper_control_points = upper_control_points
        self.lower_control_points = lower_control_points
        self.upper_surface = BezierCurve.from_nodes(self.upper_control_points)
        self.lower_surface = BezierCurve.from_nodes(self.lower_control_points)

    @staticmethod
    def parameters_required_for_shape(shape=tuple[int,int]):
        return 2 * shape[0] + 2 * shape[1] - 8

    @staticmethod
    def get_random_airfoil(mean=0.1, range_size=0.07, shape=(6,6), max_try_count=1000):
        parameters = BezierAirfoil.random_params_initializer(mean=mean, range_size=range_size, shape=shape, max_try_count=max_try_count)
        return BezierAirfoil(parameters=parameters, shape=shape)
    
    @staticmethod
    def from_normalized_parameters(parameters: list, shape=tuple[int,int]):
        upper_bounds, lower_bounds = BezierAirfoil.get_bounds(shape)
        parameters = [p * (u - l) + l for p, u, l in zip(parameters, upper_bounds, lower_bounds)]
        return BezierAirfoil(parameters=parameters, shape=shape)

    @staticmethod
    def random_params_initializer(mean=0.1, range_size=0.07, shape=(6,6), max_try_count=1000):
        for i in  range(max_try_count):
            z_te = np.random.uniform(0.0, 0.01) # Based on Bézier Parsec method
            dz_te = np.random.uniform(0.0, 0.001) # Based on Bézier Parsec method

            x_upper = np.sort(np.random.uniform(0.0, 1.0, shape[0] - 3))
            x_lower = np.sort(np.random.uniform(0.0, 1.0, shape[1] - 3))

            y_upper = list(np.random.uniform(0.0, range_size, shape[0] - 2) + mean)
            y_lower = list(np.random.uniform(0.0, -range_size, shape[1] - 2) - mean)
            
            first_y_upper = y_upper.pop(0) * 0.5 # To reduce the range of the first control point
            first_y_lower = y_lower.pop(0) * 0.5 # To reduce the range of the first control point

            parameters = [z_te, dz_te] + [first_y_upper, first_y_lower] + list(x_upper) + list(y_upper) + list(x_lower) + list(y_lower)
            
            if(BezierAirfoil.check_parameters(parameters, shape)):
                return parameters
        raise Exception("Could not generate valid airfoil")
    
    @staticmethod
    def get_bounds(shape=(6,6)):
        z_te_bounds = [0.0, 0.1]
        dz_te_bounds = [0.0, 0.01]
        y_upper_1_bounds = [0.0, 0.5]
        y_lower_1_bounds = [-0.5, 0.0]
        x_upper_bounds = [0.0, 1.0]
        x_lower_bounds = [0.0, 1.0]
        y_upper_bounds = [-0.1, 0.2]
        y_lower_bounds = [-0.2, 0.1]
        upper_bounds = [z_te_bounds[1], 
                        dz_te_bounds[1], 
                        y_upper_1_bounds[1], 
                        y_lower_1_bounds[1]] \
                    + [x_upper_bounds[1]] * (shape[0] - 3) \
                    + [y_upper_bounds[1]] * (shape[0] - 3) \
                    + [x_lower_bounds[1]] * (shape[1] - 3) \
                    + [y_lower_bounds[1]] * (shape[1] - 3)

        lower_bounds = [z_te_bounds[0],
                        dz_te_bounds[0],
                        y_upper_1_bounds[0],
                        y_lower_1_bounds[0]] \
                    + [x_upper_bounds[0]] * (shape[0] - 3) \
                    + [y_upper_bounds[0]] * (shape[0] - 3) \
                    + [x_lower_bounds[0]] * (shape[1] - 3) \
                    + [y_lower_bounds[0]] * (shape[1] - 3)
        return upper_bounds, lower_bounds

    @staticmethod
    def check_parameters(parameters: list, shape=tuple[int,int]):
        try:
            BezierAirfoil._map_parameters_to_control_points(parameters, shape)
            return True
        except Exception as e:
            return False

    @staticmethod
    def _map_parameters_to_control_points(parameters: list, shape=tuple[int,int]):
        if (shape[0] < 3 or shape[1] < 3):
            raise Exception("Invalid shape, must be at least (3,3)")
        if(len(parameters) != BezierAirfoil.parameters_required_for_shape(shape)):
            raise Exception(f"Invalid number of parameters for shape {shape}, expected {BezierAirfoil.parameters_required_for_shape(shape)} got {len(parameters)}")
        
        parameters_copy = list(parameters).copy()
        z_te = parameters_copy.pop(0)
        dz_te = parameters_copy.pop(0)
        y_upper_1 = parameters_copy.pop(0)
        y_lower_1 = parameters_copy.pop(0)
        upper_points_to_get = 2*(shape[0] - 3)
        lower_points_to_get = 2*(shape[1] - 3)
        
        upper_points = parameters_copy[:upper_points_to_get]
        lower_points = parameters_copy[upper_points_to_get:upper_points_to_get+lower_points_to_get]

        # upper control points contains a list of x2, x3, x4, ... y2, y3, y4 ... yn coordinates        
        
        # Adding (0,0) to first upper control points to assure continuity at (0,0)
        # Adding (0,_) to the second upper control point to assure slope at (0,0)
        # The last control point is (1, z_te+dz_te)
        upper_points = [0, 0] + upper_points[:upper_points_to_get//2] + [1] + [0, y_upper_1] + upper_points[upper_points_to_get//2:] + [z_te+dz_te]
        upper_control_points = np.asfortranarray(splitListInHalf(upper_points))

        # Adding (0,0) to first lower control points to assure continuity at (0,0)
        # Adding (0,_) to the second lower control point to assure slope at (0,0)
        # The last control point is (1, z_te-dz_te)
        lower_points = [0, 0] + lower_points[:lower_points_to_get//2] + [1] + [0, y_lower_1] + lower_points[lower_points_to_get//2:] + [z_te-dz_te]
        lower_control_points = np.asfortranarray(splitListInHalf(lower_points))

        upper_surface = BezierCurve.from_nodes(upper_control_points)
        lower_surface = BezierCurve.from_nodes(lower_control_points)

        if not check_surfaces(upper_surface, lower_surface):
            raise Exception("Invalid airfoil surface")

        return upper_control_points, lower_control_points

    def plot(self, ax):
        t = np.linspace(0, 1, 100)
        upper_points = self.upper_surface.evaluate_multi(t)
        lower_points = self.lower_surface.evaluate_multi(t)
        upper_control_points = self.upper_surface.nodes
        lower_control_points = self.lower_surface.nodes

        ax.plot(upper_points[0], upper_points[1], color='b', label='Upper surface')
        ax.plot(lower_points[0], lower_points[1], color='r', label='Lower surface')
        ax.scatter(upper_control_points[0], upper_control_points[1], ec='b', fc='none', label='Upper control points')
        ax.scatter(lower_control_points[0], lower_control_points[1], ec='r', fc='none', label='Lower control points')
        ax.plot(upper_control_points[0], upper_control_points[1], c='b', linestyle='--', alpha=0.5)
        ax.plot(lower_control_points[0], lower_control_points[1], c='r', linestyle='--', alpha=0.5)
        ax.set_title('Airfoil with Bézier parametrization')
        ax.legend()

    def get_coordinates(self, n_points=300):
        """
        This function returns the coordinates of an airfoil in the format required by XFoil.
        """
        coordinates = []

        t = np.linspace(0, 1, n_points//2 + 1)
        t_reversed = np.linspace(1, 0, n_points//2)
        upper_points = self.upper_surface.evaluate_multi(t_reversed)
        lower_points = self.lower_surface.evaluate_multi(t)
        
        # Dropping first point because it contains the leading edge twice
        for x,y in zip(upper_points[0], upper_points[1]):
            coordinates.append([x,y])
        for x,y in zip(lower_points[0], lower_points[1]):
            if (x == 0.0):
                continue
            coordinates.append([x,y])
        return coordinates


def splitListInHalf(list):
    half = len(list)//2
    return [list[:half], list[half:]]

def check_surfaces(upper_surface: BezierCurve, lower_surface: BezierCurve):
    """
    Checks for intersections and incorrect slopes at trailing edge
    :param name:
    :return: True if airfoil is valid
    """
    min_angle = 60 * np.pi / 180

    # Check angle at leading edge
    upper_surface_tangent_vector = upper_surface.evaluate_hodograph(0)
    lower_surface_tangent_vector = lower_surface.evaluate_hodograph(0)
    if (upper_surface_tangent_vector[1]) < np.tan(min_angle) * upper_surface_tangent_vector[0]:
        logging.debug("Upper surface slope at leading edge is too small")
        return False
    if lower_surface_tangent_vector[1] > - np.tan(min_angle) * lower_surface_tangent_vector[0]:
        logging.debug("Lower surface slope at leading edge is too small")
        return False

    intersections = upper_surface.intersect(lower_surface)
    # Can only intersect at endpoints
    if (intersections.shape[1] > 2):
        logging.debug("Upper and lower surfaces intersect")
        return False
    
    # Checks intersection with itself
    intersections = upper_surface.self_intersections()
    if (intersections.shape[1] > 0):
        logging.debug("Upper surface intersects with itself")
        return False
    
    intersections = lower_surface.self_intersections()
    if (intersections.shape[1] > 0):
        logging.debug("Lower surface intersects with itself")
        return False

    return True

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import sys
    import logging
    import numpy as np

    log_level = logging.DEBUG 
    logging.basicConfig(
        level=log_level,
        stream=sys.stdout
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
    shape=(6,6)
    parameters = [
        0, #z_te
        0, #dz_te
        0.1411396509632579, # y_upper_1
        -0.06040774190578092, # y_lower_1
        0.31520572542813763, # x_2
        0.39109318267809434, # x_3
        0.5709148615746626, # x_4
        0.34363932259611313, # y_2
        0.0424359669691409, # y_3
        0.07410454920970207, # y_4
        0.4464414301191849, # x_2
        0.47325435804708416, # x_3
        0.3638052466788326, # x_4
        -0.09693677277639359, # y_2
        -0.11485557171942154, # y_3
        0.014626604247552747, # y_4
    ]

    airfoil = BezierAirfoil(
        parameters=parameters,
        shape=shape,
        )

    # Random init
    # parameters = BezierAirfoil.get_random_parameters(
    #     mean=0.1,
    #     range_size=0.07,
    #     shape=shape,
    #     max_try_count=1000
    # )

    # airfoil = BezierAirfoil(
    #     parameters=parameters,
    #     shape=shape,
    #     )

    fix, ax = plt.subplots()
    airfoil.plot(ax)
    plt.show()
    # print(airfoil.get_coordinates())