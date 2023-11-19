import matplotlib.pyplot as plt
import numpy as np
import math
import logging
from bezier.curve import Curve as BezierCurve
from bezier.hazmat.curve_helpers import get_curvature, evaluate_hodograph
from scipy.interpolate import CubicSpline
from functools import cached_property
# from utils.airfoil import Airfoil

class BezierAirfoilNSGA2Adapter:
    """
    This class is used to adapt the BezierAirfoil class to the NSGA2 algorithm
    """
    def __init__(self, shape: tuple[int,int], max_try_count=1000):
        self.shape = shape
        self.max_try_count = max_try_count
    
    ## NSGA2 methods
    def random_params_initializer(self):
        shape = self.shape
        z_te_bounds, dz_te_bounds, y_le_bounds, y_upper_1_bounds, y_lower_1_bounds, x_upper_bounds, x_lower_bounds, y_upper_bounds, y_lower_bounds = self._get_bounds_components()
        for i in  range(self.max_try_count):
            z_te = np.random.uniform(z_te_bounds[0], z_te_bounds[0]) # Based on Bézier Parsec method
            dz_te = np.random.uniform(dz_te_bounds[1], dz_te_bounds[1]) # Based on Bézier Parsec method

            y_le = np.random.uniform(y_le_bounds[0], y_le_bounds[1])
            x_upper = np.sort(np.random.uniform(x_upper_bounds[0], x_upper_bounds[1], shape[0] - 3))
            x_lower = np.sort(np.random.uniform(x_lower_bounds[0], x_lower_bounds[1], shape[1] - 3))

            y_upper = list(np.random.uniform(y_upper_bounds[0], y_upper_bounds[1], shape[0] - 3))
            y_lower = list(np.random.uniform(y_lower_bounds[0], y_lower_bounds[1], shape[1] - 3))
            
            first_y_upper = np.random.uniform(y_upper_1_bounds[0], y_upper_1_bounds[1])
            first_y_lower = np.random.uniform(y_lower_1_bounds[0], y_lower_1_bounds[1])

            parameters = [z_te, dz_te, y_le, first_y_upper, first_y_lower] + list(x_upper) + list(y_upper) + list(x_lower) + list(y_lower)
            
            if(BezierAirfoil.check_parameters(parameters, shape)):
                return parameters
        raise Exception("Could not generate valid airfoil")
    
    def get_bounds(self):
        shape = self.shape
        z_te_bounds, dz_te_bounds, y_le_bounds, y_upper_1_bounds, y_lower_1_bounds, x_upper_bounds, x_lower_bounds, y_upper_bounds, y_lower_bounds = self._get_bounds_components()
        
        upper_bounds = [z_te_bounds[1], 
                        dz_te_bounds[1],
                        y_le_bounds[1],
                        y_upper_1_bounds[1], 
                        y_lower_1_bounds[1]] \
                    + [x_upper_bounds[1]] * (shape[0] - 3) \
                    + [y_upper_bounds[1]] * (shape[0] - 3) \
                    + [x_lower_bounds[1]] * (shape[1] - 3) \
                    + [y_lower_bounds[1]] * (shape[1] - 3)

        lower_bounds = [z_te_bounds[0],
                        dz_te_bounds[0],
                        y_le_bounds[0],
                        y_upper_1_bounds[0],
                        y_lower_1_bounds[0]] \
                    + [x_upper_bounds[0]] * (shape[0] - 3) \
                    + [y_upper_bounds[0]] * (shape[0] - 3) \
                    + [x_lower_bounds[0]] * (shape[1] - 3) \
                    + [y_lower_bounds[0]] * (shape[1] - 3)
        return upper_bounds, lower_bounds
    
    def from_parameters(self, parameters: list):
        return BezierAirfoil(parameters=parameters, shape=self.shape)

    ## Helper methods
    @staticmethod
    def _get_bounds_components():
        z_te_bounds = [-0.05, 0.05]
        dz_te_bounds = [0.0, 0.01]
        y_le_bounds = [-0.1, 0.1]
        y_upper_1_bounds = [0.0, 0.2]
        y_lower_1_bounds = [-0.2, 0.0]
        x_upper_bounds = [0.1, 0.9]
        x_lower_bounds = [0.1, 0.9]
        y_upper_bounds = [0, 0.3]
        y_lower_bounds = [-0.3, 0]
        return z_te_bounds, dz_te_bounds, y_le_bounds, y_upper_1_bounds, y_lower_1_bounds, x_upper_bounds, x_lower_bounds, y_upper_bounds, y_lower_bounds

    def get_random_airfoil(self):
        parameters = self.random_params_initializer()
        return BezierAirfoil(parameters=parameters, shape=self.shape)

class BezierAirfoil():
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
    def required_parameters(shape=tuple[int,int]):
        return 2 * shape[0] + 2 * shape[1] - 7

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
        if(len(parameters) != BezierAirfoil.required_parameters(shape)):
            raise Exception(f"Invalid number of parameters for shape {shape}, expected {BezierAirfoil.required_parameters(shape)} got {len(parameters)}")
        
        parameters_copy = list(parameters).copy()
        z_te = parameters_copy.pop(0)
        dz_te = parameters_copy.pop(0)
        y_le = parameters_copy.pop(0)
        y_upper_1 = parameters_copy.pop(0)
        y_lower_1 = parameters_copy.pop(0)
        upper_points_to_get = 2*(shape[0] - 3)
        lower_points_to_get = 2*(shape[1] - 3)
        
        upper_points = parameters_copy[:upper_points_to_get]
        lower_points = parameters_copy[upper_points_to_get:upper_points_to_get+lower_points_to_get]

        # upper control points contains a list of x2, x3, x4, ... y2, y3, y4 ... yn coordinates        
        
        # Adding (0,y_le) to first upper control points to assure continuity at (0,y_le)
        # Adding (0,y_le+y_upper_1) to the second upper control point to assure slope at (0,y_le)
        # The last control point is (1, z_te+dz_te/2)
        # Upper points is defined of a list of x0, x1, x2, ... xn coordinates followed by y0, y1, y2, ... yn coordinates.
        upper_points = \
            [0, 0] + upper_points[:upper_points_to_get//2] + [1] \
            + [y_le, y_le + y_upper_1] + upper_points[upper_points_to_get//2:] + [z_te+dz_te/2]
        upper_control_points = np.asfortranarray(splitListInHalf(upper_points))

        # Adding (0,y_le) to first lower control points to assure continuity at (0,y_le)
        # Adding (0,y_le+y_lower_1) to the second lower control point to assure slope at (0,y_le)
        # The last control point is (1, z_te-dz_te/2)
        lower_points = \
            [0, 0] + lower_points[:lower_points_to_get//2] + [1] \
            + [y_le, y_le + y_lower_1] + lower_points[lower_points_to_get//2:] + [z_te-dz_te/2]
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

        self.upper_surface.plot(100, ax=ax, color='b', )
        self.lower_surface.plot(100, ax=ax, color='r', )
        ax.plot(upper_points[0], upper_points[1], color='b', label='Upper surface')
        ax.plot(lower_points[0], lower_points[1], color='r', label='Lower surface')
        ax.plot(upper_control_points[0], upper_control_points[1], 'bo', label='Upper control points')
        ax.plot(lower_control_points[0], lower_control_points[1], 'ro', label='Lower control points')
        ax.plot(upper_control_points[0], upper_control_points[1], c='b', linestyle='--', alpha=0.5)
        ax.plot(lower_control_points[0], lower_control_points[1], c='r', linestyle='--', alpha=0.5)
        ax.set_title('Airfoil with Bézier parametrization')
        ax.set_xlim([-0.1, 1.1])
        ax.set_ylim([-0.5, 0.5])
        ax.set_aspect('equal')
        ax.legend()

    def plot_camber(self, ax, consider_chord=False, style='-'):
        general_info = self.general_info
        camber_fx = general_info['camber']['fx']
        chord_fx = general_info['chord']['fx']
        x = np.linspace(0, 1, 300)
        ax.plot(x, camber_fx(x)+chord_fx(x) if consider_chord else camber_fx(x), style, label='Camber')
        ax.set_title('Camber')
        ax.set_xlabel('x')

    def plot_thickness(self, ax, style='-'):
        general_info = self.general_info
        thickness_fx = general_info['thickness']['fx']
        x = np.linspace(0, 1, 300)
        ax.plot(x, thickness_fx(x), style, label='Thickness')
        ax.set_title('Thickness')
        ax.set_xlabel('x')

    def plot_chord(self, ax, style='-'):
        general_info = self.general_info
        chord_fx = general_info['chord']['fx']
        x = np.linspace(0, 1, 300)
        ax.plot(x, chord_fx(x), style, label='Chord')
        ax.set_title('Chord')
        ax.set_xlabel('x')

    @cached_property
    def coordinates(self):
        """
        This function returns the coordinates of an airfoil in the format required by XFoil.
        """
        coordinates = []
        n_points = 300
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

    def get_coordinates(self):
        return self.coordinates

    @cached_property
    def general_info(self):
        # Getting curvature at the leading edge
        result = {}
        for name, surface in zip(['upper', 'lower'], [self.upper_surface, self.lower_surface]):    
            t = 0
            nodes = surface.nodes
            tangent_vector = evaluate_hodograph(t, nodes)
            curvature = get_curvature(nodes, tangent_vector, t)
            result[name] = {
                'curvature': curvature,
                'tangent_vector': tangent_vector,
            }
        
        t = np.linspace(0, 1, 300)
        upper_points = self.upper_surface.evaluate_multi(t)
        lower_points = self.lower_surface.evaluate_multi(t)
        upper_points_fx = CubicSpline(upper_points[0], upper_points[1])
        lower_points_fx = CubicSpline(lower_points[0], lower_points[1])
        x = np.linspace(0, 1, 300)
        chord = np.interp(
            x, [0, 1], [upper_points_fx(0), (upper_points_fx(1) + lower_points_fx(1))/2]
            )
        camber = (upper_points_fx(x) + lower_points_fx(x))/2 - chord
        camber_fx = CubicSpline(x, camber)
        max_camber_location = x[np.argmax(camber)]
        max_camber_value = np.max(camber)

        thickness = 2 * ( upper_points_fx(x) - camber - chord)
        thickness_fx = CubicSpline(x, thickness)
        max_thickness_location = x[np.argmax(thickness)]
        max_thickness_value = np.max(thickness)

        result['camber'] = {
            'fx': camber_fx,
            'points': camber,
            'max_location': max_camber_location,
            'max_value': max_camber_value,
            'alpha_te': np.arctan(-camber_fx.derivative()(1)) * 180 / np.pi,
        }
        result['thickness'] = {
            'fx': thickness_fx,
            'points': thickness,
            'max_location': max_thickness_location,
            'max_value': max_thickness_value,
            'trailing_edge_thickness': thickness[-1],
            'beta_te': np.arctan(-thickness_fx.derivative()(1)) * 180 / np.pi,
        }
        result['chord'] = {
            'fx': CubicSpline(x, chord),
            'points': chord,
        }
        return result
    
    def get_general_info(self):
        return self.general_info

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

def get_curve_derivative(curve: BezierCurve) -> BezierCurve:
    """
    Returns the derivative of a curve
    """
    nodes = curve.nodes.copy()
    degree = curve.degree
    for i in range(degree):
        nodes[:,i] = (degree) * (nodes[:,i+1] - nodes[:,i])
    nodes = nodes[:,:-1]
    return BezierCurve.from_nodes(nodes)

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
        0, #y_le
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
    parameters = BezierAirfoilNSGA2Adapter(shape=shape, max_try_count=1000).random_params_initializer(
    )

    airfoil = BezierAirfoil(
        parameters=parameters,
        shape=shape,
        )

    fig, axs = plt.subplots(2, 1)
    camber_fx = airfoil.general_info['camber']['fx']
    thickness_fx = airfoil.general_info['thickness']['fx']
    airfoil.plot(axs[0])
    x = np.linspace(0, 1, 300)
    airfoil.plot_camber(axs[0], style='k--', consider_chord=True)
    airfoil.plot_thickness(axs[1])
    # airfoil.plot_chord(axs[0], style='k')
    axs[0].legend()
    from pprint import pprint
    pprint(airfoil.get_general_info())
    plt.show()
    # print(airfoil.get_coordinates())