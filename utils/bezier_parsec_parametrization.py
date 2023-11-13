import numpy as np
import math
from utils.bezier_curve import BezierCurve2D
import matplotlib.pyplot as plt

# Implementation of the Bezier-Parsec parametrization of airfoils described in "Bezier-PARSEC: An optimized aerofoil parameterization for design" doi:10.1016/j.advengsoft.2010.05.002

def bp_3333(
        r_le, 
        alpha_te, 
        beta_te, 
        z_te, 
        gamma_le,
        x_c,
        y_c,
        k_c,
        x_t,
        y_t,
        k_t,
        dz_te
        ):
    """"
    Transforms parameters from the Bezier-Parsec parametrization to the Bezier control points.
    Parameters:
    r_le - leading edge radius, must be negative, [-0.04, -0.001]
    alpha_te - trailing camber line angle, must be positive, [0.05, 0.1]
    beta_te - trailing wedge angle, must be positive, [0.001, 0.3]
    z_te - trailing edge vertical displacement, must be positive, [0, 0.01]
    gamma_le - leading edge direction, must be positive, [0.05, 0.1]
    x_c - location of the camber crest, must be positive, [0.2, 0.5]
    y_c - location of the camber crest, must be positive, [0, 0.2]
    k_c - curvature of the camber crest, must be negative, [-0.2, 0]
    x_t - position of the thickness crest, must be positive, [0.15, 0.4]
    y_t - position of the thickness crest, must be positive, [0.05, 0.15]
    k_t - curvature of the thickness crest, must be negative, [-0.5, 0]
    dz_te - the half thickness of the trailing edge, must be positive, [0, 0.001]
    """
    r_t = get_rt(k_t=k_t, x_t=x_t, y_t=y_t, r_le=r_le)
    r_c = get_rc(k_c=k_c, y_c=y_c, gamma_le=gamma_le, alpha_te=alpha_te, z_te=z_te)
    if r_t is None:
        return None
    if r_c is None:
        return None
    thickness_leading_edge_control_points = get_leading_edge_thickness_control_points(k_t=k_t, x_t=x_t, y_t=y_t, r_t=r_t)
    thickness_trailing_edge_control_points = get_trailing_edge_thickness_control_points(k_t=k_t, x_t=x_t, y_t=y_t, r_t=r_t, beta_te=beta_te, dz_te=dz_te)
    
    camber_leading_edge_control_points = get_leading_edge_camber_control_points(k_c=k_c, x_c=x_c, y_c=y_c, gamma_le=gamma_le, r_c=r_c)
    camber_trailing_edge_control_points = get_trailing_edge_camber_control_points(k_c=k_c, x_c=x_c, y_c=y_c, alpha_te=alpha_te, z_te=z_te, r_c=r_c)
    
    return {'thickness': (
        thickness_leading_edge_control_points,
        thickness_trailing_edge_control_points
    ), 'camber': (
        camber_leading_edge_control_points,
        camber_trailing_edge_control_points
    )}


class BezierParsecAirfoil:
    def __init__(self, r_le, 
        alpha_te, 
        beta_te, 
        z_te, 
        gamma_le,
        x_c,
        y_c,
        k_c,
        x_t,
        y_t,
        k_t,
        dz_te
        ):
        self.r_le = r_le
        self.alpha_te = alpha_te
        self.beta_te = beta_te
        self.z_te = z_te
        self.gamma_le = gamma_le
        self.x_c = x_c
        self.y_c = y_c
        self.k_c = k_c
        self.x_t = x_t
        self.y_t = y_t
        self.k_t = k_t
        self.dz_te = dz_te
        try:
            self.control_points = bp_3333(
                r_le=r_le, 
                alpha_te=alpha_te, 
                beta_te=beta_te, 
                z_te=z_te, 
                gamma_le=gamma_le,
                x_c=x_c,
                y_c=y_c,
                k_c=k_c,
                x_t=x_t,
                y_t=y_t,
                k_t=k_t,
                dz_te=dz_te
            )
        except:
            self.control_points = None
        if self.control_points is None:
            raise ValueError('Invalid parameters')
        self.thickess_curves = (
            BezierCurve2D(self.control_points['thickness'][0]),
            BezierCurve2D(self.control_points['thickness'][1])
        )
        self.camber_curves = (
            BezierCurve2D(self.control_points['camber'][0]),
            BezierCurve2D(self.control_points['camber'][1])
        )
    
    def thickness_at(self, t):
        return self.thickess_curves[0].evaluate_at_t(2 * t) if t < 0.5 else  self.thickess_curves[1].evaluate_at_t(2 * t - 1)

    def camber_at(self, t):
        return self.camber_curves[0].evaluate_at_t(2 * t) if t < 0.5 else  self.camber_curves[1].evaluate_at_t(2 * t - 1)

    def get_lower_surface_at_x(self, x):
        camber_curve = self.camber_curves[0] if x < self.x_c else self.camber_curves[1]
        thickness_curve = self.thickess_curves[0] if x < self.x_t else self.thickess_curves[1]
        camber_point = camber_curve.evaluate_at_x(x)
        thickness_point = thickness_curve.evaluate_at_x(x)
        return camber_point[1] - thickness_point[1]

    def get_upper_surface_at_x(self, x):
        camber_curve = self.camber_curves[0] if x < self.x_c else self.camber_curves[1]
        thickness_curve = self.thickess_curves[0] if x < self.x_t else self.thickess_curves[1]
        camber_point = camber_curve.evaluate_at_x(x)
        thickness_point = thickness_curve.evaluate_at_x(x)
        return camber_point[1] + thickness_point[1]         

    def __str__(self) -> str:
        return f'''BezierParsecAirfoil(
            r_le={self.r_le},
            alpha_te={self.alpha_te},
            beta_te={self.beta_te},
            z_te={self.z_te},
            gamma_le={self.gamma_le},
            x_c={self.x_c},
            y_c={self.y_c},
            k_c={self.k_c},
            x_t={self.x_t},
            y_t={self.y_t},
            k_t={self.k_t},
            dz_te={self.dz_te}
            )
            '''
    
    def get_coordinates(self):
        """
        This function returns the coordinates of an airfoil in the format required by XFoil.
        """
        coordinates = []
        n_points = 300
        # Creating n_points coordinates for the upper part of the airfoil
        x_upper = np.linspace(1, 0, math.floor(n_points / 2) + 1)
        for x in x_upper:
            y = self.get_upper_surface_at_x(x)
            coordinates.append([x, y])
        
        x_lower = np.linspace(0, 1, math.floor(n_points / 2))
        for x in x_lower:
            if(x == 0):
                continue
            y = self.get_lower_surface_at_x(x)
            coordinates.append([x, y])
        
        return coordinates

    @staticmethod
    def from_random(try_limit=10000):
        r_t = None
        for i in range(try_limit):
            x_t = np.random.uniform(0.15, 0.4)
            y_t = np.random.uniform(0.05, 0.15)
            k_t = np.random.uniform(-0.5, 0)
            r_le = np.random.uniform(-0.04, -0.001)
            try:
                r_t = get_rt(k_t=k_t, x_t=x_t, y_t=y_t, r_le=r_le)
            except:
                r_t = None
            if r_t is not None:
                break
        if r_t is None:
            return None
        r_c = None
        for i in range(try_limit):    
            alpha_te = np.random.uniform(0.05, 0.1)
            beta_te = np.random.uniform(0.001, 0.3)
            z_te = np.random.uniform(0, 0.01)
            gamma_le = np.random.uniform(0.05, 0.1)
            x_c = np.random.uniform(0.2, 0.5)
            y_c = np.random.uniform(0, 0.2)
            k_c = np.random.uniform(-0.2, 0)
            dz_te = np.random.uniform(0, 0.001)
            try:
                r_c = get_rc(k_c=k_c, y_c=y_c, gamma_le=gamma_le, alpha_te=alpha_te, z_te=z_te)
            except:
                r_c = None
            if r_c is not None:
                break
        if r_c is None:
            return None
        return BezierParsecAirfoil(
            r_le=r_le, 
            alpha_te=alpha_te, 
            beta_te=beta_te, 
            z_te=z_te, 
            gamma_le=gamma_le,
            x_c=x_c,
            y_c=y_c,
            k_c=k_c,
            x_t=x_t,
            y_t=y_t,
            k_t=k_t,
            dz_te=dz_te
        )

def get_leading_edge_thickness_control_points(k_t, x_t, y_t, r_t):
    x_0 = 0
    y_0 = 0
    x_1 = 0
    y_1 = 3 / 2 * k_t * (x_t - r_t) ** 2 + y_t
    x_2 = r_t
    y_2 = y_t
    x_3 = x_t
    y_3 = y_t
    return [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3)]

def get_trailing_edge_thickness_control_points(k_t, x_t, y_t, r_t, beta_te, dz_te):
    x_0 = x_t
    y_0 = y_t
    x_1 = 2 * x_t - r_t
    y_1 = y_t
    x_2 = 1 + (dz_te - (3 / 2 * k_t * (x_t - r_t) ** 2 + y_t)) * cot(beta_te)
    y_2 = 3 / 2 * k_t * (x_t - r_t) ** 2 + y_t
    x_3 = 1
    y_3 = dz_te
    return [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3)]

def get_leading_edge_camber_control_points(k_c, x_c, y_c, gamma_le, r_c):
    x_0 = 0
    y_0 = 0
    x_1 = r_c * cot(gamma_le)
    y_1 = r_c
    x_2 = x_c - np.sqrt(2 * (r_c - y_c) / (3 * k_c))
    y_2 = y_c
    x_3 = x_c
    y_3 = y_c
    return [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3)]

def get_trailing_edge_camber_control_points(k_c, x_c, y_c, alpha_te, z_te, r_c):
    x_0 = x_c
    y_0 = y_c
    x_1 = x_c + np.sqrt(2 * (r_c - y_c) / (3 * k_c))
    y_1 = y_c
    x_2 = 1 + (z_te - r_c) * cot(alpha_te)
    y_2 = r_c
    x_3 = 1
    y_3 = z_te
    return [(x_0, y_0), (x_1, y_1), (x_2, y_2), (x_3, y_3)]

def get_rt(k_t, x_t, y_t, r_le):
    coefficents = [
        (3 * (y_t ** 2) + 9 * k_t * (x_t ** 2) * y_t + 27 * (k_t ** 2) * (x_t ** 4) / 4),
        (2 * r_le - 18 * k_t * x_t * y_t - 27 * (k_t ** 2) * (x_t ** 3)),
        (9 * k_t * y_t + 81 * (k_t ** 2) * (x_t ** 2)/ 2),
        (-27 * (k_t ** 2) * x_t),
        ( 27 * (k_t ** 2) / 4 ),
    ]
    polynomial = np.polynomial.Polynomial(coefficents)
    roots = polynomial.roots()
    real_roots: list[float] = roots[np.isreal(roots)].real
    min_rt_value = max([0, x_t - np.sqrt(-2 * y_t / (3 * k_t))])
    max_rt_value = x_t
    rt_candidates = list(filter(lambda root: min_rt_value < root < max_rt_value, real_roots))
    if(len(rt_candidates) == 0):
        return None
    return min(rt_candidates)

def get_rc(k_c, y_c, gamma_le, alpha_te, z_te):
    cot_sum = cot(gamma_le) + cot(alpha_te)
    divisor = 3 * k_c * (cot_sum ** 2)
    delta = 16 + 6 * k_c * cot_sum * (1 - y_c * cot_sum + z_te * cot(alpha_te))
    if delta < 0:
        return None
    base = (16 + 3 * k_c * cot_sum * ( 1 + z_te * cot(alpha_te)))
    rc_candidates = [(base + 4 * np.sqrt(delta)) / divisor, (base - 4 * np.sqrt(delta)) / divisor]
    mix_r_c_value = 0
    max_r_c_value = y_c
    rc_candidates = list(filter(lambda x: mix_r_c_value < x < max_r_c_value, rc_candidates))
    if(len(rc_candidates) == 0):
        return None
    return min(rc_candidates, key=lambda x: abs(x - y_c))

def cot(x):
    return 1 / np.tan(x)

if __name__ == '__main__':
    r_le = -0.026690731549978808
    x_t = 0.3654508844604395
    y_t = 0.07807483165242644
    k_t = -0.37777645516921876
    beta_te = np.random.uniform(0.001, 0.3)

    k_c = -0.1
    x_c = 0.5
    y_c = 0.01
    gamma_le = 0.05
    alpha_te = 0.09
    z_te = 0.001
    dz_te = 0.000871157824702653

    airfoil = BezierParsecAirfoil(
        r_le=r_le, 
        alpha_te=alpha_te, 
        beta_te=beta_te, 
        z_te=z_te, 
        gamma_le=gamma_le,
        x_c=x_c,
        y_c=y_c,
        k_c=k_c,
        x_t=x_t,
        y_t=y_t,
        k_t=k_t,
        dz_te=dz_te
    )

    t = np.linspace(0, 1, 100)
    thickness = []
    camber = []
    for t_i in t:
        thickness.append(airfoil.thickness_at(t_i).tolist())
        camber.append(airfoil.camber_at(t_i).tolist())

    fig, axs = plt.subplots(3)
    axs[0].plot([point[0] for point in thickness], [point[1] for point in thickness])
    thcikness_control_points = airfoil.control_points['thickness'][0] + airfoil.control_points['thickness'][1]
    axs[0].plot([control_point[0] for control_point in thcikness_control_points], [control_point[1] for control_point in thcikness_control_points], 'or')
    axs[0].title.set_text('Thickness')

    axs[1].plot([point[0] for point in camber], [point[1] for point in camber])
    camber_control_points = airfoil.control_points['camber'][0] + airfoil.control_points['camber'][1]

    axs[1].plot([control_point[0] for control_point in camber_control_points], [control_point[1] for control_point in camber_control_points], 'or')
    axs[1].title.set_text('Camber')

    # Plotting the airfoil
    x = np.linspace(0, 1, 200)
    upper_surface = []
    lower_surface = []
    for x_i in x:
        upper_surface.append(airfoil.get_upper_surface_at_x(x_i))
        lower_surface.append(airfoil.get_lower_surface_at_x(x_i))
    axs[2].plot(x, upper_surface, label='Upper surface')
    axs[2].plot(x, lower_surface, label='Lower surface', c='#1f77b4')
    axs[2].title.set_text('Airfoil')
    plt.show()
    