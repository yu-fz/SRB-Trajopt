import numpy as np

class SRBTrajoptOptions:
    """
    Data structure for storing SRB model parameters for trajectory optimization
    """
    def __init__(self) -> None:
        # SRB model parameters
        self._dimensions = np.array([0.203, 0.254, 0.457])
        self._mass = 55.
        self._leg_extension_bounds = np.array([0.2, 0.2, 1.0]) # x,y,z
        self._mu = 0.85
        self._color = np.array([0.9608, 0.9608, 0.8627, 1.0])

        self.foot_length = 0.1
        self.foot_width = 0.05

        # Control Limits 

        # max normalized Z axis grf 
        self._max_z_grf = self._mass * 5
        self._min_com_height = 0.6

        # trajopt parameters
        self.N = 10 # number of knot points
        self.T = 2 # total time

    @property
    def dimensions(self):
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dims: np.ndarray):
        assert len(dims) == 3, "Dimensions must be a 3D vector"
        self._dimensions = dims

    @property
    def mass(self):
        return self._mass
    
    @mass.setter
    def mass(self, mass: float):
        assert type(mass) == float, "Mass must be a float"
        self._mass = mass

    @property
    def leg_extension_bounds(self):
        return self._leg_extension_bounds
    
    @leg_extension_bounds.setter
    def leg_extension_bounds(self, leg_extension_bounds: np.ndarray):
        assert len(leg_extension_bounds) == 3, "Leg extension bounds must be a 3D vector"
        self._leg_extension_bounds = leg_extension_bounds
    
    @property
    def max_z_grf(self):
        return self._max_z_grf
    
    @max_z_grf.setter
    def max_z_grf(self, max_z_grf: float):
        assert type(max_z_grf) == float, "Max Z GRF must be a float"
        self._max_z_grf = max_z_grf
    
    @property
    def min_com_height(self):
        return self._min_com_height
    
    @min_com_height.setter
    def min_com_height(self, min_com_height: float):
        assert type(min_com_height) == float, "Min COM height must be a float"
        self._min_com_height = min_com_height

    @property
    def mu(self):
        return self._mu
    
    @mu.setter
    def mu(self, mu: float):
        assert type(mu) == float, "Friction coefficient must be a float"
        assert mu >= 0.0, "Friction coefficient must be non-negative"
        self._mu = mu
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, color: np.ndarray):
        if color.shape != (4,) and color.shape != (3,):
            raise ValueError("Color must be a 3D or 4D RGB/RGBA vector")
        if len(color) == 3:
            color = np.append(color, 1.0)
        self._color = color

    

    