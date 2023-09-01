import numpy as np

class SRBTrajoptOptions:
    """
    Data structure for storing SRB model parameters for trajectory optimization
    """
    def __init__(self) -> None:
        # SRB model parameters
        self._dimensions = np.array([0.203, 0.254, 0.457])
        self._mass = 0.0
        self._min_leg_extension = 0.5
        self._max_leg_extension = 1.2
        self._max_z_grf = 0.0
        self._mu = 0.9
        self._color = np.array([0.9608, 0.9608, 0.8627, 1.0])

        self.foot_length = 0.1
        self.foot_width = 0.05

        # trajopt parameters
        self.N # number of knot points
        self.T # total time

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
    def min_leg_extension(self):
        return self._min_leg_extension
    
    @min_leg_extension.setter
    def min_leg_extension(self, min_leg_extension: float):
        assert type(min_leg_extension) == float, "Min leg extension must be a float"
        self._min_leg_extension = min_leg_extension
    
    @property
    def max_leg_extension(self):
        return self._max_leg_extension
    
    @max_leg_extension.setter
    def max_leg_extension(self, max_leg_extension: float):
        assert type(max_leg_extension) == float, "Max leg extension must be a float"
        self._max_leg_extension = max_leg_extension
    
    @property
    def max_z_grf(self):
        return self._max_z_grf
    
    @max_z_grf.setter
    def max_z_grf(self, max_z_grf: float):
        assert type(max_z_grf) == float, "Max Z GRF must be a float"
        self._max_z_grf = max_z_grf
    
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

    

    