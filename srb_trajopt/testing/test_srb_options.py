
import unittest
import numpy as np
from options.srb_trajopt_options import SRBTrajoptOptions

class TestSRBTrajoptOptions(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = SRBTrajoptOptions()

    def test_set_dimensions(self):
        self.options.dimensions = np.array([0.1, 0.2, 0.3])
        assert np.array_equal(self.options.dimensions, np.array([0.1, 0.2, 0.3]))

    def test_set_mass(self):
        self.options.mass = 0.5
        assert self.options.mass == 0.5
    
    def test_set_color(self):
        self.options.color = np.array([0.1, 0.2, 0.3, 0.4])
        assert np.array_equal(self.options.color, np.array([0.1, 0.2, 0.3, 0.4]))
        self.options.color = np.array([0.1, 0.2, 0.3])
        assert np.array_equal(self.options.color, np.array([0.1, 0.2, 0.3, 1.0]))


if __name__ == '__main__':
    unittest.main()