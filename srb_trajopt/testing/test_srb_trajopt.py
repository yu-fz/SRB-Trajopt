import unittest
import numpy as np
from trajopt.srb_trajopt import *

class TestSRBTrajopt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = SRBTrajoptOptions()

    def test_set_options(self):
        self.options.dimensions = np.array([0.1, 0.2, 0.3])
        self.options.color = np.array([0.1, 0.2, 0.3, 0.4])
        srb_trajopt = SRBTrajopt(self.options)
        assert np.array_equal(srb_trajopt.options.dimensions, np.array([0.1, 0.2, 0.3]))
        assert np.array_equal(srb_trajopt.options.color, np.array([0.1, 0.2, 0.3, 0.4]))



if __name__ == '__main__':
    unittest.main()