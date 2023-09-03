import unittest
import numpy as np
from trajopt.srb_trajopt import *

class TestSRBTrajopt(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.options = SRBTrajoptOptions()


    def test_set_options(self):
        self.options.dimensions = np.array([0.203, 0.254, 0.457])
        self.options.color = np.array([0.1, 0.2, 0.3, 0.4])
        srb_trajopt = SRBTrajopt(self.options, headless=True)
        assert np.array_equal(srb_trajopt.options.dimensions, np.array([0.203, 0.254, 0.457]))
        assert np.array_equal(srb_trajopt.options.color, np.array([0.1, 0.2, 0.3, 0.4]))

    # def test_visualizer(self):
    #     srb_trajopt = SRBTrajopt(self.options, headless=False)
    #     srb_trajopt.render_srb()

    def test_trajopt_solver(self):
        srb_trajopt = SRBTrajopt(self.options, headless=True)
        srb_trajopt.solve_trajopt()

if __name__ == '__main__':
    unittest.main()