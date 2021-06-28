import unittest
import physics.physics as phys
import igl
import numpy as np
import math

class TestPhysObject(unittest.TestCase):
    def setUp(self):
        v, f = igl.read_triangle_mesh('../data/cylinder3.obj')
        self.mesh = (v, f)
        

    def test_mass_invariant(self):
        v, f = self.mesh
        masses = []
        test_lengths = [0.1, 0.05, 0.02]
        test_lengths.sort(reverse=True)
        for grid_len in test_lengths:
            density = 1
            obj = phys.PhysObject(v, f, grid_len, translation=np.array([0, 0, 0], np.float32), density=density)
            masses.append(obj.mass)
        print(masses)
        m0 = np.mean(masses)
        for m in masses:
            diff = abs(m - m0)/masses[-1]
            self.assertLessEqual(diff, 5e-2)


