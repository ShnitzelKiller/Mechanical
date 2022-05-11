import unittest
import mechanical.physics as phys
import igl
import numpy as np
import math
import mechanical.onshape as onshape
from utils import adjacency_list, adjacency_list_from_brepdata, homogenize, connected_components, connected_components_dense, adjacency_matrix
import time

class TestDataAnalysis(unittest.TestCase):
    #def __init__(self, assembly='b1962d9eb0863de9d3befdfb_0e59c1f11f56bbf035659f09_a60cbcfa67d608b28ffb10fa.json'):
    #    super().__init__()
    #    self.assembly = assembly


    def setUp(self):
        loader = onshape.Loader('/projects/grail/benjones/cadlab/data')
        self.occs, self.mates = loader.load_flattened('b1962d9eb0863de9d3befdfb_0e59c1f11f56bbf035659f09_a60cbcfa67d608b28ffb10fa.json',geometry=False)

    def test_adjacency(self):
        start = time.time()
        print('creating adjacency list')
        adj = adjacency_list_from_brepdata(self.occs, self.mates)
        end = time.time()
        print('time taken to create adjacency matrix:',end-start)
        start = end
        adj_mat = homogenize(adj)
        end = time.time()
        print('time taken to homogenize:',end-start)

        adj_dense = adjacency_matrix(self.occs, self.mates)
        for i in range(adj_mat.shape[0]):
            for j in range(0, adj_mat.shape[1], 2):
                if adj_mat[i,j] >= 0:
                    self.assertEqual(adj_dense[i,adj_mat[i,j]], adj_mat[i,j+1], f'row {i}, column {adj_mat[i,j]}')
    
    def test_connected_components(self):
        
        adj = adjacency_list_from_brepdata(self.occs, self.mates)
        adj_mat = homogenize(adj)
        adj_dense = adjacency_matrix(self.occs, self.mates)

        start = time.time()
        num_components = connected_components(adj_mat)
        print('num components:',num_components)
        end = time.time()
        print('time taken for normal components:', end-start)

        start = time.time()
        num_components_dense = connected_components_dense(adj_dense)
        end = time.time()
        print('time taken to compute dense components')
        self.assertEqual(num_components, num_components_dense)
        start = time.time()
        num_rigid = connected_components(adj_mat, connectionType='fasten')
        print('num rigid:',num_rigid)
        end = time.time()
        print('time taken for rigid components: ',end-start)
        start = time.time()
        num_rigid_dense = connected_components_dense(adj_dense, connectionType='fasten')
        end = time.time()
        print('time taken to compute dense rigid pieces')
        self.assertEqual(num_rigid, num_rigid_dense)



class TestPhysObject(unittest.TestCase):
    def setUp(self):
        v, f = igl.read_triangle_mesh('data/cylinder3.obj')
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


