import numpy as np
import igl
import math
from numba import jit, njit, vectorize

def voxel_points(axes):
    """generate n-dimensional grid coordinates from the given set of (x, y, z...) values"""
    ndims = len(axes)
    dims = [len(x) for x in axes]
    indices = np.indices(dims)
    points = np.array([axis[index] for axis, index in zip(axes, indices)])
    points = points.reshape([ndims, np.prod(dims)])
    return points.T.copy()

@njit#(parallel=True)
def bbox_filter(V, minPt, maxPt):
    mask = np.empty(V.shape[0], np.bool8)
    for i in range(V.shape[0]):
        mask[i] = maxPt[0] > V[i,0] >= minPt[0] and maxPt[1] > V[i,1] >= minPt[1] and maxPt[2] > V[i, 2] >= minPt[2]
    return V[mask, :]

# @vectorize
# def test_in_bounds(x, a, b):
#     a <= x < b

# def bbox_filter(V, minPt, maxPt):
#     minPt = minPt[np.newaxis]
#     maxPt = maxPt[np.newaxis]
#     print('DEBUG',V.dtype, minPt.dtype, maxPt.dtype)
#     print('DEBUG',V.shape, minPt.shape, maxPt.shape)
#     mask = test_in_bounds(V, minPt, maxPt).all(1)
#     print('DEBUG passed')
#     return V[mask,:]


def sample_mesh_interior(V, F, gridLength, dtype=np.float32):
    """Return a matrix of grid points inside the mesh, and also the bounds (minPt, maxPt) of interior points"""
    minPt = np.min(V, 0)
    maxPt = np.max(V, 0)
    q = voxel_points((np.arange(minPt[0], maxPt[0] + gridLength, gridLength, dtype=dtype),
                       np.arange(minPt[1], maxPt[1] + gridLength, gridLength, dtype=dtype),
                         np.arange(minPt[2], maxPt[2] + gridLength, gridLength, dtype=dtype)))
    #print(q)
    w = igl.fast_winding_number_for_meshes(V.astype(dtype), F, q)
    #print(w.shape)
    return q[w > 0.5], [minPt, maxPt]


def sample_surface_points(mesh, n):
    points = np.zeros([n, 3], np.float)
    areas = np.zeros(mesh.cells_dict['triangle'].shape[0], np.float)
    trinormals = np.zeros([areas.shape[0], 3], np.float)
    for i,tri in enumerate(mesh.cells_dict['triangle']):
        base = mesh.points[tri[0],:]
        v1 = mesh.points[tri[1],:] - base
        v2 = mesh.points[tri[2],:] - base
        normal = np.cross(v1, v2)
        area = math.sqrt(normal.dot(normal))
        trinormals[i,:] = normal / area
        areas[i] = area
    total_area = np.sum(areas)
    areas /= total_area
    choices = [c for c in range(len(areas))]
    normals = np.zeros([n, 3], np.float)
    for i in range(n):
        index = random.choices(choices, areas)[0]
        tri = mesh.cells_dict['triangle'][index,:]
        base = mesh.points[tri[0],:]
        v1 = mesh.points[tri[1],:] - base
        v2 = mesh.points[tri[2],:] - base
        u1 = random.random()
        u2 = random.random()
        if u1 + u2 > 1:
            u1 = 1-u1
            u2 = 1-u1
        rp = base + u1 * v1 + u2 * v2
        points[i,:] = rp
        normals[i,:] = trinormals[index,:]
    return points, normals

