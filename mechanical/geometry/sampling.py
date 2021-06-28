import numpy as np
import igl

def voxel_points(axes):
    """generate n-dimensional grid coordinates from the given set of (x, y, z...) values"""
    ndims = len(axes)
    dims = [len(x) for x in axes]
    indices = np.indices(dims)
    points = np.array([axis[index] for axis, index in zip(axes, indices)])
    points = points.reshape([ndims, np.prod(dims)])
    return points.T.copy()


def sample_mesh_interior(V, F, gridLength, dtype=np.float32):
    minPt = np.min(V, 0)
    maxPt = np.max(V, 0)
    q = voxel_points((np.arange(minPt[0], maxPt[0] + gridLength, gridLength, dtype=dtype),
                       np.arange(minPt[1], maxPt[1] + gridLength, gridLength, dtype=dtype),
                         np.arange(minPt[2], maxPt[2] + gridLength, gridLength, dtype=dtype)))
    #print(q)
    w = igl.fast_winding_number_for_meshes(V.astype(dtype), F, q)
    #print(w.shape)
    return q[w > 0.5]