import numpy as np
import numpy.linalg as linalg
import math
#from numba import jit

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def slippage(points, normals, condition_number=150, normalize=True):
    """return the list of degrees of freedom as tuples of 3-vectors in the form (euler rotation, translation)"""
    if normalize:
        cm = np.mean(points, 0)
        points = points - cm
        avgscale = np.mean(linalg.norm(points, axis=1))
        points /= avgscale
    else:
        cm = np.zeros(3)
        avgscale = 1

    C = np.zeros([6, 6], np.float)
    n = points.shape[0]
    for i in range(n):
        cross = np.cross(points[i,:], normals[i,:])
        block00 = np.outer(cross, cross)
        block01 = np.outer(cross, normals[i,:])
        block11 = np.outer(normals[i,:], normals[i,:])
        C[:3, :3] += block00
        C[:3, 3:] += block01
        C[3:, :3] += block01.T
        C[3:, 3:] += block11

    w, v = linalg.eigh(C)
    #print(w)
    dofs = []
    for i in range(6):
        if w[i] == 0 or w[5]/w[i] > condition_number:
            rigidMotion = np.hstack((v[:3, i], v[3:, i] * avgscale - np.cross(v[:3, i], cm)))
            dofs.append(normalized(rigidMotion))
    return dofs
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import meshio
    import utils
    mesh = meshio.read("data/halfpipe.obj")
    #mesh = meshio.read("../tests/lego/part_5_contactonly_conservative.obj")
    print('vertices:', mesh.points.shape[0])
    #segmentMesh(mesh.points, mesh.cells_dict['triangle'], 0.5)
    points, normals = utils.sample_points(mesh, 10000)
    dofs = slippage(points, normals, normalize=True)
    for i, dof in enumerate(dofs):
        print('dof',i,':',dof)
