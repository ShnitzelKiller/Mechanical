import numpy as np
import math
import random

def sample_points(mesh, n):
    points = np.zeros([n, 3], np.float)
    areas = np.zeros(mesh.cells_dict['triangle'].shape[0], np.float)
    trinormals = np.zeros([n, 3], np.float)
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

