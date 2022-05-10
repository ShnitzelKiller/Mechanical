import wildmeshing as wm
import json
import igl
from scipy.spatial.transform import Rotation as R
from mechanical.geometry import sample_surface_points
import numpy as np

import meshplot as mp
mp.website()

def min_signed_distance_points_mesh(pc, V, F):
    S, I, C = igl.signed_distance(pc, V, F)
    return S.min()

def min_signed_distance(meshes, samples=200, include_vertices=False, debug_suffix=None, debug_plot_name=None):
    points, indices = sample_surface_points(meshes[0][0], meshes[0][1], samples)
    if include_vertices:
        points = np.vstack([points, meshes[0][0]])
    S, I, C = igl.signed_distance(points, meshes[1][0], meshes[1][1])
    if debug_plot_name is not None:
        fname = debug_plot_name + f'_{debug_suffix}.html'
        maxdim = (meshes[1][0].max(0) - meshes[1][0].min(0)).max()
        p = mp.plot(meshes[1][0], meshes[1][1])
        p.add_points(points, shading={'point_size': maxdim/10})
        p.save(fname)
    return S.min()



def min_signed_distance_symmetric(meshes, samples=200, include_vertices=False, debug_plot_name=None):
    return min(min_signed_distance(meshes, samples=samples, include_vertices=include_vertices, debug_suffix=0, debug_plot_name=debug_plot_name), min_signed_distance(list(reversed(meshes)), samples=samples, include_vertices=include_vertices, debug_suffix=1, debug_plot_name=debug_plot_name))



def motion_meshes(meshes, axis, origin, motion_type, displacement):
    """
    given a pair of meshes, return another pair of meshes displaced relative to each other based on motion type
    """

    meshes = sorted(meshes, key=lambda x: x[0].shape[0])
    #p = mp.plot(*meshes[0])
    if motion_type == 'ROTATE':
        r = R.from_rotvec(axis * displacement)
        mat = r.as_matrix()
        meshes[0] = ((mat @ (meshes[0][0] - origin).T).T + origin, meshes[0][1])
    elif motion_type == 'SLIDE':
        meshes[0] = (meshes[0][0] + axis * displacement, meshes[0][1])
    else:
        raise ValueError
    return meshes


def displaced_min_signed_distance(meshes, axis, origin=None, motion_type='SLIDE', samples=100, displacement=0.01, include_vertices=False, debug_plot_name=None):
    """
    compute the amount of overlap between meshes after motion of type [ROTATE|SLIDE] is applied along the
    specified axis.
    """
    meshes = motion_meshes(meshes, axis, origin, motion_type, displacement=displacement)
    #p.add_mesh(*meshes[0])
    #p.add_mesh(*meshes[1])
    #if motion_type == 'ROTATE':
    #    p.add_points(origin[np.newaxis,:])
    #p.save(f'debugvis_{motion_type}.html')
    
    return min_signed_distance_symmetric(meshes, samples=samples, include_vertices=include_vertices, debug_plot_name=debug_plot_name)


def mesh_overlap_volume(meshes, axis, quality=100):
    tet = wm.Tetrahedralizer(stop_quality=100)
    tet.set_meshes(list(zip(*meshes)))
    tet.tetrahedralize()
    V, F = tet.get_tet_mesh_from_csg(json.dumps({"operation":"intersection", "left": 0, "right": 1}))
    intersection = igl.volume(V, F).sum()
    smallest = min(igl.volume(*tet.get_tet_mesh_from_csg(json.dumps({"operation":"union", "left": i, "right": i}))) for i in range(2))
    return intersection, smallest


if __name__ == '__main__':
    import numpy as np
    V, _, _, F, _, _ = igl.read_obj('data/meshes/cube.obj')
    mesh1 = (V, F)
    mesh2 = (V + np.array([2, 0, 0]), F)
    meshes = [mesh1, mesh2]
    print('default overlap:',min_signed_distance_symmetric(meshes))
    print('displaced overlap:',displaced_min_signed_distance(meshes, np.array([1,0,0]), motion_type='SLIDE', displacement=0.2))
    print('displaced overlap2:',displaced_min_signed_distance(meshes, np.array([1,0,0]), motion_type='SLIDE', displacement=-0.2))
    print('rotated overlap:',displaced_min_signed_distance(meshes, np.array([1,0,0]), np.array([1,0,0]), motion_type='ROTATE', displacement=-np.pi/8))
    print('rotated overlap penetrating:',displaced_min_signed_distance(meshes, np.array([0,1,0]), np.array([1,0,0]), motion_type='ROTATE', displacement=-np.pi/8))