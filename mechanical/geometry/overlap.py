import wildmeshing as wm
import json
import igl
from scipy.spatial.transform import Rotation as R
from mechanical.geometry import sample_surface_points
import numpy as np

#import meshplot as mp
#mp.offline()


def min_signed_distance(meshes, samples=200):
    points, indices = sample_surface_points(meshes[0][0], meshes[0][1], samples)
    S, I, C = igl.signed_distance(points, meshes[1][0], meshes[1][1])
    return S.min()



def min_signed_distance_symmetric(meshes, samples=200):
    return min(min_signed_distance(meshes, samples=samples), min_signed_distance(list(reversed(meshes)), samples=samples))



def mesh_overlap_volume(meshes, axis, quality=100):
    tet = wm.Tetrahedralizer(stop_quality=100)
    tet.set_meshes(list(zip(*meshes)))
    tet.tetrahedralize()
    V, F = tet.get_tet_mesh_from_csg(json.dumps({"operation":"intersection", "left": 0, "right": 1}))
    intersection = igl.volume(V, F).sum()
    smallest = min(igl.volume(*tet.get_tet_mesh_from_csg(json.dumps({"operation":"union", "left": i, "right": i}))) for i in range(2))
    return intersection, smallest


def displaced_min_signed_distance(meshes, axis, origin=None, motion_type='SLIDE', samples=100, displacement=0.01):
    """
    compute the amount of overlap between meshes after motion of type [ROTATE|SLIDE] is applied along the
    specified axis.
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
    #p.add_mesh(*meshes[0])
    #p.add_mesh(*meshes[1])
    #if motion_type == 'ROTATE':
    #    p.add_points(origin[np.newaxis,:])
    #p.save(f'debugvis_{motion_type}.html')
    
    return min_signed_distance_symmetric(meshes, samples=samples)


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