import wildmeshing as wm
import json
import igl
from scipy.spatial.transform import Rotation as R

def mesh_overlap_volume(meshes, axis, quality=100):
    tet = wm.Tetrahedralizer(stop_quality=100)
    tet.set_meshes(list(zip(*meshes)))
    tet.tetrahedralize()
    V, F = tet.get_tet_mesh_from_csg(json.dumps({"operation":"intersection", "left": 0, "right": 1}))
    intersection = igl.volume(V, F).sum()
    smallest = min(igl.volume(*tet.get_tet_mesh_from_csg(json.dumps({"operation":"union", "left": i, "right": i}))) for i in range(2))
    return intersection, smallest


def displaced_volume(meshes, axis, motion_type='ROTATION', quality=100, displacement=0.01):
    """
    compute the amount of overlap between meshes after motion of type [ROTATION|SLIDER] is applied along the
    specified axis.
    """
    if motion_type == 'ROTATION':
        r = R.from_rotvec(axis * displacement)