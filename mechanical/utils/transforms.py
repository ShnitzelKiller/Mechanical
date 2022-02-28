from numba import jit, njit
import numpy as np

@njit
def compute_basis(norm):
    up = np.zeros(3)
    ind = 0
    mincomp = abs(norm[0])
    for i in range(1,3):
        abscomp = abs(norm[i])
        if abscomp < mincomp:
            ind = i
            mincomp = abscomp
        
    up[ind] = 1
    basis = np.empty((2, 3), dtype=norm.dtype)
    basis[1] = np.cross(norm, up)
    basis[1] /= LA.norm(basis[1])
    basis[0] = np.cross(basis[1], norm)
    return basis

@njit
def project_to_plane(point, normal):
    norm2 = np.dot(normal, normal)
    projdist = np.dot(normal, point)/norm2
    return point - projdist * normal

@njit
def homogenize_sign(vec):
    maxdim = -1
    maxabs = 0
    for i in range(len(vec)):
        vecabs = abs(vec[i])
        if vecabs > maxabs:
            maxabs = vecabs
            maxdim = i
    if vec[maxdim] < 0:
        return -vec, True
    else:
        return vec.copy(), False

@njit
def vec3_tuple(vec):
    return (vec[0],vec[1],vec[2])

@njit
def tuple_vec3(vec,tup):
    vec[0] = tup[0]
    vec[1] = tup[1]
    vec[2] = tup[2]
    return vec

@njit
def homogenize_frame(frame, z_flip_only=True):
    #homogenized x and y can only differ by a swap if the z axes of two orthogonally rotated frames differ only by sign
    x, _ = homogenize_sign(frame[:,0])
    y, _ = homogenize_sign(frame[:,1])
    z, _ = homogenize_sign(frame[:,2])

    if z_flip_only:
        vectors = sorted([vec3_tuple(x),vec3_tuple(y)])
        z_final = z
        x_final = tuple_vec3(x,vectors[0])
    else:
        vectors = sorted([vec3_tuple(x),vec3_tuple(y),vec3_tuple(z)])
        z_final = tuple_vec3(z,vectors[0])
        x_final = tuple_vec3(x,vectors[1])

    y_final = np.cross(z_final,x_final)

    frame_out = np.empty_like(frame)
    frame_out[:,0] = x_final
    frame_out[:,1] = y_final
    frame_out[:,2] = z_final
    return frame_out

def apply_transform(tf, v, is_points=True):
    "Apply the 4-matrix `tf` to a vector or column-wise matrix of vectors. If is_points, also add the translation."
    v_trans = tf[:3,:3] @ v
    if is_points:
        if v.ndim==1:
            v_trans += tf[:3,3]
        elif v.ndim==2:
            v_trans += tf[:3,3,np.newaxis]
    return v_trans

def cs_to_origin_frame(cs):
    origin = cs[:,3]
    if origin[3] != 1:
        origin /= origin[3]
    return origin[:3], cs[:3,:3]

def cs_from_origin_frame(origin, frame):
    cs = np.identity(4, np.float64)
    cs[:3,:3] = frame
    cs[:3,3] = origin
    return cs