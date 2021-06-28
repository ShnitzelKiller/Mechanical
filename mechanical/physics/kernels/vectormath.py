from numba import cuda, jit, prange, njit, int32, float32, void
import numpy as np

@njit
def quaternion_to_matrix(q):
    s = q[0]
    vx = q[1]
    vy = q[2]
    vz = q[3]
    vx2 = vx*vx
    vy2 = vy*vy
    vz2 = vz*vz
    return np.array([[1-2*vy2-2*vz2, 2*vx*vy-2*s*vz, 2*vx*vz+2*s*vy],
                     [2*vx*vy+2*s*vz, 1-2*vx2-2*vz2, 2*vy*vz-2*s*vx],
                     [2*vx*vz-2*s*vy, 2*vy*vz+2*s*vx, 1-2*vx2-2*vy2]])

@cuda.jit(device=True)
def hamilton_product_device(p, q, out):
    out0 = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    out1 = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    out2 = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    out3 = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    out[0] = out0
    out[1] = out1
    out[2] = out2
    out[3] = out3

@njit
def hamilton_product_host(p, q, out):
    out0 = p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3]
    out1 = p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2]
    out2 = p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1]
    out3 = p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0]
    out[0] = out0
    out[1] = out1
    out[2] = out2
    out[3] = out3


@cuda.jit(device=True)
def cross_product_device(p, q, out):
    a0 = p[1] * q[2] - p[2] * q[1]
    a1 = p[2] * q[0] - p[0] * q[2]
    a2 = p[0] * q[1] - p[1] * q[0]
    out[0] = a0
    out[1] = a1
    out[2] = a2

@njit
def cross_product_host(p, q, out):
    out0 = p[1] * q[2] - p[2] * q[1]
    out1 = p[2] * q[0] - p[0] * q[2]
    out2 = p[0] * q[1] - p[1] * q[0]
    out[0] = out0
    out[1] = out1
    out[2] = out2
    
@cuda.jit(device=True)
def transform_device(p, pose, out):
    """transform the point p by the pose [qw, qx, qy, qz, tx, ty, tz], where p and out are both represented quaternions with no scalar part"""
    #rotate
    hamilton_product_device(pose, p, out)
    pose[1] = -pose[1]
    pose[2] = -pose[2]
    pose[3] = -pose[3]
    hamilton_product_device(out, pose, out)
    pose[1] = -pose[1]
    pose[2] = -pose[2]
    pose[3] = -pose[3]
    #translate
    out[1] += pose[4]
    out[2] += pose[5]
    out[3] += pose[6]

@njit
def transform_host(p, pose, out):
    """transform the point p by the pose [qw, qx, qy, qz, tx, ty, tz], where p and out are both represented quaternions with no scalar part"""
    #rotate
    hamilton_product_host(pose, p, out)
    pose[1] = -pose[1]
    pose[2] = -pose[2]
    pose[3] = -pose[3]
    hamilton_product_host(out, pose, out)
    pose[1] = -pose[1]
    pose[2] = -pose[2]
    pose[3] = -pose[3]
    #translate
    out[1] += pose[4]
    out[2] += pose[5]
    out[3] += pose[6]