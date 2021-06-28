from numba import cuda, int32, int64, float32
from . import vectormath as vm
import math


TPB = 32
TPB_volume = 4
    
@cuda.jit
def populate_grid_kernel(points1, pose1, points2, pose2, grid, minPt, gridDim, cell_size):
    n1 = points1.shape[0]
    n2 = points2.shape[0]
    i = cuda.grid(1)
    transpoint = cuda.local.array(4, float32)
    pose = cuda.local.array(7, float32)
    transpoint[0] = 0
    if i<n1:
        for j in range(3):
            transpoint[j+1] = points1[i,j]
        for j in range(7):
            pose[j] = pose1[j]
        offset = 0
    elif i<n1 + n2:
        i -= n1
        for j in range(3):
            transpoint[j+1] = points2[i,j]
        for j in range(7):
            pose[j] = pose2[j]
        offset = cell_size
    else:
        return
    
    vm.transform_device(transpoint, pose, transpoint)
    px = int(math.floor((transpoint[1] - minPt[0])/gridDim))
    py = int(math.floor((transpoint[2] - minPt[1])/gridDim))
    pz = int(math.floor((transpoint[3] - minPt[2])/gridDim))
    nx = grid.shape[0]
    ny = grid.shape[1]
    nz = grid.shape[2] // (cell_size*2)
    if px >= 0 and px < nx and py >= 0 and py < ny and pz >= 0 and pz < nz:
        for j in range(cell_size):
            cellval = cuda.atomic.compare_and_swap(grid[px, py, pz*2*cell_size+offset+j:], -1, i)
            if cellval == -1:
                break
            #if grid[px, py, pz * 2 * cell_size + offset + j] < 0:
                #grid[px, py, pz * 2 * cell_size + offset + j] = i
            #    break
                
@cuda.jit
def empty_grid(grid):
    i, j, k = cuda.grid(3)
    nx = grid.shape[0]
    ny = grid.shape[1]
    nz = grid.shape[2]
    if i < nx and j < ny and k < nz:
        grid[i, j, k] = -1
                
@cuda.jit
def compute_forces(outForces1, points1, pose1, outForces2, points2, pose2, grid, minPt, gridDim, cell_size, springK, damping):
    radius = gridDim # maximum distance between interacting particles
    radius2 = radius * radius
    n1 = points1.shape[0]
    n2 = points2.shape[0]
    i = cuda.grid(1)
    transpoint = cuda.local.array(4, float32)
    pose = cuda.local.array(13, float32)
    otherpose = cuda.local.array(13, float32)
    transpoint[0] = 0
    if i<n1:
        for j in range(3):
            transpoint[j+1] = points1[i,j]
        for j in range(13):
            pose[j] = pose1[j]
            otherpose[j] = pose2[j]
        offset = 0
        otherOffset = cell_size
        otherpoints = points2
        outForces = outForces1
    elif i<n1 + n2:
        i -= n1
        for j in range(3):
            transpoint[j+1] = points2[i,j]
        for j in range(13):
            pose[j] = pose2[j]
            otherpose[j] = pose1[j]
        offset = cell_size
        otherOffset = 0
        otherpoints = points1
        outForces = outForces2
    else:
        return
    
    vm.transform_device(transpoint, pose, transpoint)
    point_cm = cuda.local.array(3, float32)
    for k in range(3):
        point_cm[k] = transpoint[k+1] - pose[k+4]
    px = int(math.floor((transpoint[1] - minPt[0])/gridDim))
    py = int(math.floor((transpoint[2] - minPt[1])/gridDim))
    pz = int(math.floor((transpoint[3] - minPt[2])/gridDim))
    nx = grid.shape[0]
    ny = grid.shape[1]
    nz = grid.shape[2] // (cell_size*2)
    if px >= 0 and px < nx and py >= 0 and py < ny and pz >= 0 and pz < nz:
        for j in range(cell_size):
            found = False
            if grid[px, py, pz * 2 * cell_size + offset + j] == i:
                found = True
                break
        if not found:
            return
        force = cuda.local.array(3, float32)
        tempforce = cuda.local.array(3, float32)
        torque = cuda.local.array(3, float32)
        temptorque = cuda.local.array(3, float32)
        omega = cuda.local.array(3, float32)
        otheromega = cuda.local.array(3, float32)
        for l in range(3):
            force[l] = 0
            torque[l] = 0
            omega[l] = pose[l+7]
            otheromega[l] = otherpose[l+7]
                    
        vel = cuda.local.array(3, float32)
        vm.cross_product_device(omega, point_cm, vel)
        
        otherpoint = cuda.local.array(4, float32)
        otherpoint_cm = cuda.local.array(3, float32)
        othervel = cuda.local.array(3, float32)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    for j in range(cell_size):
                        ox = px+dx
                        oy = py+dy
                        oz = pz+dz
                        if ox < 0 or ox >= nx or oy < 0 or oy >= ny or oz < 0 or oz >= nz:
                            continue
                        otherIndex = grid[ox,oy,oz*2*cell_size+otherOffset+j]
                        if otherIndex >= 0:
                            otherpoint[0] = 0
                            for k in range(3):
                                otherpoint[k+1] = otherpoints[otherIndex, k]
                            vm.transform_device(otherpoint, otherpose, otherpoint)
                            for k in range(3):
                                otherpoint_cm[k] = otherpoint[k+1] - otherpose[k+4]
                                #otherpoint holds the displacement from otherpoint to this point
                                otherpoint[k+1] = transpoint[k+1] - otherpoint[k+1]                                
                            
                            dist2 = otherpoint[1]*otherpoint[1]+otherpoint[2]*otherpoint[2]+otherpoint[3]*otherpoint[3]
                            dist = math.sqrt(dist2)
                            #print(dist)
                            #spring force
                            if dist2 > 0 and dist2 < radius2:
                                    springfac = springK * (radius - dist)/dist
                                    for k in range(3):
                                        tempforce[k] = springfac * otherpoint[k+1]
                                        #print('realvalue:',springfac * otherpoint[k+1],'got:',tempforce[k])
                                        #force[k] += tempforce[k]
                                    vm.cross_product_device(point_cm, tempforce, temptorque)
                                    for k in range(3):
                                        force[k] += tempforce[k]
                                        torque[k] += temptorque[k]
                                        
                            #damping force
                            #compute relative velocity of points
                            vm.cross_product_device(otheromega, otherpoint_cm, othervel)
                            for k in range(3):
                                tempforce[k] = damping * (othervel[k] - vel[k])
                            vm.cross_product_device(point_cm, tempforce, temptorque)
                            for k in range(3):
                                force[k] += tempforce[k]
                                torque[k] += temptorque[k]
                            
        outForces[i, 0] = force[0]
        outForces[i, 1] = force[1]
        outForces[i, 2] = force[2]
        outForces[i, 3] = torque[0]
        outForces[i, 4] = torque[1]
        outForces[i, 5] = torque[2]


@cuda.jit
def reduce_forces_torques(d_accum, d_f):
    i = cuda.grid(1)
    tIdx = cuda.threadIdx.x
    n = d_f.shape[0]
    sh_w = cuda.shared.array((TPB, 6), dtype=float32)
    if i < n:
        for j in range(6):
            sh_w[tIdx, j] = d_f[i, j]
    if i < 6:
        d_accum[i] = 0
    cuda.syncthreads()
    if tIdx < 6:
        component_sum = 0.0
        for j in range(cuda.blockDim.x):
            component_sum += sh_w[j, tIdx]
        cuda.atomic.add(d_accum, tIdx, component_sum)
