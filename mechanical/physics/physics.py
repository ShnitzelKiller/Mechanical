import numpy as np
import numpy.linalg as la
import time, math
from numba import cuda, jit, prange, njit
import physics.kernels.cuda_kernels as ck
import physics.kernels.vectormath as vm
import geometry.sampling as sampling

@njit (parallel=True)
def compute_inertia_tensor(points, pointmass):
    inertia_tensor = np.zeros((3, 3), dtype=points.dtype)
    N = points.shape[0]
    for i in prange(N):
        pointCM = points[i,:]
        moments = pointCM * pointCM
        inertia_tensor[0, 0] += moments[1] + moments[2]
        inertia_tensor[1, 1] += moments[0] + moments[2]
        inertia_tensor[2, 2] += moments[0] + moments[1]
        inertia_tensor[0, 1] = pointCM[0] * pointCM[1]
        inertia_tensor[0, 2] = pointCM[0] * pointCM[2]
        inertia_tensor[1, 2] = pointCM[1] * pointCM[2]
    inertia_tensor *= pointmass
    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]
    return inertia_tensor


class PhysObject:
    def __init__(self, V, F, gridDim, translation, rotation=np.array([1, 0, 0, 0], np.float32), density=1, angular_velocity=np.zeros(3, np.float32), velocity=np.zeros(3, np.float32)):
        self.points = sampling.sample_mesh_interior(V, F, gridDim)
        pointmass = density * gridDim ** 3
        cm = np.mean(self.points, 0)
        self.points -= cm
        self.mass = pointmass * self.points.shape[0]
        self.inertia_tensor = compute_inertia_tensor(self.points, pointmass)
        self.inertia_tensor_inv = la.inv(self.inertia_tensor)

        self.translation = translation
        self.rotation = rotation
        self.set_angular_velocity(angular_velocity)
        self.set_velocity(velocity)

    def get_world_space_inertia_tensor(self):
        R = vm.quaternion_to_matrix(self.rotation)
        return R @ self.inertia_tensor @ R.T

    def get_world_space_inertia_tensor_inverse(self):
        R = vm.quaternion_to_matrix(self.rotation)
        return R @ self.inertia_tensor_inv @ R.T

    def get_angular_velocity(self):
        Iinv = self.get_world_space_inertia_tensor_inverse()
        return Iinv @ self.angular_momentum

    def set_angular_velocity(self, angular_velocity):
        I = self.get_world_space_inertia_tensor()
        self.angular_momentum = I @ angular_velocity

    def get_velocity(self):
        return self.momentum / self.mass

    def set_velocity(self, velocity):
        self.momentum = velocity * self.mass

    def step(self, force, torque, dt):
        self.momentum += force * dt
        self.angular_momentum += torque * dt
        
        v = self.get_velocity()
        omega = self.get_angular_velocity()
        
        qdot = np.array([0, omega[0], omega[1], omega[2]], np.float32)
        vm.hamilton_product_host(qdot, self.rotation, qdot)
        qdot *= 0.5
        self.rotation += qdot * dt
        self.rotation /= la.norm(self.rotation)
        self.translation += v * dt



class Simulator:
    def __init__(self, phys_obj1, phys_obj2, bounds_min, bounds_max, grid_length, spring_constant, damping, cell_size=2):
        self.objects = [phys_obj1, phys_obj2]
        self.grid_length = grid_length
        self.bounds_min = bounds_min
        self.spring_constant = spring_constant
        self.damping = damping
        self.cell_size = cell_size

        res = np.ceil(((bounds_max - bounds_min) / grid_length)).astype(np.int32)
        res[2] *= cell_size * 2
        
        start = time.time()
        self.grid_d = cuda.to_device(np.full(res, -1, dtype=np.int32))
        #gpu state: [rotation, translation, angular velocity, velocity]
        self.states_d = [cuda.device_array(13, np.float32) for obj in self.objects]
        self.points_d = [cuda.to_device(obj.points) for obj in self.objects]
        self.forces_d = [cuda.device_array((obj.points.shape[0], 6), np.float32) for obj in self.objects] #[force, torque]
        self.forces_accum_d = [cuda.device_array(6, np.float32) for obj in self.objects]
        self.copy_state_to_device()
        self.bounds_min_d = cuda.to_device(bounds_min)
        end = time.time()
        print('copied data to GPU in',end-start)
        
        m = np.sum([obj.points.shape[0] for obj in self.objects])
        TPB = ck.TPB
        self.cuda_point_grid_dims = (m + TPB - 1) // TPB
        self.cuda_point_block_dims = TPB

        TPCB = ck.TPB_volume
        self.cuda_volume_grid_dims = ((res[0] + TPCB - 1) // TPCB, (res[1] + TPCB - 1) // TPCB, (res[2] + TPCB - 1) // TPCB)
        self.cuda_volume_block_dims = (TPCB, TPCB, TPCB)
        self.num_points = m
        
        self.allstates = []
        self.t = 0.0
        self.record_state()

    def copy_state_to_device(self):
        for i in range(2):
            state = np.zeros((13,), dtype=np.float32)
            state[0:4] = self.objects[i].rotation
            state[4:7] = self.objects[i].translation
            state[7:10] = self.objects[i].get_angular_velocity()
            state[10:13] = self.objects[i].get_velocity()
            self.states_d[i].copy_to_device(state)

    def record_state(self):
        self.allstates.append((self.t, [(obj.rotation.copy(), obj.translation.copy()) for obj in self.objects]))

    def step(self, dt):
        ck.empty_grid[self.cuda_volume_grid_dims, self.cuda_volume_block_dims](self.grid_d)
        ck.populate_grid_kernel[self.cuda_point_grid_dims, self.cuda_point_block_dims](self.points_d[0], self.states_d[0], self.points_d[1], self.states_d[1], self.grid_d, self.bounds_min_d, self.grid_length, self.cell_size)
        ck.compute_forces[self.cuda_point_grid_dims, self.cuda_point_block_dims](self.forces_d[0], self.points_d[0], self.states_d[0], self.forces_d[1], self.points_d[1], self.states_d[1], self.grid_d, self.bounds_min_d, self.grid_length, self.cell_size, self.spring_constant, self.damping)
        
        for j in range(2):
            cuda_grid_dim = (self.objects[j].points.shape[0] + self.cuda_point_block_dims - 1) // self.cuda_point_block_dims
            ck.reduce_forces_torques[cuda_grid_dim, self.cuda_point_block_dims](self.forces_accum_d[j], self.forces_d[j])
            forces_accum = self.forces_accum_d[j].copy_to_host()
            
            self.objects[j].step(forces_accum[:3], forces_accum[3:], dt)

        self.copy_state_to_device()
        self.t += dt
        self.record_state()

    def integrate(self, dt, steps):
        for i in range(steps):
            self.step(dt)

