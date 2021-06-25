#In[0]:

from meshplot import plot, interact
import ipywidgets
import numpy as np
import physics.vectormath as vm
import igl
import physics.physics as phys
import math

def joinmeshes(v1, f1, v2, f2):
    return np.vstack((v1, v2)), np.vstack((f1, f2 + v1.shape[0]))



def display_animation(v1, f1, v2, f2, states):
    fullv, fullf = joinmeshes(v1, f1, v2, f2)
    p = plot(fullv, fullf)
    n = len(states)
    @interact(t=ipywidgets.IntSlider(min=0, max=n-1, step=1))
    def ff(t):
        vs = []
        for j in range(2):
            v = (v1, v2)[j]
            R = vm.quaternion_to_matrix(states[t][j][0])
            x = states[t][j][1]
            newv = (R @ v.T).T + x
            vs.append(newv)
        fullv = np.vstack(vs)
        p.update_object(vertices=fullv)


#In[1]:
# ### off-center collision

v, f = igl.read_triangle_mesh('data/cylinder3.obj')
grid_len = 0.02
density = 1
obj1 = phys.PhysObject(v, f, grid_len, translation=np.array([0, 0, 0], np.float32), density=density)
obj2 = phys.PhysObject(v, f, grid_len, translation=np.array([0.5, 2.1, 0], np.float32),
                    rotation=np.array([math.sqrt(2)/2, 0, math.sqrt(2)/2, 0], np.float32), density=density,
                    velocity=np.array([0, -10, 0], np.float32))

sim = phys.Simulator(obj1, obj2, np.array([-1.1, -1.1, -1.1]), np.array([1.1, 3.1, 1.1]), 0.02, 100, 0)
sim.integrate(0.001, 1000)
print(sim.allstates)
#display_animation(v, f, v, f, sim.allstates)
