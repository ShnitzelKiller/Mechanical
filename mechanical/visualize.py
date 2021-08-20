import numpy as np
import meshplot as mp

def add_axis(plot, center, x_dir, y_dir, z_dir, scale=1):
    V = np.array([center, center+x_dir * scale, center+y_dir * scale, center+z_dir * scale])
    Ex = np.array([[0,1]])
    Ey = np.array([[0,2]])
    Ez = np.array([[0,3]])
    plot.add_edges(V, Ex, shading={'line_color':'red','line_width':5})
    plot.add_edges(V, Ey, shading={'line_color':'green','line_width':5})
    plot.add_edges(V, Ez, shading={'line_color':'blue','line_width':5})


def hsv2rgb(h, s, v):
    c = v * s
    x = c * (1 - abs((h/60) % 2 - 1))
    m = v - c
    if (h >= 0 and h < 60):
        rp = c
        gp = x
        bp = 0
    elif (h >= 60 and h < 120):
        rp = x
        gp = c
        bp = 0
    elif (h >= 120 and h < 180):
        rp = 0
        gp = c
        bp = x
    elif (h >= 180 and h < 240):
        rp = 0
        gp = x
        bp = c
    elif (h >= 240 and h < 300):
        rp = x
        gp = 0
        bp = c
    else:
        rp = c
        gp = 0
        bp = x
    r = rp + m
    g = gp + m
    b = bp + m
    return r, g, b


def get_color(index, total):
    h = index / total * 360
    intensity = 1
    rf, gf, bf = hsv2rgb(h, 1.0, 1.0)
    return rf * intensity, gf * intensity, bf * intensity

def occ_to_mesh(g):
    V = g[1].V.copy()
    F = g[1].F
    if V.shape[0] > 0:
        V = (g[0][:3,:3] @ V.T).T + g[0][:3,3]
    return V, F



def inspect(geo, mates, p=None, wireframe=False):
    maxdim = max([max(geo[i][1].V.max(0)-geo[i][1].V.min(0)) for i in geo if geo[i][1].V.shape[0] > 0])
    num_parts = len(geo)
    part_index = 0
    for i in geo:
        g = geo[i]
        V, F = occ_to_mesh(g)
        if V.shape[0] == 0:
            continue
        colors = np.array(get_color(part_index, num_parts))
        part_index += 1
        try:
            plot
        except NameError:
            if wireframe:
                if p is None:
                    plot = emptyplot()
                else:
                    plot = p
                plot.reset()
                plot.add_edges(V, F, shading={'line_color': 'red'})
            else:
                plot = mp.plot(V, F, c=colors, return_plot=True, plot=p)
        else:
            if wireframe:
                plot.add_edges(V, F, shading={'line_color': 'red'})
            else:
                plot.add_mesh(V, F, c=colors)

    for mate in mates:
        if len(mate.matedEntities)==2:
            for mated in mate.matedEntities:
                tf = geo[mated[0]][0]
                newaxes = tf[:3, :3] @ mated[1][1]
                neworigin = tf[:3,:3] @ mated[1][0] + tf[:3,3]
                add_axis(plot, neworigin, newaxes[:,0], newaxes[:,1], newaxes[:,2], scale=maxdim/10)
    return plot

def emptyplot():
    vd = np.array([[0, 0, 1],
              [0, 1, 0],
              [0, 1, 1]], dtype=np.float64)
    fd = np.array([[0, 1, 2],[2, 1, 0]])
    return mp.plot(vd, fd)