import numpy as np
import meshplot as mp

def make_edge_ring(N):
    Ec = np.empty([N, 2], np.int64)
    Ec[:,0] = np.arange(N)
    Ec[:-1,1] = Ec[1:,0]
    Ec[-1,1] = Ec[0,0]
    return Ec

def add_circle(plot, center, u, v, scale, color, N=20):
    circ = np.empty([N, 3], dtype=np.float64)
    for i in range(N):
        ang = i/N * np.pi * 2
        circ[i,:] = center + scale * (u * np.cos(ang) + v * np.sin(ang))
    Ec = make_edge_ring(N)
    plot.add_edges(circ, Ec, shading={'line_color':color,'line_width':5})

def add_square(plot, center, u, v, scale, color):
    square = np.empty([4, 3], dtype=np.float64)
    square[0,:] = center
    square[1,:] = center + u * scale
    square[2,:] = center + u * scale + v * scale
    square[3,:] = center + v * scale
    Ec = make_edge_ring(4)
    plot.add_edges(square, Ec, shading={'line_color':color,'line_width':5})

def add_axis(plot, center, x_dir, y_dir, z_dir, scale=1, mate_type=None):
    V = np.array([center, center+x_dir * scale, center+y_dir * scale, center+z_dir * scale])
    Ex = np.array([[0,1]])
    Ey = np.array([[0,2]])
    Ez = np.array([[0,3]])
    plot.add_edges(V, Ex, shading={'line_color':'red','line_width':5})
    plot.add_edges(V, Ey, shading={'line_color':'green','line_width':5})
    plot.add_edges(V, Ez, shading={'line_color':'blue','line_width':5})
    N = 20
    if mate_type == 'REVOLUTE':
        add_circle(plot, center, x_dir, y_dir, scale, 'blue', N)
    elif mate_type == 'BALL':
        add_circle(plot, center, x_dir, y_dir, scale, 'blue', N)
        add_circle(plot, center, y_dir, z_dir, scale, 'red', N)
        add_circle(plot, center, z_dir, x_dir, scale, 'green', N)
    elif mate_type == 'CYLINDRICAL':
        add_circle(plot, center, x_dir, y_dir, scale, 'cyan', N)
        add_circle(plot, center+z_dir*scale/2, x_dir, y_dir, scale, 'cyan', N)
    elif mate_type == 'FASTENED':
        add_square(plot, center, x_dir, y_dir, scale/2, 'blue')
        add_square(plot, center, y_dir, z_dir, scale/2, 'red')
        add_square(plot, center, z_dir, x_dir, scale/2, 'blue')
    elif mate_type == 'SLIDER':
        add_square(plot, center, x_dir, y_dir, scale/2, 'cyan')
        add_square(plot, center+z_dir*scale/2, x_dir, y_dir, scale/2, 'cyan')
    elif mate_type == 'PLANAR':
        add_square(plot, center, x_dir, y_dir, scale, 'blue')
    elif mate_type == 'PARALLEL':
        add_square(plot, center, x_dir, y_dir, scale, 'orange')


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


def plot_mate(geo, mate, p=None, wireframe=False):
    #badOccs = [k for k in geo if geo[k][1] is None or geo[k][1].V.shape[0] == 0]
    #print('mated parts:',me[0][0],me[1][0])
    #if me[0][0] in badOccs or me[1][0] in badOccs:
    #    print('invalid parts in mate')
    #    return None
    me = mate.matedEntities
    occs = [geo[me[i][0]] for i in range(2)]
    maxdim = max([max(geo[i[0]][1].V.max(0)-geo[i[0]][1].V.min(0)) for i in me if geo[i[0]][1].V.shape[0] > 0])

    meshes = [occ_to_mesh(occ) for occ in occs]
    if wireframe:
        p.reset()
        p.add_edges(meshes[0][0], meshes[0][1], shading={'line_color': 'red'})
        p.add_edges(meshes[1][0], meshes[1][1], shading={'line_color': 'blue'})
    else:
        mp.plot(meshes[0][0], meshes[0][1],c=np.array([1, 0, 0]), plot=p)
        p.add_mesh(meshes[1][0], meshes[1][1],c=np.array([0, 0, 1]))

    for i in range(2):
        tf = occs[i][0]
        #print(f'matedCS origin {i}: {me[i][1][0]}')
        newaxes = tf[:3, :3] @ me[i][1][1]
        neworigin = tf[:3,:3] @ me[i][1][0] + tf[:3,3]
        #print(f'transform {i}: {tf}')
        add_axis(p, neworigin, newaxes[:,0], newaxes[:,1], newaxes[:,2], scale=maxdim/2, mate_type=mate.type)
        
        mcf_origins = []
        for mc in occs[i][1].all_mate_connectors:
            mcf = mc.get_coordinate_system()
            mcf_origins.append(tf[:3,:3] @ mcf[:3,3] + tf[:3,3])
        mcf_origins = np.array(mcf_origins)
        p.add_points(mcf_origins,shading={'point_size':maxdim/30, 'point_color': 'cyan' if i == 0 else 'pink'})
        return p


def plot_assembly(geo, mates, p=None, wireframe=False, show_parts=True):
    maxdim = max([max(geo[i][1].V.max(0)-geo[i][1].V.min(0)) for i in geo if geo[i][1].V.shape[0] > 0])
    num_parts = len(geo)
    part_index = 0
    geo_colors = dict()
    for i in geo:
        g = geo[i]
        if g[1] is None:
            continue
        V, F = occ_to_mesh(g)
        if V.shape[0] == 0:
            continue
        colors = np.array(get_color(part_index, num_parts))
        geo_colors[i] = colors
        part_index += 1
        if show_parts:
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
            mcf_origins = []
            tf = g[0]
            for mc in g[1].all_mate_connectors:
                mcf = mc.get_coordinate_system()
                mcf_origins.append(tf[:3,:3] @ mcf[:3,3] + tf[:3,3])
            mcf_origins = np.array(mcf_origins)
            plot.add_points(mcf_origins,shading={'point_size':maxdim/15})

    if not show_parts:
        plot = emptyplot() if p is None else p
        plot.reset()

    mate_origins = []
    mate_colors = []
    for i,mate in enumerate(mates):
        if len(mate.matedEntities)==2:
            for mated in mate.matedEntities:
                tf = geo[mated[0]][0]
                newaxes = tf[:3, :3] @ mated[1][1]
                neworigin = tf[:3,:3] @ mated[1][0] + tf[:3,3]
                add_axis(plot, neworigin, newaxes[:,0], newaxes[:,1], newaxes[:,2], scale=maxdim/10, mate_type=mate.type)
                mate_origins.append(neworigin)
            if mate.matedEntities[0][0] in geo_colors and mate.matedEntities[1][0] in geo_colors:
                mate_color = 0.5 * (geo_colors[mate.matedEntities[0][0]] + geo_colors[mate.matedEntities[1][0]])
            else:
                mate_color = np.zeros(3)
            mate_colors.append(mate_color)
            mate_colors.append(mate_color)
    mate_origins = np.vstack(mate_origins)
    mate_colors = np.vstack(mate_colors)
    plot.add_points(mate_origins, c=mate_colors, shading={'point_size':maxdim/10})
    return plot

def emptyplot():
    vd = np.array([[0, 0, 1],
              [0, 1, 0],
              [0, 1, 1]], dtype=np.float64)
    fd = np.array([[0, 1, 2],[2, 1, 0]])
    return mp.plot(vd, fd)