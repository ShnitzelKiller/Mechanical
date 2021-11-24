import numpy as np
from mechanical.utils import get_color
import pythreejs as p3s
import ipywidgets as wg


def make_edge_ring(N):
    Ec = np.empty([N, 2], np.int64)
    Ec[:,0] = np.arange(N)
    Ec[:-1,1] = Ec[1:,0]
    Ec[-1,1] = Ec[0,0]
    return Ec


def make_segments_geometry(points, color, width=5):
    g2 = p3s.LineSegmentsGeometry(positions=np.roll(np.repeat(points, 2, axis=0), 1, axis=0).reshape((-1, 2, 3)))
    return p3s.LineSegments2(geometry=g2, material=p3s.LineMaterial(linewidth=width, color=color))


def add_circle(lst, center, u, v, scale, color, N=20):
    circ = np.empty([N, 3], dtype=np.float64)
    for i in range(N):
        ang = i/N * np.pi * 2
        circ[i,:] = center + scale * (u * np.cos(ang) + v * np.sin(ang))
    
    lst.append(make_segments_geometry(circ, color))


def add_square(lst, center, u, v, scale, color):
    square = np.empty([4, 3], dtype=np.float64)
    square[0,:] = center
    square[1,:] = center + u * scale
    square[2,:] = center + u * scale + v * scale
    square[3,:] = center + v * scale
    lst.append(make_segments_geometry(square, color))

def add_axis(center, x_dir, y_dir, z_dir, scale=1, mate_type=None):
    N = 20
    objects = []
    if mate_type == 'REVOLUTE':
        add_circle(objects, center, x_dir, y_dir, scale, 'blue', N)
    elif mate_type == 'BALL':
        add_circle(objects, center, x_dir, y_dir, scale, 'blue', N)
        add_circle(objects, center, y_dir, z_dir, scale, 'red', N)
        add_circle(objects, center, z_dir, x_dir, scale, 'green', N)
    elif mate_type == 'CYLINDRICAL':
        add_circle(objects, center, x_dir, y_dir, scale, 'cyan', N)
        add_circle(objects, center+z_dir*scale/2, x_dir, y_dir, scale, 'cyan', N)
    elif mate_type == 'FASTENED':
        add_square(objects, center, x_dir, y_dir, scale/2, 'blue')
        add_square(objects, center, y_dir, z_dir, scale/2, 'red')
        add_square(objects, center, z_dir, x_dir, scale/2, 'blue')
    elif mate_type == 'SLIDER':
        add_square(objects, center, x_dir, y_dir, scale/2, 'cyan')
        add_square(objects, center+z_dir*scale/2, x_dir, y_dir, scale/2, 'cyan')
    elif mate_type == 'PLANAR':
        add_square(objects, center, x_dir, y_dir, scale, 'blue')
    elif mate_type == 'PARALLEL':
        add_square(objects, center, x_dir, y_dir, scale, 'orange')
    elif mate_type == 'PIN_SLOT':
        add_square(objects, center, y_dir, z_dir, scale/2, 'purple')
        add_square(objects, center+x_dir*scale/2, y_dir, z_dir, scale/2, 'purple')
    return objects


def occ_to_mesh(g, tf=None):
    if tf is None:
        tf = g[0]
    V = g[1].V.copy()
    F = g[1].F
    if V.shape[0] > 0:
        V = (tf[:3,:3] @ V.T).T + tf[:3,3]
    return V, F


def plot_assembly(geo, mates, rigid_labels=None, view_width=800, view_height=600):
    max_part_dim = max([max(geo[i][1].V.max(0)-geo[i][1].V.min(0)) for i in geo if geo[i][1].V.shape[0] > 0])
    num_parts = len(geo)
    part_index = 0
    geo_colors = dict()
    occ_to_index = dict()
    part_meshes = []
    mcf_points = []
    minPt = np.full((3,), np.inf)
    maxPt = np.full((3,), -np.inf)

    for i in geo:
        g = geo[i]
        if g[1] is None:
            continue
        V, F = occ_to_mesh(g)
        if V.shape[0] == 0:
            continue
        minPt = np.minimum(minPt, V.min(0))
        maxPt = np.maximum(maxPt, V.max(0))
    
    meanPt = (minPt + maxPt)/2
    maxDim = (maxPt - minPt).max()
    scalemat = np.identity(4)
    scalemat[:3,:3] /= maxDim
    tfs = {occ: scalemat @ geo[occ][0] for occ in geo}
    mesh_data = {}
    for i in geo:
        g = geo[i]
        if g[1] is None:
            continue
        tf = tfs[i]
        V, F = occ_to_mesh(g, tf)
        if V.shape[0] == 0:
            continue
        
        mesh_data[i] = (V, F)

    if rigid_labels is not None:
        num_rigid_labels = len(set(rigid_labels))
    for ind,i in enumerate(geo):
        occ_to_index[i] = part_index
        g = geo[i]
        if g[1] is None:
            continue
        
        if i not in mesh_data:
            continue

        V, F = mesh_data[i]
        
        if rigid_labels is None:
            colors = np.array(get_color(part_index, num_parts))
        else:
            colors = np.array(get_color(rigid_labels[ind], num_rigid_labels))
        geo_colors[i] = colors
        colors_int = (colors * 255).astype(np.uint8)
        part_index += 1
        
        #render the parts meshes
        ba_dict = {}
        ba_dict['position'] = p3s.BufferAttribute(V.astype(np.float32), normalized=False)
        ba_dict['index'] = p3s.BufferAttribute(F.astype(np.uint32).ravel(), normalized=False)
        geometry = p3s.BufferGeometry(attributes=ba_dict)
        geometry.exec_three_obj_method('computeVertexNormals')
        material = p3s.MeshStandardMaterial(color=f"rgb({colors_int[0]},{colors_int[1]},{colors_int[2]})", flatShading=True)
        mesh = p3s.Mesh(geometry=geometry, material=material)
        part_meshes.append(mesh)
    
        mcf_origins = []
        tf = tfs[i]
        for mc in g[1].all_mate_connectors:
            mcf = mc.get_coordinate_system()
            mcf_origins.append(tf[:3,:3] @ mcf[:3,3] + tf[:3,3])
        mcf_origins = np.array(mcf_origins)
        pointgeometry = p3s.BufferGeometry(attributes={'position': p3s.BufferAttribute(mcf_origins.astype(np.float32), normalized=False)})
        pointsmaterial = p3s.PointsMaterial(color=f"rgb({colors_int[0]//2},{colors_int[1]//2},{colors_int[2]//2})", size=3, sizeAttenuation=False)
        points = p3s.Points(geometry=pointgeometry, material=pointsmaterial)
        mcf_points.append(points)

    all_mate_objects = []
    for i,mate in enumerate(mates):
        if mate.type == 'FASTENED' and rigid_labels is not None:
            continue
        if len(mate.matedEntities)==2:
            for mated in mate.matedEntities:
                tf = tfs[mated[0]]
                newaxes = tf[:3, :3] @ mated[1][1]
                neworigin = tf[:3,:3] @ mated[1][0] + tf[:3,3]
                mate_objects = add_axis(neworigin, newaxes[:,0], newaxes[:,1], newaxes[:,2], scale=maxDim/10, mate_type=mate.type)
                all_mate_objects += mate_objects

    camera = p3s.PerspectiveCamera( position=tuple(np.array([1, 0.6, 1])), lookAt=meanPt/maxDim, aspect=view_width/view_height)
    key_light = p3s.DirectionalLight(position=[0, 10, 10])
    back_light = p3s.DirectionalLight(position=[0, -10, -10], intensity=0.4)
    ambient_light = p3s.AmbientLight()
    scene = p3s.Scene(children=part_meshes + mcf_points + all_mate_objects + [camera, key_light, back_light, ambient_light])
    controller = p3s.OrbitControls(controlling=camera, target=tuple(meanPt/maxDim))
    renderer = p3s.Renderer(camera=camera, scene=scene, controls=[controller],
                        width=view_width, height=view_height)
    chks = [
        wg.Checkbox(False, description='wireframe'),
        wg.Checkbox(True, description='show parts'),
        wg.Checkbox(True, description='show MCFs'),
    ]
    #slider = wg.FloatSlider(description="widget scale", value=1, min=0.1, max=10)
    for mesh in part_meshes:
        wg.jslink((chks[0], 'value'), (mesh.material, 'wireframe'))
        wg.jslink((chks[1], 'value'), (mesh, 'visible'))
        #wg.jslink((slider,'value'),(mesh,'scale'))
    for mcfs in mcf_points:
        wg.jslink((chks[2], 'value'), (mcfs, 'visible'))
    
    #for matesymbol in all_mate_objects:
    hbox = wg.HBox(chks)
    vbox = wg.VBox([renderer, hbox])

    return vbox
