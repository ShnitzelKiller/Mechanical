import numpy as np
import numba
from numba import jit, njit
from intervaltree import IntervalTree, Interval
import pspart
from enum import Enum


def get_color(index, total):
    h = index / total * 360
    intensity = 1
    rf, gf, bf = hsv2rgb(h, 1.0, 1.0)
    return rf * intensity, gf * intensity, bf * intensity

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

def cluster_points(nnhash, points):
    """
    Returns a dictionary from a representative element index in each cluster to the number of elements in that cluster.
    """
    visited = set()
    clusters = dict()
    for i,point in enumerate(points):
        if i not in visited:
            nearest = nnhash.get_nearest_points(point)
            visited.add(i)
            visited = visited.union(nearest)
            clusters[i] = nearest
    return clusters


def inter_group_cluster_points(nnhash, points, group_labels, min_size=2):
    """
    Returns a dictionary from pairs of group indices to cluster indices, and a dict from those indices to sets of point indices
    """
    assert(len(points) == len(group_labels))
    visited = set()
    clusters = dict()

    for i,point in enumerate(points):
        if i not in visited:
            nearest = nnhash.get_nearest_points(point)
            visited.add(i)
            visited = visited.union(nearest)
            if len(nearest) >= min_size:
                clusters[i] = (nearest, {group_labels[n] for n in nearest})
    
    groups_to_clusters = dict()
    for clust_index in clusters:
        unique_groups = clusters[clust_index][1]
        for g in unique_groups:
            for g2 in unique_groups:
                if g < g2:
                    key = (g, g2)
                    if key not in groups_to_clusters:
                        groups_to_clusters[key] = []
                    groups_to_clusters[key].append(clust_index)
    return groups_to_clusters, {i: clusters[i][0] for i in clusters}

class SingleIntervalTree:
    def __init__(self, tree):
        self.tree = tree
    
    def __getitem__(self, ind):
        return next(iter(self.tree[ind])).data

def sizes_to_interval_tree(sizes):
    """
    Returns a data structure which, given an index, returns the id of the interval delimited by `indices` in which that index lies.
    """
    interval2part = []
    offset = 0
    for i,size in enumerate(sizes):
        newoffset = offset + size
        interval2part.append((offset, newoffset, i))
        offset = newoffset
    
    return SingleIntervalTree(IntervalTree([Interval(l, u, d) for l, u, d in interval2part if l < u]))

def find_neighbors(points1, points2, dim, eps):
    """
    Get a list of pairs of indices (i1, i2) of points in each set that are within eps of each other
    """
    N1 = len(points1)
    N2 = len(points2)
    swap = False
    #smaller set is at the end, and used as query
    if N1 < N2:
        swap = True
        N1, N2 = N2, N1
        points1, points2 = points2, points1
    hash = pspart.NNHash(points1, dim, eps)
    matches = []
    for i,point in enumerate(points2):
        neighbors = hash.get_nearest_points(point)
        for n in neighbors:
            matches.append((n, i))
    if swap:
        matches = [(match[1], match[0]) for match in matches]
    return matches

def inter_group_matches(group_sizes, points, dim, eps, hash=None, point_to_grouped_map=None):
    """
    Given a list of N `dim`-dimensional points, and a list of sizes of k consecutive groups, return a list of all matching pairs of points between different
    groups, in the form (group1, group2, ind1, ind2), where ind1 and ind2 are with respect to the corresponding groups' offset in the point array.
    if `hash` is defined, use an existing spatial hash map
    if `point_to_grouped_map` is defined, the group sizes will pertain to the output space of the map
    """
    if point_to_grouped_map is None:
        point_to_grouped_map = lambda x: x
    start_indices = [0] * len(group_sizes)
    offset = 0
    for i,size in enumerate(group_sizes[:-1]):
        offset += size
        start_indices[i+1] = offset
    if hash is None:
        hash = pspart.NNHash(points, dim, eps)
    offset2group = sizes_to_interval_tree(group_sizes)
    proposals = set()
    #get all coincident mate connectors
    for i,point in enumerate(points):
        nearest = hash.get_nearest_points(point)
        iglobal = point_to_grouped_map(i)
        group_index = offset2group[iglobal]
        for j in nearest:
            if j != i:
                jglobal = point_to_grouped_map(j)
                other_group_index = offset2group[jglobal]
                if other_group_index != group_index:
                    pi1, pi2 = group_index, other_group_index
                    mci1, mci2 = iglobal - start_indices[group_index], jglobal - start_indices[other_group_index]
                    if pi1 > pi2:
                        pi1, pi2 = pi2, pi1
                        mci1, mci2 = mci2, mci1
                    proposals.add((pi1, pi2, mci1, mci2))
    return proposals

def joinmeshes(meshes):
    F = []
    offset = 0
    for i in range(len(meshes)):
        F.append(meshes[i][1] + offset)
        offset += meshes[i][0].shape[0]
    return np.vstack([v for v,f in meshes]), np.vstack(F)

def adjacency_matrix(occs, mates):
    adj = np.zeros([len(occs), len(occs)], dtype=np.int32)
    index2occ = []
    for occ in occs:
        index2occ.append(occ)
    occ2index = dict()
    for i in range(len(occs)):
        occ2index[index2occ[i]] = i
    
    for mate in mates:
        if len(mate.matedEntities) == 2:
            if mate.type.lower().startswith('fasten'):
                mateType = 1
            else:
                mateType = 2
            ind1 = occ2index[mate.matedEntities[0][0]]
            ind2 = occ2index[mate.matedEntities[1][0]]
            adj[ind1, ind2] = mateType
            adj[ind2, ind1] = mateType
    
    return adj

class MateTypes(Enum):
    PIN_SLOT = 'PIN_SLOT'
    BALL = 'BALL'
    PARALLEL = 'PARALLEL'
    SLIDER = 'SLIDER'
    REVOLUTE = 'REVOLUTE'
    CYLINDRICAL = 'CYLINDRICAL'
    PLANAR = 'PLANAR'
    FASTENED = 'FASTENED'
    def __eq__(self, obj):
        return self.value == obj


def external_adjacency_list(occs, mates):
    """
    Adjacency list given occurrences as a list of ids, and mates as a list of 3-tuples
    of (id1, id2, type)
    """
    adj = [[] for o in occs]
    index2occ = []
    for occ in occs:
        index2occ.append(occ)
    occ2index = dict()
    for i in range(len(occs)):
        occ2index[index2occ[i]] = i
    
    for i,mate in enumerate(mates):
        ind1 = occ2index[mate[0]]
        ind2 = occ2index[mate[1]]
        adj[ind1].append((ind2, i))
        adj[ind2].append((ind1, i))
    return adj


def external_adjacency_list_from_brepdata(occs, mates):
    """
    Assumes mates have 2 mated occurrences.
    """
    mates = [(mate.matedEntities[0][0], mate.matedEntities[1][0]) for mate in mates]
    return external_adjacency_list(occs, mates)


def adjacency_list(occs, mates):
    """
    Adjacency list given occurrences as a list of ids, and mates as a list of 3-tuples
    of (id1, id2, type)
    """
    adj = [[] for o in occs]
    index2occ = []
    for occ in occs:
        index2occ.append(occ)
    occ2index = dict()
    for i in range(len(occs)):
        occ2index[index2occ[i]] = i
    
    for mate in mates:
        if mate[2].lower().startswith('fasten'):
            mateType = 1
        else:
            mateType = 2
        ind1 = occ2index[mate[0]]
        ind2 = occ2index[mate[1]]
        adj[ind1].append((ind2, mateType))
        adj[ind2].append((ind1, mateType))
    return adj

def adjacency_list_from_brepdata(occs, mates):
    mates = [(mate.matedEntities[0][0], mate.matedEntities[1][0], mate.type) for mate in mates if len(mate.matedEntities) == 2]
    return adjacency_list(occs, mates)


def homogenize(lst):
    N = len(lst)
    d = max(len(l) for l in lst)
    mat = np.full((N, d*2), -1, dtype=np.int32)
    for i,l in enumerate(lst):
        for j,pair in enumerate(l):
            mat[i,2*j:2*j+2] = pair
    return mat

@njit
def connected_components_dense(adj, connectionType = 'any'):
    """
    Find number of connected components
    ConnectionType: any -> all mates are considered
    fasten -> only fastens are considered (find rigidly connected components)
    """
    visited = np.zeros(adj.shape[0], numba.uint8)
    components = 0
    for j in range(len(visited)):
        if not visited[j]:
            components += 1
            frontier = [j]
            while len(frontier) > 0:
                curr = frontier.pop()
                if visited[curr]:
                    continue
                visited[curr] = 1
                for i,ne in enumerate(adj[:,curr]):
                    if connectionType == 'any':
                        if ne > 0:
                            frontier.append(i)
                    else:
                        if ne == 1:
                            frontier.append(i)
    return components

@njit
def connected_components(adj, connectionType = 'any'):
    """
    Find number of connected components
    ConnectionType: any -> all mates are considered
    fasten -> only fastens are considered (find rigidly connected components)
    """
    visited = np.zeros(adj.shape[0], dtype=numba.uint8)
    labeling = np.zeros(adj.shape[0], dtype=numba.int32)
    maxneighbors = adj.shape[1]//2
    components = 0
    for j in range(len(visited)):
        if not visited[j]:
            components += 1
            frontier = [j]
            while len(frontier) > 0:
                curr = frontier.pop()
                if visited[curr]:
                    continue
                visited[curr] = 1
                labeling[curr] = components - 1
                for k in range(maxneighbors):
                    i = adj[curr,k*2]
                    ne = adj[curr,k*2+1]
                    if i >= 0 and not visited[i]:
                        if connectionType == 'any':
                            if ne > 0:
                                frontier.append(i)
                        else:
                            if ne == 1:
                                frontier.append(i)
    return components, labeling

if __name__ == '__main__':
    import mechanical.onshape as brepio
    from mechanical.data import AssemblyInfo, assembly_info_from_onshape
    import os
    from mechanical.visualize2 import plot_assembly
    from ipywidgets.embed import embed_minimal_html

    #assemblyPath = '7886c69b1f149069e7a43bdb_b809b448b6da81db4b2388a0_1dc646f33c96254f526ea650.json' #skateboard
    assemblyPath = '080bda692c7362d9d8f550c0_2bb4d493fe1adaf882bc9c9f_2b9ddd420bbfde38934d34ef.json' #tesla board
    datapath = '/projects/grail/benjones/cadlab'
    loader = brepio.Loader(datapath)
    occs, mates = loader.load_flattened(assemblyPath, geometry=False)
    adj_list = adjacency_list_from_brepdata(occs, mates)
    adj = homogenize(adj_list)
    num_rigid, labeling = connected_components(adj, connectionType='fasten')
    print('num rigid:',num_rigid)

    assembly_info = assembly_info_from_onshape(occs, datapath)
    #renderer = plot_assembly(assembly_info.get_onshape(), mates, rigid_labels=labeling)
    #embed_minimal_html('export.html', views=[renderer], title='Viewer export')

    print('valid:',assembly_info.valid)
    missing = assembly_info.fill_missing_mates(mates, labeling, 0.01)
    print('missing mates:',len(missing))
    print(assembly_info.newmatestats)


if __name__ == '__main__2':
    points = np.random.randn(10, 2)
    points1 = points[:6, :]
    points2 = points[3:, :]
    points_all = np.vstack([points1, points2])
    matches = inter_group_matches([6, 7], points_all, 2, 0.0001)
    assert(matches == {(0, 1, 5, 2), (0, 1, 3, 0), (0, 1, 4, 1)})
    group_sizes = [100, 100]
    group_to_offset = [0] + group_sizes
    point_to_grouped_map = lambda i: i+1 if i < 6 else i + 150 - 6
    matches = inter_group_matches(group_sizes, points_all, 2, 0.0001, point_to_grouped_map=point_to_grouped_map)
    assert(matches == {(0, 1, 5, 51), (0, 1, 4, 50), (0, 1, 6, 52)})

    points1 = np.random.randn(10, 2)
    points2 = points1[6:, :]

    matches = find_neighbors(list(points1), list(points2), 2, 0.0001)
    assert(matches == [(6, 0), (7, 1), (8, 2), (9, 3)])

    matches = find_neighbors(list(points2), list(points1), 2, 0.0001)
    assert(matches == [(0, 6), (1, 7), (2, 8), (3, 9)])