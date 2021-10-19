import numpy as np
import numba
from numba import jit, njit
from intervaltree import IntervalTree, Interval
import pspart

class SingleIntervalTree:
    def __init__(self, tree):
        self.tree = tree
    
    def __getitem__(self, ind):
        return next(iter(self.tree[ind])).data

def start_indices_to_interval_tree(indices, size):
    """
    Returns a data structure which, given an index, returns the id of the interval delimited by `indices` in which that index lies.
    """
    interval2part = []
    for i in range(len(indices)-1):
        interval2part.append((indices[i], indices[i+1], i))
    interval2part.append((indices[-1], size, len(indices)-1))
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
    return components

if __name__ == '__main__':
    points1 = np.random.randn(10, 2)
    points2 = points1[6:, :]

    matches = find_neighbors(list(points1), list(points2), 2, 0.0001)
    assert(matches == [(6, 0), (7, 1), (8, 2), (9, 3)])

    matches = find_neighbors(list(points2), list(points1), 2, 0.0001)
    assert(matches == [(0, 6), (1, 7), (2, 8), (3, 9)])