import openmesh as om
import random
import math
from collections import deque
import sys
import matplotlib.pyplot as plt
import numpy as np

def createHalfEdge(V, F):
    """Create a half edge mesh from the input triangle mesh"""
    mesh = om.TriMesh()
    nv = V.shape[0]
    nf = F.shape[0]
    vertexIndices = np.empty(nv, dtype=object)
    for i in range(nv):
        vh = mesh.add_vertex(V[i,:])
        vertexIndices[i] = vh
    for i in range(nf):
        #print(F[i,:])
        mesh.add_face(vertexIndices[F[i,0]], vertexIndices[F[i,1]], vertexIndices[F[i,2]])
    return mesh
    

def segmentMesh(V, F, patchRadius):
    n = V.shape[0]
    mesh = createHalfEdge(V, F)
    vhs = [vh for vh in mesh.vertices()]
    visited = set()
    random.shuffle(vhs)
    print('designating sources')
    sources = []
    for i,vh in enumerate(vhs):
        if vh.idx() in visited:
            continue
        sources.append(vh)
        neighbors = deque([(vh, 0)])
        added = 0
        while neighbors:
            curr_vertex, dist = neighbors.popleft()
            visited.add(curr_vertex.idx())
            added += 1
            #print('from vertex',curr_vertex.idx(), 'dist', dist)
            for heh in mesh.voh(curr_vertex):
                nh = mesh.to_vertex_handle(heh)
                if nh.idx() not in visited:
                    diff = mesh.point(nh) - mesh.point(curr_vertex)
                    newDist = dist + math.sqrt(diff.dot(diff))
                    if newDist < patchRadius:
                        #print(nh.idx())
                        neighbors.append((nh, newDist))
        print('added',added,'neighbors')
    print('sources:',len(sources))
    #TODO: run djikstra's to get final patches
    #values are (source index, dist, vertex handle)
    known = dict()
    frontier = dict()
    for i,source in enumerate(sources):
        frontier[source.idx()] = i, 0, source
    
    while frontier:
        shortestDistance = sys.maxsize
        for id in frontier:
            dist = frontier[id][1]
            if dist < shortestDistance:
                shortestDistance = dist
                closestId = id
        source, dist, curr_vertex = frontier.pop(closestId)
        known[closestId] = source

        for heh in mesh.voh(curr_vertex):
            nh = mesh.to_vertex_handle(heh)
            if nh.idx() not in known:
                diff = mesh.point(nh) - mesh.point(curr_vertex)
                newDist = dist + math.sqrt(diff.dot(diff))
                if nh.idx() not in frontier or newDist < frontier[nh.idx()][1]:
                    frontier[nh.idx()] = source, newDist, nh

    #debug
    allpoints = mesh.points()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(allpoints[:,0], allpoints[:,1], allpoints[:,2], c=[known[vh.idx()] for vh in mesh.vertices()])
    plt.show()

    if __name__ == "__main__":
        import matplotlib.pyplot as plt
        import meshio
        mesh = meshio.read("data/cylinder.obj")
        print('vertices:', mesh.points.shape[0])
        segmentMesh(mesh.points, mesh.cells_dict['triangle'], 0.5)