import numpy as np

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
    
    for i,mate in enumerate(mates):
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

def connected_components(adj, connectionType = 'any'):
    """
    Find number of connected components
    ConnectionType: any -> all mates are considered
    fasten -> only fastens are considered (find rigidly connected components)
    """
    visited = np.zeros(adj.shape[0], bool)
    components = 0
    for j in range(len(visited)):
        if not visited[j]:
            components += 1
            frontier = [j]
            while len(frontier) > 0:
                curr = frontier.pop()
                if visited[curr]:
                    continue
                visited[curr] = True
                for i,ne in enumerate(adj[:,curr]):
                    if connectionType == 'any':
                        if ne > 0:
                            frontier.append(i)
                    else:
                        if ne == 1:
                            frontier.append(i)
    return components