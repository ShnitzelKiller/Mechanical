#from pspart import Part
import onshape.brepio as brepio
import geometry.sampling as sampling
import argparse
import meshplot as mp
import numpy as np
import os
from utils import joinmeshes

parser = argparse.ArgumentParser()
parser.add_argument('paths', nargs='+')
parser.add_argument('--num_points', type=int, default=2048)
parser.add_argument('--out_folder', default='data/pointclouds/')

args = parser.parse_args()

loader = brepio.Loader('/projects/grail/benjones/cadlab')


for fname in args.paths:
    path = fname.split('/')
    did = path[-7]
    mv = path[-3]
    eid = path[-1]
    print(did, mv, eid)

    try:
        data = loader.load(did, mv, eid)
    except KeyError:
        print('WARNING: failed to load',did,mv,eid)
        continue
    
    meshes = []
    for p in data[0]:
        print('part '+p)
        part = data[0][p][1]
        trans = data[0][p][0]
        meshes.append(((trans[:3, :3] @ part.V.T).T + trans[:3, 3], part.F))
    V, F = joinmeshes(meshes)
    points, normals = sampling.sample_surface_points(V, F, args.num_points)
    minPt = points.min(0)
    maxPt = points.max(0)
    points -= minPt
    maxdim = (maxPt - minPt).max()
    points /= maxdim

    np.savetxt(os.path.join(args.out_folder, '-'.join([did, mv, eid])) + '.txt', points)