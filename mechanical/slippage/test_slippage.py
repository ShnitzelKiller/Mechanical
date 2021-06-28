import argparse
#import meshio
import utils
import time
import matplotlib.pyplot as plt
#import meshplot
import trimesh
import trimesh.remesh as remesh
import slippage

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("meshfile", type=str, nargs='?', default="data/scissors/scissors.stl")
    parser.add_argument("num_points", type=int, nargs='?', default=1000)
    #parser.add_argument("--max_edge", type=float, default=.5)
    args = parser.parse_args()
    mesh = trimesh.load(args.meshfile)
    if not mesh.is_watertight:
        print('warning: non-watertight mesh',args.meshfile)
    print('vertices:',mesh.vertices.shape[0])

    #meshplot.plot(mesh.points, mesh.cells_dict['triangle'])
    
    #start = time.time()
    #points, normals = utils.sample_points(mesh.vertices, mesh.facets, args.num_points)
    #end = time.time()
    #print('time:',end - start)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    ax.scatter(mesh.vertices[::10,0], mesh.vertices[::10,1], mesh.vertices[::10,2])
    plt.show()
    
    slippage.segmentMesh(mesh.vertices, mesh.faces, 0.02)


if __name__ == "__main__":
    main()