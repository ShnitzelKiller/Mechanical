import argparse
import meshio
import utils
import time
import matplotlib.pyplot as plt
#import meshplot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("meshfile", type=str, nargs='?', default="data/monkey.obj")
    parser.add_argument("num_points", type=int, nargs='?', default=1000)
    args = parser.parse_args()
    mesh = meshio.read(args.meshfile)
    start = time.time()
    points, normals = utils.sample_points(mesh, args.num_points)
    end = time.time()
    print('time:',end - start)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2])
    plt.show()
    #print(points)

if __name__ == "__main__":
    main()