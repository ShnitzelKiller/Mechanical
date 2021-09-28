from intervaltree import IntervalTree, Interval
import pspart
import numpy as np
import numpy.linalg as LA
import numba
from numba import njit

def cluster_axes(nnhash, axes):
    visited = set()
    clusters = dict()
    for i in range(len(axes)):
        if i not in visited:
            nearest = list(nnhash.get_nearest_points(axes[i]))
            visited.add(i)
            for n in nearest:
                visited.add(n)
            clusters[i] = 1 + len(nearest)
    return clusters

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
        return -vec
    else:
        return vec

def mate_proposals(parts, epsilon_rel=0.001, max_groups=10):
    """
    Find list of (part1, part2, mc1, mc2) of probable mate locations given a list of (transform, pspart.Part)
    `epsilon_fac`: fraction of maximum part dimension to use as epsilon for finding neighboring mate connectors
    """

    #build initial data structures and nearest neighbor hashmap for axial directions
    maxdim = max([(part.bounding_box()[1]-part.bounding_box()[0]).max() for _, part in parts])
    mc_axis = []
    mc_plane = []
    mc_origin = []
    interval2part = []
    part2offset = dict()
    total_mcs = 0
    for i,tf_part in enumerate(parts):
        tf, part = tf_part
        for mc in part.all_mate_connectors:
            cs = mc.get_coordinate_system()
            origin = tf[:3,:3] @ cs[:3,3] + tf[:3,3]
            frame = tf[:3,:3] @ cs[:3,:3]
            z_axis = frame[:,2]
            xy_plane = frame[:,:2]
            mc_axis.append(homogenize_sign(z_axis))
            mc_plane.append(xy_plane)
            mc_origin.append(origin)
        new_total = total_mcs + len(part.all_mate_connectors)
        interval2part.append((total_mcs, new_total, i))
        part2offset[i] = total_mcs
        total_mcs = new_total
    nnhash = pspart.NNHash(mc_axis, 3, epsilon_rel)
    tree = IntervalTree([Interval(l, u, d) for l, u, d in interval2part])
    
    #for each axial cluster, find nearest neighbors of projected set of frames with the same axis direction
    proposals = set()
    result = cluster_axes(nnhash, mc_axis)
    group_indices = list(result)
    group_indices.sort(key=lambda k: result[k], reverse=True)
    for ind in group_indices[:max_groups]:
        #print('processing group',ind,':',result[ind])
        proj_dir = mc_axis[ind]
        same_dir_inds = list(nnhash.get_nearest_points(proj_dir))
        xy_plane = mc_plane[ind]
        #print('projecting')
        projected_origins = list((np.array([mc_origin[i] for i in same_dir_inds]) @ xy_plane)/maxdim)
        #print('hashing')
        proj_hash = pspart.NNHash(projected_origins, 2, epsilon_rel)
        #print('querying')
        for igroup,loc in enumerate(projected_origins):
            i = same_dir_inds[igroup]
            nearest = list(proj_hash.get_nearest_points(loc))
            part_index = next(iter(tree[i])).data
            for jgroup in nearest:
                j = same_dir_inds[jgroup]
                other_part_index = next(iter(tree[j])).data
                if other_part_index != part_index:
                    pi1, pi2 = part_index, other_part_index
                    mci1, mci2 = i - part2offset[part_index], j - part2offset[other_part_index]
                    if pi1 > pi2:
                        pi1, pi2 = pi2, pi1
                        mci1, mci2 = mci2, mci1
                    proposals.add((pi1, pi2, mci1, mci2))
    return proposals

if __name__ == '__main__':
    import os
    tol = 0.01
    datapath = '/projects/grail/benjones/cadlab'
    part = pspart.Part(os.path.join(datapath, 'data/models/58aace5054540c1fba909bab/ac500a73d2324ea4fa3af589/84dd987436323f4357c873e2/default/JFD.xt'))
    print(part)
    tf = np.identity(4)
    proposals = mate_proposals([(tf, part), (tf, part)])
    print(len(proposals))
    #mc_axis = []
    #for mc in part.all_mate_connectors:
    #    cs = mc.get_coordinate_system()
    #    #origin = tf[:3,:3] @ cs[:3,3] + tf[:3,3]
    #    z_axis = cs[:3,2]
    #    mc_axis.append(homogenize_sign(z_axis))
    #
    #result = cluster_axes(mc_axis, tol)
    #print(len(part.all_mate_connectors))
    #print(result)
    