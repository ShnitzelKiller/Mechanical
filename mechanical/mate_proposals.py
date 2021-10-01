from intervaltree import IntervalTree, Interval
import pspart
import numpy as np
import numpy.linalg as LA
import numba
from numba import njit
from scipy.spatial.transform import Rotation as R


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

def mate_proposals(parts, epsilon_rel=0.001, max_groups=10):
    """
    Find list of (part1, part2, mc1, mc2) of probable mate locations given a list of (transform, pspart.Part)
    `epsilon_fac`: fraction of maximum part dimension to use as epsilon for finding neighboring mate connectors
    """

    #build initial data structures and nearest neighbor hashmap for axial directions
    maxdim = max([(part.bounding_box()[1]-part.bounding_box()[0]).max() for _, part in parts])
    mc_frames = []
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
            
            frame = homogenize_frame(frame, z_flip_only=True)

            mc_frames.append(frame)
            mc_origin.append(origin)
        new_total = total_mcs + len(part.all_mate_connectors)
        interval2part.append((total_mcs, new_total, i))
        part2offset[i] = total_mcs
        total_mcs = new_total
    mc_axis = [frame[:,2] for frame in mc_frames]
    mc_quat = [R.from_matrix(frame).as_quat() for frame in mc_frames]
    nnhash = pspart.NNHash(mc_axis, 3, epsilon_rel)
    tree = IntervalTree([Interval(l, u, d) for l, u, d in interval2part if l < u])
    
    #for each axial cluster, find nearest neighbors of projected set of frames with the same axis direction
    proposals = set()
    result = cluster_axes(nnhash, mc_axis)
    # print(result)
    group_indices = list(result)
    group_indices.sort(key=lambda k: result[k], reverse=True)
    for ind in group_indices[:max_groups]:
        # print('processing group',ind,':',result[ind])
        proj_dir = mc_axis[ind]
        same_dir_inds = list(nnhash.get_nearest_points(proj_dir))
        xy_plane = mc_frames[ind][:,:2]
        # print('projecting')
        subset_stacked = np.array([mc_origin[i] for i in same_dir_inds])
        projected_origins = list((subset_stacked @ xy_plane)/maxdim)
        projected_z = list((subset_stacked @ proj_dir)/maxdim)
        z_quat = [np.concatenate([mc_quat[same_dir_inds[i]], [projected_z[i]]]) for i in range(len(same_dir_inds))]
        # print('hashing')
        coplanar_hash = pspart.NNHash(z_quat, 5, epsilon_rel)
        coaxial_hash = pspart.NNHash(projected_origins, 2, epsilon_rel)
        # print('querying')
        for igroup in range(len(same_dir_inds)):
            nearest_coaxial = coaxial_hash.get_nearest_points(projected_origins[igroup])
            nearest_coplanar = coplanar_hash.get_nearest_points(z_quat[igroup])
            nearest = list(nearest_coaxial.union(nearest_coplanar))
            i = same_dir_inds[igroup]
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

if __name__ == '__main__2':
    homogenized_all = []
    for axis in range(3):
        homogenized_z_all = []
        for flip in range(2):
            vec = np.zeros(3)
            vec[axis] = -1 if flip else 1

            for axis_next in range(2):
                for flip_next in range(2):
                    vec_next = np.zeros(3)
                    vec_next[(axis+1+axis_next)%3] = -1 if flip_next else 1
                    frame = np.empty((3,3))
                    frame[:,0] = vec_next
                    frame[:,2] = vec
                    frame[:,1] = np.cross(vec, vec_next)
                    homogenized = homogenize_frame(frame,z_flip_only=False)
                    assert(LA.det(homogenized) == 1)
                    homogenized_z = homogenize_frame(frame,z_flip_only=True)
                    assert(LA.det(homogenized_z) == 1)
                    homogenized_all.append(homogenized)
                    homogenized_z_all.append(homogenized_z)
        for h in homogenized_z_all[1:]:
            dist = LA.norm(h-homogenized_z_all[0])
            if dist > 0:
                print(f'error for z: dist nonzero: {dist}')
                print(h)
                print(homogenized_z_all[0])
    for h in homogenized_all[1:]:
        dist = LA.norm(h-homogenized_all[0])
        if dist > 0:
            print(f'error: dist nonzero: {dist}')

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
    