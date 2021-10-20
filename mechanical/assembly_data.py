import os
from pspart import Part, NNHash
import numpy as np
from utils import find_neighbors, inter_group_matches, cluster_points, sizes_to_interval_tree, homogenize_frame
from scipy.spatial.transform import Rotation as R


def global_bounding_box(parts, transforms=None):
    if transforms is None:
        allpoints = [part.V for part in parts if part.V.shape[0] > 0]
    else:
        allpoints = [(tf[:3,:3] @ part.V.T + tf[:3,3,np.newaxis]).T for tf, part in zip(transforms, parts) if part.V.shape[0] > 0]
    if len(allpoints) == 0:
        return None
    minPt = np.array([points.min(axis=0) for points in allpoints]).min(axis=0)
    maxPt = np.array([points.max(axis=0) for points in allpoints]).max(axis=0)
    return np.vstack([minPt, maxPt])

def apply_transform(tf, v, is_points=True):
    "Apply the 4-matrix `tf` to a vector or column-wise matrix of vectors. If is_points, also add the translation."
    v_trans = tf[:3,:3] @ v
    if is_points:
        if v.ndim==1:
            v_trans += tf[:3,3]
        elif v.ndim==2:
            v_trans += tf[:3,3,np.newaxis]
    return v_trans

def cs_to_origin_frame(cs):
    origin = cs[:,3]
    if origin[3] != 1:
        origin /= origin[3]
    return origin[:3], cs[:3,:3]

def cs_from_origin_frame(origin, frame):
    cs = np.identity(4, np.float64)
    cs[:3,:3] = frame
    cs[:3,3] = origin
    return cs

class AssemblyInfo:

    def __init__(self, part_paths, transforms, occ_ids, epsilon_rel=0.001):
        self.epsilon_rel = epsilon_rel
        self.part_paths = part_paths
        self.part_caches = []
        self.parts = []
        self.occ_to_index = {}
        self.occ_ids = occ_ids
        self.mc_origins_all = []
        self.mc_rots_all = [] #3x3 matrices
        self.mc_rots_homogenized_all = []
        self.mco_hashes = [] #spatial hashes of origins
        self.mcz_hashes = [] #spatial hashes of homogenized z direction
        self.mcr_hashes = [] #spatial hashes of origin + homogenized z dir

        for j,path in enumerate(part_paths):
            assert(os.path.isfile(path))
            with open(path) as partf:
                part_str = partf.read()
                part = Part(part_str)
                self.part_caches.append(part_str)
                self.parts.append(part)
                self.occ_to_index[occ_ids[j]] = j

        self.bbox = global_bounding_box(self.parts, transforms)
        if self.bbox is None:
            raise ValueError('invalid bounding box')
        
        minPt, maxPt = self.bbox
        self.median = self.bbox.mean(axis=0)
        dims = maxPt - minPt
        self.maxdim = max(dims)

        p_normalized = np.identity(4, dtype=float)
        p_normalized[:3,3] = -self.median
        p_normalized[3,3] = self.maxdim #todo: figure out if this is double the factor

        norm_matrices = [p_normalized @ tf for tf in transforms]
        self.normalized_parts = [Part(cached_part, mat) for cached_part, mat in zip(self.part_caches, norm_matrices)]
        
        #these may change if we choose to handle transforms differently. TODO if changing: also handle the epsilons/scaling thresholds!
        self.mate_transforms = norm_matrices
        self.part_transforms = [np.identity(4) for part in self.normalized_parts]

        #analyze mate connectors
        for tf,part in zip(self.part_transforms, self.normalized_parts):
            mc_origins = []
            mc_rots = []
            for mc in part.all_mate_connectors:
                cs = mc.get_coordinate_system()
                cs_transformed = tf @ cs
                origin, rot = cs_to_origin_frame(cs_transformed)
                mc_origins.append(origin)
                mc_rots.append(rot)

            self.mc_rots_all.append(mc_rots)
            self.mc_origins_all.append(mc_origins)
            mc_rots_homogenized = [homogenize_frame(rot, z_flip_only=True) for rot in mc_rots]
            self.mc_rots_homogenized_all.append(mc_rots_homogenized)

            mc_rays = [np.concatenate([origin,rot[:,2]]) for origin, rot in zip(mc_origins, mc_rots_homogenized)]
            origin_hash = NNHash(mc_origins, 3, epsilon_rel)
            z_hash = NNHash([rot[:,2] for rot in mc_rots_homogenized], 3, epsilon_rel)
            ray_hash = NNHash(mc_rays, 6, epsilon_rel)
            self.mco_hashes.append(origin_hash)
            self.mcz_hashes.append(z_hash)
            self.mcr_hashes.append(ray_hash)
        
        


    def find_mated_pairs(self, mate):
        """
        Returns a list of pairs of mate connector indices that could form this mate.
        """
        #NOTE: The orientations of mate connectors get messed up when we apply the part transforms in the normalization stage
        #so we have to be extra conservative in finding matching frames (xy planes of mate vs connector may be rotated arbitrarily)
        #pin slots may be completely messed up by this...

        #criteria:
        # - fastens: Any mate connectors which coincide up to xy rotations (because of the above) will do
        # - balls: Any mate connectors with the same origin as the mate origin will do
        # - sliders: Any mate connectors along a parallel axis will do
        # - revolute: Any mate connectors with coincident origins along the rotational axis will do
        # - cylindrical: Any mate connectors along the rotational axis will do
        # - planar: Any mate connectors with coincident xy planes parallel to the mate plane will do
        # - parallel: Any mate connectors with the same z axis (up to sign) as the mate frame will do (this one may be excessive without some more pruning)
        # - pin slots: Unsupported for now

        epsilon_rel2 = self.epsilon_rel * self.epsilon_rel

        if len(mate.matedEntities) != 2:
            raise ValueError
        
        part_indices = [self.occ_to_index[me[0]] for me in mate.matedEntities]

        mate_origins = []
        mate_rots_homogenized = []

        for partIndex, me in zip(part_indices, mate.matedEntities):
            origin_local = me[1][0]
            frame_local = me[1][1]
            cs = cs_from_origin_frame(origin_local, frame_local)
            cs_transformed = self.mate_transforms[partIndex] @ cs
            origin, rot = cs_to_origin_frame(cs_transformed)
            rot_homogenized = homogenize_frame(rot, z_flip_only=True)
            mate_origins.append(origin)
            mate_rots_homogenized.append(rot_homogenized)
        
        if mate.type == 'FASTENED':
            matches = find_neighbors(*[[np.concatenate([origin, rot[:,2]]) for origin, rot in zip(self.mc_origins_all[ind], self.mc_rots_homogenized_all[ind])] for ind in part_indices], 6, self.epsilon_rel)
        elif mate.type == 'BALL':
            same_origin_inds = [self.mco_hashes[ind].get_nearest_points(mate_origins[0]) for ind in part_indices]
            matches = [(i, j) for i in same_origin_inds[0] for j in same_origin_inds[1]]
        else:
            xy_plane = mate_rots_homogenized[0][:,:2]
            z_dir = mate_rots_homogenized[0][:,2]
            same_dir_inds = [self.mcz_hashes[ind].get_nearest_points(z_dir) for ind in part_indices]
            subset_stacked = [np.array([self.mc_origins_all[ind][i] for i in same_dir_inds_i]) for ind, same_dir_inds_i in zip(part_indices, same_dir_inds)]
            if mate.type == 'CYLINDRICAL' or mate.type == 'REVOLUTE' or mate.type == 'SLIDER':
                projected_origins = [list(subset_stacked_i @ xy_plane) for subset_stacked_i in subset_stacked] #origins of mcfs with the same z direction projected onto a shared xy plane
                
                if mate.type == 'SLIDER':
                    matches = find_neighbors(*projected_origins, 2, self.epsilon_rel)
                else:
                    mate_origin_proj = mate_origins[0] @ xy_plane
                    axial_indices = []
                    for inds, po in zip(same_dir_inds, projected_origins):
                        axial_indices_i = []
                        for j,point in zip(inds, po):
                            disp = point - mate_origin_proj
                            if disp.dot(disp) < epsilon_rel2:
                                axial_indices_i.append(j)
                        axial_indices.append(axial_indices_i)

                    if mate.type == 'CYLINDRICAL':
                        matches = [(i, j) for i in axial_indices[0] for j in axial_indices[1]]
                    elif mate.type == 'REVOLUTE':
                        subset_axial = [np.array([self.mc_origins_all[ind][i] for i in axial_indices_i]) for ind, axial_indices_i in zip(part_indices, axial_indices)]
                        z_positions = [list((subset_stacked_i @ z_dir)[:,np.newaxis]) for subset_stacked_i in subset_axial]
                        matches = find_neighbors(*z_positions, 1, self.epsilon_rel)
                    else:
                        assert(False)

            elif mate.type == 'PLANAR':
                z_positions = [list((subset_stacked_i @ z_dir)[:,np.newaxis]) for subset_stacked_i in subset_stacked]
                matches = find_neighbors(*z_positions, 1, self.epsilon_rel)
            elif mate.type == 'PARALLEL':
                matches = [(i, j) for i in same_dir_inds[0] for j in same_dir_inds[1]]
            elif mate.type == 'PIN_SLOT':
                matches = []
            else:
                raise ValueError
        return matches

    def num_mate_connectors(self):
        return sum([len(origins) for origins in self.mc_origins_all])
    
    def num_invalid_parts(self):
        return sum([not npart.valid for npart in self.normalized_parts])
    
    def num_normalized_parts_with_discrepancies(self):
        #num_normalized_parts_with_different_graph_size = 0
        #num_normalized_parts_with_different_num_mcs = 0
        #num_normalized_parts_with_different_mcs = 0
        num_discrepancies = 0
        for npart, part in zip(self.normalized_parts, self.parts):
            if npart.num_topologies != part.num_topologies:
                #num_normalized_parts_with_different_graph_size += 1
                num_discrepancies += 1
            elif len(npart.all_mate_connectors) != len(part.all_mate_connectors):
                #num_normalized_parts_with_different_num_mcs += 1
                #num_normalized_parts_with_different_mcs += 1
                num_discrepancies += 1
            else:
                mc_data = set()
                for mc in part.all_mate_connectors:
                    mc_dat = (mc.orientation_inference.topology_ref, mc.location_inference.topology_ref, mc.location_inference.inference_type.value)
                    mc_data.add(mc_dat)
                mc_data_normalized = set()
                for mc in npart.all_mate_connectors:
                    mc_dat = (mc.orientation_inference.topology_ref, mc.location_inference.topology_ref, mc.location_inference.inference_type.value)
                    mc_data_normalized.add(mc_dat)
                if mc_data != mc_data_normalized:
                    #num_normalized_parts_with_different_mcs += 1
                    num_discrepancies += 1
        return num_discrepancies
    
    def mate_proposals(self, max_z_groups=10):
        """
        Find list of (part1, part2, mc1, mc2) of probable mate locations
        `max_z_groups`: maximum number of clusters of mate connector z directions to consider as possible axes
        """
        mc_counts = [len(mcos) for mcos in self.mc_origins_all]
        
        offset2part = sizes_to_interval_tree(mc_counts)
        part2offset = [0] + mc_counts[:-1]

        flattened_origins = [origin for mc_origins in self.mc_origins_all for origin in mc_origins]
        flattened_rots = [rot for mc_rots in self.mc_rots_homogenized_all for rot in mc_rots]
        mc_axis = [rot[:,2] for rot in flattened_rots]
        mc_ray = [np.concatenate([origin, axis]) for origin, axis in zip(flattened_origins, mc_axis)]
        mc_quat = [R.from_matrix(rot).as_quat() for rot in flattened_rots]
        proposals = inter_group_matches(mc_counts, mc_ray, 6, self.epsilon_rel)

        #for each axial cluster, find nearest neighbors of projected set of frames with the same axis direction
        axis_hash = NNHash(mc_axis, 3, self.epsilon_rel)
        clusters = cluster_points(axis_hash, mc_axis)
        group_indices = list(clusters)
        group_indices.sort(key=lambda k: clusters[k], reverse=True)
        for ind in group_indices[:max_z_groups]:
            # print('processing group',ind,':',result[ind])
            proj_dir = mc_axis[ind]
            same_dir_inds = list(axis_hash.get_nearest_points(proj_dir))
            index_map = lambda x: same_dir_inds[x]
            xy_plane = flattened_rots[ind][:,:2]
            # print('projecting to find axes')
            subset_stacked = np.array([flattened_origins[i] for i in same_dir_inds])
            projected_origins = list(subset_stacked @ xy_plane)
            axial_proposals = inter_group_matches(mc_counts, projected_origins, 2, self.epsilon_rel, point_to_grouped_map=index_map)
            proposals = proposals.union(axial_proposals)
            # print('projecting to find planes')
            projected_z = list(subset_stacked @ proj_dir)
            z_quat = [np.concatenate([mc_quat[same_dir_ind], [z_dist]]) for same_dir_ind, z_dist in zip(same_dir_inds, projected_z)]
            planar_proposals = inter_group_matches(mc_counts, z_quat, 5, self.epsilon_rel, point_to_grouped_map=index_map)
            proposals = proposals.union(planar_proposals)
        return proposals



if __name__ == '__main__':
    import onshape.brepio as brepio
    datapath = '/projects/grail/benjones/cadlab'
    loader = brepio.Loader(datapath)
    #geo, mates = loader.load_flattened('87688bb8ccd911995ddc048c_6313170efcc25a6f36e56906_8ee5e722ed4853b12db03877.json', skipInvalid=True, geometry=False)
    geo, mates = loader.load_flattened('e4803faed1b9357f8db3722c_ce43730c0f1758f756fc271f_c00b5256d7e874e534c083e8.json', skipInvalid=True, geometry=False)
    occ_ids = list(geo.keys())
    part_paths = []
    transforms = []
    for id in occ_ids:
        path = os.path.join(datapath, 'data/models', *[geo[id][1][k] for k in ['documentId','documentMicroversion','elementId','fullConfiguration']], f'{geo[id][1]["partId"]}.xt')
        assert(os.path.isfile(path))
        part_paths.append(path)
        transforms.append(geo[id][0])

    assembly_info = AssemblyInfo(part_paths, transforms, occ_ids)
    print('mate connectors:',assembly_info.num_mate_connectors())

    for mate in mates:
        matches = assembly_info.find_mated_pairs(mate)
        print('matches for mate',mate.name,':',len(matches))
    
    proposals = assembly_info.mate_proposals()
    print(len(proposals))
    print(set([match[:2] for match in proposals]))