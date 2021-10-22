import os
import random
import time
import numpy as np
import numpy.linalg as LA
import torch
from torch_geometric.data import Batch
from pspart import Part, NNHash
from utils import find_neighbors, inter_group_matches, cluster_points, sizes_to_interval_tree, homogenize_frame
from scipy.spatial.transform import Rotation as R
from automate.nn.sbgcn import BrepGraphData


mate_types = [
            'PIN_SLOT',
            'BALL',
            'PARALLEL',
            'SLIDER',
            'REVOLUTE',
            'CYLINDRICAL',
            'PLANAR',
            'FASTENED'
        ]

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

def _transform_matches(matches, index_maps):
    return [(index_maps[0][match[0]], index_maps[1][match[1]]) for match in matches]

class AssemblyInfo:

    def __init__(self, part_paths, transforms, occ_ids, epsilon_rel=0.001, use_uvnet_features=False):
        self.epsilon_rel = epsilon_rel
        self.use_uvnet_features = use_uvnet_features
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
        self.stats = dict()
        self.mate_stats = []

        for j,path in enumerate(part_paths):
            assert(os.path.isfile(path))
            with open(path) as partf:
                part_str = partf.read()
                part = Part(part_str, uv_features=use_uvnet_features)
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
        self.normalized_parts = [Part(cached_part, mat, uv_features=use_uvnet_features) for cached_part, mat in zip(self.part_caches, norm_matrices)]
        
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
            same_dir_inds = [list(self.mcz_hashes[ind].get_nearest_points(z_dir)) for ind in part_indices]
            subset_stacked = [np.array([self.mc_origins_all[ind][i] for i in same_dir_inds_i]) for ind, same_dir_inds_i in zip(part_indices, same_dir_inds)]
            if mate.type == 'CYLINDRICAL' or mate.type == 'REVOLUTE' or mate.type == 'SLIDER':
                projected_origins = [list(subset_stacked_i @ xy_plane) for subset_stacked_i in subset_stacked] #origins of mcfs with the same z direction projected onto a shared xy plane
                
                if mate.type == 'SLIDER':
                    matches = find_neighbors(*projected_origins, 2, self.epsilon_rel) #neighbors are found in the subset of same direction mcs
                    matches = _transform_matches(matches, same_dir_inds)
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
                        matches = find_neighbors(*z_positions, 1, self.epsilon_rel) #neighbors are found in the subset of axial mcs
                        matches = _transform_matches(matches, axial_indices)
                    else:
                        assert(False)

            elif mate.type == 'PLANAR':
                z_positions = [list((subset_stacked_i @ z_dir)[:,np.newaxis]) for subset_stacked_i in subset_stacked]
                matches = find_neighbors(*z_positions, 1, self.epsilon_rel) #neighbors are found in the subset of same direction mcs
                matches = _transform_matches(matches, same_dir_inds)
            elif mate.type == 'PARALLEL':
                matches = [(i, j) for i in same_dir_inds[0] for j in same_dir_inds[1]]
            elif mate.type == 'PIN_SLOT':
                matches = []
            else:
                raise ValueError
        return part_indices[0], part_indices[1], matches

    def num_mate_connectors(self):
        return sum([len(origins) for origins in self.mc_origins_all])

    def num_topologies(self):
        return sum([part.num_topologies for part in self.normalized_parts])
    
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


    def _matches_to_proposals(self, matches, proposal_type):
        """
        from a list of matches (part1, part2, mc1, mc2) create a nested dict from part pairs to mc pairs to the proposal data
        which is of the form [proposal type, euclidian distance between mcs, and pointer to a mate, if any (initialized as empty)]
        """
        proposals = dict()
        for match in matches:
            part_pair = match[:2]
            mc_pair = match[2:]
            if part_pair not in proposals:
                mc_pairs_dict = dict()
                proposals[part_pair] = mc_pairs_dict
            else:
                mc_pairs_dict = proposals[part_pair]
            dist = LA.norm(self.mc_origins_all[part_pair[0]][mc_pair[0]] - self.mc_origins_all[part_pair[1]][mc_pair[1]])
            mc_pairs_dict[mc_pair] = [proposal_type, dist, -1]
        return proposals

    def _join_proposals(self, proposal, *proposals):
        """
        Call this with the lowest priority proposals first, as those to the right overwrite their predecessors.
        All proposals are merged into the first argument. The following input dicts may also be modified (do not reuse after calling this)
        """
        for newprop in proposals:
            for part_pair in newprop:
                new_mc_pairs_dict = newprop[part_pair]
                if part_pair not in proposal:
                    proposal[part_pair] = new_mc_pairs_dict
                else:
                    proposal[part_pair].update(new_mc_pairs_dict)
                    
        return proposal


    def mate_proposals(self, max_z_groups=10):
        """
        Find probable mate locations
        `max_z_groups`: maximum number of clusters of mate connector z directions to consider as possible axes
        Returns: Dictionary from (partId1, partId2) -> (mc1, mc2) -> (ptype, dist)
        where ptype is the type of proposal, and can be:
        - 0: coincident
        - 1: coaxial
        - 2: coplanar

        and dist is the euclidian distance between the mate connector origins.

        """
        mc_counts = [len(mcos) for mcos in self.mc_origins_all]
        
        #offset2part = sizes_to_interval_tree(mc_counts)
        #part2offset = [0] + mc_counts[:-1]

        flattened_origins = [origin for mc_origins in self.mc_origins_all for origin in mc_origins]
        flattened_rots = [rot for mc_rots in self.mc_rots_homogenized_all for rot in mc_rots]
        mc_axis = [rot[:,2] for rot in flattened_rots]
        mc_ray = [np.concatenate([origin, axis]) for origin, axis in zip(flattened_origins, mc_axis)]
        mc_quat = [R.from_matrix(rot).as_quat() for rot in flattened_rots]
        matches = inter_group_matches(mc_counts, mc_ray, 6, self.epsilon_rel)
        coincident_proposals = self._matches_to_proposals(matches, 0)

        all_axial_proposals = []
        all_planar_proposals = []

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
            axial_matches = inter_group_matches(mc_counts, projected_origins, 2, self.epsilon_rel, point_to_grouped_map=index_map)
            axial_proposals = self._matches_to_proposals(axial_matches, 1)
            all_axial_proposals.append(axial_proposals)
            
            # print('projecting to find planes')
            projected_z = list(subset_stacked @ proj_dir)
            z_quat = [np.concatenate([mc_quat[same_dir_ind], [z_dist]]) for same_dir_ind, z_dist in zip(same_dir_inds, projected_z)]
            planar_matches = inter_group_matches(mc_counts, z_quat, 5, self.epsilon_rel, point_to_grouped_map=index_map)
            planar_proposals = self._matches_to_proposals(planar_matches, 2)
            all_planar_proposals.append(planar_proposals)
        
        all_proposals = self._join_proposals(*all_planar_proposals)
        all_proposals = self._join_proposals(all_proposals, *all_axial_proposals)
        all_proposals = self._join_proposals(all_proposals, coincident_proposals)

        return all_proposals


    def create_batches(self, mates, max_z_groups=10, max_mc_pairs=100000):
        """
        Create a list of batches of BrepData objects for this assembly. Returns False if any part has too many topologies to fit in a batch.
        """
        
        self.mate_stats = [dict() for mate in mates]
        
        #datalists = []
        # curr_datalist = []
        # topo_count = 0
        # for part in self.normalized_parts:
        #     if part.num_topologies > max_topologies:
        #         return None
        #     topo_count += part.num_topologies

        #     if topo_count > max_topologies:
        #         datalists.append(curr_datalist)
        #         curr_datalist = []
        #         topo_count = 0
            
        #     data = BrepGraphData(part, self.use_uvnet_features)
        #     curr_datalist.append(data)
        # datalists.append(curr_datalist)

        #batches = [Batch.from_data_list(lst) for lst in datalists]

        datalist = [BrepGraphData(part, uvnet_features=self.use_uvnet_features) for part in self.normalized_parts]
        batch = Batch.from_data_list(datalist)
        
        #proposal indices are local to each part's mate connector set
        #get mate connector and its topological references, and offset them by the appropriate amount based on where they are in batch
        part2offset = [0] * len(self.normalized_parts)
        offset = 0
        for i,part in enumerate(self.normalized_parts[:-1]):
            offset += part.num_topologies
            part2offset[i+1] = offset

        proposal_start = time.time()
        proposals = self.mate_proposals(max_z_groups=max_z_groups)
        proposal_duration = time.time() - proposal_start
        self.stats['proposal_time'] = proposal_duration

        #record any ground truth mates that agree with the proposals
        match_start = time.time()
        missed_mates = 0
        missed_part_pairs = 0
        for i,(mate, mate_stat) in enumerate(zip(mates, self.mate_stats)):
            part1, part2, matches = self.find_mated_pairs(mate)
            if part1 > part2:
                part1, part2 = part2, part1
                matches = [(match[1], match[0]) for match in matches]
            
            mate_stat['num_matches'] = len(matches)
            mate_stat['found_by_heuristic'] = False

            part_pair = part1, part2
            if part_pair in proposals:
                num_matches = 0
                for match in matches:
                    if match in proposals[part_pair]:
                        proposals[part_pair][match][2] = i
                        num_matches += 1
                if num_matches > 0:
                    mate_stat['found_by_heuristic'] = True
            else:
                missed_part_pairs += 1
            if not mate_stat['found_by_heuristic']:
                missed_mates += 1
        match_duration = time.time() - match_start
        self.stats['match_time'] = match_duration
        self.stats['missed_mates'] = missed_mates
        self.stats['missed_part_pairs'] = missed_part_pairs


        truncation_start = time.time()
        mc_keys = []
        for part_pair in proposals:
            for mc_pair in proposals[part_pair]:
                mc_keys.append((part_pair, mc_pair, proposals[part_pair][mc_pair][2]))
        
        #truncate mc pairs (keep positive matches, and all others up to the maximum number)
        if len(mc_keys) > max_mc_pairs:
            self.stats['truncated_mc_pairs'] = True
            pair_indices_final=[]
            pair_indices_false=[]
            for i,(part_pair, mc_pair, mateIndex) in enumerate(mc_keys):
                if mateIndex >= 0:
                    pair_indices_final.append(i)
                else:
                    pair_indices_false.append(i)
            N_true = len(pair_indices_final)
            N_remainder = max_mc_pairs - N_true
            random.shuffle(pair_indices_false)
            pair_indices_final += pair_indices_false[:N_remainder]
        else:
            self.stats['truncated_mc_pairs'] = False
            pair_indices_final = list(range(len(mc_keys)))

        truncation_duration = time.time() - truncation_start
        self.stats['truncation_time'] = truncation_duration
        
        #add mc_pairs, mc_pair_type, and mc_pair_labels sets to assembly data
        #mc_pairs data we want is:
        # - or1, loc1, type1, or2, loc2, type2, proposal type, euclidian distance
        conversion_start = time.time()
        batch.mc_pairs = torch.empty((6, len(pair_indices_final)), dtype=torch.int64)
        batch.mc_proposal_feat = torch.empty((2, len(pair_indices_final)), dtype=torch.float32)
        batch.mc_pair_labels = torch.zeros(len(pair_indices_final), dtype=torch.float32)
        batch.mc_pair_type = torch.zeros(len(pair_indices_final), dtype=torch.int64)
        
        for col_index, i in enumerate(pair_indices_final):
            part_pair, mc_pair, mateId = mc_keys[i]
            mcs = [self.normalized_parts[pi].all_mate_connectors[mci] for pi, mci in zip(part_pair, mc_pair)]
            topo_offsets = [part2offset[partid] for partid in part_pair]
            batch.mc_pairs[:,col_index] = torch.tensor([mcs[0].orientation_inference.topology_ref + topo_offsets[0], mcs[0].location_inference.topology_ref + topo_offsets[0], mcs[0].location_inference.inference_type.value,
                          mcs[1].orientation_inference.topology_ref + topo_offsets[1], mcs[1].location_inference.topology_ref + topo_offsets[1], mcs[1].location_inference.inference_type.value], dtype=torch.int64)
            
            proposal_feat = proposals[part_pair][mc_pair]
            batch.mc_proposal_feat[:,col_index] = torch.tensor(proposal_feat[:2], dtype=torch.float32)
            
            if mateId >= 0:
                batch.mc_pair_labels[col_index] = 1
                batch.mc_pair_type[col_index] = mate_types.index(mates[mateId].type)
        
        conversion_duration = time.time() - conversion_start
        self.stats['conversion_time'] = conversion_duration

        return batch

#test that this assembly has all mates matched and found by heuristics
if __name__ == '__main__2':
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
    batch = assembly_info.create_batches(mates)
    assert(all([mate_stat['found_by_heuristic'] for mate_stat in assembly_info.mate_stats]))


if __name__ == '__main__':
    import onshape.brepio as brepio
    datapath = '/projects/grail/benjones/cadlab'
    loader = brepio.Loader(datapath)
    #geo, mates = loader.load_flattened('87688bb8ccd911995ddc048c_6313170efcc25a6f36e56906_8ee5e722ed4853b12db03877.json', skipInvalid=True, geometry=False)
    geo, mates = loader.load_flattened('e04c8a49d30adb3a5c0f1deb_3d9b45359a15b248f75e41a2_070617843f30f132ab9e6661.json', skipInvalid=True, geometry=False)
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
    print('topologies:', assembly_info.num_topologies())

    batch = assembly_info.create_batches(mates)

    # for mate in mates:
    #     part1, part2, matches = assembly_info.find_mated_pairs(mate)
    #     print('matches for mate',mate.name,'(', part1, part2,'):',len(matches))
    
    # proposals = assembly_info.mate_proposals()
    # print(sum([len(proposals[part_pair]) for part_pair in proposals]))
    # print(set([part_pair for part_pair in proposals]))