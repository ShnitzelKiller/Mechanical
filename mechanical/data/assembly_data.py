import logging
import os
import random
import time
import numpy as np
import numpy.linalg as LA
import torch
#from torch_geometric.data import Batch
from pspart import NNHash
import pspy
from mechanical.utils import find_neighbors, inter_group_matches, cluster_points, homogenize_frame, external_adjacency_list_from_brepdata, compute_basis, apply_transform, MateTypes, mate_types
from scipy.spatial.transform import Rotation as R
from automate import PartFeatures, part_to_graph, flatbatch
import torch_scatter
import trimesh.proximity as proximity
import trimesh.interval as interval
import trimesh
import copy

from mechanical.utils import homogenize, homogenize_sign, inter_group_cluster_points, project_to_plane

def bboxes_overlap(bbox1, bbox2, margin=0):
    overlap = True
    marginarray = np.array([-margin, margin])
    for i in range(3):
        intersects, _ = interval.intersection(bbox1[:,i] + marginarray, bbox2[:,i] + marginarray)
        overlap = overlap and intersects
    return overlap

def get_num_topologies(part):
    return sum(len(getattr(part.brep.nodes, key)) for key in ['faces', 'vertices', 'edges','loops'])

def global_bounding_box(parts, transforms=None):
    if transforms is None:
        allpoints = [part.mesh.V for part in parts if part.mesh.V.shape[0] > 0]
    else:
        allpoints = [(tf[:3,:3] @ part.mesh.V.T + tf[:3,3,np.newaxis]).T for tf, part in zip(transforms, parts) if part.mesh.V.shape[0] > 0]
    if len(allpoints) == 0:
        return None
    minPt = np.array([points.min(axis=0) for points in allpoints]).min(axis=0)
    maxPt = np.array([points.max(axis=0) for points in allpoints]).max(axis=0)
    return np.vstack([minPt, maxPt])

def _transform_matches(matches, index_maps):
    return [(index_maps[0][match[0]], index_maps[1][match[1]]) for match in matches]

def assembly_info_from_onshape(occs, datapath):
    occ_ids = list(occs)
    part_paths = []
    transforms = []

    for occ in occs:
        rel_path = os.path.join(*[occs[occ][1][k] for k in ['documentId','documentMicroversion','elementId','fullConfiguration']], f'{occs[occ][1]["partId"]}.xt')
        path = os.path.join(datapath, 'data/models', rel_path)
        assert(os.path.isfile(path))
        part_paths.append(path)
        transforms.append(occs[occ][0])
    
    return AssemblyInfo(part_paths, transforms, occ_ids)

class AssemblyInfo:

    def recompute_map(self):
        self.occ_to_index = dict()
        for i,occ in enumerate(self.occ_ids):
            self.occ_to_index[occ] = i
              

    def __init__(self, part_paths, transforms, occ_ids, epsilon_rel=0.001, use_uvnet_features=False, max_topologies=10000, include_mcfs=True, precompute=False):
        self.epsilon_rel = epsilon_rel
        self.use_uvnet_features = use_uvnet_features
        self.stats = dict()
        self.mate_stats = []
        self.invalid_occs = set()
        self.include_mcfs = include_mcfs

        self.parts = []
        self.part_paths = []
        self.part_caches = []
        self.occ_transforms = []
        self.occ_ids = []

        self.mc_origins_all = []
        self.mc_axes_all = []
        self.mc_axes_homogenized_all = []

        #self.mco_hashes = [] #spatial hashes of origins
        #self.mcz_hashes = [] #spatial hashes of homogenized z direction
        #self.mcr_hashes = [] #spatial hashes of origin + homogenized z dir

        self.computed_data_structures = False
        part_opts = pspy.PartOptions()
        part_opts.default_mcfs = include_mcfs
        part_opts.num_uv_samples = 0
        for path, occ, tf in zip(part_paths, occ_ids, transforms):
            assert(os.path.isfile(path))
            with open(path) as partf:
                part_str = partf.read()
                logging.debug(f'loading initial part {occ}')
                part = pspy.Part(part_str, part_opts)
                if part.is_valid:
                    self.part_paths.append(path)
                    self.part_caches.append(part_str)
                    self.parts.append(part)
                    self.occ_transforms.append(tf)
                    self.occ_ids.append(occ)
                else:
                    self.invalid_occs.add(occ)
        
        self.recompute_map()

        assert(all(len(x)==len(self.parts) for x in [self.part_paths, self.part_caches, \
                        self.occ_transforms, self.occ_to_index, self.occ_ids]))

        num_topologies = self.num_topologies()

        self.valid = False
        self.stats['num_total_parts'] = len(part_paths)
        self.stats['num_invalid_loaded_parts'] = len(self.invalid_occs)
        self.stats['num_topologies'] = num_topologies
        self.stats['too_big'] = num_topologies > max_topologies

        if num_topologies <= max_topologies and len(self.parts) > 0:
            self.bbox = global_bounding_box(self.parts, self.occ_transforms)
            self.stats['invalid_bbox'] = self.bbox is None
            
            if self.bbox is not None:
                self.valid = True
                minPt, maxPt = self.bbox
                self.median = self.bbox.mean(axis=0)
                dims = maxPt - minPt
                self.maxdim = max(dims)
                self.stats['maxdim'] = self.maxdim

                self.transform_parts()
                
                # if self.valid:
                #     self.precompute_geometry()

                    
        if self.valid: 
            if len(self.occ_ids) > 0:
                assert([self.occ_to_index[occi] == i for i, occi in enumerate(self.occ_ids)])
                assert([self.occ_ids[self.occ_to_index[occi]] == occi for occi in self.occ_to_index])
        self.stats['initialized'] = self.valid
        if self.valid and precompute:
            self.precompute_geometry()
        

    def transform_parts(self):
        p_normalized = np.identity(4, dtype=float)
        p_normalized[:3,3] = -self.median
        p_normalized[3,3] = self.maxdim #todo: figure out if this is double the factor

        norm_matrices = [p_normalized @ tf for tf in self.occ_transforms]
        transformed_parts = []
        for cached_part, mat, occ in zip(self.part_caches, norm_matrices, self.occ_ids):
            part_opts = pspy.PartOptions()
            part_opts.transform = True
            part_opts.transform_matrix = mat
            part_opts.default_mcfs = self.include_mcfs
            part_opts.num_uv_samples = 10 if self.use_uvnet_features else 0
            logging.debug(f'loading transformed part {occ}')
            transformed_parts.append(pspy.Part(cached_part, part_opts))

        self.stats['num_normalized_parts_with_discrepancies'] = self._num_normalized_parts_with_discrepancies(transformed_parts)

        #filter out invalid ones from the existing structures:
        num_invalid = sum([not part.is_valid for part in transformed_parts])
        self.stats['num_invalid_transformed_parts'] = num_invalid
        if num_invalid > 0:
            if num_invalid < len(self.parts):
                for part, occ in zip(transformed_parts, self.occ_ids):
                    if not part.is_valid:
                        self.invalid_occs.add(occ)
                
                self.parts, self.part_paths, self.part_caches, norm_matrices, self.occ_transforms, self.occ_ids = zip(*[group for group in zip(transformed_parts, self.part_paths, self.part_caches, norm_matrices, self.occ_transforms, self.occ_ids) if group[0].valid])
                self.recompute_map()
            else:
                self.valid = False
                return
        else:
            self.parts = transformed_parts

        #these may change if we choose to handle transforms differently. TODO if changing: also handle the epsilons/scaling thresholds!
        self.mate_transforms = norm_matrices
        self.part_transforms = [np.identity(4) for part in self.parts]


    def precompute_geometry(self):
        if not self.include_mcfs:
            raise ValueError
        if not self.computed_data_structures:
            #analyze mate connectors
            for tf,part in zip(self.part_transforms, self.parts):
                assert(part.is_valid)
                mc_origins = []
                mc_axes = []
                for mc in part.default_mcfs:
                    mc_origins.append(apply_transform(tf, mc.origin, is_points=True))
                    mc_axes.append(apply_transform(tf, mc.axis, is_points=False))

                self.mc_axes_all.append(mc_axes)
                self.mc_origins_all.append(mc_origins)
                mc_axes_homogenized = [homogenize_sign(vec)[0] for vec in mc_axes]
                self.mc_axes_homogenized_all.append(mc_axes_homogenized)

                #mc_rays = [np.concatenate([origin,axis]) for origin, axis in zip(mc_origins, mc_axes_homogenized)]
                #origin_hash = NNHash(mc_origins, 3, self.epsilon_rel)
                #z_hash = NNHash([axis for axis in mc_axes_homogenized], 3, self.epsilon_rel)
                #ray_hash = NNHash(mc_rays, 6, self.epsilon_rel)
                #self.mco_hashes.append(origin_hash)
                #self.mcz_hashes.append(z_hash)
                #self.mcr_hashes.append(ray_hash)
            self.computed_data_structures = True
            list_of_lists = [self.parts, self.part_paths, self.part_caches, \
                            self.occ_transforms, self.occ_to_index, self.occ_ids, self.mc_origins_all, \
                            self.mc_axes_all, self.mc_axes_homogenized_all, \
                            #self.mco_hashes, self.mcz_hashes, self.mcr_hashes, \
                            self.mate_transforms, self.part_transforms]
            try:
                assert(all(len(x)==len(self.parts) for x in list_of_lists))
            except AssertionError as e:
                e.args += (self.stats, [len(lst) for lst in list_of_lists])
                raise

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
        self.precompute_geometry()

        if len(mate.matedEntities) != 2:
            logging.warning('INCORRECT NUMBER OF MATED ENTITIES:',len(mate.matedEntities))
            raise ValueError

        
        try:
            part_indices = [self.occ_to_index[me[0]] for me in mate.matedEntities]
        except KeyError:
            return -1, -1, []

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
        
        if mate.type == MateTypes.FASTENED:
            matches = find_neighbors(*[[np.concatenate([origin, rot[:,2]]) for origin, rot in zip(self.mc_origins_all[ind], self.mc_rots_homogenized_all[ind])] for ind in part_indices], 6, self.epsilon_rel)
        elif mate.type == MateTypes.BALL:
            same_origin_inds = [self.mco_hashes[ind].get_nearest_points(mate_origins[0]) for ind in part_indices]
            matches = [(i, j) for i in same_origin_inds[0] for j in same_origin_inds[1]]
        else:
            xy_plane = mate_rots_homogenized[0][:,:2]
            z_dir = mate_rots_homogenized[0][:,2]
            same_dir_inds = [list(self.mcz_hashes[ind].get_nearest_points(z_dir)) for ind in part_indices]
            if any([len(same_dir_i) == 0 for same_dir_i in same_dir_inds]):
                matches = []
            else:
                subset_stacked = [np.array([self.mc_origins_all[ind][i] for i in same_dir_inds_i]) for ind, same_dir_inds_i in zip(part_indices, same_dir_inds)]
                if mate.type == MateTypes.CYLINDRICAL or mate.type == MateTypes.REVOLUTE or mate.type == MateTypes.SLIDER:
                    projected_origins = [list(subset_stacked_i @ xy_plane) for subset_stacked_i in subset_stacked] #origins of mcfs with the same z direction projected onto a shared xy plane
                    
                    if mate.type == MateTypes.SLIDER:
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
                        if any([len(axial_indices_i) == 0 for axial_indices_i in axial_indices]):
                            matches = []
                        elif mate.type == MateTypes.CYLINDRICAL:
                            matches = [(i, j) for i in axial_indices[0] for j in axial_indices[1]]
                        elif mate.type == MateTypes.REVOLUTE:
                            subset_axial = [np.array([self.mc_origins_all[ind][i] for i in axial_indices_i]) for ind, axial_indices_i in zip(part_indices, axial_indices)]
                            z_positions = [list((subset_stacked_i @ z_dir)[:,np.newaxis]) for subset_stacked_i in subset_axial]
                            matches = find_neighbors(*z_positions, 1, self.epsilon_rel) #neighbors are found in the subset of axial mcs
                            matches = _transform_matches(matches, axial_indices)
                        else:
                            assert(False)

                elif mate.type == MateTypes.PLANAR:
                    z_positions = [list((subset_stacked_i @ z_dir)[:,np.newaxis]) for subset_stacked_i in subset_stacked]
                    matches = find_neighbors(*z_positions, 1, self.epsilon_rel) #neighbors are found in the subset of same direction mcs
                    matches = _transform_matches(matches, same_dir_inds)
                elif mate.type == MateTypes.PARALLEL:
                    matches = [(i, j) for i in same_dir_inds[0] for j in same_dir_inds[1]]
                elif mate.type == MateTypes.PIN_SLOT:
                    matches = []
                else:
                    raise ValueError
        return part_indices[0], part_indices[1], matches

    def num_mate_connectors(self):
        return sum([len(origins) for origins in self.mc_origins_all])

    def num_topologies(self):
        return sum([get_num_topologies(part) for part in self.parts])
    
    def num_invalid_parts(self):
        return sum([not npart.is_valid for npart in self.parts])
    
    def _num_normalized_parts_with_discrepancies(self, norm_parts):
        num_discrepancies = 0
        for npart, part in zip(norm_parts, self.parts):
            if get_num_topologies(npart) != get_num_topologies(part):
                num_discrepancies += 1
            else:
                if self.include_mcfs:
                    if len(npart.default_mcfs) != len(part.default_mcfs):
                        num_discrepancies += 1
                    else:
                        mc_data = set()
                        for mc in part.default_mcfs:
                            mc_dat = (mc.ref.axis_ref.reference_index, mc.ref.axis_ref.reference_type, mc.ref.origin_ref.reference_index, mc.ref.origin_ref.reference_type, mc.ref.origin_ref.inference_type.value)
                            mc_data.add(mc_dat)
                        mc_data_normalized = set()
                        for mc in npart.default_mcfs:
                            mc_dat = (mc.ref.axis_ref.reference_index, mc.ref.axis_ref.reference_type, mc.ref.origin_ref.reference_index, mc.ref.origin_ref.reference_type, mc.ref.origin_ref.inference_type.value)
                            mc_data_normalized.add(mc_dat)
                        if mc_data != mc_data_normalized:
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
    

    def mc_clusters(self, max_z_groups=10):
        """
        Compute clusters of MCs that have the same direction (and span at least 2 parts), and return
        dir_data: a dict from a cluster index -> (indices of other MCs in each cluster, and their origins projected to the plane with that direction normal)
        mc_part_labels: integer part labels of each mc in the assembly
        """

        self.precompute_geometry()
        mc_part_labels = [i for i,l in enumerate(self.mc_origins_all) for x in l]

        flattened_origins = [origin for mc_origins in self.mc_origins_all for origin in mc_origins]
        flattened_axes = [axis for mc_axes in self.mc_axes_homogenized_all for axis in mc_axes]
        flattened_rots = [compute_basis(axis) for axis in flattened_axes]

        axis_hash = NNHash(flattened_axes, 3, self.epsilon_rel)
        pairs_to_clusters, dir_clusters = inter_group_cluster_points(axis_hash, flattened_axes, mc_part_labels)
        group_indices = list(dir_clusters)
        group_indices.sort(key=lambda k: len(dir_clusters[k]), reverse=True)
        #dir_to_projected_origins = {}
        #dir_to_same_dir_inds = {}
        dir_data = {}

        for ind in group_indices[:max_z_groups]:
            same_dir_inds = list(dir_clusters[ind])
            xy_plane = flattened_rots[ind].T
            proj_dir = flattened_axes[ind]
            subset_stacked = np.array([flattened_origins[i] for i in same_dir_inds])
            projected_origins = subset_stacked @ xy_plane
            projected_z = subset_stacked @ proj_dir
            dir_data[ind] = (same_dir_inds, projected_origins, projected_z)
        
        return pairs_to_clusters, dir_data, mc_part_labels


    def axis_proposals(self, max_z_groups=10):
        """
        return:
        pairs_to_axes: pair -> (MC dir cluster index -> [indices of axis clusters in all_clusters[dir]])
        all_axis_clusters: MC dir cluster index -> (MC axis cluster index -> same axis indices)
        pairs_to_dir_clusters: pair -> MC dir cluster index
        dir_data: MC dir cluster index -> ([same dir inds], [projected origins])
        mc_part_labels: MC index -> part label
        """


        pairs_to_dir_clusters, dir_data, mc_part_labels = self.mc_clusters(max_z_groups=max_z_groups)
        
        pairs_to_axes = {} #dictionary from pairs of parts to a dictionary from an axis direction MC to indices of origin MCs that are unique w.r.t. the rays they define with the axis
        all_axis_clusters = {} #dictionary from direction group indices to all clusters of projected points, as MC indices

        for ind in dir_data:
            same_dir_inds, projected_origins, _ = dir_data[ind]
            same_dir_mc_part_labels = [mc_part_labels[k] for k in same_dir_inds]
            projected_hash = NNHash(projected_origins, 2, self.epsilon_rel)
            pairs_to_projected_clusters, clusters_to_point_ids = inter_group_cluster_points(projected_hash, projected_origins, same_dir_mc_part_labels)
            #convert to global indices
            all_axis_clusters[ind] = {same_dir_inds[c] : [same_dir_inds[k] for k in clusters_to_point_ids[c]] for c in clusters_to_point_ids}
            for pair in pairs_to_projected_clusters:
                if pair not in pairs_to_axes:
                    pairs_to_axes[pair] = {}
                pairs_to_axes[pair][ind] = [same_dir_inds[c] for c in pairs_to_projected_clusters[pair]]
                for k in pairs_to_axes[pair][ind]: #remove
                    assert(k in all_axis_clusters[ind])
        
        return pairs_to_axes, all_axis_clusters, pairs_to_dir_clusters, dir_data, mc_part_labels
    

    def mate_proposals(self, max_z_groups=10, coincident_only=False):
        """
        Find probable mate locations
        `max_z_groups`: maximum number of clusters of mate connector z directions to consider as possible axes
        Returns: Dictionary from (partId1, partId2) -> (mc1, mc2) -> (ptype, dist, pointer to mate (default -1))
        where ptype is the type of proposal, and can be:
        - 0: coincident
        - 1: coaxial
        - 2: coplanar

        and dist is the euclidian distance between the mate connector origins.

        """

        pairs_to_clusters, dir_data, mc_part_labels = self.mc_clusters(max_z_groups=max_z_groups)
        
        mc_counts = [len(mcos) for mcos in self.mc_origins_all]

        flattened_origins = [origin for mc_origins in self.mc_origins_all for origin in mc_origins]
        flattened_axes = [axis for mc_axes in self.mc_axes_homogenized_all for axis in mc_axes]
        mc_ray = [np.concatenate([origin, axis]) for origin, axis in zip(flattened_origins, flattened_axes)]

        matches = inter_group_matches(mc_counts, mc_ray, 6, self.epsilon_rel)
        coincident_proposals = self._matches_to_proposals(matches, 1)
        
        if coincident_only:
            return coincident_proposals
        
        flattened_rots = [compute_basis(axis) for axis in flattened_axes]
        mc_quat = [R.from_matrix(rot).as_quat() for rot in flattened_rots]

        all_axial_proposals = []
        all_planar_proposals = []

        for ind in dir_data:
            same_dir_inds, projected_origins, projected_z = dir_data[ind]
            index_map = lambda x: same_dir_inds[x]
            
            axial_matches = inter_group_matches(mc_counts, projected_origins, 2, self.epsilon_rel, point_to_grouped_map=index_map)
            axial_proposals = self._matches_to_proposals(axial_matches, 2)
            all_axial_proposals.append(axial_proposals)
            
            z_quat = [np.concatenate([mc_quat[same_dir_ind], [z_dist]]) for same_dir_ind, z_dist in zip(same_dir_inds, projected_z)]
            planar_matches = inter_group_matches(mc_counts, z_quat, 5, self.epsilon_rel, point_to_grouped_map=index_map)
            planar_proposals = self._matches_to_proposals(planar_matches, 3)
            all_planar_proposals.append(planar_proposals)

        all_proposals = self._join_proposals(*all_planar_proposals)
        all_proposals = self._join_proposals(all_proposals, *all_axial_proposals)
        all_proposals = self._join_proposals(all_proposals, coincident_proposals)

        return all_proposals

    def part_distances(self, threshold):
        """
        Returns the relative distances between parts (as a fraction of bbox dim)
        """
        aabbs = []
        bboxes = []
        self.stats['num_degenerate_bboxes'] = 0
        for part in self.parts:
            bbox = part.bounding_box()
            if not part.is_valid:
                self.stats['num_degenerate_bboxes'] += 1
            bboxes.append(bbox)
            mesh = trimesh.Trimesh(vertices=part.mesh.V, faces=part.mesh.F)
            tree = proximity.ProximityQuery(mesh)
            aabbs.append(tree)
        
        pairs = dict()
        if self.stats['num_degenerate_bboxes'] == 0:
            N = len(self.parts)
            for i in range(N):
                for j in range(i+1,N):
                    bbox1 = bboxes[i]
                    bbox2 = bboxes[j]
                    if bboxes_overlap(bbox1, bbox2, threshold):
                        closest, dists, id = aabbs[i].on_surface(self.parts[j].mesh.V)
                        minDist = dists.min()
                        closest, dists, id = aabbs[j].on_surface(self.parts[i].mesh.V)
                        minDist = min(minDist, dists.min())
                        pairs[(i,j)] = minDist
        return pairs
    
    def find_mate_path(self, mates, part_ind1, part_ind2, threshold=1e-5, dir_proposals=None, axes_proposals=None, require_axis=False, reference_mate=None):
        """
        adj: external adjacency list
        mates: corresponding mate objects with the coordinate frames transformed to global space (This is not the default, so be careful)
        part_ind1 and part_ind2: the indices in the occurrence list of the two parts to find a path between (in sorted order)
        dir_proposals and axes_proposals: list of directions, or list of tuples of (direction, origin) of axes/rays that can serve as mate definitions
        require_axis: Always search for a shared axis between parts; not just shared direction, even for sliders
        """
        adj = external_adjacency_list_from_brepdata(self.occ_ids, mates)
        finalized = dict()

        frontier = {part_ind1: (part_ind1, -1, 0)} #partId -> ((prevPart, mate, dist)
        found=False

        while len(frontier) > 0:
            curr, (prev, mateId, dist) = min(frontier.items(), key=lambda x: x[1][2])
            del frontier[curr]
            if curr in finalized:
                continue
            finalized[curr] = (prev, mateId, dist)
            if curr == part_ind2:
                found=True
                break
            for neighbor, mateId in adj[curr]:
                if neighbor not in frontier or dist + 1 < frontier[neighbor][2]:
                    frontier[neighbor] = (curr, mateId, dist + 1)
        chain_types = []
        if found:
            lastpart = part_ind2
            axis = None
            origin = None
            slides = False
            rotates = False
            valid_chain = True

            while lastpart != part_ind1:
                prev, mateId, dist = finalized[lastpart]

                #check that each mate is compatible with the DOFs so far
                mate = mates[mateId]
                lastpart = prev
                chain_types.append(mate.type)
                
                if mate.type == MateTypes.FASTENED or mate.type == MateTypes.CYLINDRICAL or mate.type == MateTypes.SLIDER or mate.type == MateTypes.REVOLUTE:
                    #accumulate DOFs
                    if mate.type == MateTypes.SLIDER:
                        slides = True
                    elif mate.type == MateTypes.CYLINDRICAL:
                        slides = True
                        rotates = True
                    elif mate.type == MateTypes.REVOLUTE:
                        rotates = True
                    
                    #check for consistency of axes
                    if mate.type == MateTypes.REVOLUTE or mate.type == MateTypes.SLIDER or mate.type == MateTypes.CYLINDRICAL:
                        newaxis = homogenize_frame(mate.matedEntities[0][1][1], z_flip_only=True)[:,2]
                        if axis is None:
                            axis = newaxis
                            norm2 = np.dot(axis, axis)
                        else:
                            if not np.allclose(newaxis, axis, rtol=0, atol=threshold):
                                valid_chain = False
                                break
                    if mate.type == MateTypes.REVOLUTE or mate.type == MateTypes.CYLINDRICAL:
                        neworigin = mate.matedEntities[0][1][0]
                        if origin is None:
                            origin = neworigin
                            olddist = np.dot(axis, origin)/norm2
                            projectedpt = origin - olddist * axis
                        else:
                            newdist = np.dot(axis, neworigin)/norm2
                            newprojectedpt = neworigin - newdist * axis
                            if not np.allclose(projectedpt, newprojectedpt, rtol=0, atol=threshold):
                                valid_chain = False
                                break

                else:
                    valid_chain = False
                    break
            newtype = ""
            
            out_dict = {'origin': origin, 'axis': axis, 'type': "", 'chain_types': chain_types, 'valid': False, 'axis_index': -1, 'dir_index': -1}
            
            if valid_chain:
                if slides:
                    if rotates:
                        newtype = MateTypes.CYLINDRICAL
                    else:
                        newtype = MateTypes.SLIDER
                else:
                    if rotates:
                        newtype = MateTypes.REVOLUTE
                    else:
                        newtype = MateTypes.FASTENED
                out_dict['type'] = newtype.value
                if dir_proposals is not None and axes_proposals is not None:
                    found_mc_pair = False
                    if axis is not None:
                        if origin is not None:
                            #revolutes & cylindricals
                            projdist_old = np.dot(axis, origin)/norm2
                            projectedpt_old = origin - projdist_old * axis
                            for k,ax in enumerate(axes_proposals):
                                dir_homo, _ = homogenize_sign(ax[0])
                                if np.allclose(dir_homo, axis, rtol=0, atol=threshold):
                                    mcf_origin = ax[1]

                                    projdist_mcf = np.dot(axis, mcf_origin)/norm2
                                    projectedpt_mcf = mcf_origin - projdist_mcf * axis
                                    if np.allclose(projectedpt_mcf, projectedpt_old, rtol=0, atol=threshold):
                                        found_mc_pair = True
                                        out_dict['axis_index'] = k
                                        break
                        else:
                            #sliders
                            if require_axis:
                                for k,ax in enumerate(axes_proposals):
                                    dir_homo, _ = homogenize_sign(ax[0])
                                    if np.allclose(dir_homo, axis, rtol=0, atol=threshold):
                                        found_mc_pair = True
                                        out_dict['axis_index'] = k
                            else:
                                for k,dir in enumerate(dir_proposals):
                                    dir_homo, _ = homogenize_sign(dir)
                                    if np.allclose(dir_homo, axis, rtol=0, atol=threshold):
                                        found_mc_pair = True
                                        out_dict['dir_index'] = k
                                        break
                    
                    if found_mc_pair:
                        out_dict['valid'] = True

                elif reference_mate is not None:
                    out_dict['true_type'] = reference_mate.type
                    #check that there are at least the degrees of freedom required by the reference mate
                    mate_axis = reference_mate.matedEntities[0][1][1][:,2]
                    mate_origin = reference_mate.matedEntities[0][1][0]
                    mate_projected_origin = project_to_plane(mate_origin, mate_axis)
                    if origin is not None:
                        alt_projected_origin = project_to_plane(origin, mate_axis)
                    if reference_mate.type == MateTypes.SLIDER:
                        if slides and axis is not None:
                            if np.allclose(mate_axis, axis, rtol=0, atol=threshold):
                                out_dict['valid'] = True
                    elif reference_mate.type == MateTypes.CYLINDRICAL:
                        if slides and rotates and axis is not None and origin is not None:
                            if np.allclose(mate_axis, axis, rtol=0, atol=threshold) and np.allclose(mate_projected_origin, alt_projected_origin, rtol=0, atol=threshold):
                                out_dict['valid'] = True
                    elif reference_mate.type == MateTypes.REVOLUTE:
                        if rotates and axis is not None and origin is not None:
                            if np.allclose(mate_axis, axis, rtol=0, atol=threshold) and np.allclose(mate_projected_origin, alt_projected_origin, rtol=0, atol=threshold):
                                out_dict['valid'] = True
                    elif reference_mate.type == MateTypes.FASTENED:
                        out_dict['valid'] = True
                                
                                
                else:
                    out_dict['valid'] = True
            return out_dict
        return None

    def transform_mates(self, mates):
        mates2 = copy.deepcopy(mates)
        for mate, mate2 in zip(mates, mates2):
            part_indices = [self.occ_to_index[me[0]] for me in mate.matedEntities]
            newmes = []
            for partIndex, me in zip(part_indices, mate.matedEntities):
                origin_local = me[1][0]
                frame_local = me[1][1]
                cs = cs_from_origin_frame(origin_local, frame_local)
                cs_transformed = self.mate_transforms[partIndex] @ cs
                origin, rot = cs_to_origin_frame(cs_transformed)
                rot_homogenized = homogenize_frame(rot, z_flip_only=True)
                newmes.append((me[0], (origin, rot_homogenized)))
            mate2.matedEntities = newmes
        return mates2

    def validate_mates(self, mates):
        """
        Validate mates based on assembly context (whether an alternate path of mates (if any) agrees with the given one).
        Only valid for assemblies with only fastens, revolutes, sliders, and cylindrical mates.
        """
        matestats = []
        mates2 = self.transform_mates(mates)
        for j,mate in enumerate(mates2):
            mates_holdout = [m for k,m in enumerate(mates2) if k != j]
            pair = tuple(sorted((self.occ_to_index[mate.matedEntities[0][0]], self.occ_to_index[mate.matedEntities[1][0]])))
            #dir_proposals = pair_to_dirs[pair] if pair in pair_to_dirs else []
            #axes_proposals = pair_to_axes[pair] if pair in pair_to_axes else []
            pathinfo = self.find_mate_path(mates_holdout, pair[0], pair[1], threshold=self.epsilon_rel, reference_mate=mate)
            stat = dict()
            stat['has_alt_path'] = (pathinfo is not None)
            if pathinfo is not None:
                counts = {t2.value: sum(1 for t in pathinfo['chain_types'] if t == t2) for t2 in MateTypes}
                stat = {**stat, **counts}
                stat['feasible'] = pathinfo['valid']
                stat['chain_length'] = len(pathinfo['chain_types'])
                stat['alternate_axis'] = pathinfo['axis']
                stat['alternate_origin'] = pathinfo['origin']
                stat['alternate_type'] = pathinfo['type']
            else:
                stat['chain_length'] = -1
                stat['alternate_axis'] = None
                stat['alternate_origin'] = None
                stat['alternate_type'] = ""
                stat['feasible'] = True
            matestats.append(stat)
        return matestats

    def fill_missing_mates(self, mates, components, threshold, pair_to_dirs=None, pair_to_axes=None, require_axis=False):
        newmatestats = []
        #precompute transformed mate coordinate frames
        mates2 = self.transform_mates(mates)
        
        connections = {tuple(sorted((self.occ_to_index[mate.matedEntities[0][0]], self.occ_to_index[mate.matedEntities[1][0]]))) for mate in mates}
        distances = self.part_distances(threshold)

        for pair in distances:
            comp1 = components[pair[0]]
            comp2 = components[pair[1]]
            if comp1 != comp2 and pair not in connections:
                if pair in distances and distances[pair] < threshold:
                    #mc_pair, neworigin, newaxis, newtype, chain_types = self.find_mate_path(mates2, pair[0], pair[1], proposals[pair], threshold=self.epsilon_rel)
                    stat = {'part1':self.occ_ids[pair[0]],'part2':self.occ_ids[pair[1]]}
                    if pair_to_dirs is not None and pair_to_axes is not None:
                        dir_proposals = pair_to_dirs[pair] if pair in pair_to_dirs else []
                        axes_proposals = pair_to_axes[pair] if pair in pair_to_axes else []
                        if (require_axis or len(dir_proposals) == 0) and len(axes_proposals) == 0:
                            continue
                        stat['num_dir_proposals'] = len(dir_proposals)
                        stat['num_axes_proposals'] = len(axes_proposals)
                        pathinfo = self.find_mate_path(mates2, pair[0], pair[1], threshold=self.epsilon_rel, dir_proposals=dir_proposals, axes_proposals=axes_proposals, require_axis=require_axis)
                    else:
                        pathinfo = self.find_mate_path(mates2, pair[0], pair[1], threshold=self.epsilon_rel)
                    stat['added_mate'] = False
                    if pathinfo is not None:
                        counts = {t2.value: sum(1 for t in pathinfo['chain_types'] if t == t2) for t2 in MateTypes}
                        stat = {**stat, **counts}
                        stat['chain_length'] = len(pathinfo['chain_types'])
                        stat['axis'] = pathinfo['axis']
                        stat['origin'] = pathinfo['origin']
                        stat['type'] = pathinfo['type']
                        stat['axis_index'] = pathinfo['axis_index']
                        stat['dir_index'] = pathinfo['dir_index']
                        stat['added_mate'] = pathinfo['valid']
                    newmatestats.append(stat)
        return newmatestats
    
    def get_onshape(self):
        return {self.occ_ids[i]: (np.identity(4), self.parts[i]) for i in range(len(self.parts))}


    def create_batches(self):
        options = PartFeatures()
        options.mcfs = self.include_mcfs
        options.samples = self.use_uvnet_features
        datalist = [part_to_graph(part, options) for part in self.parts]
        return flatbatch(datalist)


    def create_batches_legacy(self, mates, max_z_groups=10, max_mc_pairs=100000):
        """
        Create a list of batches of BrepData objects for this assembly. Returns False if any part has too many topologies to fit in a batch.
        """
        self.stats['num_mates'] = len(mates)
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
        logging.info('Creating initial graph data')
        datalist = [BrepGraphData(part, uvnet_features=self.use_uvnet_features) for part in self.parts]
        logging.info('batching')
        batch = Batch.from_data_list(datalist)
        
        #proposal indices are local to each part's mate connector set
        #get mate connector and its topological references, and offset them by the appropriate amount based on where they are in batch
        logging.info('finding proposals')
        part2offset = [0] * len(self.parts)
        offset = 0
        for i,part in enumerate(self.parts[:-1]):
            offset += get_num_topologies(part)
            part2offset[i+1] = offset

        proposal_start = time.time()
        proposals = self.mate_proposals(max_z_groups=max_z_groups)

        #record "pooled" proposal stats for each part pair
        proposals_pooled = dict()
        for part_pair in proposals:
            mc_dict = proposals[part_pair]
            best_mc_pair = min(mc_dict, key=lambda x: mc_dict[x][1])
            proposals_pooled[part_pair] = mc_dict[best_mc_pair]

        self.stats['proposal_time'] = time.time() - proposal_start
        self.stats['num_proposals'] = sum(len(proposals[pair]) for pair in proposals)


        #record any ground truth mates that agree with the proposals
        match_start = time.time()
        missed_part_pairs = 0
        num_invalid_mates = 0
        for i,(mate, mate_stat) in enumerate(zip(mates, self.mate_stats)):
            part1, part2, matches = self.find_mated_pairs(mate)
            mate_stat['num_possible_mc_pairs'] = len(matches)
            mate_stat['has_invalid_parts'] = part1 < 0 or part2 < 0
            mate_stat['num_possible_proposed_mc_pairs'] = 0
            if mate_stat['has_invalid_parts']:
                num_invalid_mates += 1
            else:
                if part1 > part2:
                    part1, part2 = part2, part1
                    matches = [(match[1], match[0]) for match in matches]
                
                part_pair = part1, part2
                if part_pair in proposals:
                    num_matching_proposals = 0
                    for match in matches:
                        if match in proposals[part_pair]:
                            proposals[part_pair][match][2] = i
                            num_matching_proposals += 1
                    mate_stat['num_possible_proposed_mc_pairs'] = num_matching_proposals
                else:
                    missed_part_pairs += 1
            
            mate_stat['found_by_heuristic'] = mate_stat['num_possible_proposed_mc_pairs'] > 0
            mate_stat['has_matching_connectors'] = len(matches) > 0
            
        self.stats['invalid_mates'] = num_invalid_mates
        self.stats['num_mates_not_matched'] = sum([not mate_stat['has_matching_connectors'] for mate_stat in self.mate_stats])
        self.stats['num_mates_missed_by_heuristics'] = sum([not mate_stat['found_by_heuristic'] for mate_stat in self.mate_stats])
        self.stats['num_part_pairs_missed_by_heuristics'] = missed_part_pairs
        self.stats['match_time'] = time.time() - match_start


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

        self.stats['truncation_time'] = time.time() - truncation_start
        
        #add mc_pairs, mc_pair_type, and mc_pair_labels sets to assembly data
        #mc_pairs data we want is:
        # - or1, loc1, type1, or2, loc2, type2, proposal type, euclidian distance
        conversion_start = time.time()
        batch.mc_pairs = torch.empty((6, len(pair_indices_final)), dtype=torch.int64)
        batch.mc_proposal_feat = torch.empty((2, len(pair_indices_final)), dtype=torch.float32)
        batch.mc_pair_labels = torch.zeros(len(pair_indices_final), dtype=torch.float32)
        batch.mc_pair_type = torch.zeros(len(pair_indices_final), dtype=torch.int64)
        
        pair_indices_final.sort(key=lambda x: mc_keys[x][0])

        indices_by_part = dict()
        for i in pair_indices_final:
            part_pair, mc_pair, mateId = mc_keys[i]
            if part_pair not in indices_by_part:
                sublist = []
                indices_by_part[part_pair] = sublist
            else:
                sublist = indices_by_part[part_pair]
            sublist.append(i)

        self.stats['reordering_time'] = time.time() - truncation_start

        col_index = 0
        for part_pair in indices_by_part:
            key_indices = indices_by_part[part_pair]
            topo_offsets = [part2offset[partid] for partid in part_pair]
            pair_mc_lists = [self.parts[pi].default_mcfs for pi in part_pair]
            for i in key_indices:
                _, mc_pair, mateId = mc_keys[i]
                mcs = [pair_mc_list[mci] for pair_mc_list, mci in zip(pair_mc_lists, mc_pair)]
                batch.mc_pairs[:,col_index] = torch.tensor([mcs[0].orientation_inference.topology_ref + topo_offsets[0], mcs[0].location_inference.topology_ref + topo_offsets[0], mcs[0].location_inference.inference_type.value,
                            mcs[1].orientation_inference.topology_ref + topo_offsets[1], mcs[1].location_inference.topology_ref + topo_offsets[1], mcs[1].location_inference.inference_type.value], dtype=torch.int64)
                
                proposal_feat = proposals[part_pair][mc_pair]
                batch.mc_proposal_feat[:,col_index] = torch.tensor(proposal_feat[:2], dtype=torch.float32)
                
                if mateId >= 0:
                    batch.mc_pair_labels[col_index] = 1
                    batch.mc_pair_type[col_index] = mate_types.index(mates[mateId].type)
                
                col_index += 1

        assert(col_index == len(pair_indices_final))
        self.stats['conversion_time'] = time.time() - conversion_start

        batch.part_edges = torch.tensor([key for key in proposals]).T
        batch.part_pair_feats = torch.tensor([proposals_pooled[key][:2] for key in proposals], dtype=torch.float).T # row 0: ptype, 1: dist

        #fix broken bboxes
        degen_inds = (~batch.x[:,-6:].isfinite()).sum(dim=1).nonzero().flatten()
        degen_partids = batch.graph_idx.flatten()[degen_inds]
        batch.x[degen_inds,-6:] = 0

        self.stats['invalid_topo_bboxes:'] = len(degen_inds)
        self.stats['parts_with_invalid_topo_bboxes'] = len(torch.unique(degen_partids))

        return batch
    

    def validate_batch(self, batch, mates=None):
        pfuncs_onehot = batch.x[:,25]
        pfuncs = pfuncs_onehot.argmax(1)
        pfuncs_self = []
        for part in self.parts:
            graph = part.get_graph()
            pfuncs_self.append(graph.P)
        pfuncs_self = np.concatenate(pfuncs_self)

        #Todo: check mates as well

        return all(pfuncs_self == pfuncs.numpy())


if __name__ == '__main__':
    import mechanical.onshape as brepio
    datapath = '/projects/grail/benjones/cadlab'
    loader = brepio.Loader(datapath)
    geo, mates = loader.load_flattened('ca577462ab0f9ba6fb713102_e9b039af8f720b11d74037b5_e8f8d795c4941f6cf63a2a68.json', skipInvalid=True, geometry=False)
    occ_ids = list(geo.keys())
    part_paths = []
    transforms = []
    for id in occ_ids:
        path = os.path.join(datapath, 'data/models', *[geo[id][1][k] for k in ['documentId','documentMicroversion','elementId','fullConfiguration']], f'{geo[id][1]["partId"]}.xt')
        assert(os.path.isfile(path))
        part_paths.append(path)
        transforms.append(geo[id][0])
    #print(part_paths, occ_ids)
    assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, print, use_uvnet_features=True)
    print('num valid parts: ', len(assembly_info.parts))
    pairs = assembly_info.part_distances()
    print(pairs)

#test that this assembly has all mates matched and found by heuristics
if __name__ == 'createBatches':
    import mechanical.onshape as brepio
    datapath = '/projects/grail/benjones/cadlab'
    loader = brepio.Loader(datapath)
    #geo, mates = loader.load_flattened('d8a598174bcbceaf7e2194e5_a54ba742eaa71cdd4dcefbaa_f7d8ddb4a32a4bfde5a45d20.json', skipInvalid=True, geometry=False)
    geo, mates = loader.load_flattened('e4803faed1b9357f8db3722c_ce43730c0f1758f756fc271f_c00b5256d7e874e534c083e8.json', skipInvalid=True, geometry=False)
    occ_ids = list(geo.keys())
    part_paths = []
    transforms = []
    for id in occ_ids:
        path = os.path.join(datapath, 'data/models', *[geo[id][1][k] for k in ['documentId','documentMicroversion','elementId','fullConfiguration']], f'{geo[id][1]["partId"]}.xt')
        assert(os.path.isfile(path))
        part_paths.append(path)
        transforms.append(geo[id][0])
    #print(part_paths, occ_ids)
    assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, print, use_uvnet_features=True)
    print('num valid parts: ', len(assembly_info.parts))
    batch = assembly_info.create_batches([mate for mate in mates if len(mate.matedEntities) == 2])
    degen_inds = (~batch.x[:,35].isfinite()).nonzero().flatten()
    assert(len(degen_inds) == 0)
    print(batch)
    try:
        assert(all([mate_stat['found_by_heuristic'] for mate_stat in assembly_info.mate_stats]))
    except AssertionError as e:
        e.args += (assembly_info.stats,assembly_info.mate_stats)
        raise


if __name__ == '__main__2':
    import onshape.brepio as brepio
    datapath = '/projects/grail/benjones/cadlab'
    loader = brepio.Loader(datapath)
    #geo, mates = loader.load_flattened('87688bb8ccd911995ddc048c_6313170efcc25a6f36e56906_8ee5e722ed4853b12db03877.json', skipInvalid=True, geometry=False)
    #geo, mates = loader.load_flattened('e04c8a49d30adb3a5c0f1deb_3d9b45359a15b248f75e41a2_070617843f30f132ab9e6661.json', skipInvalid=True, geometry=False)
    geo, mates = loader.load_flattened('606a34c3ae6c15f66927c70c_d51a9d7f048ae0e6dd2fd019_58acb7145b75d3d8d1937568.json', skipInvalid=True, geometry=False)
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