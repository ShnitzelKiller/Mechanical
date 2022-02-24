from xml.etree.ElementInclude import include
from mechanical.data import AssemblyLoader, Stats, MeshLoader, LoadMCData, compose
import logging
import os
import h5py
import numpy as np
import torch
from mechanical.utils import homogenize_frame, homogenize_sign, cs_from_origin_frame, cs_to_origin_frame, project_to_plane
from mechanical.geometry import displaced_min_signed_distance, min_signed_distance_symmetric
from mechanical.utils import joinmeshes
from mechanical.data.util import df_to_mates, MateTypes, mates_equivalent

#import meshplot as mp
#mp.offline()
class DataVisitor:
    def __call__(self, data):
        return self.process(data)
    
    def process(self, data):
        return {}

class NoopVisitor(DataVisitor):
    def __init__(self, transforms):
        self.transforms = transforms

class BatchSaver(DataVisitor):
    def __init__(self, global_data, out_path, use_uvnet_features, epsilon_rel, max_topologies):
        self.transforms = AssemblyLoader(global_data, use_uvnet_features=use_uvnet_features, epsilon_rel=epsilon_rel, max_topologies=max_topologies)
        self.out_path = out_path

    def process(self, data):
        out_dict = {}
        stat = Stats(defaults={'invalid_bbox': False})
        stat.append(data.assembly_info.stats, data.ind)
        out_dict['assembly_stats'] = stat
        if data.assembly_info.stats['initialized'] and data.assembly_info.stats['num_invalid_loaded_parts'] == 0:
            batch = data.assembly_info.create_batches()
            torch_datapath = os.path.join(self.out_path, f'{data.ind}.dat')
            torch.save(batch, torch_datapath)
        return out_dict


class DisplacementPenalty(DataVisitor):
    def __init__(self, global_data, sliding_distance, rotation_angle, num_samples, include_vertices, meshpath):
        self.transforms = MeshLoader(meshpath)
        self.global_data = global_data
        self.sliding_distance = sliding_distance
        self.rotation_angle = rotation_angle
        self.samples = num_samples
        self.include_vertices = include_vertices
    
    def process(self, data):
        stats = Stats()
        part_subset = self.global_data.part_df.loc[[data.ind]]
        mate_subset = self.global_data.mate_df.loc[[data.ind]]
        rigidcomps = list(part_subset['RigidComponentID'])

        if part_subset.shape[0] == len(data.meshes):
            occ_ids = list(part_subset['PartOccurrenceID'])
            occ_to_index = dict()
            for i,occ in enumerate(occ_ids):
                occ_to_index[occ] = i
            
            meshes = [(mesh[0].numpy(), mesh[1].numpy()) for mesh in data.meshes]
            V = np.vstack([mesh[0] for mesh in meshes])
            minPt = V.min(axis=0)
            maxPt = V.max(axis=0)
            maxdim = (maxPt - minPt).max()
            rel_distance = self.sliding_distance * maxdim
            
            for j in range(mate_subset.shape[0]):
                mtype = mate_subset.iloc[j]['Type']
                if mtype == MateTypes.REVOLUTE or MateTypes.CYLINDRICAL or MateTypes.SLIDER:
                    sliding = False
                    rotating = False
                    if mtype == MateTypes.REVOLUTE:
                        rotating = True
                    elif mtype == MateTypes.CYLINDRICAL:
                        rotating = True
                        sliding = True
                    elif mtype == MateTypes.SLIDER:
                        sliding = True
                    
                    pid1 = occ_to_index[mate_subset.iloc[j]['Part1']]
                    pid2 = occ_to_index[mate_subset.iloc[j]['Part2']]

                    meshes_subset = [joinmeshes([mesh for k,mesh in enumerate(meshes) if rigidcomps[k] == rigidcomps[pid]]) for pid in [pid1, pid2]]
                    #meshes_subset = [meshes[pid1], meshes[pid2]]

                    #p = mp.plot(*meshes_subset[0])
                    #p.add_mesh(*meshes_subset[1])
                    #p.save("debugvis.html")

                    penalties_separation_sliding = [0,0]
                    penalties_penetration_sliding = [0,0]
                    penalties_separation_rotating = [0,0]
                    penalties_penetration_rotating = [0,0]

                    base_penalty = min_signed_distance_symmetric(meshes_subset, samples=self.samples, include_vertices = self.include_vertices)
                    base_penetration = max(0, -base_penalty) / rel_distance
                    base_separation = max(0, base_penalty) / rel_distance

                    tf = part_subset.iloc[pid1]['Transform']
                    origin = tf[:3,:3] @ mate_subset.iloc[j]['Origin1'] + tf[:3,3]
                    axis = tf[:3,:3] @ mate_subset.iloc[j]['Axes1'][:,2]

                    if sliding:
                        for k,displacement in enumerate([rel_distance, -rel_distance]):
                            sd = displaced_min_signed_distance(meshes_subset, axis, samples=self.samples, displacement=displacement, include_vertices=self.include_vertices)
                            penalties_penetration_sliding[k] = max(0, -sd) / rel_distance
                            penalties_separation_sliding[k] = max(0, sd) / rel_distance
                    if rotating:
                        for k, angle in enumerate([self.rotation_angle, -self.rotation_angle]):
                            sd = displaced_min_signed_distance(meshes_subset, axis, origin=origin, motion_type='ROTATE', samples=self.samples, displacement=angle, include_vertices=self.include_vertices)
                            penalties_penetration_rotating[k] = max(0, -sd) / rel_distance
                            penalties_separation_rotating[k] = max(0, sd) / rel_distance

                    row = {'Assembly': data.ind, 'type': mtype,
                                    'base_penetration': base_penetration,
                                    'base_separation': base_separation}

                    for k in range(2):
                        row[f'penalty_separation_sliding_{k}'] = penalties_separation_sliding[k]
                        row[f'penalty_penetration_sliding_{k}'] = penalties_penetration_sliding[k]
                        row[f'penalty_separation_rotating_{k}'] = penalties_separation_rotating[k]
                        row[f'penalty_penetration_rotating_{k}'] = penalties_penetration_rotating[k]
                        row[f'max_penalty_sliding_{k}'] = max(base_penalty, max(penalties_separation_sliding[k], penalties_penetration_sliding[k]))
                        row[f'max_penalty_rotating_{k}'] = max(base_penalty, max(penalties_separation_rotating[k], penalties_penetration_rotating[k]))
                    row[f'min_penalty_sliding'] = min(row[f'max_penalty_sliding_{k}'] for k in range(2))
                    row[f'min_penalty_rotating'] = min(row[f'max_penalty_rotating_{k}'] for k in range(2))

                    stats.append(row, index=mate_subset.iloc[j]['MateIndex'])
        return {'mate_penetration_stats': stats}

class MCDataSaver(DataVisitor):
    def __init__(self, global_data, mc_path, max_axis_groups, epsilon_rel, max_topologies, save_frames=False):
        self.transforms = AssemblyLoader(global_data, use_uvnet_features=False, epsilon_rel=epsilon_rel, max_topologies=max_topologies)
        self.max_groups = max_axis_groups
        self.mc_path = mc_path
        self.save_frames = save_frames
    
    def process(self, data):
        stats = Stats()
        assembly_info = data.assembly_info
        pairs_to_axes, all_axis_clusters, pairs_to_dir_clusters, dir_data, mc_part_labels = assembly_info.axis_proposals(max_z_groups=self.max_groups)
        for pair in pairs_to_axes:
            for dir_ind in pairs_to_axes[pair]:
                for ax in pairs_to_axes[pair][dir_ind]:
                    filtered_clusters = [[a for a in all_axis_clusters[dir_ind][ax] if mc_part_labels[a] == part] for part in pair]
                    stats.append({'dir_ind': dir_ind, 'axis_id': ax,
                                'part1': pair[0], 'part2': pair[1],
                                'part1_occ': assembly_info.occ_ids[pair[0]], 'part2_occ': assembly_info.occ_ids[pair[1]],
                                'num_MCs_1': len(filtered_clusters[0]), 'num_MCs_2': len(filtered_clusters[1]),
                                'assembly': data.ind})
        

        with h5py.File(os.path.join(self.mc_path, f"{data.ind}.hdf5"), "w") as f:
            f.create_dataset('mc_part_labels', data=mc_part_labels)

            if self.save_frames:
                mc_frames = f.create_group('mc_frames')
                for k in range(len(assembly_info.mc_origins_all)):
                    partgroup = mc_frames.create_group(str(k))
                    partgroup.create_dataset('origins', data=np.array(assembly_info.mc_origins_all[k]))
                    #rots_quat = np.array([R.from_matrix(rot).as_quat() for rot in assembly_info.mc_rots_all[k]])
                    #partgroup.create_dataset('rots', data=rots_quat)

            dir_groups = f.create_group('dir_clusters')
            for c in dir_data:
                dir = dir_groups.create_group(str(c))
                dir.create_dataset('indices', data=np.array(dir_data[c][0], dtype=np.int32))
                ax = dir.create_group('axial_clusters')
                for c2 in all_axis_clusters[c]:
                    ax.create_dataset(str(c2), data=np.array(all_axis_clusters[c][c2], dtype=np.int32))
            pairdata = f.create_group('pair_data') #here, indices mean the UNIQUE representative indices for retrieving each axis from the mc data
            for pair in pairs_to_dir_clusters:
                key = f'{pair[0]},{pair[1]}'
                pair_group = pairdata.create_group(key)
                dirgroup = pair_group.create_group('dirs')
                #dirgroup.create_dataset('values', data = np.array(pairs_to_dirs[pair]))
                dirgroup.create_dataset('indices', data = np.array(list(pairs_to_dir_clusters[pair]), dtype=np.int32))
                ax_group = pair_group.create_group('axes')
                if pair in pairs_to_axes:
                    for dir_ind in pairs_to_axes[pair]:
                        ax_cluster = ax_group.create_group(str(dir_ind))
                        #ax_cluster.create_dataset('values', data = np.array(origins))
                        ax_cluster.create_dataset('indices', data = np.array(pairs_to_axes[pair][dir_ind], dtype=np.int32))

        return {'mc_stats': stats}


def load_axes(path):
    with h5py.File(path, 'r') as f:
        pair_to_dirs = {}
        pair_to_axes = {}
        pair_data = f['pair_data']
        for key in pair_data.keys():
            pair = tuple(int(k) for k in key.split(','))
            inds = np.array(pair_data[key]['dirs']['indices'])
            pair_to_dirs[pair] = inds
            pair_to_axes[pair] = []
            for dir_index in pair_data[key]['axes'].keys():
                for ax_ind in pair_data[key]['axes'][dir_index]['indices']:
                    pair_to_axes[pair].append((int(dir_index), ax_ind))
    return pair_to_dirs, pair_to_axes #pair -> dir, and pair -> (dir, origin)


class MateChecker(DataVisitor):
    def __init__(self, global_data, mc_path, epsilon_rel, max_topologies, validate_feasibility=False, check_alternate_paths=False):
        self.transforms = compose(LoadMCData(mc_path), AssemblyLoader(global_data, use_uvnet_features=False, epsilon_rel=epsilon_rel, max_topologies=max_topologies, precompute=True))
        self.global_data = global_data
        self.epsilon_rel = epsilon_rel
        self.validate_feasibility = validate_feasibility
        self.check_alternate_paths = check_alternate_paths

    def process(self, data):
        mate_stats = Stats()
        all_stats = Stats()
        mate_subset = self.global_data.mate_df.loc[data.ind]
        part_subset = self.global_data.part_df.loc[data.ind]
        allowed_mates = {MateTypes.FASTENED.value, MateTypes.REVOLUTE.value, MateTypes.CYLINDRICAL.value, MateTypes.SLIDER.value}    
        if all([k in allowed_mates for k in mate_subset['Type']]):
            mates = df_to_mates(mate_subset)
            rigid_comps = list(part_subset['RigidComponentID'])
            assembly_info = data.assembly_info
            if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:
                curr_mate_stats = []
                mc_origins_flat = np.array([origin for mc_origins in assembly_info.mc_origins_all for origin in mc_origins])
                mc_axes_flat = np.array([axis for mc_axes in assembly_info.mc_axes_homogenized_all for axis in mc_axes])
                pairs_to_dirs, pairs_to_axes = load_axes(data.mcfile)
                for j,mate in enumerate(mates):
                    mate_stat = {'Assembly': data.ind, 'dir_index': -1, 'axis_index': -1, 'has_any_axis':False, 'has_same_dir_axis':False, 'type': mate.type}
                    part_indices = [assembly_info.occ_to_index[me[0]] for me in mate.matedEntities]
                    mate_origins = []
                    mate_dirs = []
                    for partIndex, me in zip(part_indices, mate.matedEntities):
                        origin_local = me[1][0]
                        frame_local = me[1][1]
                        cs = cs_from_origin_frame(origin_local, frame_local)
                        cs_transformed = assembly_info.mate_transforms[partIndex] @ cs
                        origin, rot = cs_to_origin_frame(cs_transformed)
                        rot_homogenized = homogenize_frame(rot, z_flip_only=True)
                        mate_origins.append(origin)
                        mate_dirs.append(rot_homogenized[:,2])
                    
                    found = False

                    mate_stat['dirs_agree'] = np.allclose(mate_dirs[0], mate_dirs[1], rtol=0, atol=self.epsilon_rel)
                    projected_mate_origins = [project_to_plane(origin, mate_dirs[0]) for origin in mate_origins]
                    mate_stat['axes_agree'] = mate_stat['dirs_agree'] and np.allclose(projected_mate_origins[0], projected_mate_origins[1], rtol=0, atol=self.epsilon_rel)
                    key = tuple(sorted(part_indices))

                    mate_stat['has_any_axis'] = key in pairs_to_axes

                    if mate_stat['dirs_agree']:
                        if key in pairs_to_dirs:
                            for dir_ind in pairs_to_dirs[key]:
                                dir = mc_axes_flat[dir_ind]
                                dir_homo, _ = homogenize_sign(dir)
                                if np.allclose(mate_dirs[0], dir_homo, rtol=0, atol=self.epsilon_rel):
                                    mate_stat['dir_index'] = dir_ind
                                    break
                        if key in pairs_to_axes:
                            for dir_ind, origin_ind in pairs_to_axes[key]:
                                dir = mc_axes_flat[dir_ind]
                                dir_homo, _ = homogenize_sign(dir)
                                if np.allclose(mate_dirs[0], dir_homo, rtol=0, atol=self.epsilon_rel):
                                    mate_stat['has_same_dir_axis'] = True
                                    break
                    
                    if mate_stat['axes_agree']:
                        if key in pairs_to_axes:
                            for dir_ind, origin_ind in pairs_to_axes[key]:
                                dir = mc_axes_flat[dir_ind]
                                origin = mc_origins_flat[origin_ind]
                                dir_homo, _ = homogenize_sign(dir)
                                if np.allclose(mate_dirs[0], dir_homo, rtol=0, atol=self.epsilon_rel):
                                    projected_origin = project_to_plane(origin, mate_dirs[0])
                                    if np.allclose(projected_mate_origins[0], projected_origin, rtol=0, atol=self.epsilon_rel):
                                        mate_stat['axis_index'] = origin_ind
                                        break

                    if mate.type == MateTypes.FASTENED:
                        found = True
                    elif mate.type == MateTypes.SLIDER:
                        found = mate_stat['dir_index'] != -1
                        
                    elif mate.type == MateTypes.CYLINDRICAL or mate.type == MateTypes.REVOLUTE:
                        found = mate_stat['axis_index'] != -1
                    
                    if self.validate_feasibility:
                        mate_stat['rigid_comp_attempting_motion'] = rigid_comps[part_indices[0]] == rigid_comps[part_indices[1]] and mate.type != MateTypes.FASTENED
                        mate_stat['valid'] = found and not mate_stat['rigid_comp_attempting_motion']
                    else:
                        mate_stat['valid'] = found

                    mate_stats.append(mate_stat, mate_subset.iloc[j]['MateIndex'])
                
                if self.check_alternate_paths:
                    validation_stats = assembly_info.validate_mates(mates)
                    assert(len(validation_stats) == len(curr_mate_stats))
                    final_stats = [{**stat, **vstat} for stat, vstat in zip(curr_mate_stats, validation_stats)]
                    curr_mate_stats = final_stats
                
                if self.validate_feasibility:
                    num_incompatible_mates = 0
                    num_moving_mates_between_same_rigid = 0
                    num_multimates_between_rigid = 0
                    rigid_pairs_to_mate = dict()
                    for pair in data.pair_to_index:
                        rigid_pair = tuple(sorted([rigid_comps[k] for k in pair]))
                        if rigid_pair[0] != rigid_pair[1]:
                            if rigid_pair not in rigid_pairs_to_mate:
                                rigid_pairs_to_mate[rigid_pair] = data.pair_to_index[pair]
                            else:
                                num_multimates_between_rigid += 1
                                prevMate = mates[rigid_pairs_to_mate[rigid_pair]]
                                currMate = mates[data.pair_to_index[pair]]
                                prevMate = assembly_info.transform_mates([prevMate])[0]
                                currMate = assembly_info.transform_mates([currMate])[0]
                                if not mates_equivalent(prevMate, currMate, self.epsilon_rel):
                                    num_incompatible_mates += 1
                        else:
                            if mates[data.pair_to_index[pair]].type != MateTypes.FASTENED:
                                num_moving_mates_between_same_rigid += 1
                    feasible = (num_moving_mates_between_same_rigid == 0) and (num_incompatible_mates == 0) and all([mate_stat['valid'] for mate_stat in curr_mate_stats])
                    stat = {'num_moving_mates_between_same_rigid':num_moving_mates_between_same_rigid,
                            'num_incompatible_mates_between_rigid':num_incompatible_mates,
                            'num_multimates_between_rigid':num_multimates_between_rigid,
                            'feasible': feasible}
                    all_stats.append(stat, data.ind)
        return {'mate_check_stats': mate_stats, 'assembly_stats': all_stats}
