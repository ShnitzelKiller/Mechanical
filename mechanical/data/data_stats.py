from xml.etree.ElementInclude import include

from attr import validate
from mechanical.data import Stats
from mechanical.data.transforms import *
import logging
import os
import h5py
import numpy as np
import numpy.linalg as LA
import torch
from mechanical.data.dataloaders import Data
from mechanical.geometry import displaced_min_signed_distance, min_signed_distance_symmetric
from mechanical.utils import homogenize_frame, homogenize_sign, cs_from_origin_frame, cs_to_origin_frame, project_to_plane, apply_homogeneous_transform
from mechanical.utils import joinmeshes, df_to_mates, MateTypes, mates_equivalent, mate_types

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
    def __init__(self, global_data, out_path, use_uvnet_features, epsilon_rel, max_topologies, dry_run):
        self.transforms = [AssemblyLoader(global_data, use_uvnet_features=use_uvnet_features, epsilon_rel=epsilon_rel, max_topologies=max_topologies)]
        self.out_path = out_path
        self.dry_run = dry_run

    def process(self, data):
        out_dict = {}
        stat = Stats(defaults={'invalid_bbox': False})
        stat.append(data.assembly_info.stats, data.ind)
        out_dict['pspy_stats'] = stat
        if data.assembly_info.stats['initialized'] and data.assembly_info.stats['num_invalid_loaded_parts'] == 0:
            batch = data.assembly_info.create_batches()
            if not self.dry_run:
                torch_datapath = os.path.join(self.out_path, f'{data.ind}.dat')
                torch.save(batch, torch_datapath)
        return out_dict


class DisplacementPenalty(DataVisitor):
    def __init__(self, global_data, sliding_distance, rotation_angle, num_samples, include_vertices, meshpath, compute_all, augmented_mates, batch_path=None, mc_path=None):
        self.transforms = [MeshLoader(meshpath)]
        self.global_data = global_data
        self.sliding_distance = sliding_distance
        self.rotation_angle = rotation_angle
        self.samples = num_samples
        self.include_vertices = include_vertices
        self.compute_all = compute_all
        self.augmented_mates = augmented_mates
        if augmented_mates:
            self.transforms.append(LoadBatch(batch_path))
            self.mc_path = mc_path

    
    def process_mate(self, mtype, pid1, pid2, meshes, rigidcomps, rel_distance, axis, origin, assembly_index):
        row = {}
        if self.compute_all or mtype == MateTypes.REVOLUTE or MateTypes.CYLINDRICAL or MateTypes.SLIDER:
            if self.compute_all:
                sliding = True
                rotating = True
            else:
                sliding = False
                rotating = False
                if mtype == MateTypes.REVOLUTE:
                    rotating = True
                elif mtype == MateTypes.CYLINDRICAL:
                    rotating = True
                    sliding = True
                elif mtype == MateTypes.SLIDER:
                    sliding = True
        

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


            if sliding:
                for k,displacement in enumerate([rel_distance, -rel_distance]):
                    sd = displaced_min_signed_distance(meshes_subset, axis, motion_type='SLIDE', samples=self.samples, displacement=displacement, include_vertices=self.include_vertices)
                    penalties_penetration_sliding[k] = max(0, -sd) / rel_distance
                    penalties_separation_sliding[k] = max(0, sd) / rel_distance
            if rotating:
                for k, angle in enumerate([self.rotation_angle, -self.rotation_angle]):
                    sd = displaced_min_signed_distance(meshes_subset, axis, origin=origin, motion_type='ROTATE', samples=self.samples, displacement=angle, include_vertices=self.include_vertices)
                    penalties_penetration_rotating[k] = max(0, -sd) / rel_distance
                    penalties_separation_rotating[k] = max(0, sd) / rel_distance

            row = {'Assembly': assembly_index, 'type': mtype,
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
        return row
    
    def process(self, data):
        stats = Stats()
        augmented_stats = Stats()
        part_subset = self.global_data.part_df.loc[[data.ind]]
        mate_subset = self.global_data.mate_df.loc[[data.ind]]
        rigidcomps = list(part_subset['RigidComponentID'])
        if self.augmented_mates:
            newmate_stats_df = self.global_data.newmate_df
            if data.ind in newmate_stats_df.index:
                newmate_subset = newmate_stats_df.loc[[data.ind]]

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
                pid1 = occ_to_index[mate_subset.iloc[j]['Part1']]
                pid2 = occ_to_index[mate_subset.iloc[j]['Part2']]
                tf = part_subset.iloc[pid1]['Transform']
                origin = tf[:3,:3] @ mate_subset.iloc[j]['Origin1'] + tf[:3,3]
                axis = tf[:3,:3] @ mate_subset.iloc[j]['Axes1'][:,2]
                mtype = mate_subset.iloc[j]['Type']

                row = self.process_mate(mtype, pid1, pid2, meshes, rigidcomps, rel_distance, axis, origin, data.ind)
                stats.append(row, mate_subset.iloc[j]['MateIndex'])
            
            if self.augmented_mates and data.ind in newmate_stats_df.index:
                with h5py.File(os.path.join(self.mc_path, f'{data.ind}.hdf5'),'r') as f:
                    norm_mat = np.array(f['normalization_matrix'])
                inv_tf = LA.inv(norm_mat)
                for j in range(newmate_subset.shape[0]):
                    if newmate_subset.iloc[j]['added_mate']:
                        pid1 = occ_to_index[newmate_subset.iloc[j]['part1']]
                        pid2 = occ_to_index[newmate_subset.iloc[j]['part2']]
                        axis_index = newmate_subset.iloc[j]['axis_index']
                        axis = data.batch.mcfs[axis_index,:3].numpy()
                        origin = data.batch.mcfs[axis_index,3:].numpy()
                        origin = apply_homogeneous_transform(inv_tf, origin)
                        mtype = newmate_subset.iloc[j]['type']
                        row = self.process_mate(mtype, pid1, pid2, meshes, rigidcomps, rel_distance, axis, origin, data.ind)
                        augmented_stats.append(row, newmate_subset.iloc[j]['NewMateIndex'])


        return {'mate_penetration_stats': stats, 'augmented_mate_penetration_stats': augmented_stats}

class MCDataSaver(DataVisitor):
    def __init__(self, global_data, mc_path, max_axis_groups, epsilon_rel, max_topologies, save_frames=False, save_dirs=False, dry_run=False):
        self.transforms = [AssemblyLoader(global_data, use_uvnet_features=False, epsilon_rel=epsilon_rel, max_topologies=max_topologies)]
        self.max_groups = max_axis_groups
        self.mc_path = mc_path
        self.save_frames = save_frames
        self.save_dirs = save_dirs
        self.dry_run = dry_run
    
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
        
        if not self.dry_run:
            with h5py.File(os.path.join(self.mc_path, f"{data.ind}.hdf5"), "w") as f:
                f.create_dataset('mc_part_labels', data=mc_part_labels)

                if self.save_frames:
                    mc_frames = f.create_group('mc_frames')
                    for k in range(len(assembly_info.mc_origins_all)):
                        partgroup = mc_frames.create_group(str(k))
                        partgroup.create_dataset('origins', data=np.array(assembly_info.mc_origins_all[k]))
                        partgroup.create_dataset('axes', data=np.array(assembly_info.mc_axes_all[k]))
                        #rots_quat = np.array([R.from_matrix(rot).as_quat() for rot in assembly_info.mc_rots_all[k]])
                        #partgroup.create_dataset('rots', data=rots_quat)
                
                f.create_dataset('mc_counts', data=[len(part.default_mcfs) for part in assembly_info.parts])

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
                    if self.save_dirs:
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
            if len(inds) > 0:
                pair_to_dirs[pair] = inds
            axes = []
            for dir_index in pair_data[key]['axes'].keys():
                for ax_ind in pair_data[key]['axes'][dir_index]['indices']:
                    axes.append((int(dir_index), ax_ind))
            if axes:
                pair_to_axes[pair] = axes
    return pair_to_dirs, pair_to_axes #pair -> dir, and pair -> (dir, origin)


class MateChecker(DataVisitor):
    def __init__(self, global_data, mc_path, epsilon_rel, max_topologies, validate_feasibility=False, check_alternate_paths=False):
        self.transforms = [LoadMCData(mc_path), AssemblyLoader(global_data, use_uvnet_features=False, epsilon_rel=epsilon_rel, max_topologies=max_topologies, precompute=True)]
        self.global_data = global_data
        self.epsilon_rel = epsilon_rel
        self.validate_feasibility = validate_feasibility
        self.check_alternate_paths = check_alternate_paths

    def process(self, data):
        mate_stats = Stats()
        all_stats = Stats()
        mate_subset = self.global_data.mate_df.loc[[data.ind]]
        part_subset = self.global_data.part_df.loc[[data.ind]]
        allowed_mates = {MateTypes.FASTENED.value, MateTypes.REVOLUTE.value, MateTypes.CYLINDRICAL.value, MateTypes.SLIDER.value}

        num_discrepant_mc_counts = 0
        if all([k in allowed_mates for k in mate_subset['Type']]):
            mates = df_to_mates(mate_subset)
            rigid_comps = list(part_subset['RigidComponentID'])
            assembly_info = data.assembly_info
            if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:
                with h5py.File(data.mcfile,'r') as f:
                    for part, siz in zip(assembly_info.parts, f['mc_counts']):
                        if len(part.default_mcfs) != siz:
                            num_discrepant_mc_counts += 1
                stat = {'num_discrepant_mc_counts': num_discrepant_mc_counts}

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
                    
                    mate_valid = False

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
                                        
                    if mate_stat['has_any_axis'] and mate_stat['axis_index'] < 0:
                        mate_stat['axis_index'] = pairs_to_axes[key][0][1]

                    if mate.type == MateTypes.FASTENED:
                        mate_valid = mate_stat['has_any_axis']
                    elif mate.type == MateTypes.SLIDER:
                        mate_valid = mate_stat['dir_index'] != -1 and mate_stat['has_same_dir_axis']
                        
                    elif mate.type == MateTypes.CYLINDRICAL or mate.type == MateTypes.REVOLUTE:
                        mate_valid = mate_stat['axis_index'] != -1
                    
                    if self.validate_feasibility:
                        mate_stat['rigid_comp_attempting_motion'] = rigid_comps[part_indices[0]] == rigid_comps[part_indices[1]] and mate.type != MateTypes.FASTENED
                        mate_stat['valid'] = mate_valid and not mate_stat['rigid_comp_attempting_motion']
                    else:
                        mate_stat['valid'] = mate_valid

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
                    rigidstat = {'num_moving_mates_between_same_rigid':num_moving_mates_between_same_rigid,
                            'num_incompatible_mates_between_rigid':num_incompatible_mates,
                            'num_multimates_between_rigid':num_multimates_between_rigid,
                            'feasible': feasible}
                    stat = {**stat, **rigidstat}
                all_stats.append(stat, data.ind)
        return {'mate_check_stats': mate_stats, 'assembly_check_stats': all_stats}

class CombinedAxisMateChecker(DataVisitor):
    def __init__(self, global_data, mc_path, epsilon_rel, max_topologies, validate_feasibility=False, check_alternate_paths=False, max_axis_groups=10, save_frames=False, save_dirs=True, dry_run=False):
        self.mate_checker = MateChecker(global_data, mc_path, epsilon_rel, max_topologies, validate_feasibility=validate_feasibility, check_alternate_paths=check_alternate_paths)
        self.axis_saver = MCDataSaver(global_data, mc_path, max_axis_groups, epsilon_rel, max_topologies, save_frames=save_frames, save_dirs=save_dirs, dry_run=dry_run)
        self.transforms = self.mate_checker.transforms
    
    def process(self, data):
        out_dict1 = self.axis_saver(data)
        out_dict2 = self.mate_checker(data)
        return {**out_dict1, **out_dict2}
    

class CombinedAxisBatchAugment(DataVisitor):
    def __init__(self, global_data, mc_path, batch_path, epsilon_rel, max_topologies, validate_feasibility=False, check_alternate_paths=False, max_axis_groups=10, save_frames=False, save_dirs=True, dry_run=False, distance_threshold=.01, require_axis=True, matched_axes_only=True, use_uvnet_features=True, append_pair_data=True):
        self.transforms = [LoadMCData(mc_path),
            AssemblyLoader(global_data, use_uvnet_features=use_uvnet_features, epsilon_rel=epsilon_rel, max_topologies=max_topologies, precompute=True),
            ComputeDistances(distance_threshold=distance_threshold)]
        
        self.batch_saver = BatchSaver(global_data, batch_path, use_uvnet_features, epsilon_rel, max_topologies, dry_run)
        self.axis_saver = MCDataSaver(global_data, mc_path, max_axis_groups, epsilon_rel, max_topologies, save_frames=save_frames, save_dirs=save_dirs, dry_run=dry_run)
        self.mate_checker = MateChecker(global_data, mc_path, epsilon_rel, max_topologies, validate_feasibility=validate_feasibility, check_alternate_paths=check_alternate_paths)
        self.distance_checker = DistanceChecker(global_data, distance_threshold, append_pair_data, mc_path, epsilon_rel, max_topologies)
        self.augmenter = MateAugmentation(global_data, mc_path, distance_threshold, require_axis, matched_axes_only, epsilon_rel, max_topologies)
    
    def process(self, data):
        out_dict1 = self.axis_saver(data)
        out_dict2 = self.mate_checker(data)
        out_dict3 = self.augmenter(data)
        out_dict4 = self.batch_saver(data)
        out_dict5 = self.distance_checker(data)
        return {**out_dict1, **out_dict2, **out_dict3, **out_dict4, **out_dict5}
    

class MCCountChecker(DataVisitor):
    def __init__(self, mc_path, batch_path, individual):
        self.transforms = [LoadMCData(mc_path), LoadBatch(batch_path)]
        self.individual = individual
    
    def process(self, data):
        stat = Stats()
        with h5py.File(data.mcfile,'r') as f:
            counts = np.array(f['mc_counts'])
        totalcount = counts.sum()
        batch_totalcount = data.batch.mcfs.shape[0]
        row = {'mc_count': totalcount, 'batch_mc_count': batch_totalcount, 'agrees': totalcount == batch_totalcount}
        if self.individual:
            batch_counts = [(data.batch.mcfs_batch == i).sum().item() for i in data.batch.mcfs_batch.unique()]
            row['agrees'] = all([c1 == c2 for c1, c2 in zip(counts, batch_counts)])
        stat.append(row, data.ind)
        return {'mc_count_stats': stat}
        

class DistanceChecker(DataVisitor):
    def __init__(self, global_data, distance_threshold, append_pair_data, mc_path, epsilon_rel, max_topologies, coincidence=False):
        self.transforms = [AssemblyLoader(global_data, use_uvnet_features=False, epsilon_rel=epsilon_rel, max_topologies=max_topologies), ComputeDistances(distance_threshold=distance_threshold)]
        self.global_data = global_data
        self.distance_threshold = distance_threshold
        self.append_pair_data = append_pair_data
        self.mc_path = mc_path
        self.coincidence = coincidence

    def process(self, data):
        all_mate_stats = Stats()
        all_assembly_stats = Stats()

        assembly_info = data.assembly_info
        mate_subset = self.global_data.mate_df.loc[[data.ind]]
        part_subset = self.global_data.part_df.loc[[data.ind]]

        connections = {tuple(sorted((assembly_info.occ_to_index[mate_subset.iloc[l]['Part1']], assembly_info.occ_to_index[mate_subset.iloc[l]['Part2']]))): l for l in range(mate_subset.shape[0])}
        
        if self.coincidence:
            proposals = assembly_info.mate_proposals(coincident_only=True)

        #distances = assembly_info.part_distances(self.distance_threshold)
        distances = data.distances
        
        if self.append_pair_data:
            mcpath = os.path.join(self.mc_path, f'{data.ind}.hdf5')
            if os.path.isfile(mcpath):
                with h5py.File(mcpath,'r+') as f:
                    pair_data = f['pair_data']
                    for key in pair_data.keys():
                        pair = tuple(sorted([int(k) for k in key.split(',')]))
                        if pair in distances:
                            pair_data[key].attrs['distance'] = distances[pair]
                        else:
                            pair_data[key].attrs['distance'] = np.inf

        stat = assembly_info.stats

        if assembly_info.stats['num_degenerate_bboxes'] == 0:
            stat['num_unconnected_close'] = 0
            stat['num_unconnected_coincident'] = 0
            stat['num_unconnected_close_and_coincident'] = 0
            stat['num_connected_far'] = 0
            stat['num_connected_not_coincident'] = 0
            stat['num_connected_far_or_not_coincident'] = 0

            if self.coincidence:
                allpairs = {pair for pair in distances}.union({pair for pair in proposals})
            else:
                allpairs = distances

            for pair in allpairs:
                comp1 = part_subset.iloc[pair[0]]['RigidComponentID']
                comp2 = part_subset.iloc[pair[1]]['RigidComponentID']
                if comp1 != comp2 and pair not in connections:
                    if pair in distances and distances[pair] < self.distance_threshold:
                        stat['num_unconnected_close'] += 1
                    if self.coincidence:
                        if pair in proposals:
                            stat['num_unconnected_coincident'] += 1
                        if pair in distances and distances[pair] < self.distance_threshold and pair in proposals:
                            stat['num_unconnected_close_and_coincident'] += 1

            
            for pair in connections:
                mate_stat = {'connected_far': False, 'connected_not_coincident': False}
                if pair not in distances or distances[pair] >= self.distance_threshold:
                    stat['num_connected_far'] += 1
                    mate_stat['connected_far'] = True
                if self.coincidence:
                    if pair not in proposals or pair not in distances or distances[pair] >= self.distance_threshold:
                        stat['num_connected_far_or_not_coincident'] += 1
                    if pair not in proposals:
                        stat['num_connected_not_coincident'] += 1
                        mate_stat['connected_not_coincident'] = True
                mate_stat['type'] = mate_subset.iloc[connections[pair]]['Type']
                mate_stat['assembly'] = data.ind
                all_mate_stats.append(mate_stat, mate_subset.iloc[connections[pair]]['MateIndex'])
        else:
            stat['num_unconnected_close'] = -1
            stat['num_unconnected_coincident'] = -1
            stat['num_unconnected_close_and_coincident'] = -1
            stat['num_connected_far'] = -1
            stat['num_connected_not_coincident'] = -1
            stat['num_connected_far_or_not_coincident'] = -1
            
        all_assembly_stats.append(stat, data.ind)

        return {'assembly_distance_stats': all_assembly_stats, 'mate_distance_stats': all_mate_stats}


class MateAugmentation(DataVisitor):
    def __init__(self, global_data, mc_path, distance_threshold, require_axis, matched_axes_only, epsilon_rel, max_topologies):
        self.transforms = [AssemblyLoader(global_data, use_uvnet_features=False, epsilon_rel=epsilon_rel, max_topologies=max_topologies, precompute=True), ComputeDistances(distance_threshold=distance_threshold)]
        self.global_data = global_data
        self.matched_axes_only = matched_axes_only
        self.mc_path = mc_path
        self.distance_threshold = distance_threshold
        self.require_axis = require_axis

    def process(self, data):
        assembly_info = data.assembly_info
        part_subset = self.global_data.part_df.loc[[data.ind]]
        mate_subset = self.global_data.mate_df.loc[[data.ind]]
        stats = Stats()
        assstats = Stats()
        if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:
            mates = df_to_mates(mate_subset)

            if self.matched_axes_only:
                mcfile = os.path.join(self.mc_path, f'{data.ind}.hdf5')
                pair_to_dirs, pair_to_axes = load_axes(mcfile)
                
                with h5py.File(mcfile,'r') as f:
                    counts = np.array(f['mc_counts'])
                assstats.append({'mc_counts_agree':all([count == len(axs) for count, axs in zip(counts, assembly_info.mc_axes_all)])}, data.ind)
                stat = assembly_info.fill_missing_mates(mates, data.distances, list(part_subset['RigidComponentID']), self.distance_threshold, pair_to_dirs=pair_to_dirs, pair_to_axes=pair_to_axes, require_axis=self.require_axis)
            else: 
                stat = assembly_info.fill_missing_mates(mates, data.distances, list(part_subset['RigidComponentID']), self.distance_threshold)
            for st in stat:
                st['Assembly'] = data.ind
            for st in stat:
                stats.append(st)
        return {'newmate_stats':stats, 'assembly_newmate_stats': assstats}

class TransformSaver(DataVisitor):
    def __init__(self, global_data, mc_path):
        self.global_data = global_data
        self.transforms = [AssemblyLoader(global_data, load_geometry=True, include_mcfs=False)]
        self.mc_path = mc_path

    def process(self, data):
        with h5py.File(os.path.join(self.mc_path, f'{data.ind}.hdf5'),'r+') as f:
            f.create_dataset('normalization_matrix', data=data.assembly_info.norm_transform)

class MateLabelSaver(DataVisitor):
    def __init__(self, global_data, mc_path, augmented_labels, dry_run, indices_only=False):
        self.transforms = [AssemblyLoader(global_data, load_geometry=False, pair_data=True)]
        self.mc_path = mc_path
        self.global_data = global_data
        self.augmented_labels = augmented_labels
        self.dry_run = dry_run
        self.indices_only = indices_only

        self.allowed_assemblies = set(global_data.mate_check_df['Assembly'])

    def process(self, data):
        stat = Stats()
        if data.ind in self.allowed_assemblies:
            part_subset = self.global_data.part_df.loc[[data.ind]]
            mate_subset = self.global_data.mate_df.loc[[data.ind]]
            newmate_stats_df = self.global_data.newmate_df
            mate_check_df = self.global_data.mate_check_df

            h5fname = os.path.join(self.mc_path, f'{data.ind}.hdf5')
            if os.path.isfile(h5fname):
                rigid_comps = list(part_subset['RigidComponentID'])
                if self.augmented_labels:
                    augmented_pairs_to_index = dict()
                    if data.ind in newmate_stats_df.index:
                        newmate_subset = newmate_stats_df.loc[[data.ind]]
                        for m in range(newmate_subset.shape[0]):
                            part_indices = tuple(sorted([data.occ_to_index[newmate_subset.iloc[m][k]] for k in ['part1','part2']]))
                            augmented_pairs_to_index[part_indices] = m
                fmode = 'r' if self.dry_run else 'r+'
                with h5py.File(h5fname,fmode) as f:
                    pair_data = f['pair_data']
                    for key in pair_data.keys():
                        pair = tuple(sorted([int(k) for k in key.split(',')]))
                        mateType = -1
                        augmentedType = -1
                        mateDirInd = -1
                        mateAxisInd = -1
                        augmentedDirInd = -1
                        augmentedAxisInd = -1
                        mate_index = -1
                        augmented_index = -1
                        if pair in data.pair_to_index:
                            mate_subset_index = data.pair_to_index[pair]
                            mate_index = mate_subset.iloc[mate_subset_index]['MateIndex']
                            mateType = mate_types.index(mate_subset.iloc[mate_subset_index]['Type'])
                            mate_check_row = mate_check_df.loc[mate_index]
                            mateDirInd = mate_check_row['dir_index']
                            mateAxisInd = mate_check_row['axis_index']

                        if self.augmented_labels:
                            if pair in augmented_pairs_to_index:
                                augmented_subset_index = augmented_pairs_to_index[pair]
                                if newmate_subset.iloc[augmented_subset_index]['added_mate']:
                                    augmented_index = newmate_subset.iloc[augmented_subset_index]['NewMateIndex']
                                    augmentedType = mate_types.index(newmate_subset.iloc[augmented_subset_index]['type'])
                                    augmentedDirInd = newmate_subset.iloc[augmented_subset_index]['dir_index']
                                    augmentedAxisInd = newmate_subset.iloc[augmented_subset_index]['axis_index']
                        
                        #densify fasten mates
                        if rigid_comps[pair[0]] == rigid_comps[pair[1]]:
                            augmentedType = mate_types.index(MateTypes.FASTENED.value)

                        if not self.dry_run:
                            pair_data[key].attrs['mate_index'] = mate_index
                            pair_data[key].attrs['augmented_mate_index'] = augmented_index
                            if not self.indices_only:
                                if self.augmented_labels:
                                    pair_data[key].attrs['augmented_type'] = augmentedType
                                    pair_data[key].attrs['augmented_dirIndex'] = augmentedDirInd
                                    pair_data[key].attrs['augmented_axisIndex'] = augmentedAxisInd
                                pair_data[key].attrs['type'] = mateType
                                pair_data[key].attrs['dirIndex'] = mateDirInd
                                pair_data[key].attrs['axisIndex'] = mateAxisInd
                        else:
                            assert(pair_data[key].attrs['type'] == mateType)
                            assert(pair_data[key].attrs['dirIndex'] == mateDirInd)
                            assert(pair_data[key].attrs['axisIndex'] == mateAxisInd)
                            if augmentedType == mate_types.index(MateTypes.FASTENED.value):
                                assert(pair_data[key].attrs['augmented_type'] == augmentedType)
        else:
            stat.append({'skipped': True}, data.ind)
        return {'add_label_stats': stat}


class DataChecker(DataVisitor):
    def __init__(self, global_data, mc_path, torch_path, distance_threshold):
        self.transforms = [AssemblyLoader(global_data, load_geometry=False, pair_data=True)]
        self.mc_path = mc_path
        self.torch_path = torch_path
        self.global_data = global_data
        self.distance_threshold = distance_threshold
        self.mate_check_df = global_data.mate_check_df.set_index('Assembly')

    def process(self, data):
        pspy_df = self.global_data.pspy_df
        
        newmate_df = self.global_data.newmate_df
        mate_subset = self.global_data.mate_df.loc[[data.ind]]
        mcfile = os.path.join(self.mc_path, f'{data.ind}.hdf5')
        torchfile = os.path.join(self.torch_path, f'{data.ind}.dat')
        stat = Stats(defaults={'all_mates_valid': False, 'has_all_distances': False, 'has_all_type_labels': False})

        connections = {tuple(sorted((data.occ_to_index[mate_subset.iloc[l]['Part1']], data.occ_to_index[mate_subset.iloc[l]['Part2']]))): l for l in range(mate_subset.shape[0])}
        pspy_initialized = pspy_df.loc[data.ind, 'initialized']
        pspy_valid = False
        if pspy_initialized:
            pspy_valid = pspy_df.loc[data.ind, 'num_normalized_parts_with_discrepancies'] == 0 and pspy_df.loc[data.ind, 'num_invalid_loaded_parts'] == 0 and pspy_df.loc[data.ind, 'num_invalid_transformed_parts'] == 0
        
        has_mate_check = data.ind in self.mate_check_df.index

        num_attempted_augments = 0 if data.ind not in newmate_df.index else newmate_df.loc[[data.ind]].shape[0]

        row = {'has_mate_check': has_mate_check, 'attempted_augmentation': num_attempted_augments, 'pspy_initialized': pspy_initialized, 'pspy_valid': pspy_valid}
        if has_mate_check:
            row['all_mates_valid'] = self.mate_check_df.loc[[data.ind],'valid'].all()

        row['has_torch_file'] = os.path.isfile(torchfile)
        row['has_mc_file'] = os.path.isfile(mcfile)
        if row['has_mc_file']:
            if data.ind in newmate_df.index:
                newmate_subset = newmate_df.loc[[data.ind]]
                part_ids = [tuple(sorted([data.occ_to_index[occ1], data.occ_to_index[occ2]])) for occ1, occ2 in zip(newmate_subset['part1'], newmate_subset['part2'])]
                newmate_subset['part1_id'] = [p[0] for p in part_ids]
                newmate_subset['part2_id'] = [p[1] for p in part_ids]
                newmate_subset_by_pair = newmate_subset.set_index(['part1_id', 'part2_id'])
            has_type_label = True
            has_distance = True
            num_unconnected_close_with_axis = 0
            num_densified_fastens = 0
            num_augmented_missing_mates = 0
            missing_augmented_labels = 0
            num_missing_mates = 0
            num_connected_coaxial_far = 0
            with h5py.File(mcfile, 'r') as f:
                pair_data = f['pair_data']
                for key in pair_data:
                    if len(pair_data[key]['axes'].keys()) > 0:
                        pair = tuple(sorted([int(k) for k in key.split(',')]))
                        if 'distance' not in pair_data[key].attrs:
                            has_distance = False

                        else:
                            if pair in connections and pair_data[key].attrs['distance'] >= self.distance_threshold:
                                num_connected_coaxial_far += 1
                            if pair_data[key].attrs['distance'] < self.distance_threshold:
                                if pair in connections:
                                    if 'type' not in pair_data[key].attrs:
                                        has_type_label = False
                                else:
                                    num_unconnected_close_with_axis += 1
                                    if 'augmented_type' in pair_data[key].attrs and pair_data[key].attrs['augmented_type'] == mate_types.index(MateTypes.FASTENED.value):
                                        num_densified_fastens += 1
                                    elif data.ind in newmate_df.index and pair in newmate_subset_by_pair.index and newmate_subset_by_pair.loc[pair, 'added_mate']:
                                        if 'augmented_type' in pair_data[key].attrs:
                                            num_augmented_missing_mates += 1
                                        else:
                                            missing_augmented_labels += 1
                                            num_missing_mates += 1
                                    else:
                                        num_missing_mates += 1

                        
            row['has_all_distances'] = has_distance
            if has_distance:
                row['missing_augmented_labels'] = missing_augmented_labels
                row['num_missing_mates'] = num_missing_mates #number of close parts with shared axes but no mate or augmented mate
                row['num_augmented_missing_mates'] = num_augmented_missing_mates #number of close parts with shared axes that is augmented
                row['num_densified_fastens'] = num_densified_fastens
                row['has_all_type_labels'] = has_type_label
                row['num_unconnected_close_with_axis'] = num_unconnected_close_with_axis
                row['num_connected_coaxial_far'] = num_connected_coaxial_far

        valid = False
        if row['has_mc_file'] and row['has_torch_file'] and has_distance and has_mate_check and pspy_valid and has_type_label and row['num_missing_mates'] == 0 and row['all_mates_valid'] and row['num_connected_coaxial_far'] == 0 and row['missing_augmented_labels'] == 0:
            valid = True
        
        row['valid'] = valid

        stat.append(row, data.ind)
        return {'final_stat': stat}


class UVSampleChecker(DataVisitor):

    def __init__(self, batch_path):
        self.transforms = [LoadBatch(batch_path)]
    

    def process(self, data):
        face_samples = data.batch.face_samples
        edge_samples = data.batch.edge_samples

        has_face_samples=True
        has_edge_samples=True
        if face_samples.shape[0] > 0:
            max_face_pos_norm = face_samples[:,:3,:,:].norm(dim=1).max().item()
            max_face_norm_norm = face_samples[:,3:6,:,:].norm(dim=1).max().item()
            max_face_crv_norm = face_samples[:,6:8,:,:].norm(dim=1).max().item()
            max_face_mask_norm = face_samples[:,8,:,:].abs().max().item()
            max_face_norm = face_samples.norm(dim=1).max().item()
        else:
            has_face_samples=False
            max_face_pos_norm = 0
            max_face_norm_norm = 0
            max_face_norm = 0
            max_face_crv_norm = 0
            max_face_mask_norm = 0
        if edge_samples.shape[0] > 0:
            max_edge_pos_norm = edge_samples[:,:3,:].norm(dim=1).max().item()
            max_edge_norm_norm = edge_samples[:,3:6,:].norm(dim=1).max().item()
            max_edge_crv_norm = edge_samples[:,6,:].abs().max().item()
            max_edge_norm = edge_samples.norm(dim=1).max().item()
        else:
            has_edge_samples=False
            max_edge_pos_norm = 0
            max_edge_norm_norm = 0
            max_edge_norm = 0
            max_edge_crv_norm = 0


        stat = Stats()
        stat.append({'max_face_crv_norm': max_face_crv_norm, 'max_edge_crv_norm': max_edge_crv_norm, 'max_face_mask_norm': max_face_mask_norm, 'max_face_pos_norm': max_face_pos_norm, 'max_face_norm_norm': max_face_norm_norm, 'max_edge_pos_norm': max_edge_pos_norm, 'max_edge_norm_norm': max_edge_norm_norm, 'max_face_norm': max_face_norm, 'max_edge_norm': max_edge_norm, 'has_face_samples': has_face_samples, 'has_edge_samples': has_edge_samples}, data.ind)

        return {'uv_stats': stat}

