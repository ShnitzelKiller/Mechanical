from xml.etree.ElementInclude import include
from mechanical.data import AssemblyLoader, Stats, MeshLoader
import logging
import os
import h5py
import numpy as np
import torch
from mechanical.utils import MateTypes
from mechanical.geometry import displaced_min_signed_distance, min_signed_distance_symmetric
from mechanical.utils import joinmeshes
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
                dir.create_dataset('indices', data=np.array(dir_data[c][0]))
                ax = dir.create_group('axial_clusters')
                for c2 in all_axis_clusters[c]:
                    ax.create_dataset(str(c2), data=np.array(all_axis_clusters[c][c2]))
            pairdata = f.create_group('pair_data') #here, indices mean the UNIQUE representative indices for retrieving each axis from the mc data
            for pair in pairs_to_dir_clusters:
                key = f'{pair[0]},{pair[1]}'
                pair_group = pairdata.create_group(key)
                dirgroup = pair_group.create_group('dirs')
                #dirgroup.create_dataset('values', data = np.array(pairs_to_dirs[pair]))
                dirgroup.create_dataset('indices', data = np.array(list(pairs_to_dir_clusters[pair])))
                ax_group = pair_group.create_group('axes')
                if pair in pairs_to_axes:
                    for dir_ind in pairs_to_axes[pair]:
                        ax_cluster = ax_group.create_group(str(dir_ind))
                        #ax_cluster.create_dataset('values', data = np.array(origins))
                        ax_cluster.create_dataset('indices', data = np.array(pairs_to_axes[pair][dir_ind]))

        return {'mc_stats': stats}