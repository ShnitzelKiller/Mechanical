from mechanical.data import AssemblyLoader, Stats, MeshLoader
import logging
import os
import h5py
import numpy as np
import torch
from mechanical.utils import MateTypes
from mechanical.geometry import displaced_min_signed_distance, min_signed_distance_symmetric

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
        stat = Stats()
        stat.append(data.assembly_info.stats, data.ind)
        out_dict['assembly_stats'] = stat
        batch = data.assembly_info.create_batches()
        torch_datapath = os.path.join(self.out_path, f'{data.ind}.dat')
        torch.save(batch, torch_datapath)
        return out_dict


class DisplacementPenalty(DataVisitor):
    def __init__(self, global_data, sliding_distance, rotation_angle, num_samples, meshpath):
        self.transforms = MeshLoader(meshpath)
        self.global_data = global_data
        self.sliding_distance = sliding_distance
        self.rotation_angle = rotation_angle
        self.samples = num_samples
    
    def process(self, data):
        stats = Stats()
        part_subset = self.global_data.part_df.loc[[data.ind]]
        mate_subset = self.global_data.mate_df.loc[[data.ind]]
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

                    meshes_subset = [meshes[pid1], meshes[pid2]]

                    penalty_separation_sliding = 0
                    penalty_penetration_sliding = 0
                    penalty_separation_rotating = 0
                    penalty_penetration_rotating = 0

                    base_penalty = min_signed_distance_symmetric(meshes_subset, samples=self.samples)
                    base_penetration = max(0, -base_penalty) / rel_distance
                    base_separation = max(0, base_penalty) / rel_distance

                    if sliding:
                        penalty = displaced_min_signed_distance(meshes_subset, mate_subset.iloc[j]['Axes1'][:,2], samples=self.samples, displacement=rel_distance)
                        penalty_neg = displaced_min_signed_distance(meshes_subset, mate_subset.iloc[j]['Axes1'][:,2], samples=self.samples, displacement=-rel_distance)
                        penalty_separation_sliding = max(0, max(penalty, penalty_neg) / rel_distance)
                        penalty_penetration_sliding = max(0, -min(penalty, penalty_neg) / rel_distance)

                    if rotating:
                        penalty = displaced_min_signed_distance(meshes_subset, mate_subset.iloc[j]['Axes1'][:,2], origin=mate_subset.iloc[j]['Origin1'], motion_type='ROTATE', samples=self.samples, displacement=self.rotation_angle)
                        penalty_neg = displaced_min_signed_distance(meshes_subset, mate_subset.iloc[j]['Axes1'][:,2], origin=mate_subset.iloc[j]['Origin1'], motion_type='ROTATE', samples=self.samples, displacement=-self.rotation_angle)
                        penalty_separation_rotating = max(0, max(penalty, penalty_neg) / rel_distance)
                        penalty_penetration_rotating = max(0, -min(penalty, penalty_neg) / rel_distance)
                    
                    stats.append({'Assembly': data.ind, 'type': mtype,
                                    'base_penetration': base_penetration,
                                    'base_separation': base_separation,
                                    'penalty_separation_sliding': penalty_separation_sliding,
                                    'penetration_sliding': penalty_penetration_sliding,
                                    'separation_rotating': penalty_separation_rotating,
                                    'penetration_rotating': penalty_penetration_rotating,
                                    'penetration_total': max(base_penetration, max(penalty_penetration_rotating, penalty_penetration_sliding)),
                                    'separation_total': max(base_separation, max(penalty_separation_rotating, penalty_separation_sliding))}, index=mate_subset.iloc[j]['MateIndex'])
        return {'mate_penetration_stats': stats}