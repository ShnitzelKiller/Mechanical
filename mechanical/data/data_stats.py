from mechanical.data import AssemblyLoader, Stats, MeshLoader
import logging
import os
import h5py
import numpy as np
import torch
from mechanical.utils import MateTypes
from mechanical.geometry import displaced_min_signed_distance, min_signed_distance_symmetric
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
                    #p = mp.plot(*meshes_subset[0])
                    #p.add_mesh(*meshes_subset[1])
                    #p.save("debugvis.html")

                    penalties_separation_sliding = [0,0]
                    penalties_penetration_sliding = [0,0]
                    penalties_separation_rotating = [0,0]
                    penalties_penetration_rotating = [0,0]

                    base_penalty = min_signed_distance_symmetric(meshes_subset, samples=self.samples)
                    base_penetration = max(0, -base_penalty) / rel_distance
                    base_separation = max(0, base_penalty) / rel_distance

                    tf = part_subset.iloc[pid1]['Transform']
                    origin = tf[:3,:3] @ mate_subset.iloc[j]['Origin1'] + tf[:3,3]
                    axis = tf[:3,:3] @ mate_subset.iloc[j]['Axes1'][:,2]

                    if sliding:
                        for k,displacement in enumerate([rel_distance, -rel_distance]):
                            sd = displaced_min_signed_distance(meshes_subset, axis, samples=self.samples, displacement=displacement)
                            penalties_penetration_sliding[k] = max(0, -sd) / rel_distance
                            penalties_separation_sliding[k] = max(0, sd) / rel_distance
                    if rotating:
                        for k, angle in enumerate([self.rotation_angle, -self.rotation_angle]):
                            sd = displaced_min_signed_distance(meshes_subset, axis, origin=origin, motion_type='ROTATE', samples=self.samples, displacement=angle)
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