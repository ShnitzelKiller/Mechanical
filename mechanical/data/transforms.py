from onshape.brepio import Mate
import os
import json
import numpy as np
from mechanical.data import AssemblyInfo
import logging
import h5py
from scipy.spatial.transform import Rotation as R
import torch
import functools
from mechanical.utils.data import df_to_mates


def compose(*fs):
    return functools.reduce(lambda f, g: lambda *a, **kw: f(g(*a, **kw)), fs)


class MeshLoader:
    def __init__(self, meshpath):
        self.meshpath = meshpath
    
    def __call__(self, data):
        fname = f'{data.ind}.dat'
        mesh_datapath = os.path.join(self.meshpath, fname)
        data.meshes = torch.load(mesh_datapath)
        return data

class AssemblyLoader:
    def __init__(self, global_data, datapath='/projects/grail/benjones/cadlab', use_uvnet_features=False, epsilon_rel=0.001, max_topologies=5000, pair_data=False, include_mcfs=True, precompute=False, load_geometry=True):
        self.global_data = global_data
        self.datapath = datapath
        self.use_uvnet_features = use_uvnet_features
        self.epsilon_rel = epsilon_rel
        self.max_topologies = max_topologies
        self.pair_data = pair_data
        self.include_mcfs = include_mcfs
        self.precompute = precompute
        self.load_geometry = load_geometry


    def __call__(self, data):
        part_subset = self.global_data.part_df.loc[data.ind]

        occ_ids = list(part_subset['PartOccurrenceID'])
        part_paths = []
        rel_part_paths = []
        transforms = []

        #TEMPORARY: Load correct transforms, and also save them separately
        #if args.mode == 'generate':
        with open(os.path.join(self.datapath, 'data/flattened_assemblies', self.global_data.assembly_df.loc[data.ind, "AssemblyPath"] + '.json')) as f:
            assembly_def = json.load(f)

        part_occs = assembly_def['part_occurrences']
        occ_to_index = dict()
        for i,occ in enumerate(occ_ids):
            occ_to_index[occ] = i
        
        if self.load_geometry:
            tf_dict = dict()
            for occ in part_occs:
                tf = np.array(occ['transform']).reshape(4, 4)
                # npyname = f'/fast/jamesn8/assembly_data/transform64_cache/{data.ind}.npy'
                # if not os.path.isfile(npyname):
                #     np.save(npyname, tf)
                tf_dict[occ['id']] = tf

            for j in range(part_subset.shape[0]):
                rel_path = os.path.join(*[part_subset.iloc[j][k] for k in ['did','mv','eid','config']], f'{part_subset.iloc[j]["PartId"]}.xt')
                path = os.path.join(self.datapath, 'data/models', rel_path)
                assert(os.path.isfile(path))
                rel_part_paths.append(rel_path)
                part_paths.append(path)
                #if args.mode == 'generate':
                transforms.append(tf_dict[occ_ids[j]])
                #else:
                #    transforms.append(part_subset.iloc[j]['Transform'])
            
            data.assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, epsilon_rel=self.epsilon_rel, use_uvnet_features=self.use_uvnet_features, max_topologies=self.max_topologies, include_mcfs=self.include_mcfs, precompute=self.precompute)
        else:
            data.occ_to_index = occ_to_index
        
        if self.pair_data:
            pair_to_index = dict()
            mate_subset = self.global_data.mate_df.loc[[data.ind]]
            mates = df_to_mates(mate_subset)
            for m,mate in enumerate(mates):
                part_indices = [occ_to_index[me[0]] for me in mate.matedEntities]
                pair_to_index[tuple(sorted(part_indices))] = m
            data.pair_to_index = pair_to_index
        return data

class LoadMCData:
    def __init__(self, mc_path):
        self.mc_path = mc_path
    
    def __call__(self, data):
        data.mcfile = os.path.join(self.mc_path, f"{data.ind}.hdf5")
        return data

class LoadBatch:
    def __init__(self, batch_path):
        self.batch_path = batch_path
    
    def __call__(self, data):
        data.batch = torch.load(os.path.join(self.batch_path, f'{data.ind}.dat'))
        return data

class ComputeDistances:
    def __init__(self, distance_threshold):
        self.distance_threshold = distance_threshold

    def __call__(self, data):
        data.distances = data.assembly_info.part_distances(self.distance_threshold)
        return data