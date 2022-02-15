from onshape.brepio import Mate
import os
import json
import numpy as np
from mechanical.data import AssemblyInfo
import logging
import h5py
from scipy.spatial.transform import Rotation as R
import torch


def df_to_mates(mate_subset):
    mates = []
    for j in range(mate_subset.shape[0]):
        mate_row = mate_subset.iloc[j]
        mate = Mate(occIds=[mate_row[f'Part{p+1}'] for p in range(2)],
                origins=[mate_row[f'Origin{p+1}'] for p in range(2)],
                rots=[mate_row[f'Axes{p+1}'] for p in range(2)],
                name=mate_row['Name'],
                mateType=mate_row['Type'])
        mates.append(mate)
    return mates

class MeshLoader:
    def __init__(self, meshpath):
        self.meshpath = meshpath
    
    def __call__(self, data):
        fname = f'{data.ind}.dat'
        mesh_datapath = os.path.join(self.meshpath, fname)
        data.meshes = torch.load(mesh_datapath)
        return data

class AssemblyLoader:
    def __init__(self, global_data, datapath='/projects/grail/benjones/cadlab', use_uvnet_features=False, epsilon_rel=0.001, max_topologies=5000, pair_data=False, include_mcfs=True):
        self.global_data = global_data
        self.datapath = datapath
        self.use_uvnet_features = use_uvnet_features
        self.epsilon_rel = epsilon_rel
        self.max_topologies = max_topologies
        self.pair_data = pair_data
        self.include_mcfs = include_mcfs


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
        
        occ_to_index = dict()
        for i,occ in enumerate(occ_ids):
            occ_to_index[occ] = i
        data.assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, epsilon_rel=self.epsilon_rel, use_uvnet_features=self.use_uvnet_features, max_topologies=self.max_topologies, include_mcfs=self.include_mcfs)
        if self.pair_data:
            pair_to_index = dict()
            mate_subset = self.global_data.mate_df.loc[[data.ind]]
            mates = df_to_mates(mate_subset)
            for m,mate in enumerate(mates):
                part_indices = [occ_to_index[me[0]] for me in mate.matedEntities]
                pair_to_index[tuple(sorted(part_indices))] = m
            data.pair_to_index = pair_to_index
        return data

class GetMCData:
    def __init__(self, global_data, max_z_groups, mc_path, precomputed=True):
        self.global_data = global_data
        self.max_groups = max_z_groups
        self.mc_path = mc_path
        self.precomputed = precomputed

    def __call__(self, data):
        part_subset = self.global_data.part_df.loc[[data.ind]]
        mcfile = os.path.join(self.mc_path, f"{data.ind}.hdf5")
        if self.precomputed:
            if os.path.isfile(mcfile):
                data.mcfile = mcfile
        elif data.assembly_info.valid and len(data.assembly_info.parts) == part_subset.shape[0]:
            axes, pairs_to_dirs, pairs_to_axes, dir_clusters, dir_to_ax_clusters = data.assembly_info.mate_proposals(max_z_groups=self.max_groups, axes_only=True)
            with h5py.File(mcfile, "w") as f:
                mc_frames = f.create_group('mc_frames')
                for k in range(len(data.assembly_info.mc_origins_all)):
                    partgroup = mc_frames.create_group(str(k))
                    partgroup.create_dataset('origins', data=np.array(data.assembly_info.mc_origins_all[k]))
                    rots_quat = np.array([R.from_matrix(rot).as_quat() for rot in data.assembly_info.mc_rots_all[k]])
                    partgroup.create_dataset('rots', data=rots_quat)
                dir_groups = f.create_group('dir_clusters')
                for c in dir_clusters:
                    dir = dir_groups.create_group(str(c))
                    dir.create_dataset('indices', data=np.array(list(dir_clusters[c])))
                    ax = dir.create_group('axial_clusters')
                    if c in dir_to_ax_clusters:
                        for c2 in dir_to_ax_clusters[c]:
                            ax.create_dataset(str(c2), data=np.array(list(dir_to_ax_clusters[c][c2])))
                pairdata = f.create_group('pair_data') #here, indices mean the UNIQUE representative indices for retrieving each axis from the mc data
                for pair in pairs_to_dirs:
                    key = f'{pair[0]},{pair[1]}'
                    group = pairdata.create_group(key)
                    dirgroup = group.create_group('dirs')
                    dirgroup.create_dataset('values', data = np.array(pairs_to_dirs[pair]))
                    dirgroup.create_dataset('indices', data = np.array(list(axes[pair])))
                    ax_group = group.create_group('axes')
                    for dir_ind, origins in pairs_to_axes[pair]:
                        ax_cluster = ax_group.create_group(str(dir_ind))
                        ax_cluster.create_dataset('values', data = np.array(origins))
                        ax_cluster.create_dataset('indices', data = np.array(axes[pair][dir_ind]))
            data.mcfile = mcfile

        return data

