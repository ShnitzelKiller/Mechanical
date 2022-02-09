from onshape.brepio import Mate
import os
import json
import numpy as np
from mechanical.data import AssemblyInfo

class Data:
    def __init__(self, ind):
        self.ind = ind

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

class AssemblyLoader:
    def __init__(self, assembly_df, part_df, mate_df, datapath, use_uvnet_features, epsilon_rel, max_topologies):
        self.assembly_df = assembly_df
        self.part_df = part_df
        self.mate_df = mate_df
        self.datapath = datapath
        self.use_uvnet_features = use_uvnet_features
        self.epsilon_rel = epsilon_rel
        self.max_topologies = max_topologies


    def __call__(self, ind):
        part_subset = self.part_df.loc[ind]
        mate_subset = self.mate_df.loc[[ind]]

        occ_ids = list(part_subset['PartOccurrenceID'])
        part_paths = []
        rel_part_paths = []
        transforms = []

        #TEMPORARY: Load correct transforms, and also save them separately
        #if args.mode == 'generate':
        with open(os.path.join(self.datapath, 'data/flattened_assemblies', self.assembly_df.loc[ind, "AssemblyPath"] + '.json')) as f:
            assembly_def = json.load(f)
        part_occs = assembly_def['part_occurrences']
        tf_dict = dict()
        for occ in part_occs:
            tf = np.array(occ['transform']).reshape(4, 4)
            # npyname = f'/fast/jamesn8/assembly_data/transform64_cache/{ind}.npy'
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
        pair_to_index = dict()
        mates = df_to_mates(mate_subset)
        for m,mate in enumerate(mates):
            part_indices = [occ_to_index[me[0]] for me in mate.matedEntities]
            pair_to_index[tuple(sorted(part_indices))] = m
        data = Data(ind)
        data.pair_to_index = pair_to_index
        data.assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, print, epsilon_rel=self.epsilon_rel, use_uvnet_features=self.use_uvnet_features, max_topologies=self.max_topologies)
        return data