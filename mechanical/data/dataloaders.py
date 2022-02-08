import numpy as np
import os
import pandas as ps
import json
from onshape.brepio import Mate
from mechanical.data import AssemblyInfo

class Stats:
    def __init__(self, format='parquet', key=None):
        self.rows = []
        self.indices = []
        self.checkpoint = 0
        self.format = format
        self.key = key
    
    def append(self, row, index=None):
        self.rows.append(row)
        if index is not None:
            self.indices.append(index)
    
    def to_df(self, incremental=False):
        if not incremental:
            self.checkpoint = 0
        df = ps.DataFrame(self.rows[self.checkpoint:], index=self.indices[self.checkpoint:] if self.indices else None)
        if incremental:
            self.checkpoint = len(self.rows)
        return df

    def save_df(self, path, incremental=False):
        df = self.to_df(incremental=incremental)
        if incremental:
            path = path + f'_{self.checkpoint}'
        if self.format == 'parquet':
            df.to_parquet(path + '.parquet')
        elif self.format == 'hdf':
            df.to_hdf(path + '.h5', self.key)
    
    def combine(self, other):
        self.rows += other.rows
        self.indices += other.indices
    
    

class Dataset:
    def __init__(self, index_file, stride, stats_path, start_index=0, stop_index=-1):
        with open(index_file,'r') as f:
            self.index = [int(l.rstrip()) for l in f.readlines()]
        self.stride = stride
        self.stats_path = stats_path
        self.stats = dict()
        self.start_index = start_index
        self.stop_index = stop_index if stop_index >= 0 else len(self.index)
    
    def map_data(self, transforms, action):
        """
        action: function that returns a dictionary from names to Stats objects
        """
        for i,ind in enumerate(self.index[self.start_index:self.stop_index]):
            data = transforms(ind)
            results = action(data)
            self.log_results(results)
            if (i+self.start_index+1) % self.stride == 0:
                self.save_results(incremental=True)
        self.save_results(incremental=False)

    def log_results(self, results):
        for key in results:
            if key in self.stats:
                self.stats[key].combine(results[key])
            else:
                self.stats[key] = results[key]
    
    def save_results(self, incremental):
        for key in self.stats:
            self.stats[key].save_df(os.path.join(self.stats_path, key), incremental=incremental)

## functions for loading the assembly with PSPart
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