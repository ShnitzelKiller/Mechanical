import os
import pandas as ps
from pandas import DataFrame
import mechanical.onshape as onshape
import json
import numpy as np
from mechanical.utils import adjacency_list, adjacency_list_from_brepdata, connected_components, homogenize, adjacency_matrix
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5')
    parser.add_argument('--path', default='/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/full_assembly_data')
    parser.add_argument('--postprocess', choices=['segmentation'])
    parser.add_argument('--progress_interval', type=int, default=100)
    args = parser.parse_args()

    datapath = '/projects/grail/benjones/cadlab'
    loader = onshape.Loader(datapath)
    NAME = args.name
    PATH = args.path
    FULLNAME = os.path.join(PATH, NAME)
    POSTPROCESS = args.postprocess
    PROGRESS_INTERVAL = args.progress_interval
    
    if POSTPROCESS is None:
        with open('data/dataset/subassembly_filters/filter_list.txt') as f:
            filter_list = f.readlines()
        print('making filter set')
        filter_set = set()
        for l in filter_list:
            filter_set.add(l.strip())
        del filter_list

        valid_dict = dict()

        assembly_rows = []
        assembly_indices = []

        part_rows = []
        mate_rows = []

        j = 0
        for entry in os.scandir(os.path.join(datapath,'data','flattened_assemblies')):
            
            if not entry.name.endswith('.json') or entry.name in filter_set:
                continue

            occs, mates = loader.load_flattened(entry.name,geometry=False, skipInvalid=True)

            adj_list = adjacency_list_from_brepdata(occs, mates)
            if len(adj_list) > 0:
                adj = homogenize(adj_list)
            else:
                adj = np.zeros([0,0],np.int32)
            num_lone = adj.shape[0] if adj.shape[1] == 0 else (adj[:,0] < 0).sum()
            #num_connections = np.sum(adj, 0)
            #num_lone = len([c for c in num_connections if c == 0])

            num_components, _ = connected_components(adj)
            num_rigid, labeling = connected_components(adj, connectionType='fasten')
            assert(len(labeling) == len(occs))

            num_part_mates = 0
            for mate in mates:
                if len(mate.matedEntities) == 2:
                    num_part_mates += 1
                    axes = []
                    origins = []
                    for me in mate.matedEntities:
                        #tf = occs[me[0]][0]
                        #newaxes = tf[:3, :3] @ me[1][1]
                        #neworigin = tf[:3,:3] @ me[1][0] + tf[:3,3]
                        axes.append(me[1][1])
                        origins.append(me[1][0])
                    mate_rows.append([np.int32(j), mate.matedEntities[0][0], mate.matedEntities[1][0], mate.type, origins[0].astype(np.float32), axes[0].astype(np.float32), origins[1].astype(np.float32), axes[1].astype(np.float32), mate.name])

            assembly_rows.append([os.path.splitext(entry.name)[0], np.int32(num_components), np.int32(num_rigid), np.int32(num_lone), np.int32(len(occs)), np.int32(num_part_mates), np.int32(len(mates))])
            assembly_indices.append(np.int32(j)) 
            
            for p, occ in enumerate(occs):
                part = occs[occ][1]
                did = part['documentId']
                mv = part['documentMicroversion']
                eid = part['elementId']
                config = part['fullConfiguration']
                if 'partId' in part:
                    pid = part['partId']
                else:
                    pid = ''
                filepath = os.path.join(did, mv, eid, config, f'{pid}.xt')
                
                if filepath in valid_dict:
                    isValid = valid_dict[filepath]
                else:
                    filepath = os.path.join(datapath, 'data/models', filepath)
                    isValid = os.path.isfile(filepath)
                    valid_dict[filepath] = isValid
                
                part_rows.append([np.int32(j), occ, did, mv, eid, config, pid, occs[occ][0], isValid, labeling[p]])

            j += 1

        print('building dataframes...')
        assembly_df = DataFrame(assembly_rows, index=assembly_indices, columns=['AssemblyPath','ConnectedComponents','RigidPieces','LonePieces', 'NumParts', 'NumBinaryPartMates', 'TotalMates'])
        mate_df = DataFrame(mate_rows, columns=['Assembly','Part1','Part2','Type','Origin1','Axes1','Origin2','Axes2','Name'])
        part_df = DataFrame(part_rows, columns=['Assembly','PartOccurrenceID','did','mv','eid','config','PartId', 'Transform', 'HasGeometry', 'RigidComponentID'])

        print('saving dataframes...')
        assembly_df.to_hdf(FULLNAME,'assembly')
        mate_df.to_hdf(FULLNAME,'mate')
        part_df.to_hdf(FULLNAME,'part')

    elif POSTPROCESS == 'segmentation':
        print('adding rigid component segmentation to data...')
        part_df = ps.read_hdf(FULLNAME, 'part')
        mate_df = ps.read_hdf(FULLNAME, 'mate')
        assembly_df = ps.read_hdf(FULLNAME, 'assembly')

        part_df_indexed = part_df.set_index('Assembly')
        mate_df.set_index('Assembly', inplace=True)

        labelings = []

        for k,ass in enumerate(part_df_indexed.index.unique()):
            if k % PROGRESS_INTERVAL == 0:
                print(f'processing assembly {k}/{assembly_df.shape[0]}')
            num_mates = assembly_df.loc[ass, 'NumBinaryPartMates']
            num_parts = assembly_df.loc[ass, 'NumParts']
            part_subset = part_df_indexed.loc[ass,'PartOccurrenceID']
            if isinstance(part_subset, ps.Series):
                assert(num_parts == len(part_subset))
            else:
                assert(isinstance(part_subset, str))
                assert(num_parts == 1)
            if num_mates > 0 and num_parts > 1:
                mate_subset = mate_df.loc[ass,['Part1', 'Part2', 'Type']]
                records = mate_subset.to_records(index=False) if mate_subset.ndim == 2 else [list(mate_subset)]
                adj_list = adjacency_list(list(part_subset), records)
                if len(adj_list) > 0:
                    adj = homogenize(adj_list)
                else:
                    adj = np.zeros([0,0],np.int32)
                num_rigid, labeling = connected_components(adj, connectionType='fasten')
                assert(assembly_df.loc[ass, 'RigidPieces'] == num_rigid)
                labelings += list(labeling)
            else:
                labelings += list(range(num_parts))
        
        print('creating updated dataframe')
        part_df['RigidComponentID'] = labelings
        part_df.to_hdf(FULLNAME+'_'+POSTPROCESS+'.h5', 'part')



if __name__ == '__main__':
    main()