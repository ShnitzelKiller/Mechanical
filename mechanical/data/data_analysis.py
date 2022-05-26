import os
import pandas as ps
from pandas import DataFrame
import mechanical.onshape as onshape
import json
import numpy as np
from mechanical.utils import adjacency_list, adjacency_list_from_brepdata, connected_components, homogenize, adjacency_matrix
import argparse
from tqdm import tqdm
import logging


def find_redundant_mates(mate_df):
    """
    Indices of mates with the same part, and those with duplicate parts
    """
    mate_df_filtered = mate_df[mate_df['Part1'] != mate_df['Part2']]
    flattened_mateinfo = mate_df_filtered[['Origin1','Origin2','Axes1','Axes2']].apply(lambda x: [tuple(l.flatten()) for l in x])
    flattened_mateinfo['Assembly'] = mate_df_filtered['Assembly']
    flattened_mateinfo['Part1'] = mate_df_filtered['Part1']
    flattened_mateinfo['Part2'] = mate_df_filtered['Part2']
    flattened_mateinfo['Type'] = mate_df_filtered['Type']
    return flattened_mateinfo.duplicated(keep='first')


def find_multimates(mate_df):
    mate_df_indexed = mate_df.copy()
    mate_df_indexed['MateID'] = [str(tup[0]) + '-' + '-'.join(sorted(tup[1:])) for tup in zip(mate_df_indexed['Assembly'], mate_df_indexed['Part1'], mate_df_indexed['Part2'])]
    mate_df_indexed.set_index('MateID', inplace=True)
    return mate_df_indexed.index.duplicated(False)


def find_nonduplicate_assemblies(assembly_df, deduped_df):
    """
    all assemblies that are not duplicates (based on deduplication) and have all loaded geometry. The deduped_df should be in cadlab/data/deduped_assembly_list.parquet
    """
    deduped_df.drop('is_unique', axis=1, inplace=True)
    deduped_df['AssemblyPath'] = ['_'.join(me) for me in zip(deduped_df['assembly_documentId'], deduped_df['assembly_documentMicroversion'], deduped_df['assembly_elementId'])]
    deduped_df.set_index('AssemblyPath', inplace=True)
    deduped_df_joined = assembly_df.join(deduped_df, on='AssemblyPath', how='inner')
    return deduped_df_joined.index

def find_valid_assemblies(part_df):
    """
    All assemblies with all geometry (index only includes assemblies with at least 1 part)
    """
    return part_df.groupby('Assembly')['HasGeometry'].agg(all)


def main():
    logging.basicConfig(filename='generate_dataframes.log', level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)
    parser.add_argument('--datapath',type=str, default='/projects/grail/benjones/cadlab/data')
    parser.add_argument('--path', default='/fast/jamesn8/assembly_data/')
    parser.add_argument('--postprocess', choices=['segmentation', 'filtering'])
    parser.add_argument('--prefilter',action='store_true')
    parser.add_argument('--deduped_df', default='/projects/grail/benjones/cadlab/data/deduped_assembly_list.parquet')
    parser.add_argument('--use_workspaces', action='store_true')
    args = parser.parse_args()

    datapath = args.datapath
    loader = onshape.Loader(datapath)
    NAME = args.name
    PATH = args.path
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    FULLNAME = os.path.join(PATH, NAME)
    POSTPROCESS = args.postprocess
    
    if POSTPROCESS is None:
        filter_set = set()
        if args.prefilter:
            with open('data/dataset/subassembly_filters/filter_list.txt') as f:
                filter_list = f.readlines()
            logging.info('making filter set')
            for l in filter_list:
                filter_set.add(l.strip())
            del filter_list

        valid_dict = dict()

        assembly_rows = []
        assembly_indices = []

        part_rows = []
        mate_rows = []

        logging.info('scanning assemblies...')
        j = 0
        for num_processed,entry in tqdm(enumerate(os.scandir(os.path.join(datapath,'flattened_assemblies')))):
            
            if not entry.name.endswith('.json') or entry.name in filter_set:
                continue

            did, wid, eid = os.path.split(entry.name)[1].split('_')
                
            with open(os.path.join(args.datapath, 'elements', did, wid, eid),'r') as f:
                doc = json.load(f)
            assname = doc['name']

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
                    mate_rows.append([np.int32(j), mate.matedEntities[0][0], mate.matedEntities[1][0], mate.type, origins[0], axes[0], origins[1], axes[1], mate.name])

            assembly_rows.append([os.path.splitext(entry.name)[0], np.int32(num_components), np.int32(num_rigid), np.int32(num_lone), np.int32(len(occs)), np.int32(num_part_mates), np.int32(len(mates)), assname])
            assembly_indices.append(np.int32(j)) 
            
            for p, occ in enumerate(occs):
                part = occs[occ][1]
                did = part['documentId']
                eid = part['elementId']
                config = part['fullConfiguration']
                if args.use_workspaces:
                    with open(os.path.join(args.datapath, 'documents', f'{did}.json'),'r') as f:
                        doc = json.load(f)
                    mv = doc['defaultWorkspace']['id']
                else:
                    mv = part['documentMicroversion']

                if 'partId' in part:
                    pid = part['partId']
                else:
                    pid = ''
                filepath = os.path.join(did, mv, eid, config, f'{pid}.xt')
                
                if filepath in valid_dict:
                    isValid = valid_dict[filepath]
                else:
                    filepath = os.path.join(datapath, 'models', filepath)
                    isValid = os.path.isfile(filepath)
                    valid_dict[filepath] = isValid
                
                part_rows.append([np.int32(j), occ, did, mv, eid, config, pid, occs[occ][0], isValid, labeling[p]])

            j += 1

        logging.info('building dataframes...')
        assembly_df = DataFrame(assembly_rows, index=assembly_indices, columns=['AssemblyPath','ConnectedComponents','RigidPieces','LonePieces', 'NumParts', 'NumBinaryPartMates', 'TotalMates', 'Name'])
        mate_df = DataFrame(mate_rows, columns=['Assembly','Part1','Part2','Type','Origin1','Axes1','Origin2','Axes2','Name'])
        part_df = DataFrame(part_rows, columns=['Assembly','PartOccurrenceID','did','wid' if args.use_workspaces else 'mv','eid','config','PartId', 'Transform', 'HasGeometry', 'RigidComponentID'])

        logging.info('saving dataframes...')
        assembly_df.to_hdf(FULLNAME,'assembly')
        mate_df.to_hdf(FULLNAME,'mate')
        part_df.to_hdf(FULLNAME,'part')

    elif POSTPROCESS == 'segmentation':
        part_df = ps.read_hdf(FULLNAME, 'part')
        if 'RigidComponentID' in part_df.keys():
            logging.warning('Segmentation data already present!')
        else:
            logging.info('adding rigid component segmentation to data...')
            mate_df = ps.read_hdf(FULLNAME, 'mate')
            assembly_df = ps.read_hdf(FULLNAME, 'assembly')

            part_df_indexed = part_df.set_index('Assembly')
            mate_df.set_index('Assembly', inplace=True)

            labelings = []

            for k,ass in tqdm(enumerate(part_df_indexed.index.unique())):
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
            
            logging.info('creating updated dataframe')
            part_df['RigidComponentID'] = labelings
            part_df.to_hdf(FULLNAME+'_'+POSTPROCESS+'.h5', 'part')

    elif POSTPROCESS == 'filtering':
        part_df = ps.read_hdf(FULLNAME, 'part')
        mate_df = ps.read_hdf(FULLNAME, 'mate')
        assembly_df = ps.read_hdf(FULLNAME, 'assembly')

        duplicate_mates = find_redundant_mates(mate_df)
        mate_df_filtered = mate_df[~duplicate_mates]

        multimates = find_multimates(mate_df_filtered)
        mate_duplicates = mate_df[multimates]
        multimate_groups = mate_duplicates.groupby(by='MateID')
        aggregated = multimate_groups.agg({'Type':lambda x: 'FASTENED' if 'FASTENED' in list(x) else '-'.join(sorted(list(x))),'Assembly':lambda x:x[0]})



        deduped_df = ps.read_parquet(args.deduped_df)
        unique_assemblies = find_nonduplicate_assemblies(assembly_df, deduped_df)
        assembly_df = assembly_df[unique_assemblies]

        valid_assemblies = find_valid_assemblies(part_df)
        assembly_df = assembly_df[assembly_df['NumParts'] > 0][valid_assemblies]

        



if __name__ == '__main__':
    main()