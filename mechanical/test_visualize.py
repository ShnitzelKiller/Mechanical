import onshape as brepio
from mechanical.visualize2 import plot_assembly, filter_assembly
import os
import pandas as ps
from mechanical.utils.data import newmate_df_to_mates
import torch


if __name__ == '__main__':
    df_name = '/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5'
    df_name_part = '/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5_segmentation.h5'
    
    newmate_df = ps.read_parquet('/fast/jamesn8/assembly_data/assembly_torch2_fixsize/full_pipeline/newmate_stats.parquet')
    newmate_df_by_ass = newmate_df.copy()
    newmate_df_by_ass['NewmateIndex'] = newmate_df_by_ass.index
    newmate_df_by_ass.set_index('Assembly', inplace=True)

    mate_df = ps.read_hdf(df_name,'mate')
    mate_df_by_ass = mate_df.copy()
    mate_df_by_ass = mate_df.set_index('Assembly')
    mate_df_by_ass['MateIndex'] = mate_df.index

    assembly_df = ps.read_hdf(df_name,'assembly')
    part_df = ps.read_hdf(df_name_part,'part')
    has_geometry = part_df.groupby('Assembly')['HasGeometry'].agg(all)
    assembly_df['HasAllGeometry'] = has_geometry

    part_df.set_index('Assembly', inplace=True)

    datapath = '/projects/grail/benjones/cadlab'
    loader = brepio.Loader(datapath)

    batch_path = '/fast/jamesn8/assembly_data/assembly_torch2_fixsize/full_pipeline/batches'

    sample = 1030



    assemblypath = assembly_df.loc[sample, "AssemblyPath"]
    did, mv, eid = assemblypath.split('_')
    geo, mates = loader.load_flattened(assemblypath + '.json', skipInvalid=True, use_pspy=True)
    mates = [mate for mate in mates if len(mate.matedEntities) == 2]

    mate_counts = dict()
    for mate in mates:
        if len(mate.matedEntities) == 2:
            if mate.type not in mate_counts:
                mate_counts[mate.type] = 0
            mate_counts[mate.type] += 1
    num_connected = assembly_df.loc[sample, "ConnectedComponents"]
    num_rigid = assembly_df.loc[sample, "RigidPieces"]
    if num_connected > 1:
        print('warning:',num_connected,'connected components')
    print('rigid pieces:',num_rigid)
    print('total parts:',len(geo))
    print(f'mates: {len(mates)}: ',mate_counts)

    augmented_start = len(mates)
    if sample in mate_df_by_ass.index:
        mate_subset = mate_df_by_ass.loc[[sample]]
        #augmented mates
        newmate_subset = newmate_df_by_ass.loc[[sample]]
        batch = torch.load(os.path.join(batch_path, f'{sample}.dat'))
        augmented_mates = newmate_df_to_mates(newmate_subset, batch)
        mates += augmented_mates
    
    print(f'augmented mates: {len(augmented_mates)}')
    
    geo_to_id = {occ: i for i,occ in enumerate(geo)}
    print(geo_to_id)
    print([(len(mate.matedEntities),mate.type) for mate in mates])
    pairs = [(geo_to_id[mate.matedEntities[0][0]],geo_to_id[mate.matedEntities[1][0]]) for mate in mates]
    choices = [(f'{"" if i < augmented_start else "augmented "}mate ({pairs[i][0]},{pairs[i][1]}) (ID {(mate_subset.iloc[i]["MateIndex"] if i < augmented_start else newmate_subset.iloc[i-augmented_start]["NewmateIndex"])}) ({mates[i].type}) ({mates[i].name})',i) for i in range(len(mates)) if len(mates[i].matedEntities) == 2]
    choices = [('fullAssembly', -1)] + choices

    rigid_labels = list(part_df.loc[sample, 'RigidComponentID'])
    renderer = plot_assembly(geo, mates, rigid_labels = rigid_labels)
