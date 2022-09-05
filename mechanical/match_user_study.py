import pandas as ps
import os
import numpy as np
import numpy.linalg as LA

if __name__ == '__main__':

    dataroot = '/fast/jamesn8/assembly_data/assembly_torch2_fixsize'
    study_gen_name = 'test_user_study_full_pipeline'
    study_df_path = '/projects/grail/jamesn8/projects/mechanical/data_test/dataframes/assemblies_test_named.hdf5'
    assembly_df = ps.read_hdf(study_df_path,'assembly')
    part_df = ps.read_hdf(study_df_path,'part')
    mate_df = ps.read_hdf(study_df_path,'mate')
    final_df = ps.read_parquet(os.path.join(dataroot, study_gen_name, 'final_stat.parquet'))
    assembly_df_old = ps.read_hdf('/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5','assembly')
    part_df_old = ps.read_hdf('/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5_segmentation.h5','part')
    #mate_df_old = ps.read_hdf('/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5', 'mate')

    part_df_old_by_ass = part_df_old.set_index('Assembly')
    part_df_by_ass = part_df.set_index('Assembly')
    part_df_old_by_ass['PartIndex'] = part_df_old.index
    matches = []
    dists = []
    oldasses = []
    for ass in assembly_df.index:
        name = assembly_df.loc[ass, 'Name']
        oldindex = int(name.split(' ')[1])
        oldparts = part_df_old_by_ass.loc[oldindex]
        newparts = part_df_by_ass.loc[ass]
        assert(oldparts.shape[0] == newparts.shape[0])
        
        visited = set()
        for i in range(newparts.shape[0]):
            mindiff = np.inf
            match = -1
            for j in range(oldparts.shape[0]):
                if oldparts.iloc[j]['PartIndex'] not in visited:
                    tf1 = newparts.iloc[i]['Transform']
                    tf2 = oldparts.iloc[j]['Transform']
                    diff = LA.norm(tf1-tf2)
                    if diff < mindiff:
                        mindiff = diff
                        match = oldparts.iloc[j]['PartIndex']
            visited.add(match)
            matches.append(match)
            dists.append(mindiff)
        oldasses.append(oldindex)
                
    part_df['oldpart'] = matches
    part_df['mindiff'] = mindiff
    assembly_df['OriginalIndex'] = oldasses