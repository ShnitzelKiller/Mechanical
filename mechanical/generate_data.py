from argparse import ArgumentParser
import pandas as ps
from assembly_data import AssemblyInfo
import os
import time
from onshape.brepio import Mate
import torch
import shutil

def main():
    parser = ArgumentParser(allow_abbrev=False, conflict_handler='resolve')
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--stop_index', type=int, default=-1)
    parser.add_argument('--epsilon_rel', type=float, default=0.001)
    parser.add_argument('--max_groups', type=int, default=10)
    parser.add_argument('--max_topologies', type=int, default=5000)
    parser.add_argument('--max_mc_pairs', type=int, default=100000)
    parser.add_argument('--stride', type=int, default=500)
    parser.add_argument('--save_stats', type=bool, default=True)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--use_uvnet_features', type=bool, default=True)
    parser.add_argument('--last_checkpoint', action='store_true')
    
    #parser.add_argument('--prob_threshold', type=float, default=0.7)

    args = parser.parse_args()
    #PROB_THRESHOLD = args.prob_threshold

    start_index = args.start_index
    stop_index = args.stop_index
    epsilon_rel = args.epsilon_rel
    max_groups = args.max_groups
    max_topologies = args.max_topologies
    max_mc_pairs = args.max_mc_pairs
    stride = args.stride
    #debug_run=False
    save_stats=args.save_stats
    use_uvnet_features = args.use_uvnet_features
    clear_previous_run = args.clear

    statspath = os.path.join(args.out_path, 'stats')
    outdatapath = os.path.join(args.out_path, 'data')

    if clear_previous_run and os.path.isdir(statspath):
        print('clearing')
        shutil.rmtree(statspath)
        shutil.rmtree(outdatapath)

    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.isdir(statspath):
        os.mkdir(statspath)
        os.mkdir(outdatapath)
    
    logfile = os.path.join(statspath, 'log.txt')
    resumefile = os.path.join(statspath, 'recovery.txt')
    def LOG(st):
        with open(logfile,'a') as logf:
            logf.write(st + '\n')
            logf.flush()
    
    def record_val(val):
        with open(resumefile, 'w') as f:
            f.write(val + '\n')
            f.flush()
    
    def read_val():
        with open(resumefile,'r') as f:
            return int(f.read())
    
    if args.last_checkpoint:
        start_index = read_val()

    replace_keys={'truncated_mc_pairs': False, 'invalid_bbox': False}

    LOG('=========RESTARTING=========')
    LOG(str(args))
    LOG('loading dataframes...')
    datapath = '/projects/grail/benjones/cadlab'
    df_name = '/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5'
    assembly_df = ps.read_hdf(df_name,'assembly')
    mate_df = ps.read_hdf(df_name,'mate')
    part_df = ps.read_hdf(df_name,'part')
    mate_df['MateIndex'] = mate_df.index
    part_df['PartIndex'] = part_df.index
    mate_df.set_index('Assembly', inplace=True)
    part_df.set_index('Assembly', inplace=True)


    with open('fully_connected_moving_no_multimates.txt','r') as f:
        assembly_indices = [int(l.rstrip()) for l in f.readlines()]

    last_ckpt = 0
    last_mate_ckpt = 0
    all_stats = []
    all_mate_stats = []
    processed_indices = []
    mate_indices = []
    invalid_part_paths_and_transforms = []

    run_start_time = time.time()
    if stop_index < 0:
        stop_index = len(assembly_indices)
    for num_processed,ind in enumerate(assembly_indices[start_index:stop_index]):
        print(f'num_processed: {num_processed+start_index}/{len(assembly_indices)}')
        LOG(f'{num_processed+start_index}/{len(assembly_indices)}: processing {assembly_df.loc[ind,"AssemblyPath"]} at {time.time()-run_start_time}')

        part_subset = part_df.loc[ind]
        mate_subset = mate_df.loc[ind]
        if mate_subset.ndim == 1:
            mate_subset = ps.DataFrame([mate_subset], columns=mate_subset.keys())
        
        part_paths = []
        transforms = []
        occ_ids = []

        for j in range(part_subset.shape[0]):
            path = os.path.join(datapath, 'data/models', *[part_subset.iloc[j][k] for k in ['did','mv','eid','config']], f'{part_subset.iloc[j]["PartId"]}.xt')
            assert(os.path.isfile(path))
            part_paths.append(path)
            transforms.append(part_subset.iloc[j]['Transform'])
            occ_ids.append(part_subset.iloc[j]['PartOccurrenceID'])

        assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, epsilon_rel=epsilon_rel, use_uvnet_features=use_uvnet_features, max_topologies=max_topologies)
        num_topologies = assembly_info.num_topologies()
        LOG(f'initialized AssemblyInfo with {num_topologies} topologies')
        LOG(f'topologies per part: {[part.num_topologies for part in assembly_info.parts]}')
        for j,part in enumerate(assembly_info.parts):
            if part.num_topologies != len(part.node_types):
                LOG(f'part {j} has valid bit {part.valid}, and length of node type vector is {len(part.node_types)}')
            #assert(part.num_topologies == len(part.node_types))
        
        skipped = True
        if assembly_info.stats['initialized']:
            #LOG(f'normalization_matrices:{assembly_info.norm_matrices}')
            num_invalid_parts = assembly_info.num_invalid_parts()
            assert(num_invalid_parts == 0)
            skipped = False
            mates = []
            for j in range(mate_subset.shape[0]):
                mate_row = mate_subset.iloc[j]
                mate = Mate(occIds=[mate_row[f'Part{p+1}'] for p in range(2)],
                        origins=[mate_row[f'Origin{p+1}'] for p in range(2)],
                        rots=[mate_row[f'Axes{p+1}'] for p in range(2)],
                        name=mate_row['Name'],
                        mateType=mate_row['Type'])
                mates.append(mate)
            
            LOG(f'creating batch data...')
            batch = assembly_info.create_batches(mates, max_z_groups=max_groups, max_mc_pairs=max_mc_pairs)
            LOG(str(batch))
            fname = f'{ind}.dat'
            torch_datapath = os.path.join(outdatapath, fname)
            torch.save(batch, torch_datapath)
            del batch

        if save_stats:
            stats = assembly_info.stats
            mate_stats = assembly_info.mate_stats

            stats['num_mates'] = mate_subset.shape[0]
            stats['num_mcs'] = assembly_info.num_mate_connectors()
            stats['num_invalid_frames'] = sum([mate_stat['num_matches'] == 0 for mate_stat in mate_stats])
            stats['num_mates_not_detected'] = sum([not mate_stat['found_by_heuristic'] for mate_stat in mate_stats])
            #stats['too_big'] = tooBig
            stats['skipped'] = skipped
            
            all_stats.append(stats)
            processed_indices.append(ind)

            if not skipped:
                all_mate_stats += mate_stats
                mate_indices += list(mate_subset['MateIndex'])

            if (num_processed+start_index+1) % stride == 0:
                stat_df_mini = ps.DataFrame(all_stats[last_ckpt:], index=processed_indices[last_ckpt:])
                stat_df_mini.fillna(value=replace_keys, inplace=True)
                stat_df_mini.to_parquet(os.path.join(statspath, f'stats_{num_processed + start_index}.parquet'))
                mate_stat_df_mini = ps.DataFrame(all_mate_stats[last_mate_ckpt:], index=mate_indices[last_mate_ckpt:])
                mate_stat_df_mini.to_parquet(os.path.join(statspath, f'mate_stats_{ind}.parquet'))
                last_mate_ckpt = len(mate_indices)
                last_ckpt = len(processed_indices)
                record_val(num_processed + start_index + 1)

        del assembly_info

    stats_df = ps.DataFrame(all_stats, index=processed_indices)
    stats_df.fillna(value=replace_keys, inplace=True)
    print(all_stats)
    print(stats_df)
    stats_df.to_parquet(os.path.join(statspath, 'stats_all.parquet'))
    mate_stats_df = ps.DataFrame(all_mate_stats, index=mate_indices)
    mate_stats_df.to_parquet(os.path.join(statspath, 'mate_stats_all.parquet'))

if __name__ == '__main__':
    main()