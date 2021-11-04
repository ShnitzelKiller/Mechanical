from argparse import ArgumentParser
import pandas as ps
from mechanical.data import AssemblyInfo, mate_types
import os
import time
from onshape.brepio import Mate
import torch

import json
import numpy as np
import random

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

class Logger:
    def __init__(self, filepath):
        self.filepath = filepath
    
    def __call__(self, msg):
        with open(self.filepath,'a') as logf:
            logf.write(msg + '\n')
            logf.flush()
    
    def clear(self):
        if os.path.isfile(self.filepath):
            os.remove(self.filepath)


def check_datalists(out_path, name_lst, boundary_check_value, uv_only, filter_list, log):
    name = name_lst[0]
    datapath = os.path.join(out_path, 'data')
    if filter_list is not None:
        with open(filter_list,'r') as f:
            exclude = set(l.split(' ')[0].rstrip() for l in f.readlines())
    if len(name_lst) > 1:
        train_file = os.path.join(out_path, name_lst[1])
        validation_file = os.path.join(out_path, name_lst[2])
    else:
        train_file = os.path.join(out_path, name + '_train.flist')
        validation_file = os.path.join(out_path, name + '_validation.flist')
    print('filtering',train_file,validation_file)
    allfiles = [train_file, validation_file]
    for k,fname in enumerate(allfiles):
        newdatalist = []
        with open(fname,'r') as f:
            datalist = [os.path.join(datapath, l.rstrip()) for l in f.readlines()]
        for i,d in enumerate(datalist):
            if i % 1000 == 0:
               print(f'processed {i}/{len(datalist)} {["train", "val"][k]} batches')
            batch = torch.load(d)
            invalid = True
            if filter_list is not None:
                invalid = os.path.splitext(os.path.split(d)[1])[0] in exclude
            else:
                tensors = []
                if not uv_only:
                    tensors.append(batch.x)
                else:
                    tensors.append(batch.crv_feat)
                    tensors.append(batch.srf_feat)
                
                dataname = f'{["train", "val"][k]} {os.path.split(d)[1]}'
                if any([t.flatten().min() < -boundary_check_value or t.flatten().max() > boundary_check_value for t in tensors if t.shape[0] > 0]):
                    msg = f'OOB {dataname} with value {[(t.flatten().min().item(), t.flatten().max().item()) for t in tensors]}'
                    print(msg)
                    log(msg)
                elif batch.mc_pair_labels.sum().item() < 1:
                    msg = f'EMPTY {dataname}'
                    print(msg)
                    log(msg)
                else:
                    invalid = False
            if not invalid:
                newdatalist.append(d)
        if len(name_lst) > 3:
            newfile = os.path.join(out_path, name_lst[3] + ['_train','_validation'][k] + '.flist')
            print(f'saving filtered datalist as {newfile}')
            #newfile = os.path.splitext(fname)[0] + '_fixed.flist'
            with open(newfile,'w') as f:
                f.writelines([os.path.split(d)[1]+'\n' for d in newdatalist])

            

def generate_datalists(out_path, validation_split, name):
    datapath = os.path.join(out_path, 'data')
    datalist = [entry.name for entry in os.scandir(datapath) if entry.name.endswith('.dat')]
    random.shuffle(datalist)
    split_ind = int(len(datalist) * validation_split)
    validation_datalist = datalist[:split_ind]
    train_datalist = datalist[split_ind:]
    with open(os.path.join(out_path, name + '_validation.flist'),'w') as f:
        f.writelines([data + '\n' for data in validation_datalist])
    with open(os.path.join(out_path, name + '_train.flist'),'w') as f:
        f.writelines([data + '\n' for data in train_datalist])


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
    parser.add_argument('--generate_datalists', type=str, default=None)
    parser.add_argument('--check_datalists', nargs='+', type=str, default=None)
    parser.add_argument('--boundary_check_value', type=float, default=100)
    parser.add_argument('--filter_list', type=str, default=None)
    parser.add_argument('--uv_features_only', type=bool, default=True)
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--check_batches', action='store_true')
    parser.add_argument('--add_mate_labels', action='store_true')
    
    #parser.add_argument('--prob_threshold', type=float, default=0.7)

    args = parser.parse_args()
    #PROB_THRESHOLD = args.prob_threshold

    if args.generate_datalists is not None:
        generate_datalists(args.out_path, args.validation_split, args.generate_datalists)
        exit(0)

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

    logfile = os.path.join(statspath, 'log.txt')
    resultsfile = os.path.join(statspath, 'results.txt')
    resumefile = os.path.join(statspath, 'recovery.txt')

    LOG = Logger(logfile)
    LOG('=========RESTARTING=========')
    LOG(str(args))

    if args.check_batches or args.add_mate_labels:
        LOG_results = Logger(resultsfile)
        LOG_results.clear()

    if args.check_datalists is not None:
        check_datalists(args.out_path, args.check_datalists, args.boundary_check_value, args.uv_features_only, args.filter_list, LOG)
        exit(0)
    
    # if clear_previous_run and os.path.isdir(statspath):
    #     print('clearing')
    #     shutil.rmtree(statspath)
    #     shutil.rmtree(outdatapath)

    if not os.path.isdir(args.out_path):
        os.mkdir(args.out_path)
    if not os.path.isdir(statspath):
        os.mkdir(statspath)
        os.mkdir(outdatapath)
    
    
    def record_val(val):
        with open(resumefile, 'w') as f:
            f.write(str(val) + '\n')
            f.flush()
    
    def read_val():
        with open(resumefile,'r') as f:
            return int(f.read())
    
    if args.last_checkpoint:
        start_index = read_val()

    replace_keys={'truncated_mc_pairs': False, 'invalid_bbox': False}

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

    run_start_time = time.time()
    if stop_index < 0:
        stop_index = len(assembly_indices)
    for num_processed,ind in enumerate(assembly_indices[start_index:stop_index]):
        print(f'num_processed: {num_processed+start_index}/{len(assembly_indices)}')
        LOG(f'{num_processed+start_index}/{len(assembly_indices)}: processing {ind} ({assembly_df.loc[ind,"AssemblyPath"]}) at {time.time()-run_start_time}')

        part_subset = part_df.loc[ind]
        mate_subset = mate_df.loc[ind]
        if mate_subset.ndim == 1:
            mate_subset = ps.DataFrame([mate_subset], columns=mate_subset.keys())
        
        part_paths = []
        transforms = []
        occ_ids = list(part_subset['PartOccurrenceID'])

        #TEMPORARY: Load correct transforms, and also save them separately
        if not args.add_mate_labels:
            with open(os.path.join(datapath, 'data/flattened_assemblies', assembly_df.loc[ind, "AssemblyPath"] + '.json')) as f:
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
                path = os.path.join(datapath, 'data/models', *[part_subset.iloc[j][k] for k in ['did','mv','eid','config']], f'{part_subset.iloc[j]["PartId"]}.xt')
                assert(os.path.isfile(path))
                part_paths.append(path)
                occ_id = part_subset.iloc[j]['PartOccurrenceID']
                transforms.append(tf_dict[occ_id])
        
        fname = f'{ind}.dat'
        torch_datapath = os.path.join(outdatapath, fname)

        if args.add_mate_labels or args.check_batches:
            if os.path.isfile(torch_datapath):
                batch = torch.load(torch_datapath)

                num_parts = torch.unique(batch.graph_idx.flatten()).size(0)
                if num_parts != part_subset.shape[0]:
                    LOG_results(f'{ind} contains {num_parts} graph ids but assembly has {part_subset.shape[0]} parts')
                    print('skipping',ind)
                    continue

                if args.add_mate_labels:
                    occ_to_index = dict()
                    for i,occ in enumerate(occ_ids):
                        occ_to_index[occ] = i
                    
                    mates = df_to_mates(mate_subset)
                    pair_to_type = dict()

                    for mate in mates:
                        part_indices = [occ_to_index[me[0]] for me in mate.matedEntities]
                        pair_to_type[tuple(sorted(part_indices))] = mate_types.index(mate.type)
                    
                    pair_types = torch.full((batch.part_edges.shape[1],), -1, dtype=torch.int64)
                    for k in range(batch.part_edges.shape[1]):
                        key = tuple(t.item() for t in batch.part_edges[:,k])
                        assert(key[0] < key[1])
                        if key in pair_to_type:
                            pair_types[k] = pair_to_type[key]
                    
                    left_partids = batch.graph_idx.flatten()[batch.mc_pairs[0]]
                    right_partids = batch.graph_idx.flatten()[batch.mc_pairs[3]]
                    for lid, rid, typ, lbl in zip(left_partids, right_partids, batch.mc_pair_type, batch.mc_pair_labels):
                        if lbl == 1:
                            key = (lid.item(), rid.item())
                            assert(key[0] < key[1])
                            assert(pair_to_type[key] == typ.item())
                    
                    batch.pair_types = pair_types
                    torch.save(batch, torch_datapath)
            

                    

        else:

            assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, LOG, epsilon_rel=epsilon_rel, use_uvnet_features=use_uvnet_features, max_topologies=max_topologies)
            num_topologies = assembly_info.num_topologies()
            LOG(f'initialized AssemblyInfo with {num_topologies} topologies')
            # LOG(f'topologies per part: {[part.num_topologies for part in assembly_info.parts]}')
            for j,part in enumerate(assembly_info.parts):
                if part.num_topologies != len(part.node_types):
                    LOG(f'part {j} has valid bit {part.valid}, and length of node type vector is {len(part.node_types)}')
                #assert(part.num_topologies == len(part.node_types))
            
            skipped = True
            has_mc_pairs = False
            if assembly_info.stats['initialized']:
                #LOG(f'normalization_matrices:{assembly_info.norm_matrices}')
                num_invalid_parts = assembly_info.num_invalid_parts()
                assert(num_invalid_parts == 0)
                if len(assembly_info.parts) > 1:
                    skipped = False
                    mates = df_to_mates(mate_subset)
                    
                    LOG(f'creating batch data...')
                    batch = assembly_info.create_batches(mates, max_z_groups=max_groups, max_mc_pairs=max_mc_pairs)
                    LOG(str(batch))
                    if batch.mc_pair_labels.size(0) > 0:
                        has_mc_pairs = True
                        assert(batch.part_edges.size(-1) > 0)
                        torch.save(batch, torch_datapath)
                    else:
                        LOG('skipping due to no mc_pairs')
                    del batch

            if save_stats:
                stats = assembly_info.stats
                mate_stats = assembly_info.mate_stats

                stats['num_mcs'] = assembly_info.num_mate_connectors()
                stats['skipped'] = skipped
                stats['has_mc_pairs'] = has_mc_pairs
                
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
    
    if not args.add_mate_labels and not args.check_batches:
        stats_df = ps.DataFrame(all_stats, index=processed_indices)
        stats_df.fillna(value=replace_keys, inplace=True)
        stats_df.to_parquet(os.path.join(statspath, 'stats_all.parquet'))
        mate_stats_df = ps.DataFrame(all_mate_stats, index=mate_indices)
        mate_stats_df.to_parquet(os.path.join(statspath, 'mate_stats_all.parquet'))

if __name__ == '__main__':
    main()