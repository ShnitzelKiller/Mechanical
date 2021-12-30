from argparse import ArgumentParser, Action
import pandas as ps
from mechanical.data import AssemblyInfo, assembly_data, mate_types, cs_from_origin_frame, cs_to_origin_frame
import os
import time
from onshape.brepio import Mate
import torch
import json
import numpy as np
import random
from enum import Enum, auto
from pspart import Part
import h5py
from scipy.spatial.transform import Rotation as R
from utils import MateTypes, homogenize_frame, homogenize_sign, project_to_plane

class Mode(Enum):
    CHECK_BATCHES = "CHECK_BATCHES"
    CHECK_BOUNDS = "CHECK_BOUNDS"
    ADD_MATE_LABELS = "ADD_MATE_LABELS"
    ADD_RIGID_LABELS = "ADD_RIGID_LABELS"
    GENERATE = "GENERATE"
    DISTANCE = "DISTANCE"
    AUGMENT = "AUGMENT" #try to fill in the missing mates between nearby parts if possible (only accounting for simple mate types)
    ADD_OCC_DATA = "ADD_OCC_DATA"
    ADD_MESH_DATA = "ADD_MESH_DATA"
    ADD_MC_DATA = "ADD_MC_DATA"
    ADD_MC_DEFINITIONS = "ADD_MC_DEFINITIONS"
    CHECK_MATES = "CHECK_MATES" #check the assemblies with only simple mate types for whether those mates adhere to shared axes inferred from MCs

def load_axes(path):
    with h5py.File(path, 'r') as f:
        pair_to_dirs = {}
        pair_to_axes = {}
        pair_data = f['pair_data']
        for key in pair_data.keys():
            pair = tuple(int(k) for k in key.split(','))
            dirs = np.array(pair_data[key]['dirs']['values'])
            dir_index_to_value = {dir_ind : dir for dir_ind, dir in zip(pair_data[key]['dirs']['indices'], dirs)}
            pair_to_dirs[pair] = dirs
            pair_to_axes[pair] = []
            for dir_index in pair_data[key]['axes'].keys():
                dir = dir_index_to_value[int(dir_index)]
                for ax_origin in pair_data[key]['axes'][dir_index]['values']:
                    pair_to_axes[pair].append((dir, ax_origin))
    return pair_to_dirs, pair_to_axes #pair -> dir, and pair -> (dir, origin)



class EnumAction(Action):
    """
    Argparse action for handling Enums
    """
    def __init__(self, **kwargs):
        # Pop off the type value
        enum_type = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum_type is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum_type, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum_type))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum_type

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        value = self._enum(values)
        setattr(namespace, self.dest, value)


def check_mates(batch, mates, pair_to_index):
    left_partids = batch.graph_idx.flatten()[batch.mc_pairs[0]]
    right_partids = batch.graph_idx.flatten()[batch.mc_pairs[3]]
    for lid, rid, typ, lbl in zip(left_partids, right_partids, batch.mc_pair_type, batch.mc_pair_labels):
        if lbl == 1:
            key = (lid.item(), rid.item())
            if not (key[0] < key[1]) or mate_types.index(mates[pair_to_index[key]].type) != typ.item():
                return False

    return True

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

def mate_self_consistent(mate, tol):
    if mate.type == MateTypes.FASTENED:
        return True
    elif mate.type == MateTypes.SLIDER:
        dirs = mate.get_axes()
        return np.allclose(dirs[0], dirs[1], rtol=0, atol=tol)
    else:
        dirs = mate.get_axes()
        if np.allclose(dirs[0], dirs[1], rtol=0, atol=tol):
            projpoints = mate.get_projected_origins(dirs[0])
            return np.allclose(projpoints[0], projpoints[1], rtol=0, atol=tol)
        else:
            return False

def mates_equivalent(mate1, mate2, tol):
    if mate1.type != mate2.type:
        return False
    else:
        if mate1.type == MateTypes.FASTENED:
            return True
        elif mate1.type == MateTypes.SLIDER:
            return np.allclose(mate1.get_axes()[0], mate2.get_axes()[0], rtol=0, atol=tol)
        else:
            axis1 = mate1.get_axes()[0]
            axis2 = mate2.get_axes()[0]
            if np.allclose(axis1, axis2, rtol=0, atol=tol):
                projpoint1 = mate1.get_projected_origins(axis1)[0]
                projpoint2 = mate2.get_projected_origins(axis2)[0]
                return np.allclose(projpoint1, projpoint2, rtol=0, atol=tol)
            else:
                return False

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
    parser.add_argument('--index_file', default='/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/fully_connected_moving_no_multimates.txt')
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--stop_index', type=int, default=-1)
    parser.add_argument('--epsilon_rel', type=float, default=0.001)
    parser.add_argument('--max_groups', type=int, default=10)
    parser.add_argument('--max_topologies', type=int, default=5000)
    parser.add_argument('--max_mc_pairs', type=int, default=100000)
    parser.add_argument('--stride', type=int, default=500)
    parser.add_argument('--no_stats', dest='save_stats', action='store_false')
    parser.add_argument('--save_stats', dest='save_stats', action='store_true')
    parser.set_defaults(save_stats=True)
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--no_uvnet_features', dest='use_uvnet_features', action='store_false')
    parser.add_argument('--use_uvnet_features', dest='use_uvnet_features', action='store_true')
    parser.set_defaults(use_uvnet_features=True)
    parser.add_argument('--last_checkpoint', action='store_true')
    parser.add_argument('--generate_datalists', type=str, default=None)
    parser.add_argument('--check_datalists', nargs='+', type=str, default=None)
    parser.add_argument('--boundary_check_value', type=float, default=100)
    parser.add_argument('--filter_list', type=str, default=None)
    parser.add_argument('--uv_features_only', type=bool, default=True) #this doesn't work
    parser.add_argument('--validation_split', type=float, default=0.2)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--mode', type=Mode, action=EnumAction, required=True)
    parser.add_argument('--name', default='stats')
    #distance mode options
    parser.add_argument('--add_distances_to_batches', action='store_true')
    
    #augmentation mode options
    parser.add_argument('--matched_axes_only', action='store_true')
    parser.add_argument('--require_axis', action='store_true')
    #distance & augmentation mode options
    parser.add_argument('--distance_threshold', type=float, default=0.01)
    #mate check options
    parser.add_argument('--check_alternate_paths', action='store_true')
    parser.add_argument('--validate_feasibility', action='store_true')
    #parser.add_argument('--prob_threshold', type=float, default=0.7)

    #adding mate labels options
    parser.add_argument('--pair_data', action='store_true')
    parser.add_argument('--augmented_labels', action='store_true')

    args = parser.parse_args()
    #PROB_THRESHOLD = args.prob_threshold

    if args.generate_datalists is not None:
        generate_datalists(args.out_path, args.validation_split, args.generate_datalists)
        exit(0)

    DRY_RUN = args.dry_run
    start_index = args.start_index
    stop_index = args.stop_index
    epsilon_rel = args.epsilon_rel
    distance_threshold = args.distance_threshold
    max_groups = args.max_groups
    max_topologies = args.max_topologies
    max_mc_pairs = args.max_mc_pairs
    stride = args.stride
    #debug_run=False
    save_stats=args.save_stats
    use_uvnet_features = args.use_uvnet_features
    add_distances_to_batches = args.add_distances_to_batches
    PAIR_DATA = args.pair_data
    AUGMENTED_LABELS = args.augmented_labels
    print('using uvnet features:',use_uvnet_features)
    print('saving stats:',save_stats)
    clear_previous_run = args.clear

    
    # if args.mode == Mode.DISTANCE:
    #     statsname = 'stats_distance'
    # elif args.mode == Mode.AUGMENT:
    #     if args.matched_axes_only:
    #         statsname = 'stats_augment_matchedonly'
    #     else:
    #         statsname = 'stats_augment'
    # elif args.mode == Mode.ADD_MC_DATA:
    #     statsname = 'stats_mc'
    # elif args.mode == Mode.CHECK_MATES:
    #     statsname = 'stats_check_mates'
    # else:
    #     statsname = 'stats'
    statsname = args.name
    statspath = os.path.join(args.out_path, statsname)
    if not os.path.isdir(statspath):
        os.mkdir(statspath)

    outdatapath = os.path.join(args.out_path, 'data')

    if args.mode == Mode.ADD_MESH_DATA:
        meshpath = os.path.join(args.out_path, 'mesh')
        if not os.path.isdir(meshpath):
            os.mkdir(meshpath)
    
    mcpath = os.path.join(args.out_path, 'mc_data')
    if args.mode == Mode.ADD_MC_DATA:
        if not os.path.isdir(mcpath):
            os.mkdir(mcpath) 
    
    if args.mode == Mode.GENERATE:
        if not os.path.isdir(args.out_path):
            os.mkdir(args.out_path)
            os.mkdir(outdatapath)
    
    if args.mode == Mode.DISTANCE or args.mode == Mode.AUGMENT or args.mode == Mode.ADD_MC_DATA:
        if not os.path.isdir(statspath):
            os.mkdir(statspath)

    logfile = os.path.join(statspath, 'log.txt')
    resultsfile = os.path.join(statspath, 'results.txt')
    resumefile = os.path.join(statspath, 'recovery.txt')

    LOG = Logger(logfile)
    LOG('=========RESTARTING=========')
    LOG(str(args))

    if args.mode == Mode.CHECK_BATCHES or args.mode == Mode.ADD_MATE_LABELS or args.mode == Mode.CHECK_BOUNDS or args.mode == Mode.ADD_RIGID_LABELS or args.mode == Mode.ADD_OCC_DATA or args.mode == Mode.ADD_MESH_DATA:
        LOG_results = Logger(resultsfile)
        LOG_results.clear()

    if args.mode == Mode.CHECK_BOUNDS:
        if args.check_datalists is not None:
            check_datalists(args.out_path, args.check_datalists, args.boundary_check_value, args.uv_features_only, args.filter_list, LOG_results)
        else:
            print('need to specify datalists to check, and name for filtered list')
            exit(1)
        exit(0)
    
    
    # if clear_previous_run and os.path.isdir(statspath):
    #     print('clearing')
    #     shutil.rmtree(statspath)
    #     shutil.rmtree(outdatapath)

    LOG('loading dataframes...')
    datapath = '/projects/grail/benjones/cadlab'
    df_name = '/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5'
    updated_df_name = '/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5_segmentation.h5'
    assembly_df = ps.read_hdf(df_name,'assembly')
    mate_df = ps.read_hdf(df_name,'mate')
    part_df = ps.read_hdf(updated_df_name,'part')
    mate_df['MateIndex'] = mate_df.index
    part_df['PartIndex'] = part_df.index
    mate_df.set_index('Assembly', inplace=True)
    part_df.set_index('Assembly', inplace=True)
    
    if args.mode == Mode.ADD_MATE_LABELS:
        #mate_check_df = ps.read_parquet('/fast/jamesn8/assembly_data/assembly_torch2_fixsize/stats_check_mates/mate_stats_all.parquet')
        mate_check_df = ps.read_hdf('/fast/jamesn8/assembly_data/assembly_torch2_fixsize/stats_check_mates_all_proposals_validate_rigidcomps_more_axis_stats/mate_stats_all.h5','mates')
        newmate_stats_df = ps.read_hdf('/fast/jamesn8/assembly_data/assembly_torch2_fixsize/stats_augment_matchedonly/newmate_stats_all.h5','newmates')
        newmate_stats_df.set_index('Assembly', inplace=True)
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


    with open(args.index_file,'r') as f:
        assembly_indices = [int(l.rstrip()) for l in f.readlines()]

    last_ckpt = 0
    last_mate_ckpt = 0
    all_stats = []
    all_mate_stats = []
    processed_indices = []
    mate_indices = []
    all_newmate_stats = []

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
        

        occ_ids = list(part_subset['PartOccurrenceID'])
        part_paths = []
        rel_part_paths = []
        transforms = []

        #TEMPORARY: Load correct transforms, and also save them separately
        #if args.mode == 'generate':
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
            rel_path = os.path.join(*[part_subset.iloc[j][k] for k in ['did','mv','eid','config']], f'{part_subset.iloc[j]["PartId"]}.xt')
            path = os.path.join(datapath, 'data/models', rel_path)
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
        
        fname = f'{ind}.dat'
        torch_datapath = os.path.join(outdatapath, fname)

        if args.mode == Mode.ADD_MATE_LABELS or args.mode == Mode.CHECK_BATCHES or args.mode == Mode.ADD_RIGID_LABELS or args.mode == Mode.ADD_OCC_DATA or args.mode == Mode.ADD_MESH_DATA:
            if os.path.isfile(torch_datapath):
                
                batch = torch.load(torch_datapath)
                num_parts = torch.unique(batch.graph_idx.flatten()).size(0)

                if num_parts != part_subset.shape[0]:
                    LOG_results(f'{ind} contains {num_parts} graph ids but assembly has {part_subset.shape[0]} parts')
                    print('skipping',ind)
                    continue

                if args.mode == Mode.ADD_MATE_LABELS:
                    if PAIR_DATA:
                        h5fname = os.path.join(args.out_path, 'mc_data', f'{ind}.hdf5')
                        if os.path.isfile(h5fname):
                            rigid_comps = list(part_subset['RigidComponentID'])
                            if AUGMENTED_LABELS:
                                augmented_pairs_to_index = dict()
                                newmate_subset = newmate_stats_df.loc[ind]
                                if ind in newmate_subset.index:
                                    if newmate_subset.ndim == 1:
                                        newmate_subset = ps.DataFrame([newmate_subset], columns=newmate_subset.keys())
                                    for m in range(newmate_subset.shape[0]):
                                        part_indices = tuple(sorted([occ_to_index[newmate_subset.iloc[m][k]] for k in ['part1','part2']]))
                                        augmented_pairs_to_index[part_indices] = m
                        
                            with h5py.File(h5fname,'r+') as f:
                                pair_data = f['pair_data']
                                for key in pair_data.keys():
                                    pair = tuple(sorted([int(k) for k in key.split(',')]))
                                    mateType = -1
                                    augmentedType = -1
                                    mateDirInd = -1
                                    mateAxisInd = -1
                                    augmentedDirInd = -1
                                    augmentedAxisInd = -1
                                    if pair in pair_to_index:
                                        mate_index = pair_to_index[pair]
                                        mateType = mate_types.index(mates[mate_index].type)
                                        mate_check_row = mate_check_df.loc[mate_subset.iloc[mate_index]['MateIndex']]
                                        mateDirInd = mate_check_row['dir_index']
                                        mateAxisInd = mate_check_row['axis_index']

                                    if AUGMENTED_LABELS:
                                        if pair in augmented_pairs_to_index:
                                            augmented_index = augmented_pairs_to_index[pair]
                                            if newmate_subset.iloc[augmented_index]['added_mate']:
                                                augmentedType = mate_types.index(newmate_subset.iloc[augmented_index]['type'])
                                                augmentedDirInd = newmate_subset.iloc[augmented_index]['dir_index']
                                                augmentedAxisInd = newmate_subset.iloc[augmented_index]['axis_index']
                                    
                                    #densify fasten mates
                                    if rigid_comps[pair[0]] == rigid_comps[pair[1]]:
                                        augmentedType = mate_types.index('FASTENED')
                                
                                    pair_data[key].attrs['type'] = mateType
                                    pair_data[key].attrs['dirIndex'] = mateDirInd
                                    pair_data[key].attrs['axisIndex'] = mateAxisInd
                                    pair_data[key].attrs['augmented_type'] = augmentedType
                                    pair_data[key].attrs['augmented_dirIndex'] = augmentedDirInd
                                    pair_data[key].attrs['augmented_axisIndex'] = augmentedAxisInd
                        else:
                            LOG(f'missing {h5fname}')
                    else:
                        pair_types = torch.full((batch.part_edges.shape[1],), -1, dtype=torch.int64)
                        for k in range(batch.part_edges.shape[1]):
                            key = tuple(t.item() for t in batch.part_edges[:,k])
                            assert(key[0] < key[1])
                            if key in pair_to_index:
                                pair_types[k] = mate_types.index(mates[pair_to_index[key]].type)
                        
                        assert(check_mates(batch, mates, pair_to_index))
                        if not DRY_RUN:
                            batch.pair_types = pair_types
                            torch.save(batch, torch_datapath)

                elif args.mode == Mode.ADD_RIGID_LABELS:
                    if not hasattr(batch, 'rigid_labels') and not DRY_RUN:
                        batch.rigid_labels = torch.tensor(np.array(part_subset['RigidComponentID']), dtype=torch.int64)
                        torch.save(batch, torch_datapath)
                elif args.mode == Mode.ADD_OCC_DATA:
                    batch.tfs = torch.stack([torch.tensor(tf, dtype=torch.float) for tf in transforms])
                    batch.paths = rel_part_paths
                    batch.assembly = assembly_df.loc[ind, 'AssemblyPath']

                    #populate mate frames
                    mate_frames = torch.empty((len(mates), 2, 4, 4), dtype=torch.float) #mate frame tensor
                    pair_indices = torch.full((batch.part_edges.shape[1],), -1, dtype=torch.int64) #part pair to mate index tensor
                    for k in range(batch.part_edges.shape[1]):
                        key = tuple(t.item() for t in batch.part_edges[:,k])
                        assert(key[0] < key[1])
                        if key in pair_to_index:
                            pair_indices[k] = pair_to_index[key]
                    #assert(check_mates(batch, mates, pair_to_index))
                    for m, mate in enumerate(mates):
                        mate_frame_np = np.stack([cs_from_origin_frame(me[1][0], me[1][1]) for me in mate.matedEntities])
                        mate_frames[m] = torch.from_numpy(mate_frame_np).float()
                    if not DRY_RUN:
                        batch.mate_frames = mate_frames
                        batch.pair_indices = pair_indices
                        torch.save(batch, torch_datapath)
                elif args.mode == Mode.ADD_MESH_DATA:
                    mesh_datapath = os.path.join(meshpath, fname)
                    parts = [Part(os.path.join(datapath, 'data/models', pth)) for pth in batch.paths]
                    pairs = []
                    for part, tf in zip(parts, batch.tfs):
                        V = torch.from_numpy(part.V).float()
                        F = torch.from_numpy(part.F)
                        V_tf = (tf[:3,:3] @ V.T).T + tf[:3,3]
                        pairs.append((V_tf, F))
                    torch.save(pairs, mesh_datapath)


                #TODO: add mode check_batches with additional sanity checks
            else:
                pass
                #LOG_results(f'{torch_datapath} missing from directory')

        elif args.mode == Mode.GENERATE or args.mode == Mode.DISTANCE or args.mode == Mode.AUGMENT or args.mode == Mode.ADD_MC_DATA or args.mode == Mode.CHECK_MATES or args.mode == Mode.ADD_MC_DEFINITIONS:
            if args.mode == Mode.AUGMENT:
                if os.path.isfile(torch_datapath):
                    assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, LOG, epsilon_rel=epsilon_rel, use_uvnet_features=use_uvnet_features, max_topologies=max_topologies)
                    if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:
                        mates = df_to_mates(mate_subset)

                        if args.matched_axes_only:
                            pair_to_dirs, pair_to_axes = load_axes(os.path.join(args.out_path, 'mc_data', f'{ind}.hdf5'))
                            stat = assembly_info.fill_missing_mates(mates, list(part_subset['RigidComponentID']), distance_threshold, pair_to_dirs=pair_to_dirs, pair_to_axes=pair_to_axes, require_axis=args.require_axis)
                        else: 
                            stat = assembly_info.fill_missing_mates(mates, list(part_subset['RigidComponentID']), distance_threshold)
                        for st in stat:
                            st['Assembly'] = ind
                        all_newmate_stats += stat

                        if save_stats:

                            if (num_processed+start_index+1) % stride == 0:

                                newmate_stats_df_mini = ps.DataFrame(all_newmate_stats[last_ckpt:])
                                newmate_stats_df_mini.to_hdf(os.path.join(statspath, f'newmate_stats_{num_processed + start_index}.h5'), 'newmates')
                                last_ckpt = len(all_newmate_stats)
                                record_val(num_processed + start_index + 1)

            elif args.mode == Mode.DISTANCE:
                if os.path.isfile(torch_datapath):
                    assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, LOG, epsilon_rel=epsilon_rel, use_uvnet_features=use_uvnet_features, max_topologies=max_topologies)
                    if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:

                        # mates_to_match = df_to_mates(mate_subset)
                        # connections = dict()
                        # for mate in mates_to_match:
                        #     mate.type = 'FASTENED' #pretend they're fastened so that the mated pair search is limited to coincident connectors
                        #     mc_pairs = assembly_info.find_mated_pairs(mate)

                        #     pair = tuple(sorted((assembly_info.occ_to_index[mate.matedEntities[0][0]], assembly_info.occ_to_index[mate.matedEntities[1][0]])))
                        #     connections[pair] = len(mc_pairs)

                        connections = {tuple(sorted((assembly_info.occ_to_index[mate_subset.iloc[l]['Part1']], assembly_info.occ_to_index[mate_subset.iloc[l]['Part2']]))): l for l in range(mate_subset.shape[0])}
                        proposals = assembly_info.mate_proposals(coincident_only=True)


                        distances = assembly_info.part_distances(distance_threshold)

                        if add_distances_to_batches:
                            batch = torch.load(torch_datapath)
                            if batch.part_pair_feats.shape[0] < 3:
                                dists_torch = torch.zeros(batch.part_edges.shape[1], dtype=torch.float32)
                                for r in range(batch.part_edges.shape[1]):
                                    pair = tuple(t.item() for t in batch.part_edges[:,r])
                                    if pair in distances:
                                        dists_torch[r] = distances[pair]
                                    else:
                                        dists_torch[r] = float("Inf")
                                
                                batch.part_pair_feats = torch.vstack([batch.part_pair_feats, dists_torch])
                            #torch.save(batch, torch_datapath)
                        
                        if PAIR_DATA:
                            with h5py.File(os.path.join(args.out_path, 'mc_data', f'{ind}.hdf5'),'r+') as f:
                                pair_data = f['pair_data']
                                for key in pair_data.keys():
                                    pair = tuple(sorted([int(k) for k in key.split(',')]))
                                    if pair in distances:
                                        pair_data[key].attrs['distance'] = distances[pair]
                                    else:
                                        pair_data[key].attrs['distance'] = np.inf

                        stat = assembly_info.stats

                        if assembly_info.stats['num_degenerate_bboxes'] == 0:
                            stat['num_unconnected_close'] = 0
                            stat['num_unconnected_coincident'] = 0
                            stat['num_unconnected_close_and_coincident'] = 0
                            stat['num_connected_far'] = 0
                            stat['num_connected_not_coincident'] = 0
                            stat['num_connected_far_or_not_coincident'] = 0
                            allpairs = {pair for pair in distances}.union({pair for pair in proposals})

                            for pair in allpairs:
                                comp1 = part_subset.iloc[pair[0]]['RigidComponentID']
                                comp2 = part_subset.iloc[pair[1]]['RigidComponentID']
                                if comp1 != comp2 and pair not in connections:
                                    if pair in distances and distances[pair] < distance_threshold:
                                        stat['num_unconnected_close'] += 1
                                    if pair in proposals:
                                        stat['num_unconnected_coincident'] += 1
                                    if pair in distances and distances[pair] < distance_threshold and pair in proposals:
                                        stat['num_unconnected_close_and_coincident'] += 1

                            
                            for pair in connections:
                                mate_stat = {'connected_far': False, 'connected_not_coincident': False}
                                if pair not in proposals or pair not in distances or distances[pair] >= distance_threshold:
                                    stat['num_connected_far_or_not_coincident'] += 1
                                if pair not in distances or distances[pair] >= distance_threshold:
                                    stat['num_connected_far'] += 1
                                    mate_stat['connected_far'] = True
                                if pair not in proposals:
                                    stat['num_connected_not_coincident'] += 1
                                    mate_stat['connected_not_coincident'] = True
                                mate_stat['type'] = mate_subset.iloc[connections[pair]]['Type']
                                mate_stat['assembly'] = ind
                                mate_indices.append(mate_subset.iloc[connections[pair]]['MateIndex'])
                                all_mate_stats.append(mate_stat)
                        else:
                            stat['num_unconnected_close'] = -1
                            stat['num_unconnected_coincident'] = -1
                            stat['num_unconnected_close_and_coincident'] = -1
                            stat['num_connected_far'] = -1
                            stat['num_connected_not_coincident'] = -1
                            stat['num_connected_far_or_not_coincident'] = -1
                            
                        
                        all_stats.append(stat)
                        processed_indices.append(ind)
                        if save_stats:

                            if (num_processed+start_index+1) % stride == 0:
                                stat_df_mini = ps.DataFrame(all_stats[last_ckpt:], index=processed_indices[last_ckpt:])
                                stat_df_mini.to_parquet(os.path.join(statspath, f'stats_{num_processed + start_index}.parquet'))
                                mate_stat_df_mini = ps.DataFrame(all_mate_stats[last_mate_ckpt:], index=mate_indices[last_mate_ckpt:])
                                mate_stat_df_mini.to_parquet(os.path.join(statspath, f'mate_stats_{num_processed + start_index}.parquet'))
                                
                                last_ckpt = len(processed_indices)
                                last_mate_ckpt = len(mate_indices)
                                record_val(num_processed + start_index + 1)
            
            elif args.mode == Mode.ADD_MC_DEFINITIONS:
                assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, LOG, epsilon_rel=epsilon_rel, use_uvnet_features=use_uvnet_features, max_topologies=max_topologies)
                if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:
                    h5file = os.path.join(mcpath, f"{ind}.hdf5")
                    if os.path.isfile(h5file):
                        with h5py.File(os.path.join(mcpath, f"{ind}.hdf5"), "r+") as f:
                            mc_group = f.create_group('mc_definitions')
                            for p,part in enumerate(assembly_info.parts):
                                mcs = part.all_mate_connectors
                                assert(f['mc_frames'][str(p)]['origins'].shape[0] == len(mcs))
                                assert(f['mc_frames'][str(p)]['rots'].shape[0] == len(mcs))
                                mc_dataset = np.empty((len(mcs), 3), dtype=np.int32)
                                for r in range(len(mcs)):
                                    mc_dataset[r,0] = mcs[r].orientation_inference.topology_ref
                                    mc_dataset[r,1] = mcs[r].location_inference.topology_ref
                                    mc_dataset[r,2] = mcs[r].location_inference.inference_type.value
                                mc_group.create_dataset(str(p), data=mc_dataset)

                            
                                

            elif args.mode == Mode.ADD_MC_DATA:
                assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, LOG, epsilon_rel=epsilon_rel, use_uvnet_features=use_uvnet_features, max_topologies=max_topologies)
                if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:
                    axes, pairs_to_dirs, pairs_to_axes, dir_clusters, dir_to_ax_clusters = assembly_info.mate_proposals(max_z_groups=args.max_groups, axes_only=True)
                    with h5py.File(os.path.join(mcpath, f"{ind}.hdf5"), "w") as f:
                        mc_frames = f.create_group('mc_frames')
                        for k in range(len(assembly_info.mc_origins_all)):
                            partgroup = mc_frames.create_group(str(k))
                            partgroup.create_dataset('origins', data=np.array(assembly_info.mc_origins_all[k]))
                            rots_quat = np.array([R.from_matrix(rot).as_quat() for rot in assembly_info.mc_rots_all[k]])
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
            elif args.mode == Mode.CHECK_MATES:
                allowed_mates = {MateTypes.FASTENED.value, MateTypes.REVOLUTE.value, MateTypes.CYLINDRICAL.value, MateTypes.SLIDER.value}    
                if all([k in allowed_mates for k in mate_subset['Type']]):
                    mates = df_to_mates(mate_subset)
                    rigid_comps = list(part_subset['RigidComponentID'])
                    assembly_info = AssemblyInfo(part_paths, transforms, occ_ids, LOG, epsilon_rel=epsilon_rel, use_uvnet_features=use_uvnet_features, max_topologies=max_topologies)
                    if assembly_info.valid and len(assembly_info.parts) == part_subset.shape[0]:
                        curr_mate_stats = []
                        pairs_to_dirs, pairs_to_axes = load_axes(os.path.join(args.out_path, 'mc_data', f'{ind}.hdf5'))
                        for j,mate in enumerate(mates):
                            mate_stat = {'Assembly': ind, 'dir_index': -1, 'axis_index': -1, 'has_any_axis':False, 'has_same_dir_axis':False, 'type': mate.type}
                            part_indices = [assembly_info.occ_to_index[me[0]] for me in mate.matedEntities]
                            mate_origins = []
                            mate_dirs = []
                            for partIndex, me in zip(part_indices, mate.matedEntities):
                                origin_local = me[1][0]
                                frame_local = me[1][1]
                                cs = cs_from_origin_frame(origin_local, frame_local)
                                cs_transformed = assembly_info.mate_transforms[partIndex] @ cs
                                origin, rot = cs_to_origin_frame(cs_transformed)
                                rot_homogenized = homogenize_frame(rot, z_flip_only=True)
                                mate_origins.append(origin)
                                mate_dirs.append(rot_homogenized[:,2])
                            
                            found = False

                            mate_stat['dirs_agree'] = np.allclose(mate_dirs[0], mate_dirs[1], rtol=0, atol=epsilon_rel)
                            projected_mate_origins = [project_to_plane(origin, mate_dirs[0]) for origin in mate_origins]
                            mate_stat['axes_agree'] = mate_stat['dirs_agree'] and np.allclose(projected_mate_origins[0], projected_mate_origins[1], rtol=0, atol=epsilon_rel)
                            key = tuple(sorted(part_indices))

                            mate_stat['has_any_axis'] = key in pairs_to_axes

                            if mate_stat['dirs_agree']:
                                if key in pairs_to_dirs:
                                    for k,dir in enumerate(pairs_to_dirs[key]):
                                        dir_homo, _ = homogenize_sign(dir)
                                        if np.allclose(mate_dirs[0], dir_homo, rtol=0, atol=epsilon_rel):
                                            mate_stat['dir_index'] = k
                                            break
                                if key in pairs_to_axes:
                                    for k, (dir, origin) in enumerate(pairs_to_axes[key]):
                                        dir_homo, _ = homogenize_sign(dir)
                                        if np.allclose(mate_dirs[0], dir_homo, rtol=0, atol=epsilon_rel):
                                            mate_stat['has_same_dir_axis'] = True
                                            break
                            
                            if mate_stat['axes_agree']:
                                if key in pairs_to_axes:
                                    for k, (dir, origin) in enumerate(pairs_to_axes[key]):
                                        dir_homo, _ = homogenize_sign(dir)
                                        if np.allclose(mate_dirs[0], dir_homo, rtol=0, atol=epsilon_rel):
                                            projected_origin = project_to_plane(origin, mate_dirs[0])
                                            if np.allclose(projected_mate_origins[0], projected_origin, rtol=0, atol=epsilon_rel):
                                                mate_stat['axis_index'] = k
                                                break

                            if mate.type == MateTypes.FASTENED:
                                found = True
                            elif mate.type == MateTypes.SLIDER:
                                found = mate_stat['dir_index'] != -1
                                
                            elif mate.type == MateTypes.CYLINDRICAL or mate.type == MateTypes.REVOLUTE:
                                found = mate_stat['axis_index'] != -1
                            
                            if args.validate_feasibility:
                                mate_stat['rigid_comp_attempting_motion'] = rigid_comps[part_indices[0]] == rigid_comps[part_indices[1]] and mate.type != MateTypes.FASTENED
                                mate_stat['valid'] = found and not mate_stat['rigid_comp_attempting_motion']
                            else:
                                mate_stat['valid'] = found

                            curr_mate_stats.append(mate_stat)
                            mate_indices.append(mate_subset.iloc[j]['MateIndex'])
                        
                        if args.check_alternate_paths:
                            validation_stats = assembly_info.validate_mates(mates)
                            assert(len(validation_stats) == len(curr_mate_stats))
                            final_stats = [{**stat, **vstat} for stat, vstat in zip(curr_mate_stats, validation_stats)]
                            curr_mate_stats = final_stats
                        
                        if args.validate_feasibility:
                            num_incompatible_mates = 0
                            num_moving_mates_between_same_rigid = 0
                            num_multimates_between_rigid = 0
                            rigid_pairs_to_mate = dict()
                            for pair in pair_to_index:
                                rigid_pair = tuple(sorted([rigid_comps[k] for k in pair]))
                                if rigid_pair[0] != rigid_pair[1]:
                                    if rigid_pair not in rigid_pairs_to_mate:
                                        rigid_pairs_to_mate[rigid_pair] = pair_to_index[pair]
                                    else:
                                        num_multimates_between_rigid += 1
                                        prevMate = mates[rigid_pairs_to_mate[rigid_pair]]
                                        currMate = mates[pair_to_index[pair]]
                                        prevMate = assembly_info.transform_mates([prevMate])[0]
                                        currMate = assembly_info.transform_mates([currMate])[0]
                                        if not mates_equivalent(prevMate, currMate, epsilon_rel):
                                            num_incompatible_mates += 1
                                else:
                                    if mates[pair_to_index[pair]].type != MateTypes.FASTENED:
                                        num_moving_mates_between_same_rigid += 1
                            feasible = (num_moving_mates_between_same_rigid == 0) and (num_incompatible_mates == 0) and all([mate_stat['valid'] for mate_stat in curr_mate_stats])
                            stat = {'num_moving_mates_between_same_rigid':num_moving_mates_between_same_rigid,
                                    'num_incompatible_mates_between_rigid':num_incompatible_mates,
                                    'num_multimates_between_rigid':num_multimates_between_rigid,
                                    'feasible': feasible}
                            all_stats.append(stat)
                            processed_indices.append(ind)

                                
                                

                        all_mate_stats += curr_mate_stats
                if save_stats:
                    if (num_processed+start_index+1) % stride == 0:
                        if args.validate_feasibility:
                            stat_df_mini = ps.DataFrame(all_stats[last_ckpt:], index=processed_indices[last_ckpt:])
                            stat_df_mini.to_parquet(os.path.join(statspath, f'stats_{num_processed+start_index}.parquet'))
                            last_ckpt = len(processed_indices)
                        mate_stat_df_mini = ps.DataFrame(all_mate_stats[last_mate_ckpt:], index=mate_indices[last_mate_ckpt:])
                        mate_stat_df_mini.to_hdf(os.path.join(statspath, f'mate_stats_{ind}.h5'), 'mates')
                        last_mate_ckpt = len(mate_indices)
                        record_val(num_processed + start_index + 1)

            elif args.mode == Mode.GENERATE:
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
        
    if save_stats:
        if args.mode == Mode.GENERATE:
            stats_df = ps.DataFrame(all_stats, index=processed_indices)
            stats_df.fillna(value=replace_keys, inplace=True)
            stats_df.to_parquet(os.path.join(statspath, 'stats_all.parquet'))
            mate_stats_df = ps.DataFrame(all_mate_stats, index=mate_indices)
            mate_stats_df.to_parquet(os.path.join(statspath, 'mate_stats_all.parquet'))
        
        elif args.mode == Mode.DISTANCE:
            stats_df = ps.DataFrame(all_stats, index=processed_indices)
            stats_df.to_parquet(os.path.join(statspath, 'stats_all.parquet'))
            mate_stats_df = ps.DataFrame(all_mate_stats, index=mate_indices)
            mate_stats_df.to_parquet(os.path.join(statspath, 'mate_stats_all.parquet'))

        elif args.mode == Mode.AUGMENT:
            newmate_stats_df = ps.DataFrame(all_newmate_stats)
            newmate_stats_df.to_hdf(os.path.join(statspath, 'newmate_stats_all.h5'), 'newmates')
        elif args.mode == Mode.CHECK_MATES:
            if args.validate_feasibility:
                stat_df = ps.DataFrame(all_stats, index=processed_indices)
                stat_df.to_parquet(os.path.join(statspath, 'stats_all.parquet'))
            mate_stats_df = ps.DataFrame(all_mate_stats, index=mate_indices)
            mate_stats_df.to_hdf(os.path.join(statspath, 'mate_stats_all.h5'), 'mates')
    else:
        print('indices:',processed_indices)
        print(all_stats)

if __name__ == '__main__':
    main()