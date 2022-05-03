from argparse import ArgumentParser
import logging
import os
import sys
import pandas as ps
from mechanical.data import Dataset, GlobalData
from mechanical.data.data_stats import *
from enum import Enum
from mechanical.utils import EnumAction
import math
import random
class Mode(Enum):
    SAVE_BATCHES = "SAVE_BATCHES"
    PENETRATION = "PENETRATION"
    SAVE_AXIS_DATA = "SAVE_AXIS_DATA"
    CHECK_MATES = "CHECK_MATES"
    SAVE_AXIS_AND_CHECK_MATES = "SAVE_AXIS_AND_CHECK_MATES"
    FULL_PIPELINE = "FULL_PIPELINE"
    COMPARE_MC_COUNTS = "COMPARE_MC_COUNTS"
    ANALYZE_DISTANCES = "ANALYZE_DISTANCES"
    AUGMENT_MATES = "AUGMENT_MATES"
    ADD_MATE_LABELS = "ADD_MATE_LABELS"
    FINALIZE_DATASET = "FINALIZE_DATASET"
    CHECK_SAMPLES = "CHECK_SAMPLES"
    ADD_NORMALIZATION_MATRICES = "ADD_NORMALIZATION_MATRICES"
    DUPLICATE_ASSEMBLIES = "DUPLICATE_ASSEMBLIES"
    COPY_TABS = "COPY_TABS"

parser = ArgumentParser()

#data loading args
parser.add_argument('--index_file', nargs='+', default='/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/simple_valid_dataset.txt')
parser.add_argument('--dataroot', default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/')
parser.add_argument('--scraped_dataroot', default='/projects/grail/benjones/cadlab/data')
parser.add_argument('--name', required=True)
parser.add_argument('--targetname', default=None)
parser.add_argument('--stride',type=int, default=100)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--stop_index', type=int, default=-1)
parser.add_argument('--mode', type=Mode, action=EnumAction, required=True)
parser.add_argument('--resume',action='store_true')

#general args
parser.add_argument('--max_topologies', type=int, default=5000)
parser.add_argument('--epsilon_rel', type=float, default=0.001)
parser.add_argument('--no_uvnet_features', dest='use_uvnet_features', action='store_false')
parser.add_argument('--use_uvnet_features', dest='use_uvnet_features', action='store_true')
parser.add_argument('--dry_run', action='store_true')
parser.add_argument('--simple_mcfs', action='store_true')
parser.set_defaults(use_uvnet_features=True)

#penetration args:
parser.add_argument('--sliding_distance',type=float, default=.025, help='distance as a fraction of assembly maxdim')
parser.add_argument('--rotation_angle',type=float, default=math.pi/16)
parser.add_argument('--num_samples',type=int, default=100)
parser.add_argument('--include_vertices', action='store_true')
parser.add_argument('--compute_all_motions',action='store_true')
parser.add_argument('--simulate_all_axes',action='store_true')
parser.add_argument('--simulate_augmented_mates',action='store_true')

#axis args:
parser.add_argument('--max_axis_groups',type=int, default=100)
parser.add_argument('--save_mc_frames', action='store_true')
parser.add_argument('--save_dirs',action='store_true')

#check mates args/check mc args:
parser.add_argument('--validate_feasibility', action='store_true')
parser.add_argument('--check_alternate_paths', action='store_true')
parser.add_argument('--mc_path', type=str, default=None, help='path to desired MC dataset')
parser.add_argument('--batch_path', type=str, default=None, help='path to batch dataset, if checking those')
parser.add_argument('--check_individual_part_mcs', dest='check_individual_part_mcs', action='store_true')
parser.add_argument('--no_individual_part_mcs', dest='check_individual_part_mcs', action='store_false')
parser.set_defaults(check_individual_part_mcs=True)

#distance
parser.add_argument('--distance_threshold',type=float, default=0.01)
parser.add_argument('--append_pair_distance_data', dest='append_pair_distance_data', action='store_true')
parser.add_argument('--no_append_pair_distance_data', dest='append_pair_distance_data', action='store_false')
parser.set_defaults(append_pair_distance_data=True)

#augmentation
parser.add_argument('--require_axis', action='store_true', dest='require_axis', help='only augment mates that have a shared axis, not just direction even if a slider')
parser.add_argument('--no_require_axis', dest='require_axis', action='store_true')
parser.set_defaults(require_axis=True)

#ground truth labeling
#finalize dataset
parser.add_argument('--newmate_df_path',default=None)
parser.add_argument('--mate_check_df_path',default=None)
parser.add_argument('--pspy_df_path', default=None)
parser.add_argument('--augmented_labels', action='store_true')
parser.add_argument('--indices_only', action='store_true')
parser.add_argument('--no_augmented_labels', dest='augmented_labels', action='store_false')
parser.set_defaults(augmented_labels=True)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--split', nargs=3, type=int, default=[8, 1, 1])

#document API args
parser.add_argument('--target_document', default='086f239f99f9cf9b2e216f10')
parser.add_argument('--target_workspace', default='d23963f4c6e45364d606b0bf')
parser.add_argument('--dup_df_path', default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/duplicator/assembly_duplication_stats.parquet')


args = parser.parse_args()

def main():
    if args.targetname is None:
        targetname = args.name
    else:
        targetname = args.targetname

    statspath = os.path.join(args.dataroot, args.name)
    outpath = os.path.join(args.dataroot, targetname)
    if not os.path.isdir(outpath):
        os.mkdir(outpath)
    
    file_handler = logging.FileHandler(filename=os.path.join(outpath, 'log.txt'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logging.info(f'args: {args}')

    dataset = Dataset(args.index_file, args.stride, outpath, args.start_index, args.stop_index)

    mate_check_df_path = args.mate_check_df_path if args.mate_check_df_path is not None else os.path.join(statspath, 'mate_check_stats.parquet')
    newmate_df_path = args.newmate_df_path if args.newmate_df_path is not None else os.path.join(statspath, 'newmate_stats.parquet')
    pspy_df_path = args.pspy_df_path if args.pspy_df_path is not None else os.path.join(statspath, 'pspy_stats.parquet')
    mc_path = args.mc_path if args.mc_path is not None else os.path.join(statspath, 'axis_data')
    batch_path = args.batch_path if args.batch_path is not None else os.path.join(statspath, 'batches')


    globaldata = GlobalData(mate_check_df_path=mate_check_df_path, newmate_df_path=newmate_df_path, pspy_df_path=pspy_df_path)

    if args.mode == Mode.SAVE_BATCHES:
        batchpath = os.path.join(outpath, 'batches')
        if not os.path.isdir(batchpath):
            os.mkdir(batchpath)
        action = BatchSaver(globaldata, batchpath, args.use_uvnet_features, args.epsilon_rel, args.max_topologies, dry_run=args.dry_run, simple_mcfs=args.simple_mcfs)
    elif args.mode == Mode.DUPLICATE_ASSEMBLIES:
        newjsonpath = os.path.join(outpath, 'json')
        if not os.path.isdir(newjsonpath):
            os.mkdir(newjsonpath)
        action = MatelessAssemblyDuplicator(globaldata, args.scraped_dataroot, newjsonpath=newjsonpath)
    elif args.mode == Mode.COPY_TABS:
        action = TabCopier(globaldata, args.scraped_dataroot, args.target_document, args.target_workspace, args.dup_df_path)
    elif args.mode == Mode.PENETRATION:
        meshpath = os.path.join(args.dataroot, 'mesh')
        action = DisplacementPenalty(globaldata, args.sliding_distance, args.rotation_angle, args.num_samples, args.include_vertices, meshpath, compute_all=args.compute_all_motions, augmented_mates=args.simulate_augmented_mates, batch_path=batch_path, mc_path=mc_path, all_axes=args.simulate_all_axes, part_distance_threshold=args.distance_threshold)
    elif args.mode == Mode.CHECK_SAMPLES:
        action = UVSampleChecker(args.batch_path)
    elif args.mode == Mode.SAVE_AXIS_DATA:
        mc_path = os.path.join(outpath, 'axis_data')
        if not os.path.isdir(mc_path):
            os.mkdir(mc_path)
        action = MCDataSaver(globaldata, mc_path, args.max_axis_groups, args.epsilon_rel, args.max_topologies, save_frames=args.save_mc_frames, save_dirs=args.save_dirs, dry_run=args.dry_run, simple_mcfs=args.simple_mcfs)
    
    elif args.mode == Mode.CHECK_MATES:
        action = MateChecker(globaldata, mc_path, args.epsilon_rel, args.max_topologies, args.validate_feasibility, args.check_alternate_paths, simple_mcfs=args.simple_mcfs)
    
    elif args.mode == Mode.SAVE_AXIS_AND_CHECK_MATES:
        mc_path = os.path.join(outpath, 'axis_data')
        if not os.path.isdir(mc_path):
            os.mkdir(mc_path)
        action = CombinedAxisMateChecker(globaldata, mc_path, args.epsilon_rel, args.max_topologies, args.validate_feasibility, args.check_alternate_paths, args.max_axis_groups, save_frames=args.save_mc_frames, save_dirs=True, dry_run=args.dry_run, simple_mcfs=args.simple_mcfs)
    
    elif args.mode == Mode.FULL_PIPELINE:
        mc_path = os.path.join(outpath, 'axis_data')
        if not os.path.isdir(mc_path):
            os.mkdir(mc_path)
        
        batchpath = os.path.join(outpath, 'batches')
        if not os.path.isdir(batchpath):
            os.mkdir(batchpath)
        
        action = CombinedAxisBatchAugment(globaldata, mc_path, batchpath, args.epsilon_rel, args.max_topologies, args.validate_feasibility, args.check_alternate_paths, args.max_axis_groups, save_frames=args.save_mc_frames, save_dirs=True, dry_run=args.dry_run, distance_threshold=args.distance_threshold, require_axis=args.require_axis, use_uvnet_features=args.use_uvnet_features, append_pair_data=args.append_pair_distance_data, simple_mcfs=args.simple_mcfs)

    elif args.mode == Mode.COMPARE_MC_COUNTS:
        action = MCCountChecker(mc_path, batch_path, args.check_individual_part_mcs)
    
    elif args.mode == Mode.ANALYZE_DISTANCES:
        action = DistanceChecker(globaldata, args.distance_threshold, args.append_pair_distance_data, mc_path, args.epsilon_rel, args.max_topologies, simple_mcfs=args.simple_mcfs)

    elif args.mode == Mode.AUGMENT_MATES:
        action = MateAugmentation(globaldata, mc_path, args.distance_threshold, args.require_axis, True, args.epsilon_rel, args.max_topologies, simple_mcfs=args.simple_mcfs)
    
    elif args.mode == Mode.ADD_MATE_LABELS:
        action = MateLabelSaver(globaldata, mc_path, args.augmented_labels, args.dry_run, indices_only=args.indices_only)
    
    elif args.mode == Mode.ADD_NORMALIZATION_MATRICES:
        action = TransformSaver(globaldata, mc_path)

    elif args.mode == Mode.FINALIZE_DATASET:
        action = DataChecker(globaldata, mc_path, batch_path, args.distance_threshold)

    dataset.map_data(action)

    if args.mode == Mode.FULL_PIPELINE:
        logging.info('Saving mate labels final data check')
        action2 = MateLabelSaver(globaldata, mc_path, args.augmented_labels, args.dry_run)
        dataset.map_data(action2)
        action3 = DataChecker(globaldata, mc_path, batch_path, args.distance_threshold)
        logging.info('Final data check')
        dataset.map_data(action3)

    if args.mode == Mode.FINALIZE_DATASET or args.mode == Mode.FULL_PIPELINE:
        logging.info('writing training files')
        final_df = ps.read_parquet(os.path.join(args.dataroot, args.name, 'final_stat.parquet'))
        datalist = list(final_df[lambda df: df['valid']].index)
        with open(os.path.join(args.dataroot, args.name, 'full.txt'),'w') as f:
            f.writelines([str(l) + '\n' for l in datalist])
        
        random.seed(args.seed)
        random.shuffle(datalist)

        split = args.split
        N = len(datalist)
        splittotal = sum(split)
        train_i = int(split[0] / splittotal * N)
        test_i = int((split[0] + split[1]) / splittotal * N)

        with open(os.path.join(args.dataroot, args.name, 'train.txt'),'w') as f:
            f.writelines([str(l) + '\n' for l in datalist[:train_i]])
        
        with open(os.path.join(args.dataroot, args.name, 'test.txt'),'w') as f:
            f.writelines([str(l) + '\n' for l in datalist[train_i:test_i]])
        
        with open(os.path.join(args.dataroot, args.name, 'validation.txt'),'w') as f:
            f.writelines([str(l) + '\n' for l in datalist[test_i:]])


if __name__ == '__main__':
    main()