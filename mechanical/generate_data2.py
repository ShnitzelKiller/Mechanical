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
class Mode(Enum):
    SAVE_BATCHES = "SAVE_BATCHES"
    PENETRATION = "PENETRATION"
    SAVE_AXIS_DATA = "SAVE_AXIS_DATA"
    CHECK_MATES = "CHECK_MATES"
    SAVE_AXIS_AND_CHECK_MATES = "SAVE_AXIS_AND_CHECK_MATES"
    COMPARE_MC_COUNTS = "COMPARE_MC_COUNTS"

parser = ArgumentParser()

#data loading args
parser.add_argument('--index_file', nargs='+', default='/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/simple_valid_dataset.txt')
parser.add_argument('--dataroot', default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/')
parser.add_argument('--name', required=True)
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
parser.set_defaults(use_uvnet_features=True)

#penetration args:
parser.add_argument('--sliding_distance',type=float, default=.05, help='distance as a fraction of assembly maxdim')
parser.add_argument('--rotation_angle',type=float, default=math.pi/16)
parser.add_argument('--num_samples',type=int, default=100)
parser.add_argument('--include_vertices', action='store_true')

#axis args:
parser.add_argument('--max_axis_groups',type=int, default=10)
parser.add_argument('--save_mc_frames', action='store_true')
parser.add_argument('--save_dirs',action='store_true')

#check mates args/check mc args:
parser.add_argument('--validate_feasibility', action='store_true')
parser.add_argument('--check_alternate_paths', action='store_true')
parser.add_argument('--mc_path', type=str, default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/new_axes_100groups_and_mate_check/axis_data', help='path to desired MC dataset')
parser.add_argument('--batch_path', type=str, default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/pspy_batches/batches', help='path to batch dataset, if checking those')
parser.add_argument('--check_individual_part_mcs', action='store_true')

args = parser.parse_args()

def main():
    statspath = os.path.join(args.dataroot, args.name)
    if not os.path.isdir(statspath):
        os.mkdir(statspath)
    
    file_handler = logging.FileHandler(filename=os.path.join(statspath, 'log.txt'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=logging.INFO, 
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logging.info(f'args: {args}')

    dataset = Dataset(args.index_file, args.stride, statspath, args.start_index, args.stop_index)
    globaldata = GlobalData()

    if args.mode == Mode.SAVE_BATCHES:
        batchpath = os.path.join(statspath, 'batches')
        if not os.path.isdir(batchpath):
            os.mkdir(batchpath)
        action = BatchSaver(globaldata, batchpath, args.use_uvnet_features, args.epsilon_rel, args.max_topologies, dry_run=args.dry_run)

    elif args.mode == Mode.PENETRATION:
        meshpath = os.path.join(args.dataroot, 'mesh')
        action = DisplacementPenalty(globaldata, args.sliding_distance, args.rotation_angle, args.num_samples, args.include_vertices, meshpath)
    
    elif args.mode == Mode.SAVE_AXIS_DATA:
        mc_path = os.path.join(statspath, 'axis_data')
        if not os.path.isdir(mc_path):
            os.mkdir(mc_path)
        action = MCDataSaver(globaldata, mc_path, args.max_axis_groups, args.epsilon_rel, args.max_topologies, save_frames=args.save_mc_frames, save_dirs=args.save_dirs, dry_run=args.dry_run)
    
    elif args.mode == Mode.CHECK_MATES:
        action = MateChecker(globaldata, args.mc_path, args.epsilon_rel, args.max_topologies, args.validate_feasibility, args.check_alternate_paths)
    
    elif args.mode == Mode.SAVE_AXIS_AND_CHECK_MATES:
        mc_path = os.path.join(statspath, 'axis_data')
        if not os.path.isdir(mc_path):
            os.mkdir(mc_path)
        action = CombinedAxisMateChecker(globaldata, mc_path, args.epsilon_rel, args.max_topologies, args.validate_feasibility, args.check_alternate_paths, args.max_axis_groups, save_frames=args.save_mc_frames, save_dirs=True, dry_run=args.dry_run)
    
    elif args.mode == Mode.COMPARE_MC_COUNTS:
        action = MCCountChecker(args.mc_path, args.batch_path, args.check_individual_part_mcs)

    dataset.map_data(action)



if __name__ == '__main__':
    main()