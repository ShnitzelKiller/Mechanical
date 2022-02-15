from argparse import ArgumentParser
import logging
import os
import sys
import pandas as ps
from mechanical.data import Dataset, BatchSaver, GlobalData, DisplacementPenalty
from enum import Enum
from mechanical.utils import EnumAction
import math
class Mode(Enum):
    SAVE_BATCHES = "SAVE_BATCHES"
    PENETRATION = "PENETRATION"

parser = ArgumentParser()
parser.add_argument('--index_file', nargs='+', default='/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/simple_valid_dataset.txt')
parser.add_argument('--dataroot', default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/')
parser.add_argument('--name', required=True)
parser.add_argument('--stride',type=int, default=100)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--stop_index', type=int, default=-1)
parser.add_argument('--mode', type=Mode, action=EnumAction, required=True)

#penetration args:
parser.add_argument('--sliding_distance',type=float, default=.05, help='distance as a fraction of assembly maxdim')
parser.add_argument('--rotation_angle',type=float, default=math.pi/16)
parser.add_argument('--num_samples',type=int, default=100)

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

    dataset = Dataset(args.index_file, args.stride, statspath, args.start_index, args.stop_index)
    globaldata = GlobalData()

    if args.mode == Mode.SAVE_BATCHES:
        batchpath = os.path.join(statspath, 'batches')
        if not os.path.isdir(batchpath):
            os.mkdir(batchpath)
        action = BatchSaver(globaldata, batchpath, False, .001, 5000)
    elif args.mode == Mode.PENETRATION:
        meshpath = os.path.join(args.dataroot, 'mesh')
        action = DisplacementPenalty(globaldata, args.sliding_distance, args.rotation_angle, args.num_samples, meshpath)
    
    dataset.map_data(action)



if __name__ == '__main__':
    main()