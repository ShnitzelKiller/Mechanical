from argparse import ArgumentParser
import logging
import os
import pandas as ps
from mechanical.data import Dataset, BatchSaver, GlobalData
from enum import Enum
from mechanical.utils import EnumAction
class Mode(Enum):
    SAVE_BATCHES = "SAVE_BATCHES"

parser = ArgumentParser()
parser.add_argument('--index_file', default='/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/simple_valid_dataset.txt')
parser.add_argument('--dataroot', default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/')
parser.add_argument('--name', required=True)
parser.add_argument('--stride',type=int, default=100)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--stop_index', type=int, default=-1)
parser.add_argument('--mode', type=Mode, action=EnumAction, required=True)

args = parser.parse_args()

def main():
    statspath = os.path.join(args.dataroot, args.name)
    if not os.path.isdir(statspath):
        os.mkdir(statspath)
    logging.basicConfig(filename=os.path.join(statspath, 'log.txt'), level=1)

    globaldata = GlobalData()
    dataset = Dataset(args.index_file, args.stride, statspath, args.start_index, args.stop_index)

    if args.mode == Mode.SAVE_BATCHES:
        batchpath = os.path.join(statspath, 'batches')
        if not os.path.isdir(batchpath):
            os.mkdir(batchpath)
        action = BatchSaver(globaldata, batchpath, False, .001, 5000)
    
    dataset.map_data(action)



if __name__ == '__main__':
    main()