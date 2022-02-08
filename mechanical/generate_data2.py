from argparse import ArgumentParser
import logging
import os
import pandas as ps
from mechanical.data import Dataset, AssemblyLoader, Stats

parser = ArgumentParser()
parser.add_argument('--index_file', default='/projects/grail/jamesn8/projects/mechanical/Mechanical/data/dataset/simple_valid_dataset.txt')
parser.add_argument('--dataroot', default='/fast/jamesn8/assembly_data/assembly_torch2_fixsize/')
parser.add_argument('--name', required=True)
parser.add_argument('--stride',type=int, default=100)
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--stop_index', type=int, default=-1)
args = parser.parse_args()

def main():
    statspath = os.path.join(args.dataroot, args.name)
    if not os.path.isdir(statspath):
        os.mkdir(statspath)
    logging.basicConfig(filename=os.path.join(statspath, 'log.txt'), level=1)

    logging.info('Loading dataframes...')

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

    logging.info('done')

    dataset = Dataset(args.index_file, args.stride, statspath, args.start_index, args.stop_index)

    #debug
    transforms = AssemblyLoader(assembly_df, part_df, mate_df, datapath, False, .001, 5000)
    def action(data):
        out_dict = {}
        stat = Stats()
        stat.append({'num_parts': len(data.assembly_info.parts)}, data.ind)
        out_dict['assembly_stats'] = stat
        return out_dict
    dataset.map_data(transforms, action)



if __name__ == '__main__':
    main()