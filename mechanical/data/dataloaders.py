import os
import pandas as ps
import logging


class GlobalData:
    def __init__(self):
        self.df_name = '/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5'
        self.updated_df_name = '/fast/jamesn8/assembly_data/assembly_data_with_transforms_all.h5_segmentation.h5'

    @property
    def assembly_df(self):
        if not hasattr(self, '_assembly_df'):
            logging.info('Loading assembly dataframe')
            self._assembly_df = ps.read_hdf(self.df_name,'assembly')
            logging.info('done')
        return self._assembly_df
    
    @property
    def mate_df(self):
        if not hasattr(self, '_mate_df'):
            logging.info('Loading mate dataframe')
            self._mate_df = ps.read_hdf(self.df_name,'mate')
            self._mate_df['MateIndex'] = self._mate_df.index
            self._mate_df.set_index('Assembly', inplace=True)
            logging.info('done')
        return self._mate_df
    
    @property
    def part_df(self):
        if not hasattr(self, '_part_df'):
            logging.info('Loading part dataframe')
            self._part_df = ps.read_hdf(self.updated_df_name,'part')
            self._part_df['PartIndex'] = self._part_df.index
            self._part_df.set_index('Assembly', inplace=True)
            logging.info('done')
        return self._part_df


class Stats:
    def __init__(self, format='parquet', key=None):
        self.rows = []
        self.indices = []
        self.checkpoint = 0
        self.format = format
        self.key = key
    
    def append(self, row, index=None):
        self.rows.append(row)
        if index is not None:
            self.indices.append(index)
    
    def to_df(self, incremental=False):
        if not incremental:
            self.checkpoint = 0
        if len(self.rows) - self.checkpoint > 0:
            df = ps.DataFrame(self.rows[self.checkpoint:], index=self.indices[self.checkpoint:] if self.indices else None)
            if incremental:
                self.checkpoint = len(self.rows)
            return df
        else:
            return None

    def save_df(self, path, incremental=False):
        df = self.to_df(incremental=incremental)
        if df is not None:
            if incremental:
                path = path + f'_{self.checkpoint}'
            if self.format == 'parquet':
                df.to_parquet(path + '.parquet')
            elif self.format == 'hdf':
                df.to_hdf(path + '.h5', self.key)
    
    def combine(self, other):
        self.rows += other.rows
        self.indices += other.indices
    
    

class Dataset:
    def __init__(self, index_file, stride, stats_path, start_index=0, stop_index=-1):
        with open(index_file,'r') as f:
            self.index = [int(l.rstrip()) for l in f.readlines()]
        self.stride = stride
        self.stats_path = stats_path
        self.stats = dict()
        self.start_index = start_index
        self.stop_index = stop_index if stop_index >= 0 else len(self.index)
    
    def map_data(self, action):
        """
        action: function that returns a dictionary from names to Stats objects
        """
        for i,ind in enumerate(self.index[self.start_index:self.stop_index]):
            logging.info(f'processing {i+self.start_index}/{len(self.index)} ({ind})')
            data = action.transforms(ind)
            results = action(data)
            if results is not None:
                self.log_results(results)
            if (i+self.start_index+1) % self.stride == 0:
                self.save_results(incremental=True)
        self.save_results(incremental=False)

    def log_results(self, results):
        for key in results:
            if key in self.stats:
                self.stats[key].combine(results[key])
            else:
                self.stats[key] = results[key]
    
    def save_results(self, incremental):
        for key in self.stats:
            self.stats[key].save_df(os.path.join(self.stats_path, key), incremental=incremental)

