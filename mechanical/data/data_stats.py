from mechanical.data import AssemblyLoader, Stats
import logging
import os
import h5py
import numpy as np
import torch


class DataVisitor:
    def __call__(self, data):
        return self.process(data)
    
    def process(self, data):
        return {}

class NoopVisitor(DataVisitor):
    def __init__(self, transforms):
        self.transforms = transforms

class BatchSaver(DataVisitor):
    def __init__(self, global_data, out_path, use_uvnet_features, epsilon_rel, max_topologies):
        self.transforms = AssemblyLoader(global_data, use_uvnet_features=use_uvnet_features, epsilon_rel=epsilon_rel, max_topologies=max_topologies)
        self.out_path = out_path

    def process(self, data):
        out_dict = {}
        stat = Stats()
        stat.append(data.assembly_info.stats, data.ind)
        out_dict['assembly_stats'] = stat
        batch = data.assembly_info.create_batches()
        torch_datapath = os.path.join(self.out_path, f'{data.ind}.dat')
        torch.save(batch, torch_datapath)
        return out_dict
