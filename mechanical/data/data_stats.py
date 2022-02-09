from mechanical.data import AssemblyLoader, Stats
import logging

class TestVisitor:
    def __init__(self, global_data, use_uvnet_features, epsilon_rel, max_topologies):
        self.transforms = AssemblyLoader(global_data, use_uvnet_features=use_uvnet_features, epsilon_rel=epsilon_rel, max_topologies=max_topologies)
    
    def __call__(self, data):
        out_dict = {}
        stat = Stats()
        stat.append({'num_parts': len(data.assembly_info.parts)}, data.ind)
        out_dict['assembly_stats'] = stat
        return out_dict