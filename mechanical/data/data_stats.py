from mechanical.data import AssemblyLoader, Stats

class TestVisitor:
    def __init__(self,assembly_df, part_df, mate_df, datapath, use_uvnet_features, epsilon_rel, max_topologies):
        self.transforms = AssemblyLoader(assembly_df, part_df, mate_df, datapath, use_uvnet_features, epsilon_rel, max_topologies)
    
    def __call__(self, data):
        out_dict = {}
        stat = Stats()
        stat.append({'num_parts': len(data.assembly_info.parts)}, data.ind)
        out_dict['assembly_stats'] = stat
        return out_dict