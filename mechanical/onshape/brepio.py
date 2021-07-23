import json
from pspart import Part
import os
import numpy as np

def frame_from_mated_cs(mated_cs):
    origin = np.array(mated_cs['origin'])
    frame = np.array([mated_cs['xAxis'], mated_cs['yAxis'], mated_cs['zAxis']]).T
    return origin, frame

class Mate:
    def __init__(self, json):
        self.type = json['featureData']['mateType']
        self.matedEntities = [(me['matedOccurrence'][-1], frame_from_mated_cs(me['matedCS'])) for me in json['featureData']['matedEntities']]

class Loader:
    def __init__(self, datapath):
        self.datapath = datapath
    
    def load(self, did, mv, eid):
        """
        Load parts and mates given the document, multiversion, and element ids.
        Parts are a dict occurrence path -> (rigid transform, Part object)
        """
        path = os.path.join(self.datapath, 'data/assemblies', did, mv, eid, 'default.json')

        with open(path) as f:
            assembly_def = json.load(f)

        occs = assembly_def['definition']['rootAssembly']['occurrences']

        instances = assembly_def['definition']['rootAssembly']['instances']
        if 'subAssemblies' in assembly_def['definition']:
            for sub in assembly_def['definition']['subAssemblies']:
                instances = instances + sub['instances']
        part_dict = dict([(inst['id'], inst) for inst in instances if inst['type'] == 'Part'])

        occs = [(part_dict[occ['path'][-1]], np.array(occ['transform']).reshape(4,4), occ['path'][-1]) for occ in occs]

        def part_from_occ(occ):
            pd = occ[0]
            did = pd['documentId']
            mv = pd['documentMicroversion']
            eid = pd['elementId']
            config = pd['fullConfiguration']
            pid = pd['partId']
            return Part(os.path.join(self.datapath, 'data/models/', did, mv, eid, config, f'{pid}.xt'))


        parts = [part_from_occ(occ) for occ in occs]
        geo = dict([(po[1][2], (po[1][1], po[0])) for po in zip(parts, occs)])

        mates = assembly_def['mates']
        mates = [Mate(mate) for mate in mates]

        return geo, mates
