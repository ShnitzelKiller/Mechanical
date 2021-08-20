import json
from pspart import Part
import os
import numpy as np

def frame_from_mated_cs(mated_cs):
    origin = np.array(mated_cs['origin'])
    frame = np.array([mated_cs['xAxis'], mated_cs['yAxis'], mated_cs['zAxis']]).T
    return origin, frame

class Mate:
    def __init__(self, mate, flattened=False):
        if flattened:
            self.matedEntities = [(me['matedOccurrence'], frame_from_mated_cs(me['matedCS'])) for me in mate['matedEntities'] if me['occurrenceType'] == 'Part']
        else:
            mate = mate['featureData']
            self.matedEntities = [(me['matedOccurrence'][-1], frame_from_mated_cs(me['matedCS'])) for me in mate['matedEntities']]
        self.type = mate['mateType']



class Loader:
    def __init__(self, datapath):
        self.datapath = datapath
    
    def load_flattened(self, path, geometry=True):
        #_,fname = os.path.split(path)
        #name, ext = os.path.splitext(fname)
        #did, mv, eid = name.split('_')
        with open(os.path.join(self.datapath, 'data/flattened_assemblies', path)) as f:
            assembly_def = json.load(f)
        part_occs = assembly_def['part_occurrences']
        for occ in part_occs:
            if len(occ['partId']) == 0:
                print('empty part ID for part ',occ['id'])
                raise KeyError

        def part_from_occ(occ, checkOnly=False):
            did = occ['documentId']
            mv = occ['documentMicroversion']
            eid = occ['elementId']
            config = occ['fullConfiguration']
            pid = occ['partId']
            filepath = os.path.join(self.datapath, 'data/models/', did, mv, eid, config, f'{pid}.xt')
            if not os.path.isfile(filepath):
                print('part not found',filepath)
                raise FileNotFoundError(filepath)
            if not checkOnly:
                return Part(filepath)
        
        [part_from_occ(occ, checkOnly=True) for occ in part_occs] #check for exceptions first
        
        if geometry:
            part_occ_dict = dict([(occ['id'], (np.array(occ['transform']).reshape(4, 4), part_from_occ(occ))) for occ in part_occs])
        else:
            part_occ_dict = dict([(occ['id'], np.array(occ['transform']).reshape(4, 4)) for occ in part_occs])
        mates = [Mate(mate, flattened=True) for mate in assembly_def['mates']]

        return part_occ_dict, mates

    def load(self, did=None, mv=None, eid=None, path=None, geometry=True):
        """
        Load parts and mates given the document, multiversion, and element ids.
        Parts are a dict occurrence path -> (rigid transform, Part object)
        """
        if path is None and all(id is not None for id in (did, mv, eid)):
            path = os.path.join(self.datapath, 'data/assemblies', did, mv, eid, 'default.json')
        elif path is not None:
            basepath, _ = os.path.split(path)
            basepath, eid = os.path.split(basepath)
            basepath, mv = os.path.split(basepath)
            _, did = os.path.split(basepath)
        else:
            raise ValueError

        with open(path) as f:
            assembly_def = json.load(f)

        occs = assembly_def['definition']['rootAssembly']['occurrences']

        instances = assembly_def['definition']['rootAssembly']['instances']
        if 'subAssemblies' in assembly_def['definition']:
            for sub in assembly_def['definition']['subAssemblies']:
                instances = instances + sub['instances']

        #dict from instance ID to instance
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


        if geometry:
            parts = [part_from_occ(occ) for occ in occs]
            geo = dict([(po[1][2],(po[1][1], po[0])) for po in zip(parts, occs)]) #(occID, (transform, part))
        else:
            geo = dict([(occ[2], occ[1]) for occ in occs])

        mates = assembly_def['mates']
        mates = [Mate(mate) for mate in mates]

        return geo, mates
