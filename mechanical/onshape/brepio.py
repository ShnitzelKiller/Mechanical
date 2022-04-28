import json
from pspart import Part, MateConnector
import pspy
import os
import numpy as np
from mechanical.utils.transforms import project_to_plane
from mechanical.data import Mate

class Loader:
    def __init__(self, datapath):
        self.datapath = datapath
    
    def load_flattened(self, path, geometry=True, skipInvalid=False, loadWorkspace=False, use_pspy=False):
        #_,fname = os.path.split(path)
        #name, ext = os.path.splitext(fname)
        #did, mv, eid = name.split('_')
        with open(os.path.join(self.datapath, 'data/flattened_assemblies', path)) as f:
            assembly_def = json.load(f)
        part_occs = assembly_def['part_occurrences']
        if not skipInvalid:
            for occ in part_occs:
                if len(occ['partId']) == 0:
                    raise KeyError('empty key')

        def part_from_occ(occ, checkOnly=False, silent=False):
            did = occ['documentId']
            mv = occ['documentMicroversion']
            eid = occ['elementId']
            config = occ['fullConfiguration']
            pid = occ['partId']
            filepath = os.path.join(self.datapath, 'data/models/', did, mv, eid, config, f'{pid}.xt')
            if not os.path.isfile(filepath):
                if silent:
                    return None
                else:
                    raise FileNotFoundError(filepath)
            if not checkOnly:
                if use_pspy:
                    part_opts = pspy.PartOptions()
                    part_opts.num_uv_samples = 0
                    return pspy.Part(filepath, part_opts)
                else:
                    return Part(filepath)
        
        if geometry:
            if not skipInvalid:
                [part_from_occ(occ, checkOnly=True) for occ in part_occs] #check for exceptions first
            part_occ_dict = dict([(occ['id'], (np.array(occ['transform']).reshape(4, 4), part_from_occ(occ, silent=skipInvalid))) for occ in part_occs])
        else:
            part_occ_dict = dict([(occ['id'], (np.array(occ['transform']).reshape(4, 4), occ)) for occ in part_occs])
        mates = [Mate(mate, flattened=True) for mate in assembly_def['mates']]

        if loadWorkspace:
            did = assembly_def['documentId']
            mv = assembly_def['documentMicroversion']
            documentFname = os.path.join(self.datapath, f'data/documents/{did}/{mv}.json')
            if os.path.isfile(documentFname):
                with open(documentFname) as f:
                    dj = json.load(f)
                    workspaceId = dj['data']['defaultWorkspace']['id']
            else:
                workspaceId = None
            return part_occ_dict, mates, workspaceId

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
