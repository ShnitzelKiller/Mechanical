import mechanical.onshape as brepio
import meshplot as mp
import numpy as np
import os
import pandas as ps
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_path', default='/projects/grail/jamesn8/projects/mechanical/data/dataset/subassembly_filters/filter_list.txt')

    datapath = '/projects/grail/benjones/cadlab/data'
    loader = brepio.Loader(datapath)

    filter_set = set()
    scan = os.scandir(os.path.join(datapath,'data/flattened_assemblies'))
    for i,entry in enumerate(scan):
        if not entry.name.endswith('json'):
            continue
        if i % 1000 == 0:
            print(f'processed {i}')
        with open(entry.path) as f:
            fa = json.load(f)
        for sub_assembly in fa['assembly_occurrences']:
            if sub_assembly['fullConfiguration'] != 'default':
                continue
            did = sub_assembly['documentId']
            mv = sub_assembly['documentMicroversion']
            eid = sub_assembly['elementId']
            filter_set.add(f'{did}_{mv}_{eid}')
    
        with open('filter_list.txt','w') as filter_list:
            filter_list.writelines([f'{l}.json\n' for l in filter_set])
