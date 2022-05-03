from mechanical.onshape.api import Onshape
from pathlib import Path
import json

class AssemblyDuplicator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.api = Onshape()
    

    def copy_assembly_without_mates(self, did, mv, eid, prefix='', newdid=None, newwid=None):
        data_path = self.data_dir / 'assemblies' / did / mv / eid / 'default.json'
        document_path = self.data_dir / 'documents' / did / f'{mv}.json'
        if data_path.exists() and document_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                assembly_definition = json.load(f)

            with open(document_path,'r', encoding='utf-8') as f:
                document_definition = json.load(f)
            

            wid = document_definition['data']['defaultWorkspace']['id']

            elements_response = self.get_elements(did,wid)

            element_name = None
            for element in elements_response:
                if element['id'] == eid:
                    element_name = element['name']
                    break
            
            if element_name is None:
                raise Exception(f'element not in document')

            name = document_definition["data"]["name"]

            if newdid is None:
                newname = f'{prefix} {name} Mateless'
                
                new_assembly_json = self.duplicate_workspace(did, wid, newname)
                
                newdid = new_assembly_json['newDocumentId']
                newwid = new_assembly_json['newWorkspaceId']

            new_elements_response = self.get_elements(newdid, newwid)
            neweid = None
            for element in new_elements_response:
                if element['name'] == element_name:
                    neweid = element['id']
                    break
            if neweid is None:
                raise Exception(f'element not copied with the same name???')
            
            newass = self.create_assembly(newdid, newwid, f'{newname} MAIN')
            target_eid = newass['id']
            adef = self.get_assembly(newdid, newwid, neweid)

            assemblies = [adef['rootAssembly']] + adef['subAssemblies']
            part_instances = {}
            new_occs = {}
            for assembly in assemblies:
                for instance in assembly['instances']:
                    if instance['type'] == 'Part':
                        part_instances[instance['id']] = instance

            for occurrence in adef['rootAssembly']['occurrences']:
                if occurrence['path'][-1] in part_instances:
                    tf = occurrence['transform']
                    instance = part_instances[occurrence['path'][-1]]
                    odid =  instance['documentId']
                    oeid = instance['elementId']
                    opid = instance['partId']
                    copy_response = self.copy_instance_with_transform(newdid, newwid, target_eid, odid, oeid, opid, tf)
                    new_occs[occurrence['path'][-1]] = copy_response
            return newdid, newwid, neweid, target_eid, name, newname, adef, new_occs
        else:
            raise ValueError('document not found')
            



    def copy_assembly_and_delete_mates(self, did, mv, eid, prefix='', newdid=None, newwid=None):
        newname = None

        data_path = self.data_dir / 'assemblies' / did / mv / eid / 'default.json'
        document_path = self.data_dir / 'documents' / did / f'{mv}.json'
        if data_path.exists() and document_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                assembly_definition = json.load(f)

            with open(document_path,'r', encoding='utf-8') as f:
                document_definition = json.load(f)
            
            wid = document_definition['data']['defaultWorkspace']['id']

            elements_response = self.get_elements(did,wid)

            element_name = None
            for element in elements_response:
                if element['id'] == eid:
                    element_name = element['name']
                    break
            
            if element_name is None:
                raise Exception(f'element not in document')

            if newdid is None:
                newname = f'{prefix} {document_definition["data"]["name"]} Mateless'
                
                new_assembly_json = self.duplicate_workspace(did, wid, newname)
                
                newdid = new_assembly_json['newDocumentId']
                newwid = new_assembly_json['newWorkspaceId']

            new_elements_response = self.get_elements(newdid, newwid)
            neweid = None
            for element in new_elements_response:
                if element['name'] == element_name:
                    neweid = element['id']
                    break
            if neweid is None:
                raise Exception(f'element not copied with the same name???')
            
            features = self.get_features(newdid, newwid, neweid)
            for feature in features['features']:
                if feature['message']['featureType'] == 'mate':
                    fid = feature['message']['featureId']
                    self.delete_feature(newdid, newwid, neweid, fid)
        
            return newdid, newwid, neweid, newname

            
        else:
            raise ValueError


    def copy_element(self, did, wid, source_did, source_wid, source_eid):
        endpoint = f'elements/copyelement/{did}/workspace/{wid}'
        query = {
            "documentIdSource": source_did,
            "elementIdSource": source_eid,
            "workspaceIdSource": source_wid
            }
        
        response = self.api.request('post', f'/api/{endpoint}', body=query)
        self._check_errors('copying element was not successful', response)
        return response.json()


    def copy_instance_with_transform(self, did, wid, eid, source_did, source_eid, part_id, transform):
        endpoint = f'assemblies/d/{did}/w/{wid}/e/{eid}/transformedinstances'
        query = {
            "transformGroups": [
                {
                "instances": [
                    {
                    "documentId": source_did,
                    "elementId": source_eid,
                    "includePartTypes": [
                        "PARTS"
                    ],
                    "isAssembly": False,
                    "isHidden": False,
                    "isSuppressed": False,
                    "isWholePartStudio": False,
                    "partId": part_id
                    }
                ],
                "transform": transform

                }
            ]
            }
        response = self.api.request('post', f'/api/{endpoint}', body=query)
        self._check_errors('copying instance was not successful', response)
        return response.json() if response.text else None


    def copy_instance(self, did, wid, eid, source_did, source_eid, part_id):
        endpoint = f'assemblies/d/{did}/w/{wid}/e/{eid}/instances'
        query = {
        "documentId": source_did,
        "elementId": source_eid,
            "partId": part_id,
        }
        response = self.api.request('post', f'/api/{endpoint}', body=query)
        self._check_errors('copying instance was not successful', response)
        return response.json() if response.text else None


    def duplicate_workspace(self, did, wid, newname):
        endpoint = f'documents/{did}/workspaces/{wid}/copy'
        query = {'isPublic':True, 'newName': newname}

        response = self.api.request('post', f'/api/{endpoint}', body=query)
        self._check_errors('duplicating workspace was not successful', response)
        return response.json()

    
    def create_assembly(self, did, wid, name):
        endpoint = f'assemblies/d/{did}/w/{wid}'
        query = {
                'name':name,
            }
        response = self.api.request('post', f'/api/{endpoint}', body=query)
        self._check_errors('creating assembly was not successful', response)
        return response.json()
        
    
    def get_assembly(self, did, wid, eid):
        
        query = {
                'includeMateConnectors':True,
                'includeMateFeatures': True
            }
        endpoint = f'assemblies/d/{did}/w/{wid}/e/{eid}/'
        response = self.api.request('get', f'/api/{endpoint}', query)
        self._check_errors(f'Request for assembly /d/{did}/w/{wid}/e/{eid} was not successful', response)
        return response.json()


    def get_elements(self, did, wid):
        endpoint = f'documents/d/{did}/w/{wid}/elements'
        response = self.api.request('get', f'/api/{endpoint}')
        self._check_errors('requesting elements was not successful', response)
        return response.json()

    
    def get_features(self, did, wid, eid):
        endpoint = f'assemblies/d/{did}/w/{wid}/e/{eid}/features'
        response = self.api.request('get',f'/api/{endpoint}')
        self._check_errors('requesting features was not successful', response)
        return response.json()

    
    def delete_feature(self, did, wid, eid, fid):
        endpoint = f'assemblies/d/{did}/w/{wid}/e/{eid}/features/featureid/{fid}'
        response = self.api.request('delete',f'/api/{endpoint}')
        self._check_errors(f'deleting feature d/{did}/w/{wid}/e/{eid}/f/{fid} was not successful', response)
        return response.json()


    def _check_errors(self, msg, response):
        if response.status_code < 200 or response.status_code >= 300:
            raise Exception(f'{msg}; CODE: {response.status_code}; JSON: {response.text}')

if __name__ == '__main__':

    dup = AssemblyDuplicator('/projects/grail/benjones/cadlab/data')

    #newdid, newwid, neweid, newname = dup.copy_assembly_and_delete_mates('d8a303e6739db91d4aab5bc5','150b669cde452a8c0eceeda0','6a06a9a7db2c8b85d3601940', newdid='63dd6441d8c1932d211ba4a1', newwid='eaea8bfef788195bd549952b')
    dup.copy_assembly_without_mates('d8a303e6739db91d4aab5bc5','150b669cde452a8c0eceeda0','6a06a9a7db2c8b85d3601940', prefix='TEST')

