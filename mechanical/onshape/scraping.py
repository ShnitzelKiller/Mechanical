from mechanical.onshape.api import Onshape
from pathlib import Path
import json

class AssemblyDuplicator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.api = Onshape()
    

    def process_assembly(self, did, mv, eid, prefix='', newdid=None, newwid=None):
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
                
                endpoint = f'documents/{did}/workspaces/{wid}/copy'
                query = {'isPublic':True, 'newName': newname}

                response = self.api.request('post', f'/api/{endpoint}', body=query)
                if response.status_code < 200 or response.status_code >= 300:
                    raise Exception(f'duplicating assembly was not successful: {response.text}')
                
                
                new_assembly_json = response.json()
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
        
    
    def get_assembly(self, did, mv, eid):
        
        query = {
                'includeMateConnectors':True,
                'includeMateFeatures': True
            }
        endpoint = f'assemblies/d/{did}/m/{mv}/e/{eid}/'
        response = self.api.request('get', f'/api/{endpoint}', query)
        if response.status_code < 200 or response.status_code >= 300:
            raise Exception(f'Request for assembly features {did}/{mv}/{eid} was not successful.: {response.text}')
        json_data = response.json()
        return json_data

    def get_elements(self, did, wid):
        endpoint = f'documents/d/{did}/w/{wid}/elements'
        response = self.api.request('get', f'/api/{endpoint}')
        if response.status_code < 200 or response.status_code >= 300:
            raise Exception(f'requesting elements was not successful.: {response.text}')
        return response.json()
    
    def get_features(self, did, wid, eid):
        endpoint = f'assemblies/d/{did}/w/{wid}/e/{eid}/features'
        response = self.api.request('get',f'/api/{endpoint}')
        if response.status_code < 200 or response.status_code >= 300:
            raise Exception(f'requesting features was not successful.: {response.text}')
        return response.json()
    
    def delete_feature(self, did, wid, eid, fid):
        endpoint = f'assemblies/d/{did}/w/{wid}/e/{eid}/features/featureid/{fid}'
        response = self.api.request('delete',f'/api/{endpoint}')
        if response.status_code < 200 or response.status_code >= 300:
            raise Exception(f'deleting feature d/{did}/w/{wid}/e/{eid}/f/{fid} was not successful: {response.text}')

if __name__ == '__main__':

    dup = AssemblyDuplicator('/projects/grail/benjones/cadlab/data')

    newdid, newwid, neweid, newname = dup.process_assembly('d8a303e6739db91d4aab5bc5','150b669cde452a8c0eceeda0','6a06a9a7db2c8b85d3601940', newdid='63dd6441d8c1932d211ba4a1', newwid='eaea8bfef788195bd549952b')

    print(newdid, newwid, neweid, newname)