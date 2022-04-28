from mechanical.onshape import Onshape
from pathlib import Path
import json

class AssemblyDuplicator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.api = Onshape()
    

    def process_assembly(self, did, mv, eid):

        data_path = self.data_dir / 'assemblies' / did / mv / eid / 'default.json'
        document_path = self.data_dir / 'documents' / did / f'{mv}.json'
        if data_path.exists() and document_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                assembly_definition = json.load(f)

            with open(document_path,'r', encoding='utf-8') as f:
                document_definition = json.load(f)
            
            wid = document_definition['data']['defaultWorkspace']['id']
            newname = f'{document_definition["data"]["name"]} Mateless'
            
            endpoint = f'documents/{did}/workspaces/{wid}/copy'
            query = {'isPublic':True, 'newName': newname}

            response = self.api.request('post', f'/api/{endpoint}', body=query)
            if response.status_code < 200 or response.status_code >= 300:
                raise Exception(f'Request for assembly features {did}/{mv}/{eid} was not successful.')
            json_data = response.json()
            return json_data

        else:
            raise ValueError
        

if __name__ == '__main__':

    dup = AssemblyDuplicator('/projects/grail/benjones/cadlab/data')

    response = dup.process_assembly('d8a303e6739db91d4aab5bc5','150b669cde452a8c0eceeda0','6a06a9a7db2c8b85d3601940')

    print(response)