from enum import Enum
import numpy as np
from mechanical.utils.transforms import compute_basis, apply_homogeneous_transform, project_to_plane
import numpy.linalg as LA


def frame_from_mated_cs(mated_cs):
    origin = np.array(mated_cs['origin'])
    frame = np.array([mated_cs['xAxis'], mated_cs['yAxis'], mated_cs['zAxis']]).T
    return origin, frame

class Mate:
    def __init__(self, mate_json=None, flattened=False, mcs=None, occIds=None, mateType='UNKNOWN', name='', origins=None, rots=None):
        if mate_json is None:
            if occIds is None:
                raise ValueError
            if mcs is None:
                if origins is None or rots is None:
                    raise ValueError
                self.matedEntities = [(occ, (origin, rot)) for occ, origin, rot in zip(occIds, origins, rots)]
            else:
                frames = [mc.get_coordinate_system() for mc in mcs]
                self.matedEntities = [(occ, (cs[:3,3], cs[:3,:3])) for occ, cs in zip(occIds, frames)]
            self.type = mateType
            self.name = name
        else:
            if flattened:
                self.matedEntities = [(me['matedOccurrence'], frame_from_mated_cs(me['matedCS'])) for me in mate_json['matedEntities'] if me['occurrenceType'] == 'Part']
            else:
                mate_json = mate_json['featureData']
                self.matedEntities = [(me['matedOccurrence'][-1], frame_from_mated_cs(me['matedCS'])) for me in mate_json['matedEntities']]
            self.type = mate_json['mateType']
            self.name = mate_json['name']
    
    def get_axes(self):
        return [me[1][1][:,2] for me in self.matedEntities]

    def get_origins(self):
        return [me[1][0] for me in self.matedEntities]

    def get_projected_origins(self, dir):
        return [project_to_plane(origin, dir) for origin in self.get_origins()]

def df_to_mates(mate_subset):
    mates = []
    for j in range(mate_subset.shape[0]):
        mate_row = mate_subset.iloc[j]
        mate = Mate(occIds=[mate_row[f'Part{p+1}'] for p in range(2)],
                origins=[mate_row[f'Origin{p+1}'] for p in range(2)],
                rots=[mate_row[f'Axes{p+1}'] for p in range(2)],
                name=mate_row['Name'],
                mateType=mate_row['Type'])
        mates.append(mate)
    return mates

def newmate_df_to_mates(newmate_subset, batch, norm_tf):
    inv_tf = LA.inv(norm_tf)
    mates = []
    for j in range(newmate_subset.shape[0]):
        if newmate_subset.iloc[j]['added_mate']:
            newmate_row = newmate_subset.iloc[j]
            axis_index = newmate_subset.iloc[j]['axis_index']
            origin = batch.mcfs[axis_index,3:].numpy()
            origin = apply_homogeneous_transform(inv_tf, origin)
            axis = batch.mcfs[axis_index,:3].numpy()
            axis = inv_tf[:3,:3] @ axis
            axes = compute_basis(axis)
            axes = np.concatenate([axes.T, axis[:,np.newaxis]], axis=1)
            mate = Mate(occIds=[newmate_row[f'part{p+1}'] for p in range(2)],
                    origins=[origin for p in range(2)],
                    rots=[axes for p in range(2)],
                    name=f'Augmented {j}',
                    mateType=newmate_row['type'])
            mates.append(mate)
    return mates

class MateTypes(Enum):
    PIN_SLOT = 'PIN_SLOT'
    BALL = 'BALL'
    PARALLEL = 'PARALLEL'
    SLIDER = 'SLIDER'
    REVOLUTE = 'REVOLUTE'
    CYLINDRICAL = 'CYLINDRICAL'
    PLANAR = 'PLANAR'
    FASTENED = 'FASTENED'
    def __eq__(self, obj):
        return self.value == obj

mate_types = [m.value for m in list(MateTypes)]

def mates_equivalent(mate1, mate2, tol):
    if mate1.type != mate2.type:
        return False
    else:
        if mate1.type == MateTypes.FASTENED:
            return True
        elif mate1.type == MateTypes.SLIDER:
            return np.allclose(mate1.get_axes()[0], mate2.get_axes()[0], rtol=0, atol=tol)
        else:
            axis1 = mate1.get_axes()[0]
            axis2 = mate2.get_axes()[0]
            if np.allclose(axis1, axis2, rtol=0, atol=tol):
                projpoint1 = mate1.get_projected_origins(axis1)[0]
                projpoint2 = mate2.get_projected_origins(axis2)[0]
                return np.allclose(projpoint1, projpoint2, rtol=0, atol=tol)
            else:
                return False