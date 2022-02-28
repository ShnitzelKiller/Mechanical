from mechanical.onshape import Mate
from enum import Enum
import numpy as np

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