import numpy as np
from numba import jit
from time import time




def build_Exact(params):
    
    H              = np.zeros( (params.DIM_H,params.DIM_H), dtype=np.complex128 )
    mol_diag_inds  = (np.arange(params.N_MOL), np.arange(params.N_MOL))
    phot_diag_inds = (np.arange(params.N_MOL,params.N_MOL+1),np.arange(params.N_MOL,params.N_MOL+1))
    Coupling_inds1 = (np.arange(params.N_MOL), np.arange(params.N_MOL,params.DIM_H))
    Coupling_inds2 = (np.arange(params.N_MOL,params.DIM_H),np.arange(params.N_MOL))

    if ( params.HAM == "JC" ):
        H[mol_diag_inds]  = params.E_MOL
        H[phot_diag_inds] = params.WC - 1j * 0.5  * params.CAV_LOSS
        H[Coupling_inds1] = params.WC * params.A0 * params.MU_MOL
        H[Coupling_inds2] = params.WC * params.A0 * params.MU_MOL
    return H

@jit(nopython=True)
def H_JC_on_vec( Hvec, vec, N_MOL, E_MOL_SHIFT, MU_MOL_SCALED, WC_SHIFTED, dH ):

    # DIAGONAL ELEMENTS
    Hvec[:N_MOL] += E_MOL_SHIFT * vec[:N_MOL]
    
    # LAST COLUMN
    Hvec[:N_MOL] += MU_MOL_SCALED * vec[-1]

    # BOTTOM ROW
    Hvec[N_MOL] += np.sum( MU_MOL_SCALED * vec[:N_MOL], axis=0 )
    Hvec[N_MOL] += WC_SHIFTED * vec[-1]

    return Hvec/dH


def H_JC_on_vec_batch( Hvec, vec, N_MOL, E_MOL_SHIFT, MU_MOL_SCALED, WC_SHIFTED, dH ):

    # DIAGONAL ELEMENTS
    Hvec[:N_MOL,:] += np.einsum("a,aR->aR", E_MOL_SHIFT[:], vec[:N_MOL,:] )
    
    # LAST COLUMN
    Hvec[:N_MOL,:] += np.einsum("a,R->aR", MU_MOL_SCALED[:], vec[-1,:] )

    # BOTTOM ROW
    Hvec[N_MOL,:] += np.einsum("a,aR->R", MU_MOL_SCALED[:], vec[:N_MOL,:] )
    Hvec[N_MOL,:] += WC_SHIFTED * vec[-1,:]

    return Hvec/dH