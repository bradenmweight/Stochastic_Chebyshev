import numpy as np
from numba import jit, prange
from time import time

@jit(nopython=True)
def get_H_nm_norm( N, EMOL, MU, CAV_LOSS, WC, A0, dH, E0, n, m ):
    if ( n == m ): # DIAGONAL ELEMENTS
        if ( n >= 0 and n < N ): # <g,e,0|H|g,e,0>
            H  = EMOL[n-1,1] - EMOL[n-1,0]
            return (H - E0)/dH
        elif (n == N):             # <g,g,1|H|g,g,1>
            H  = WC - CAV_LOSS/2 * 1.0j
            return (H - E0)/dH
     # OFF-DIAGONAL ELEMENTS
    elif ( n == N and m < N ): # <g,g,1|H|g,e,0>
        H = WC * A0 * MU[m-1,0,1]
        return H/dH
    elif ( m == N and n < N ): # <g,e,0|H|g,g,1>
        H = WC * A0 * MU[n-1,0,1]
        return H/dH
    else:
        return 0.0

@jit(nopython=True)
def get_H_vec_norm_Hermitian( N, CAV_LOSS, dEMOL_SHIFT, MU_SCALED, WC, E0, dH, vec ):
    Hvec = np.zeros( (N+1), dtype=np.complex128 )

    # DIAGONAL ELEMENTS
    Hvec[:N] += dEMOL_SHIFT[:] * vec[:N]
    
    # LAST COLUMN
    Hvec[:N] += MU_SCALED[:,0,1] * vec[-1]

    # BOTTOM ROW
    Hvec[N] += np.sum( MU_SCALED[:,0,1] * vec[:N], axis=0 )
    Hvec[N] += (WC-CAV_LOSS/2*1.0j-E0)*vec[-1]

    return Hvec/dH




def get_H_vec_norm_Hermitian___BATCH( N, EGS, dEMOL_SHIFT, MU_SCALED, CAV_LOSS, WC, A0, dH, E0, vec, shape ):
    #Hvec = np.zeros( shape, dtype=np.complex64 )
    Hvec = np.zeros( (N+1), dtype=np.complex64 )
    vec  = vec[:,0]

    # DIAGONAL ELEMENTS
    #Hvec[:N] += dEMOL_SHIFT[:,None] * vec[:N]
    Hvec[:N] += dEMOL_SHIFT[:] * vec[:N]
    
    # LAST COLUMN
    #Hvec[:N] += MU_SCALED[:,0,1,None] * vec[-1]
    Hvec[:N] += MU_SCALED[:,0,1] * vec[-1]

    # BOTTOM ROW
    # Hvec[N] += np.sum( MU_SCALED[:,0,1,None] * vec[:N], axis=0 )
    #Hvec[N] += (WC-CAV_LOSS/2*1.0j-E0)*vec[-1]
    Hvec[N] += np.sum( MU_SCALED[:,0,1] * vec[:N], axis=0 )
    Hvec[N] += (WC-CAV_LOSS/2*1.0j-E0)*vec[-1]

    return Hvec.reshape( shape )/dH