import numpy as np
from numba import jit, prange
from time import time

@jit(nopython=True,fastmath=True)
def get_H_nm_norm( N, EMOL, MU, CAV_LOSS, WC_MODES, A0_MODES, NMODE, dH, E0, n, m ):
    if ( n == m ): # DIAGONAL ELEMENTS
        if ( n < N ): # <g,e,0|H|g,e,0>
            H  = EMOL[n-1,1] - EMOL[n-1,0]
            return (H - E0)/dH
        elif (n >= N):             # <g,g,1|H|g,g,1>
            H = WC_MODES[n-NMODE]
            return (H - E0)/dH
     # OFF-DIAGONAL ELEMENTS
    elif ( n >= N and m < N ): # <g,g,1|H|g,e,0>
        H = WC_MODES[n-NMODE] * A0_MODES[n-NMODE] * MU[m-1,0,1]
        return H/dH
    elif ( m >= N and n < N ): # <g,e,0|H|g,g,1>
        H = WC_MODES[m-NMODE] * A0_MODES[m-NMODE] * MU[n-1,0,1]
        return H/dH
    else:
        return 0.0

@jit(nopython=True)
def get_H_vec_norm_Hermitian( N, NMODE, dEMOL_SHIFT, MU, WC, A0, dH, vec ):
    Hvec = np.zeros( (N+NMODE) )

    # MOLECULAR ENERGIES (FIRST N DIAGONAL ELEMENTS OF H)
    Hvec[:N] += dEMOL_SHIFT[:] * vec[:N]

    # CAVITY ENERGY (LAST NMODE DIAGONAL ELEMENTS OF H)
    Hvec[N:] += (WC[:] - 1.000) * vec[N:] # I Hard-coded this shift to 1.0
    
    # BI-LINEAR INTERACTION (UPPER RIGHT BLOCK OF H)
    Hvec[:N] +=  MU[:,0,1] * np.sum( WC[:] * A0[:] * vec[N:] )

    # BI-LINEAR INTERACTION (LOWER LEFT BLOCK OF H)
    Hvec[N:] += np.sum( MU[:,0,1] * vec[:N] ) * WC[:] * A0[:]
    
    return Hvec/dH

