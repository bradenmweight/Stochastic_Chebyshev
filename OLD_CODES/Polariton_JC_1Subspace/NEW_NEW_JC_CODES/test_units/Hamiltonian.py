import numpy as np
from numba import jit, prange
from time import time

@jit(nopython=True,fastmath=True)
def get_H_nm_norm( N, EGS, E, MU, CAV_LOSS, WC, A0, dH, E0, n, m ):
    if ( n == m ): # DIAGONAL ELEMENTS
        if ( n >= 0 and n < N ): # <g,e,0|H|g,e,0>
            H  = E[n-1,1] - E[n-1,0]
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
def get_H_vec_norm_Hermitian( N, EGS, E, MU, CAV_LOSS, WC, A0, dH, E0, vec ):
    Hvec         = np.zeros( (N+1), dtype=np.complex64 )

    # DIAGONAL ELEMENTS
    Hvec[:N] += ( E[:,1]-E[:,0] - E0 ) * vec[:N]
    
    # LAST COLUMN
    Hvec[:N] += WC * A0 * MU[:,0,1] * vec[-1]

    # BOTTOM ROW
    Hvec[N]   += WC * A0 * np.sum( MU[:,0,1] * vec[:N] )
    Hvec[N]   += (WC-CAV_LOSS/2*1.0j-E0)*vec[-1]
    
    return Hvec/dH

