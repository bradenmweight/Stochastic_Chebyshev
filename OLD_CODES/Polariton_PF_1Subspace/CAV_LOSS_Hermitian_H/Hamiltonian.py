import numpy as np
from numba import jit, prange
from time import time

@jit(nopython=True,fastmath=True)
def get_GS_DSE( A0, WC, MU, MU2_AA ):
    N = len(MU)
    DSE_GS = np.sum( MU2_AA[:,0,0] )
    for A in range( N ):
        for B in range( A+1, N ):
            DSE_GS += 2 * MU[A,0,0] * MU[B,0,0]
    return WC * A0**2 * DSE_GS

@jit(nopython=True,fastmath=True)
def get_DIAG_DSE( A0, WC, MU, DSE_GS, MU2_AA ):
    N = len(MU)
    DSE_DIAG = np.zeros( (N), dtype=np.complex64 )
    for n in range( N ):
        DSE_DIAG[n]  = MU2_AA[n,1,1] - MU2_AA[n,0,0]
        DSE_DIAG[n] += MU2_AA[n,1,1] - MU2_AA[n,0,0]
        #TMP = 0.0
        #for m in range( n+1, N ):
        #    TMP += MU[m,0,0]
        #DSE_DIAG[n] += 2 * ( MU[n,1,1] - MU[n,0,0] ) * TMP
        DSE_DIAG[n] += ( MU[n,1,1] - MU[n,0,0] ) * ( np.sum( MU[:,0,0] ) - MU[n,0,0] ) 
    return DSE_GS + WC * A0**2 * DSE_DIAG

@jit(nopython=True,fastmath=True)
def get_0n_DSE( A0, WC, MU, DSE_GS, MU2_AA ):
    N       = len(MU)   
    DSE_0n  = np.zeros( (N), dtype=np.complex64 )
    DSE_0n += MU2_AA[:,0,1] + 2 * MU[:,0,1] * (np.sum(MU[:,0,0]) - MU[:,0,0] )
    return WC * A0**2 * DSE_0n

@jit(nopython=True,fastmath=True)
def get_H_nm_norm( N, EGS, E, MU, CAV_LOSS, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, n, m ):
    H = 0.0 + 0j
    if ( n == m and n == 0 ):
        H += 1.0 - E0
    elif ( n == m and n == 1 ):
        H += WC - E0
    elif ( n == 0 and m == 1 ):
        H += 0.1 - CAV_LOSS/2*1.0j
    elif ( n == 1 and m == 0 ):
        H += 0.1 + CAV_LOSS/2*1.0j
    return H/dH



@jit(nopython=True)
def get_H_vec_norm_Hermitian( N, EGS, E, MU, CAV_LOSS, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, vec ):
    Hvec         = np.zeros( (N+2), dtype=np.complex64 )
    Hvec[0] = (1.0 - E0) * vec[0] + (0.1 - CAV_LOSS/2*1.0j) * vec[1]
    Hvec[1] = (WC  - E0) * vec[1] + (0.1 + CAV_LOSS/2*1.0j) * vec[0]
    return Hvec/dH