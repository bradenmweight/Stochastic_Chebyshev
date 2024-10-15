import numpy as np
from numba import jit, prange
from time import time

"""
N-molecule, Single-mode PF Hamiltonian
H_PF            = H_el + H_ph + H_el-ph + H_DSE
<g,g,0|H|g,g,0> = sum(E_GS) + DSE
<e,g,0|H|e,g,0> = <g,g,0|H|g,g,0> - E[0,0] + E[0,1] + DSE
<g,e,0|H|g,e,0> = <g,g,0|H|g,g,0> - E[1,0] + E[1,1] + DSE
<g,g,1|H|g,g,1> = <g,g,0|H|g,g,0> +  wc + DSE
<g,g,1|H|e,g,0> = wc A0 <g,g|mu|e,g>
<g,g,1|H|g,e,0> = wc A0 <g,g|mu|g,e>
"""

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
    DSE_DIAG = np.zeros( (N) )
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
    DSE_0n  = np.zeros( (N) )
    DSE_0n += MU2_AA[:,0,1] + 2 * MU[:,0,1] * (np.sum(MU[:,0,0]) - MU[:,0,0] )
    return WC * A0**2 * DSE_0n

@jit(nopython=True,fastmath=True)
def get_H_nm_norm( N, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, n, m ):
    H = 0.0
    if ( n == m ): # DIAGONAL ELEMENTS
        if ( n == 0 ):               # <g,g,0|H|g,g,0>
            H  = EGS
            H += DSE_GS
            return (H - E0)/dH
        elif ( n >= 1 and n < N+1 ): # <g,e,0|H|g,e,0>
            H  = EGS - E[n-1,0] + E[n-1,1]
            H += DSE_DIAG[n-1]
            return (H - E0)/dH
        elif (n == N+1):             # <g,g,1|H|g,g,1>
            H  = EGS
            H += WC
            H += DSE_GS
            return (H - E0)/dH
     # OFF-DIAGONAL ELEMENTS
    elif ( n == 0 and m not in [0,N+1] ): # <g,g,0|H|g,e,0>
        H  = DSE_0n[m-1]
        return H/dH
    elif ( m == 0 and n not in [0,N+1] ): # <g,e,0|H|g,g,0>
        H  = DSE_0n[n-1]
        return H/dH
    elif ( n not in [0, N+1] and m not in [0,N+1] and n != m ): # <g,e,0|H|e,g,0>
        H = 2 * WC * A0**2 * MU[n-1,0,1] * MU[m-1,0,1]
        return H/dH
    elif ( n == N+1 and m >= 1 and m < N+1 ): # <g,g,1|H|g,e,0>
        H = WC * A0 * MU[m-1,0,1]
        return H/dH
    elif ( m == N+1 and n >= 1 and n < N+1 ): # <g,e,0|H|g,g,1>
        H = WC * A0 * MU[n-1,0,1]
        return H/dH
    elif ( m == 0 and n == N+1 ): # <g,g,1|H|g,g,0> # TODO CHECK THIS
        H = WC * A0 * np.sum( MU[:,0,0] )
        return H/dH
    elif ( n == 0 and m == N+1 ): # <g,g,0|H|g,g,1> # TODO CHECK THIS
        H = WC * A0 * np.sum( MU[:,0,0] )
        return H/dH
    else:
        return 0.0


@jit(nopython=True,fastmath=True)
def get_H_vec_norm_SLOW( N, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, vec ):
    Hvec         = np.zeros( (N+2) )

    # TOP ROW
    Hvec[0]     += (EGS+DSE_GS-E0)*vec[0] # <g0|g0>
    Hvec[0]     += np.sum( DSE_0n[:] * vec[1:N+1] ) # <g0|e0> # TODO CHECK THIS
    Hvec[0]     += WC * A0 * np.sum( MU[:,0,0] ) * vec[N+1] #<g0|g1> # TODO CHECK THIS

    for n in range( 1, N+1 ): # MIDDLE ROWS

        # FIRST COLUMN
        Hvec[n] += DSE_0n[n-1] * vec[0]

        # LEFT-TO-DIAGONAL COLUMNS
        for m in range( 2, n ):
            Hvec[n] += 2 * WC * A0**2 * MU[n-1,0,1] * MU[m-1,0,1] * vec[m]
        #Hvec[n] += 2 * WC * A0**2 * MU[n-1,0,1] * (np.sum( MU[:n-1,0,1] * vec[1:n] ) )

        # DIAGONAL ELEMENT
        Hvec[n] += (EGS - E[n-1,0] + E[n-1,1] - E0) * vec[n]
        Hvec[n] += (DSE_DIAG[n-1] - E0 ) * vec[n]

        # DIAGONAL-TO-RIGHT COLUMNS
        for m in range( n+1, N+1 ):
            Hvec[n] += 2 * WC * A0**2 * MU[n-1,0,1] * MU[m-1,0,1] * vec[m]
        #Hvec[n] += 2 * WC * A0**2 * MU[n-1,0,1] * (np.sum( MU[n:,0,1] * vec[n+1:-1] ) )
    
        # LAST COLUMN
        Hvec[n] += WC * A0 * MU[n-1,0,1] * vec[-1]
    
    # BOTTOM ROW
    Hvec[N+1] += WC * A0 * np.sum( MU[:,0,0] ) * vec[0] #<g0|g1> # TODO CHECK THIS
    Hvec[N+1] += WC * A0 * np.sum( MU[:,0,1] * vec[1:N+1] ) # TODO CHECK THIS
    Hvec[N+1] += (EGS+WC+DSE_GS-E0)*vec[-1]
    
    return Hvec/dH


@jit(nopython=True)
def get_H_vec_norm( N, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, vec ):
    Hvec         = np.zeros( (N+2) )

    # TOP ROW
    Hvec[0]     += (EGS+DSE_GS-E0)*vec[0] # <g0|g0>
    Hvec[0]     += np.sum( DSE_0n[:] * vec[1:N+1] ) # <g0|e0> # TODO CHECK THIS
    Hvec[0]     += WC * A0 * np.sum( MU[:,0,0] ) * vec[N+1] #<g0|g1> # TODO CHECK THIS

    ### BEGIN MIDDLE ROWS ###
    # FIRST COLUMN
    Hvec[1:N+1] += DSE_0n[:] * vec[0]

    # NON-FIRST, NON-LAST, NON-DIAGONAL COLUMNS
    Hvec[1:N+1] += 2 * WC * A0**2 * MU[:,0,1] * (np.sum( MU[:,0,1] * vec[1:N+1] ) )
    Hvec[1:N+1] -= 2 * WC * A0**2 * MU[:,0,1] * MU[:,0,1] * vec[1:N+1] # Remove diagonal ones

    # DIAGONAL ELEMENT
    Hvec[1:N+1] += (EGS - E[:,0] + E[:,1] - E0) * vec[1:N+1]
    Hvec[1:N+1] += (DSE_DIAG[:] - E0 ) * vec[1:N+1]
    
    # LAST COLUMN
    Hvec[1:N+1] += WC * A0 * MU[:,0,1] * vec[-1]

    ### END MIDDLE ROWS ###

    # BOTTOM ROW
    Hvec[N+1] += WC * A0 * np.sum( MU[:,0,0] ) * vec[0] #<g0|g1> # TODO CHECK THIS
    Hvec[N+1] += WC * A0 * np.sum( MU[:,0,1] * vec[1:N+1] ) # TODO CHECK THIS
    Hvec[N+1] += (EGS+WC+DSE_GS-E0)*vec[-1]
    
    return Hvec/dH

