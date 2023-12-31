import numpy as np
from numba import jit, prange

"""
N-molecule, Single-mode JC Hamiltonian
H_PF            = H_el + H_ph + H_el-ph (no RWs)
<g,g,0|H|g,g,0> = sum(E)
<e,g,0|H|e,g,0> = <g,g,0|H|g,g,0> - E[0,0] + E[0,1]
<g,e,0|H|g,e,0> = <g,g,0|H|g,g,0> - E[1,0] + E[1,1]
<g,g,1|H|g,g,1> = <g,g,0|H|g,g,0> +  wc
<g,g,1|H|e,g,0> = wc A0 <g,g|mu|e,g>
<g,g,1|H|g,e,0> = wc A0 <g,g|mu|g,e>
"""

@jit(nopython=True,fastmath=True)
def get_H_nm( N, EGS, E, MU, WC, A0, dH, E0, n, m ):
    if ( n == m ):
        if ( n == 0 ):
            return EGS
        elif ( n >= 1 and n < N+1 ):
            return EGS - E[n-1,0] + E[n-1,1]
        elif (n == N+1):
            return EGS + WC
    elif ( n == N+1 and m >= 1 and m < N+1 ):
        return WC * A0 * MU[m-1,0,1]
    elif ( m == N+1 and n >= 1 and n < N+1 ):
        return WC * A0 * MU[n-1,0,1]
    else:
        return 0.0

@jit(nopython=True,fastmath=True)
def get_H_nm_norm( N, EGS, E, MU, WC, A0, dH, E0, n, m ):
    if ( n == m ):
        if ( n == 0 ):
            return (EGS-E0)/dH
        elif ( n >= 1 and n < N+1 ):
            return (EGS - E[n-1,0] + E[n-1,1] - E0)/dH
        else:
            return (EGS + WC - E0)/dH
    elif ( n == N+1 and m >= 1 and m < N+1 ):
        return WC * A0 * MU[m-1,0,1]/dH
    elif ( m == N+1 and n >= 1 and n < N+1 ):
        return WC * A0 * MU[n-1,0,1]/dH
    else:
        return 0.0

@jit(nopython=True,fastmath=True)
def get_H_vec( N, EGS, E, MU, WC, A0, dH, E0 ):
    Hvec        = np.zeros( (N+2) )
    Hvec[0]     = EGS*vec[0]
    Hvec[1:N+1] = EGS - E[:,0] + E[:,1] + WC*A0*MU[:,0,1]*vec[-1]
    Hvec[N+1]   = np.sum( WC*A0*MU[:,0,1]*vec[1:N+1] ) + WC*vec[-1]
    return Hvec

@jit(nopython=True,fastmath=True)
def get_H_vec_norm( N, EGS, E, MU, WC, A0, dH, E0, vec ):
    Hvec        = np.zeros( (N+2) )
    Hvec[0]     = (EGS-E0)*vec[0]
    Hvec[1:N+1] = (EGS - E[:,0] + E[:,1] - E0)*vec[1:N+1] + WC*A0*MU[:,0,1]*vec[-1]
    Hvec[N+1]   = WC*A0*np.sum(MU[:,0,1]*vec[1:N+1]) + (EGS+WC-E0)*vec[-1]
    return Hvec/dH


