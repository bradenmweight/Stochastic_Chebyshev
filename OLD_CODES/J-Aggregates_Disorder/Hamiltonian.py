import numpy as np
from numba import jit

@jit(nopython=True)
def get_H_nm( E0,J,n,m,N,ESIG ):
    if ( n == m ):
        return ESIG[n]
    elif ( n == m-1 or n == m+1 ):
        return J
    elif ( n == N-1 and m == 0 ):
       return J
    elif ( m == N-1 and n == 0 ):
       return J
    else:
        return 0.0

@jit(nopython=True)
def get_H_nm_norm( E0,J,n,m,N,dH,ESIG ):
    if ( n == m ):
        return (ESIG[n] - E0)/dH
    elif ( n == m-1 or n == m+1 ):
        return J/dH
    elif ( n == N-1 and m == 0 ):
       return J/dH
    elif ( m == N-1 and n == 0 ):
       return J/dH
    else:
        return 0.0

@jit(nopython=True)
def get_H_vec_norm( E0,J,vec,dH,N,ESIG ):

    ### OPTION 1 ###
    # Hvec = np.zeros( (N) )
    # for n in range( N ):
    #     for m in range( N ):
    #         if ( n == m ):
    #             Hvec[n] += (E0 - E0)/dH * vec[m]
    #         elif ( n == m-1 or n == m+1 ):
    #             Hvec[n] += J/dH * vec[m]
    #         elif ( n == N-1 and m == 0 ):
    #             Hvec[n] += J/dH * vec[m]
    #         elif ( m == N-1 and n == 0 ):
    #             Hvec[n] += J/dH * vec[m]
    
    ### OPTION 2 ###
    # Hvec = np.zeros( (N) )
    # for n in range( N-1 ):
    #     Hvec[n] = J/dH * (vec[n-1] + vec[n+1])
    # Hvec[N-1] = J/dH * (vec[N-2] + vec[0])

    ### OPTION 3 ###
    Hvec = ESIG * vec + J/dH * ( np.roll( vec,-1 ) + np.roll( vec,1 ) )
    return Hvec







