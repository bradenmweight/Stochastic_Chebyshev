import numpy as np

from Hamiltonian import get_H_nm, get_H_nm_norm

def do_H_EXACT( N, NPTS, E0, GAM, J, EMIN, EMAX, dE, dH ):
    SIZE = N**2 * 8 * 10**-9
    if ( SIZE > 0.1 ): # in GB
        print("Matrix too large for exact solution. %1.3f GB > 10 GB" % SIZE)
        return None
    H = np.zeros( (N,N) )
    for n in range( N ):
        for m in range( N ):
            #H[n,m] = get_H_nm( E0,J,n,m,N )
            H[n,m] = get_H_nm_norm( E0,J,n,m,N,dH )
    Ei, Ui = np.linalg.eigh( H )
    #print(Ei)
    #print(Ei * dH + E0)
    Ei = Ei * dH + E0

    DOS  = np.zeros( NPTS )
    for pt in range( NPTS ):
        E       = EMIN + pt*dE
        DOS[pt] = np.sum( np.exp( -(E - Ei[:])**2 / 2 / GAM**2 ) )
    #return DOS / GAM / np.sqrt(2 * np.pi)
    return DOS 