import numpy as np

from Hamiltonian import get_H_nm_norm, get_H_nm_norm_TRANSITION

def do_H_EXACT( N, NPTS, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, EMIN, EMAX, EGRID, dH, E0, GAM ):
    SIZE = N**2 * 8 * 10**-9
    if ( SIZE > 0.1 ): # in GB
        print("Matrix too large for exact solution. %1.3f GB > 10 GB" % SIZE)
        return None
    H = np.zeros( (N+2,N+2) )
    for n in range( N+2 ):
        for m in range( N+2 ):
            H[n,m] = get_H_nm_norm( N, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, n, m )
            #H[n,m] = get_H_nm_norm_TRANSITION( N, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, n, m )
    Ei, Ui = np.linalg.eigh( H )
    Ei = Ei * dH + E0

    print("H:\n", H*dH )
    #print("H:\n", H*dH - (H*dH)[0,0] * np.identity(N+2) )
    print("Eigs:\n", Ei)

    DOS  = np.zeros( NPTS )
    for pt in range( NPTS ):
        DOS[pt] = np.sum( np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
    #return DOS / GAM / np.sqrt(2 * np.pi)
    return DOS 