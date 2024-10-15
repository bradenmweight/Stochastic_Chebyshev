import numpy as np

from Hamiltonian import get_H_nm_norm

def do_H_EXACT( N, NEL, NPTS, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, EMIN, EMAX, EGRID, dH, E0, GAM ):
    SIZE = N**2 * 8 * 10**-9
    if ( SIZE > 0.1 ): # in GB
        print("Matrix too large for exact solution. %1.3f GB > 10 GB" % SIZE)
        return None
    H = np.zeros( (N*(NEL-1)+2,N*(NEL-1)+2) )
    for n in range( N*(NEL-1)+2 ):
        for m in range( N*(NEL-1)+2 ):
            H[n,m] = get_H_nm_norm( N, EL, EGS, E, MU, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, n, m )
    Ei, Ui = np.linalg.eigh( H )
    Ei = Ei * dH + E0

    print("H:\n", H*dH )
    print("Eigs:\n", Ei)

    DOS  = np.zeros( NPTS )
    for pt in range( NPTS ):
        DOS[pt] = np.sum( np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
    #return DOS / GAM / np.sqrt(2 * np.pi)
    return DOS 