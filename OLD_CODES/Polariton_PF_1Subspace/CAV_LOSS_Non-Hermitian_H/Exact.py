import numpy as np

from Hamiltonian import get_H_nm_norm

def do_H_EXACT( N, NPTS, EGS, E, MU, CAV_LOSS, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, EMIN, EMAX, EGRID, dH, E0, GAM ):
    SIZE = N**2 * 8 * 10**-9
    if ( SIZE > 0.1 ): # in GB
        print("Matrix too large for exact solution. %1.3f GB > 10 GB" % SIZE)
        return None, None, None
    H = np.zeros( (N+2,N+2), dtype=complex )
    for n in range( N+2 ):
        for m in range( N+2 ):
            H[n,m] = get_H_nm_norm( N, EGS, E, MU, CAV_LOSS, MU2_AA, DSE_GS, DSE_DIAG, DSE_0n, WC, A0, dH, E0, n, m )
    Ei, Ui = np.linalg.eig( H ) # np.linalg.eigh( H )
    idx    = Ei.argsort()   
    Ei     = Ei[idx]
    Ui     = Ui[:,idx]
    Ei     = Ei * dH + E0
    print("H:\n", H*dH )
    #print("H:\n", H*dH - (H*dH)[0,0] * np.identity(N+2) )
    print("Eigs:\n", Ei)


    N_OP   = np.zeros( (N+2,N+2) )
    N_OP[-1,-1] = 1
    E_OP   = np.identity( (N+2) )
    E_OP[0,0] = 0
    E_OP[-1,-1] = 0
    PHOT = np.einsum( "aj,ab,bj->j", Ui.real, N_OP, Ui.real )
    ELEC = np.einsum( "aj,ab,bj->j", Ui.real, E_OP, Ui.real )

    DOS_T  = np.zeros( NPTS, dtype=np.complex64 )
    DOS_M  = np.zeros( NPTS, dtype=np.complex64 )
    DOS_P  = np.zeros( NPTS, dtype=np.complex64 )
    for pt in range( NPTS ):
        DOS_T[pt] = np.sum( np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
        DOS_M[pt] = np.sum( ELEC[:] * np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
        DOS_P[pt] = np.sum( PHOT[:] * np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
    return DOS_T, DOS_M, DOS_P