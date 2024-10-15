import numpy as np

from Hamiltonian import get_H_nm_norm

def do_H_EXACT( N, NPTS, EMOL, MU, CAV_LOSS, WC_MODES, A0_SCALED, NMODE, EGRID, dH, E0, GAM ):
    SIZE = N**2 * 8 * 10**-9
    if ( SIZE > 0.1 ): # in GB
        print("Matrix too large for exact solution. %1.3f GB > 10 GB" % SIZE)
        return None, None, None
    H = np.zeros( (N+NMODE,N+NMODE) )
    for n in range( N+NMODE ):
        for m in range( N+NMODE ):
            H[n,m] = get_H_nm_norm( N, EMOL, MU, CAV_LOSS, WC_MODES, A0_SCALED, NMODE, dH, E0, n, m )
    Ei, Ui = np.linalg.eigh( H ) # np.linalg.eigh( H )
    Ei     = Ei * dH + E0
    print("H:\n", H*dH )
    #print("H:\n", H*dH - (H*dH)[0,0] * np.identity(N+2) )
    print("Eigs:\n", Ei)

    N_OP   = np.zeros( (N+NMODE,N+NMODE) )
    photon_inds = ( np.arange(N,N+NMODE), np.arange(N,N+NMODE) )
    N_OP[photon_inds] = 1
    E_OP   = np.identity( (N+NMODE) )
    E_OP[photon_inds] = 0
    PHOT = np.einsum( "aj,ab,bj->j", np.conjugate(Ui), N_OP, Ui )
    ELEC = np.einsum( "aj,ab,bj->j", np.conjugate(Ui), E_OP, Ui )

    DOS_T  = np.zeros( NPTS )
    DOS_M  = np.zeros( NPTS )
    DOS_P  = np.zeros( NPTS )
    for pt in range( NPTS ):
        DOS_T[pt] = np.sum( 1.00000 * np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
        DOS_M[pt] = np.sum( ELEC[:] * np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
        DOS_P[pt] = np.sum( PHOT[:] * np.exp( -(EGRID[pt] - Ei[:])**2 / 2 / GAM**2 ) )
    return DOS_T, DOS_M, DOS_P