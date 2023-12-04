import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.chebyshev import chebfit
from matplotlib import pyplot as plt
from numba import jit
import subprocess as sp

from Hamiltonian import get_H_vec_norm
from Exact import do_H_EXACT


def get_Globals():
    global N, E, MU, WC, A0, A0_SCALED, EGS, E0, SIGE
    N         = 100_000 # Number of molecules
    E         = np.zeros( (N,2) )
    MU        = np.zeros( (N,2,2) )
    SIGE      = 0.5
    E[:,1]    = 1.0 + np.random.normal( 0.0, SIGE, size=N )
    MU[:,0,1] = 1.0
    WC        = 1.0
    A0        = 0.2
    A0_SCALED = A0 / np.sqrt(N)
    EGS       = np.sum(E[:,0])
    E0        = 0.0 # CODE ONLY WORKS WHEN CENTER IS 0.0

    global M, P # Matter and Photon projectors
    M        = np.ones( (N+2) )
    M[0]     = 0 # Is GS matter, photon, or neither ?
    M[N+1]   = 0 # Excited photon is not matter
    P        = np.ones( (N+2) )
    P[0]     = 0 # GS is not photon.
    P[1:N+1] = np.zeros( (N) ) # Excited matter is not photon.


    global NPTS, GAM, dE, EMIN, EMAX
    NPTS   = 400
    GAM    = 0.02
    EMIN   = -0.2 # E0 - A0*15
    EMAX   = 1.5 # E0 + A0*15
    dE     = (EMAX-EMIN)/NPTS

    global N_STOCHASTIC, N_CHEB, dH
    N_STOCHASTIC = 250
    N_CHEB       = 400
    dH     = (EMAX-EMIN)

    global DATA_DIR
    DATA_DIR = "PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)



def do_STOCHASTIC_DOS():

    DOS = np.zeros( (NPTS,3) ) # Total, Matter, Photonic
    for pt in range( NPTS ):
        print( "%1.0f of %1.0f" % (pt, NPTS) )
        Ept = EMIN + pt*dE
        c_l = np.zeros( N_CHEB, dtype=complex )
        c_l = get_coeffs( Ept, E0, N_CHEB, c_l ).real # TODO IF WE ADD COMPLEX HAMILTONIAN, WE NEED TO CHANGE THIS
        for _ in range( N_STOCHASTIC ): # WE COULD PARALLELIZE THIS EASILY -- FOR LARGE DIMENSION, WE ONLY NEED A COUPLE...
            r_vec = np.random.randint(0,high=2,size=N+2)*2. - 1 # {-1,1} with N elements
            r_vec = r_vec
            v0 = r_vec
            v1 = get_H_vec_norm( N, EGS, E, MU, WC, A0_SCALED, dH, E0, r_vec )
            DOS[pt,0] += c_l[0] * np.dot( r_vec, r_vec ) # <r|r>
            DOS[pt,1] += c_l[0] * np.dot( M, r_vec ) * np.dot( r_vec, M ) # <M|r> * <r|M>
            DOS[pt,2] += c_l[0] * np.dot( P, r_vec ) * np.dot( r_vec, P ) # <P|r> * <r|P>
            DOS[pt,0] += c_l[1] * np.dot( r_vec, v1 ) # <r|T_1(H)|r>
            DOS[pt,1] += c_l[1] * np.dot( M, v1 ) * np.dot( r_vec, M ) # <M|T_1(H)|r> * <r|M>
            DOS[pt,2] += c_l[1] * np.dot( P, v1 ) * np.dot( r_vec, P ) # <P|T_1(H)|r> * <r|P>
            for l in range( 2, N_CHEB ):
                v0, v1, RN = get_vec_Tn_vec( N, EGS, E, MU, WC, A0_SCALED, dH, E0, r_vec, v0, v1, l )
                DOS[pt,:] += c_l[l] * RN[:]

        plt.plot( np.arange(N_CHEB), c_l[:] )
    plt.xlabel("Chebyshev Expansion Coefficient", fontsize=15)
    plt.ylabel("Value of Expansion Coefficient, $c_n(E)$", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/EXPANSION_COEFFS_NC_%1.0f_GAM_%1.3f.jpg" % (DATA_DIR,N_CHEB,GAM), dpi=300)
    plt.clf()


    return DOS / N_STOCHASTIC / N


@jit(nopython=True)
def get_vec_Tn_vec( N, EGS, E, MU, WC, A0_SCALED, dH, E0, r_vec, v0, v1, nmax ):
    """
    Returns: <r|T_n(H)|r>
    Chebyshev Recurssion Relations:
    |v_0>   = |v>
    |v_1>   = H|v_0>
    |v_n+1> = 2H|v_n> - |v_n-1>
    """
    v2 = 2 * get_H_vec_norm( N, EGS, E, MU, WC, A0_SCALED, dH, E0, v1 ) - v0 # Recurssion Relation


    # <r| * (T_n(H)|r>)
    # <M| * (T_n(H)|r>) * <r|M>
    # <P| * (T_n(H)|r>) * <r|P>
    RESULTS = np.array([ np.dot( r_vec, v2 ), \
                         np.dot( M, v2 ) * np.dot( r_vec, M ), \
                         np.dot( P, v2 ) * np.dot( r_vec, P ) ])

    return v1, v2, RESULTS


@jit(nopython=True)
def get_coeffs( E,E0,N_CHEB,c_l ):
    theta = np.linspace(0,2*np.pi,1000) + 0j
    dth   = theta[1] - theta[0]
    for l in range( N_CHEB ):
        #F       = np.exp( -( dH*np.cos(theta) - E)**2 / 2 / GAM**2 ) # This works if E0 = 0
        F       = np.exp( -( dH*np.cos(theta) - E)**2 / 2 / GAM**2 )
        c_l[l]  = np.dot( F, np.exp(1j * l * theta) ) * dth # Fourier Kernel
        c_l[l] *= (2 - (l == 0))
    return c_l

def plot_DOS( EXACT, STOCHASTIC ):
    ENERGY = np.linspace(EMIN,EMAX,NPTS)

    ### TOTAL DOS ###
    plt.plot( ENERGY, STOCHASTIC[:,0] / np.max(STOCHASTIC[:,0]), "o", c="red", label="STOCHASTIC" )
    if ( EXACT is not None ):
        plt.plot( ENERGY, EXACT / np.max(EXACT), "-", c="black", label="EXACT" )
    plt.legend()
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Density of States (Arb. Units)", fontsize=15)
    if ( N >= 1000 ):
        N_STRING = "".join( [s + ","*((j-2)%3==0 and j!=0 and j!=len(str(N))-1) for j,s in enumerate(str(N)[::-1]) ] )[::-1]
    else:
        N_STRING = str(N)
    plt.title("$N_\mathrm{sites} = %s$   $N_\mathrm{random} = %1.0f$   $N_\mathrm{Chebyshev} = %1.0f$" % (N_STRING,N_STOCHASTIC,N_CHEB), fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/DOS_TOTAL_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIG_%1.3f.jpg" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE), dpi=300)
    plt.clf()

    ### MATTER-PROJECTED DOS ###
    plt.plot( ENERGY, STOCHASTIC[:,1] / np.max(STOCHASTIC[:,1]), "o", c="red", label="STOCHASTIC" )
    plt.legend()
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Density of States (Arb. Units)", fontsize=15)
    if ( N >= 1000 ):
        N_STRING = "".join( [s + ","*((j-2)%3==0 and j!=0 and j!=len(str(N))-1) for j,s in enumerate(str(N)[::-1]) ] )[::-1]
    else:
        N_STRING = str(N)
    plt.title("$N_\mathrm{sites} = %s$   $N_\mathrm{random} = %1.0f$   $N_\mathrm{Chebyshev} = %1.0f$" % (N_STRING,N_STOCHASTIC,N_CHEB), fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIG_%1.3f.jpg" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE), dpi=300)
    plt.clf()


    ### PHOTON-PROJECTED DOS ###
    plt.plot( ENERGY, STOCHASTIC[:,2] / np.max(STOCHASTIC[:,2]), "o", c="red", label="STOCHASTIC" )
    plt.legend()
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Density of States (Arb. Units)", fontsize=15)
    if ( N >= 1000 ):
        N_STRING = "".join( [s + ","*((j-2)%3==0 and j!=0 and j!=len(str(N))-1) for j,s in enumerate(str(N)[::-1]) ] )[::-1]
    else:
        N_STRING = str(N)
    plt.title("$N_\mathrm{sites} = %s$   $N_\mathrm{random} = %1.0f$   $N_\mathrm{Chebyshev} = %1.0f$" % (N_STRING,N_STOCHASTIC,N_CHEB), fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIG_%1.3f.jpg" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE), dpi=300)
    plt.clf()

def main():
    get_Globals()
    DOS_EXACT      = do_H_EXACT( N, NPTS, EGS, E, MU, WC, A0_SCALED, EMIN, EMAX, dE, dH, E0, GAM )
    DOS_STOCHASTIC = do_STOCHASTIC_DOS()

    plot_DOS( DOS_EXACT, DOS_STOCHASTIC )


if ( __name__ == "__main__" ):
    main()
