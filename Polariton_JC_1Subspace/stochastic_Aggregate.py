import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.chebyshev import chebfit
from matplotlib import pyplot as plt
from numba import jit, prange
import subprocess as sp
from time import time

from Hamiltonian import get_H_vec_norm
from Exact import do_H_EXACT


def get_Globals():
    global N, E, MU, WC, A0, A0_SCALED, EGS, E0, SIGE
    N         = 2 # Number of molecules
    E         = np.zeros( (N,2) )
    MU        = np.zeros( (N,2,2) )
    SIGE      = 0.0
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

    global GAM, N_STOCHASTIC, N_CHEB, dH, EMIN, EMAX
    GAM    = 0.02
    N_STOCHASTIC = 100
    N_CHEB       = 250
    EMIN   = -0.2 # Must be lower than smallest eigenvalue
    EMAX   = 1.5 # Must be higher than largest eigenvalue
    dH     = (EMAX-EMIN)

    global NPTS, EGRID
    NPTS   = 200 # Number of plotted points
    E01_AVE = np.average( E[:,1] - E[:,0] ) # Average matter excitation
    EGRID  = np.linspace(E01_AVE-A0-5*GAM-2*SIGE,E01_AVE+A0+5*GAM+2*SIGE,NPTS) # These are the plotted points

    global DATA_DIR
    DATA_DIR = "PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)



def do_STOCHASTIC_DOS():

    DOS = np.zeros( (NPTS,3) ) # Total, Matter, Photonic
    for pt in range( NPTS ): # SHOULD WE PARALLIZE THIS ? THIS IS THE MOST TIME-CONSUMING PART
        print( "%1.0f of %1.0f" % (pt, NPTS) )
        c_l = np.zeros( N_CHEB, dtype=complex )
        T0 = time()
        c_l = get_coeffs( EGRID[pt], E0, N_CHEB, c_l ).real # TODO IF WE ADD COMPLEX HAMILTONIAN, WE NEED TO CHANGE THIS
        print("Time to get coefficients: %1.3f s" % (time() - T0))
        T0 = time()
        DOS = do_Stochastic_Chebyshev( DOS, c_l, N, EGS, E, pt, MU, WC, A0_SCALED, dH, E0, M, P )
        print("Stochastic Time: %1.3f s" % (time() - T0) )


        # Plot the Chebyshev expansion coefficients -- check for convergence. N_CHEB ~ dH/GAM
        plt.plot( np.arange(N_CHEB), c_l[:] )
    plt.xlabel("Chebyshev Expansion Coefficient", fontsize=15)
    plt.ylabel("Value of Expansion Coefficient, $c_n(E)$", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/EXPANSION_COEFFS_NC_%1.0f_GAM_%1.3f.jpg" % (DATA_DIR,N_CHEB,GAM), dpi=300)
    plt.clf()

    return DOS / N_STOCHASTIC / N


@jit(nopython=True)
def do_Stochastic_Chebyshev( DOS, c_l, N, EGS, E, pt, MU, WC, A0_SCALED, dH, E0, M, P ):
    for _ in range( N_STOCHASTIC ):
        r_vec = np.random.randint(0,high=2,size=N+2)*2. - 1
        v0 = r_vec
        v1 = get_H_vec_norm( N, EGS, E, MU, WC, A0_SCALED, dH, E0, r_vec )
        DOS[pt,0] += c_l[0] * np.dot( r_vec, r_vec ) # <r|r>
        DOS[pt,1] += c_l[0] * np.dot( M, r_vec ) * np.dot( r_vec, M ) # <M|r> * <r|M>
        DOS[pt,2] += c_l[0] * np.dot( P, r_vec ) * np.dot( r_vec, P ) # <P|r> * <r|P>
        DOS[pt,0] += c_l[1] * np.dot( r_vec, v1 ) # <r|T_1(H)|r>
        DOS[pt,1] += c_l[1] * np.dot( M, v1 ) * np.dot( r_vec, M ) # <M|T_1(H)|r> * <r|M>
        DOS[pt,2] += c_l[1] * np.dot( P, v1 ) * np.dot( r_vec, P ) # <P|T_1(H)|r> * <r|P>
        for l in range( 2, N_CHEB ):
            v0, v1, RN = get_vec_Tn_vec( N, EGS, E, MU, WC, A0_SCALED, dH, E0, r_vec, M, P, v0, v1 )
            DOS[pt,:] += c_l[l] * RN[:]
    return DOS



@jit(nopython=True)
def get_vec_Tn_vec( N, EGS, E, MU, WC, A0_SCALED, dH, E0, r_vec, M, P, v0, v1 ):
    """
    Returns: <r|T_n(H)|r>
    Chebyshev Recurssion Relations:
    |v_0>   = |v>
    |v_1>   = H|v_0>
    |v_n+1> = 2H|v_n> - |v_n-1>
    """

    v2 = 2 * get_H_vec_norm( N, EGS, E, MU, WC, A0_SCALED, dH, E0, v1 ) - v0 # Recurssion Relation
    
    DOS_T = np.dot( r_vec, v2 )                  # <r| * (T_n(H)|r>)
    DOS_M = np.dot( M, v2 ) * np.dot( r_vec, M ) # <M| * (T_n(H)|r>) * <r|M>
    DOS_P = np.dot( P, v2 ) * np.dot( r_vec, P ) # <P| * (T_n(H)|r>) * <r|P>

    return v1, v2, np.array([DOS_T,DOS_M,DOS_P])


@jit(nopython=True,parallel=True,fastmath=True)
def get_coeffs( E,E0,N_CHEB,c_l ):
    theta = np.linspace(0,2*np.pi,1000) + 0j
    dth   = theta[1] - theta[0]
    #for l in range( N_CHEB ):
    for l in prange( N_CHEB ): # Numba parallelization -- Does it improve anything ?
        #F       = np.exp( -( dH*np.cos(theta) - E)**2 / 2 / GAM**2 ) # This works if E0 = 0
        F       = np.exp( -( dH*np.cos(theta) - E)**2 / 2 / GAM**2 )
        c_l[l]  = np.dot( F, np.exp(1j * l * theta) ) * dth # Fourier Kernel
        c_l[l] *= (2 - (l == 0))
    return c_l

def plot_DOS( EXACT, STOCHASTIC ):

    ### TOTAL DOS ###
    plt.plot( EGRID, STOCHASTIC[:,0] / np.max(STOCHASTIC[:,0]), "o", c="red", label="STOCHASTIC" )
    if ( EXACT is not None ):
        plt.plot( EGRID, EXACT / np.max(EXACT), "-", c="black", label="EXACT" )
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
    plt.plot( EGRID, STOCHASTIC[:,1] / np.max(STOCHASTIC[:,1]), "o", c="red", label="STOCHASTIC" )
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
    plt.plot( EGRID, STOCHASTIC[:,2] / np.max(STOCHASTIC[:,2]), "o", c="red", label="STOCHASTIC" )
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
    DOS_EXACT      = do_H_EXACT( N, NPTS, EGS, E, MU, WC, A0_SCALED, EMIN, EMAX, EGRID, dH, E0, GAM )
    DOS_STOCHASTIC = do_STOCHASTIC_DOS()

    plot_DOS( DOS_EXACT, DOS_STOCHASTIC )


if ( __name__ == "__main__" ):
    main()
