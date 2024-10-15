import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from numpy.polynomial.chebyshev import chebfit
from matplotlib import pyplot as plt
from numba import jit
import subprocess as sp

from Hamiltonian import get_H_vec_norm
from Exact import do_H_EXACT


def get_Globals():
    global N, E0, J, NPTS, GAM, dE, EMIN, EMAX, dH
    N    = 100_000
    E0   = 0.0 # CODE ONLY WORKS WHEN CENTER IS 0.0
    J    = 0.1
    NPTS = 50
    GAM  = 0.05
    EMIN = E0 - J*5
    EMAX = E0 + J*5
    dE   = (EMAX-EMIN)/NPTS
    dH   = (EMAX-EMIN)


    global N_STOCHASTIC, N_CHEB
    N_STOCHASTIC = 100
    N_CHEB       = 100

    global DATA_DIR
    DATA_DIR = "PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)



def do_STOCHASTIC_DOS():

    DOS = np.zeros( NPTS )
    for pt in range( NPTS ):
        print( "%1.0f of %1.0f" % (pt, NPTS) )
        E   = EMIN + pt*dE
        c_l = np.zeros( N_CHEB, dtype=complex )
        c_l = get_coeffs( E, J, N_CHEB, c_l ).real
        for _ in range( N_STOCHASTIC ): # WE COULD PARALLELIZE THIS EASILY
            ###r_vec = np.random.random(size=N)*2 - 1 # (-1,1) with N elements
            ###r_vec = np.random.random(size=N) # (0,1) with N elements
            r_vec = np.random.randint(0,high=2,size=N)*2. - 1 # {-1,1} with N elements
            v0 = r_vec
            v1 = get_H_vec_norm( E0,J,r_vec,dH,N )
            DOS[pt] += c_l[0] * np.dot( r_vec, r_vec )
            DOS[pt] += c_l[1] * np.dot( r_vec, v1 )
            for l in range( 2, N_CHEB ):
                v0, v1, RN = get_vec_Tn_vec( l, r_vec, N, v0, v1 )
                DOS[pt] += c_l[l] * RN

        plt.plot( np.arange(N_CHEB), c_l[:] )
    plt.xlabel("Chebyshev Expansion Coefficient", fontsize=15)
    plt.ylabel("Value of Expansion Coefficient, $c_n(E)$", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/EXPANSION_COEFFS_NC_%1.0f_GAM_%1.3f.jpg" % (DATA_DIR,N_CHEB,GAM), dpi=300)
    plt.clf()


    return DOS / N_STOCHASTIC / N







@jit(nopython=True)
def get_vec_Tn_vec( nmax, r_vec, N, v0, v1 ):
    """
    Returns: <r|T_n(H)|r>
    Chebyshev Recurssion Relations:
    |v_0>   = |v>
    |v_1>   = H|v_0>
    |v_n+1> = 2H|v_n> - |v_n-1>
    """
    v2 = 2 * get_H_vec_norm( E0,J,v1,dH,N ) - v0 # Recurssion Relation
    return v1, v2, np.dot( r_vec, v2 ) # <r| * (T_n(H)|r>)

@jit(nopython=True)
def get_coeffs( E, J, N_CHEB, c_l ):
    theta = np.linspace(0,2*np.pi,1000) + 0j
    dth   = theta[1] - theta[0]
    for l in range( N_CHEB ):
        #F       = np.exp( -(dH*np.cos(theta) - E0 - E)**2 / 2 / GAM**2 )
        #F       = np.exp( -( np.cos(theta) - (E + E0/2)/dH)**2 / 2 / GAM**2 )
        F       = np.exp( -( dH*np.cos(theta) - E)**2 / 2 / GAM**2 )
        c_l[l]  = np.dot( F, np.exp(1j * l * theta) ) * dth # Fourier Kernel
        c_l[l] *= (2 - (l == 0))
    return c_l
#Ei * dH + E0

def plot_DOS( EXACT, STOCHASTIC ):
    ENERGY = np.linspace(EMIN,EMAX,NPTS)
    plt.plot( ENERGY, STOCHASTIC / np.max(STOCHASTIC), "o", c="red", label="STOCHASTIC" )
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
    plt.savefig("%s/DOS_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_J_%1.3f.jpg" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,J), dpi=300)
    plt.clf()

def main():
    get_Globals()
    DOS_EXACT      = do_H_EXACT( N, NPTS, E0, GAM, J, EMIN, EMAX, dE, dH )
    DOS_STOCHASTIC = do_STOCHASTIC_DOS()

    plot_DOS( DOS_EXACT, DOS_STOCHASTIC )


if ( __name__ == "__main__" ):
    main()
