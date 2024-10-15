import numpy as np
from matplotlib import pyplot as plt
from numba import jit, prange
import subprocess as sp
from time import time
import multiprocessing as mp
import random
import sys

from Hamiltonian import get_H_vec_norm_Hermitian
from Exact import do_H_EXACT


def get_Globals():

    global NCPUS
    NCPUS     = 100 # Number of CPUs

    global N, EMOL, dEMOL_SHIFT, MU, MU_SCALED, WC, WC_MODES, A0, A0_MODES, A0_SCALED, EGS, E0, SIGE, SIGG, CAV_LOSS, NMODE
    N         =   int(sys.argv[1]) # Number of molecules
    A0        = float(sys.argv[2])
    WC        = float(sys.argv[3])
    CAV_LOSS  = float(sys.argv[4])
    SIGE      = float(sys.argv[5]) # Energy disorder (width / energy)
    SIGG      = float(sys.argv[6]) # Coupling disorder (width / dipole scaling), 0 < SIGG < 1 
    NMODE     = 11 # Modes for simulating cavity loss
  
    EMOL       = np.zeros( (N,2) )
    MU         = np.zeros( (N,2,2) )   
    MU[:,1,0] += np.ones( N )
    MU[:,0,1] += np.ones( N )
    MU[:,0,0] += np.ones( N )/100
    MU[:,1,1] += np.ones( N )/100    

    EMOL[:,1] += 1.0 + np.random.normal( 0.0, SIGE, size=N )

    # Dipole disorder
    if ( SIGG > 0 ):
        THETA = 2 * np.pi * random.random()
        scale = np.cos( THETA )
        MU   = np.array( [MU[A] * scale for A in range(N)] )

    global Mi, Pi
    Mi        = ( np.arange(1,N) )
    Pi        = ( np.arange(N,N+NMODE) )

    global GAM, N_STOCHASTIC, N_CHEB, dH, EMIN, EMAX
    GAM    = 0.010
    N_STOCHASTIC = 5000 # Num. random vectors. Conserving Error: N = 1000 and Nr = 5000 --> E. N = 10**6 and E --> Nr = 5
    N_CHEB       = 200  # Number of Chebyshev expansion coefficients
    EMIN   =  0.8 - SIGE # Must be lower than smallest eigenvalue
    EMAX   =  1.2 + SIGE # Must be higher than largest eigenvalue
    dH     = (EMAX-EMIN)

    global NPTS, EGRID
    NPTS    = 100 # Number of plotted points
    EGRID   = np.linspace(EMIN,EMAX,NPTS) # These are the plotted points

    global DATA_DIR
    DATA_DIR = "PLOTS_DATA"
    sp.call(f"mkdir -p {DATA_DIR}", shell=True)


    if ( CAV_LOSS == 0.0 ):
        WC_MODES    = np.ones( NMODE ) * WC
        A0_MODES    = WC_MODES*0
        A0_MODES[0] = A0 # Choose to couple only to the first mode -- Others are useless time-wasters
    else:
        WC_MODES   = np.linspace( WC-CAV_LOSS*5, WC+CAV_LOSS*5, NMODE)
        dwc        = WC_MODES[1] - WC_MODES[0]
        L          = dwc * A0 * (1/np.pi) * CAV_LOSS / ((WC - WC_MODES)**2 + CAV_LOSS**2)
        #L          = dwc * A0 * (1/CAV_LOSS) * np.sqrt(1/2/np.pi) * np.exp( -(WC - WC_MODES)**2/2/CAV_LOSS**2 )
        A0_MODES   = A0 * np.sqrt( L ) # Wang et al., J. Chem. Phys. 154, 104109 (2021); doi: 10.1063/5.0036283
        # print( A0, CAV_LOSS )
        # print( A0_MODES[:] / np.sqrt(L[:]) )
        # print( np.allclose(A0_MODES[:] / np.sqrt(L[:]), np.ones(NMODE)*A0) )
        # plt.plot( WC_MODES, A0_MODES )
        # plt.savefig("test.jpg",dpi=300)
        # exit()

    if ( WC-CAV_LOSS*5 < EMIN ):
        print("CAVITY MODE OUTSIDE ENERGY WINDOW !!!!!!")

    A0_SCALED = A0_MODES / np.sqrt(N)
    E0        = np.average(EMOL[:,1] - EMOL[:,0]) # CODE ONLY WORKS WHEN CENTER IS 0.0
    dEMOL_SHIFT = (EMOL[:,1] - EMOL[:,0]) - E0
    MU_SCALED  = MU # Be careful with the DSE term


def test_Chebyshev():
    for pt in range( NPTS ):
        print(pt+1, "of", NPTS)
        c_l     = np.zeros( N_CHEB, dtype=np.complex64 )
        c_l     = get_coeffs( EGRID[pt], E0, N_CHEB, c_l ).real # TODO IF WE ADD COMPLEX HAMILTONIAN, WE NEED TO CHANGE THIS
        plt.plot( np.arange(N_CHEB), np.real(c_l[:]) )
    plt.xlabel("Chebyshev Expansion Coefficient", fontsize=15)
    plt.ylabel("Value of Expansion Coefficient, $c_n(E)$", fontsize=15)
    plt.tight_layout()
    plt.savefig("EXPANSION_COEFFS_NC_%1.0f_GAM_%1.3f.jpg" % (N_CHEB,GAM), dpi=300)
    plt.clf()
    exit()

def do_STOCHASTIC_DOS():

    #test_Chebyshev() # Make the convergence plot for each energy without doing the stochastic trace

    DOS_FULL = np.zeros( (NPTS,3) ) # Total, Matter, Photonic
    for pt in range( NPTS ): # SHOULD WE PARALLIZE THIS ? THIS IS THE MOST TIME-CONSUMING PART
        print( "%1.0f of %1.0f" % (pt, NPTS) )
        c_l = np.zeros( N_CHEB, dtype=np.complex128 )
        #T0 = time()
        c_l     = get_coeffs( EGRID[pt], E0, N_CHEB, c_l ).real # TODO IF WE ADD COMPLEX HAMILTONIAN, WE NEED TO CHANGE THIS
        #print("Time to get coefficients: %1.3f s" % (time() - T0))
        T0 = time()
        DOS_FULL[pt,:] = do_Stochastic_Chebyshev( DOS_FULL[pt,:], c_l, N, dEMOL_SHIFT, pt, MU_SCALED, CAV_LOSS, WC, A0_SCALED, dH, E0 )
        print("Stochastic Time: %1.3f s" % (time() - T0) )

        # Plot the Chebyshev expansion coefficients -- check for convergence. N_CHEB ~ dH/GAM
        plt.plot( np.arange(N_CHEB), np.real(c_l[:]) )
        #plt.plot( np.arange(N_CHEB), np.imag(c_l[:]), "-", c="red" ) # This is zero... Should it be ?
    plt.xlabel("Chebyshev Expansion Coefficient", fontsize=15)
    plt.ylabel("Value of Expansion Coefficient, $c_n(E)$", fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/EXPANSION_COEFFS_NC_%1.0f_GAM_%1.3f.jpg" % (DATA_DIR,N_CHEB,GAM), dpi=300)
    plt.clf()

    return DOS_FULL / N_STOCHASTIC / 2 / np.pi


def do_STOCHASTIC_DOS_PARALLEL( pt ):
    #print( f"Working on grid point {pt} of {NPTS}" )
    DOS_FULL = np.zeros( 3 ) # Total, Matter, Photonic
    c_l      = np.zeros( N_CHEB, dtype=np.complex128 )
    c_l      = get_coeffs( EGRID[pt], E0, N_CHEB, c_l ).real
    DOS_FULL = do_Stochastic_Chebyshev( DOS_FULL, c_l, N, dEMOL_SHIFT, pt, MU_SCALED, CAV_LOSS, WC, A0_SCALED, dH, E0 )
    return DOS_FULL / N_STOCHASTIC / 2 / np.pi

@jit(nopython=True)
def do_Stochastic_Chebyshev( DOS, c_l, N, dEMOL_SHIFT, pt, MU_SCALED, CAV_LOSS, WC, A0_SCALED, dH, E0 ):
    for r in range( N_STOCHASTIC ):
        #print(r)
        #Tr = time()
        r_vec = np.array([ random.randint(0,1)*2.-1. for n in range(N+NMODE) ])
        v0, v1, DOS = get_T0_T1_vec( DOS, pt, c_l, N, dEMOL_SHIFT, MU_SCALED, CAV_LOSS, WC, A0_SCALED, dH, E0, r_vec )
        for l in range( 2, N_CHEB ):
            v0, v1, RN = get_vec_Tn_vec( N, dEMOL_SHIFT, MU_SCALED, CAV_LOSS, WC, A0_SCALED, dH, E0, r_vec, v0, v1 )
            DOS += c_l[l] * RN[:]
        #print( "\tr %s of %s  Time: %1.2f s  Remaining Time: %1.2f hrs"  % (r, N_STOCHASTIC, time() - Tr, (time() - Tr)*( N_STOCHASTIC - r-1 )/3600) )
    return DOS


@jit(nopython=True,fastmath=True)
def get_T0_T1_vec( DOS, pt, c_l, N, dEMOL_SHIFT, MU_SCALED, CAV_LOSS, WC, A0_SCALED, dH, E0, r_vec ):
    v0 = r_vec
    v1 = get_H_vec_norm_Hermitian( N, NMODE, dEMOL_SHIFT, MU_SCALED, WC_MODES, A0_SCALED, dH, v0 )
    DOS[0] += c_l[0] * np.dot(     np.conjugate(r_vec) ,     r_vec ) # <r|r>
    #DOS[1] += c_l[0] * np.dot( M * np.conjugate(r_vec) , M * r_vec ) # <M|r> * <r|M>
    #DOS[2] += c_l[0] * np.dot( P * np.conjugate(r_vec) , P * r_vec ) # <P|r> * <r|P>
    DOS[1] += c_l[0] * np.dot( np.conjugate(r_vec[Mi]) , r_vec[Mi] ) # <M|r> * <r|M>
    DOS[2] += c_l[0] * np.dot( np.conjugate(r_vec[Pi]) , r_vec[Pi] ) # <P|r> * <r|P>    
    DOS[0] += c_l[1] * np.dot(     np.conjugate(r_vec) ,     v1 )    # <r|T_1(H)|r>
    #DOS[1] += c_l[1] * np.dot( M * np.conjugate(r_vec) , M * v1 )    # <M|T_1(H)|r> * <r|M>
    #DOS[2] += c_l[1] * np.dot( P * np.conjugate(r_vec) , P * v1 )    # <P|T_1(H)|r> * <r|P>
    DOS[1] += c_l[1] * np.dot( np.conjugate(r_vec[Mi]) , v1[Mi] )    # <M|T_1(H)|r> * <r|M>
    DOS[2] += c_l[1] * np.dot( np.conjugate(r_vec[Pi]) , v1[Pi] )    # <P|T_1(H)|r> * <r|P>
    return v0, v1, DOS

@jit(nopython=True)
def get_vec_Tn_vec( N, dEMOL_SHIFT, MU_SCALED, CAV_LOSS, WC, A0_SCALED, dH, E0, r_vec, v0, v1 ):
    """
    Returns: <r|T_n(H)|r_n>
    Chebyshev Recurssion Relations:
    |v_0>   = |r>
    |v_1>   = H|v_0>
    \\vdots
    |v_n+1> = 2H|v_n> - |v_n-1>
    """

    #T0 = time()
    TMP = get_H_vec_norm_Hermitian( N, NMODE, dEMOL_SHIFT, MU_SCALED, WC_MODES, A0_SCALED, dH, v1 )
    v2  = 2 * TMP - v0 # Recurssion Relation
    #print("Time (H @ vec):", time() - T0)

    #T0 = time()
    R_T = np.dot(     np.conjugate(r_vec),     v2 )          # <r| * (T_n(H)|r>)
    #R_M = np.dot( M * np.conjugate(r_vec), M * v2 ) # <M| * (T_n(H)|r>) * <r|M>
    #R_P = np.dot( P * np.conjugate(r_vec), P * v2 ) # <P| * (T_n(H)|r>) * <r|P>
    R_M = np.dot( np.conjugate(r_vec[Mi]), v2[Mi] ) # <M| * (T_n(H)|r>) * <r|M>
    R_P = np.dot( np.conjugate(r_vec[Pi]), v2[Pi] ) # <P| * (T_n(H)|r>) * <r|P>
    #print("Time (Projection):", time() - T0)

    return v1, v2, np.array([R_T,R_M,R_P])

@jit(nopython=True,fastmath=True)
def get_coeffs( Ept,E0,N_CHEB,c_l ):
    theta    = np.linspace(0,2*np.pi,2000) + 0.j
    dth      = theta[1] - theta[0]
    F        = np.exp( -( dH*np.cos(theta) - (Ept-E0))**2 / 2 / GAM**2 ) + 0.j
    c_l[:]   =  F @ np.exp(1j * np.outer(theta,np.arange(N_CHEB))) * dth # Fourier Kernel
    #c_l[0]  *= 1 #/ np.sqrt(2 * np.pi) #/ GAM
    c_l[1:] *= 2 #/ np.sqrt(2 * np.pi) #/ GAM
    return c_l


def plot_DOS( EXACT, STOCHASTIC ):

    ### TOTAL DOS ###
    plt.plot( EGRID.real, np.abs(STOCHASTIC[:,0]), "o", c="black", label="STOCHASTIC" )
    if ( EXACT[0] is not None ):
        plt.plot( EGRID.real, np.abs(EXACT[0,:]), "-", lw=6, alpha=0.25, c="black", label="EXACT" )
        np.savetxt("%s/DOS_TOTAL_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT_ABS.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, np.abs(EXACT[0,:])])
        np.savetxt("%s/DOS_TOTAL_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, EXACT[0,:]])
    plt.legend()
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Density of States (Arb. Units)", fontsize=15)
    if ( N >= 1000 ):
        N_STRING = "".join( [s + ","*((j-2)%3==0 and j!=0 and j!=len(str(N))-1) for j,s in enumerate(str(N)[::-1]) ] )[::-1]
    else:
        N_STRING = str(N)
    #plt.title("$N_\mathrm{sites} = %s$   $N_\mathrm{random} = %1.0f$   $N_\mathrm{Chebyshev} = %1.0f$" % (N_STRING,N_STOCHASTIC,N_CHEB), fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/DOS_TOTAL_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), dpi=300)
    plt.clf()
    np.savetxt("%s/DOS_TOTAL_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_ABS.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, np.abs(STOCHASTIC[:,0])])
    np.savetxt("%s/DOS_TOTAL_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, STOCHASTIC[:,0]])


    ### MATTER DOS ###
    plt.plot( EGRID.real, np.abs(STOCHASTIC[:,1]), "o", c="black", label="STOCHASTIC" )
    #plt.plot( EGRID, STOCHASTIC[:,0], "o", c="red", label="STOCHASTIC" )
    if ( EXACT[0] is not None ):
        plt.plot( EGRID.real, np.abs(EXACT[1,:]), "-", lw=6, alpha=0.25, c="black", label="EXACT" )
        np.savetxt("%s/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT_ABS.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, np.abs(EXACT[1,:])])
        np.savetxt("%s/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, EXACT[1,:]])
    plt.legend()
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Density of States (Arb. Units)", fontsize=15)
    if ( N >= 1000 ):
        N_STRING = "".join( [s + ","*((j-2)%3==0 and j!=0 and j!=len(str(N))-1) for j,s in enumerate(str(N)[::-1]) ] )[::-1]
    else:
        N_STRING = str(N)
    #plt.title("$N_\mathrm{sites} = %s$   $N_\mathrm{random} = %1.0f$   $N_\mathrm{Chebyshev} = %1.0f$" % (N_STRING,N_STOCHASTIC,N_CHEB), fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), dpi=300)
    plt.clf()
    np.savetxt("%s/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_ABS.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, np.abs(STOCHASTIC[:,1])])
    np.savetxt("%s/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, STOCHASTIC[:,1]])



    ### PHOTON DOS ###
    plt.plot( EGRID.real, np.abs(STOCHASTIC[:,2]) / NMODE, "o", c="black", label="STOCHASTIC" )
    #plt.plot( EGRID, STOCHASTIC[:,0], "o", c="red", label="STOCHASTIC" )
    if ( EXACT[0] is not None ):
        plt.plot( EGRID.real, np.abs(EXACT[2,:]) , "-", lw=6, alpha=0.25, c="black", label="EXACT" )
        np.savetxt("%s/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT_ABS.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, np.abs(EXACT[2,:])])
        np.savetxt("%s/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, EXACT[2,:]])
    plt.legend()
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Density of States (Arb. Units)", fontsize=15)
    if ( N >= 1000 ):
        N_STRING = "".join( [s + ","*((j-2)%3==0 and j!=0 and j!=len(str(N))-1) for j,s in enumerate(str(N)[::-1]) ] )[::-1]
    else:
        N_STRING = str(N)
    #plt.title("$N_\mathrm{sites} = %s$   $N_\mathrm{random} = %1.0f$   $N_\mathrm{Chebyshev} = %1.0f$" % (N_STRING,N_STOCHASTIC,N_CHEB), fontsize=15)
    plt.tight_layout()
    plt.savefig("%s/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), dpi=300)
    plt.clf()
    np.savetxt("%s/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_ABS.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, np.abs(STOCHASTIC[:,2])])
    np.savetxt("%s/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (DATA_DIR,N,N_STOCHASTIC,N_CHEB,GAM,A0,WC,SIGE,SIGG,CAV_LOSS), np.c_[EGRID, STOCHASTIC[:,2]])


def main():
    get_Globals()
    DOS_EXACT_T, DOS_EXACT_M, DOS_EXACT_P = do_H_EXACT( N, NPTS, EMOL, MU, CAV_LOSS, WC_MODES, A0_SCALED, NMODE, EGRID, dH, E0, GAM )
    DOS_EXACT = np.array([DOS_EXACT_T, DOS_EXACT_M, DOS_EXACT_P])

    T0 = time()
    if ( NCPUS == 1 ):
        DOS_STOCHASTIC = do_STOCHASTIC_DOS()
    elif ( NCPUS >= 2 ):
        print (f"There will be {NCPUS} cores with {NPTS} grid points.")
        with mp.Pool(processes=NCPUS) as pool:
            DOS_STOCHASTIC = pool.map(do_STOCHASTIC_DOS_PARALLEL,np.arange(NPTS))
            DOS_STOCHASTIC = np.array( DOS_STOCHASTIC )
    print("Total Simulation Time: %1.3f" % (time() - T0))

    plot_DOS( DOS_EXACT, DOS_STOCHASTIC )


if ( __name__ == "__main__" ):
    main()
