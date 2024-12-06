import numpy as np
from matplotlib import pyplot as plt
import sys
import subprocess as sp

DATA_DIR = "data/"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

color_list = ["black", "red", "blue", "green", "orange", "purple", "cyan", "magenta"]

sys.path.insert(0, '../../../')
from main import Params

A0        = 0.05
WC        = 1.0 # Cavity Average Transition Energy
E0        = 1.0 # Molecular Average Transition Energy
LOSS_LIST = [0.0, 0.05, 0.1, 0.2, 0.4]

N_CHEB = 1000
GAM    = 0.01

N_MOL = 1
EMIN  = E0 - 2*A0*np.sqrt(N_MOL)
EMAX  = E0 + 2*A0*np.sqrt(N_MOL)
dH    = 2.0 # EMAX - EMIN
EGRID = np.linspace( EMIN, EMAX, 200 )
dE    = EGRID[1] - EGRID[0]

DOS_CHEB  = np.zeros( ( len(LOSS_LIST), len(EGRID), 3 ), dtype=np.complex128)
DOS_EXACT = np.zeros( ( len(LOSS_LIST), len(EGRID), 3 ), dtype=np.complex128)

for CAVi,CAV in enumerate( LOSS_LIST ):
    print( "Working on cavity loss rate: %1.3f a.u." % CAV )
    print("EMIN = %1.2f  EMAX = %1.2f" % ( EGRID[0], EGRID[-1] ) )
    print( "dH = %1.3f" % dH )
    print( "dE = %1.3f" % dE )
    

    for Ei,Ept in enumerate(EGRID):
        print( "Working on grid point %d of %d" % (Ei, len(EGRID)) )
        params = Params(    batch_size=100000, N_STOCHASTIC=100000, # (N_STOCHASTIC // batch_size) should be an integer
                            N_MOL=N_MOL,
                            F_type="1_over_e", # "gaussian" or "1_over_e" or "lorentzian"
                            N_CHEB=N_CHEB,
                            GAM=GAM,
                            dH=dH, 
                            WC=WC,
                            E0=E0,
                            A0=A0, 
                            Ept=Ept, 
                            CAV_LOSS=CAV,
                            CAV_LOSS_TYPE="stochastic", # "stochastic" or "non-hermitian"
                            P_type="lorentzian" ) # "gaussian" or "lorentzian"
        params.run()

        if ( params.F_type == "gaussian" or params.F_type == "lorentzian" ):
            DOS_CHEB[CAVi,Ei,:]  = np.real( params.DOS )
        elif ( params.F_type == "1_over_e" ):
            DOS_CHEB[CAVi,Ei,:]  = -1 * np.imag( params.DOS )

    # Normalize the DOS
    TDOS          = np.sum( DOS_CHEB[CAVi,:,0] ) * dE / (N_MOL + 1)
    MDOS          = np.sum( DOS_CHEB[CAVi,:,1] ) * dE / (N_MOL)
    PDOS          = np.sum( DOS_CHEB[CAVi,:,2] ) * dE
    DOS_CHEB[CAVi,:,0] = DOS_CHEB[CAVi,:,0] / TDOS
    DOS_CHEB[CAVi,:,1] = DOS_CHEB[CAVi,:,1] / MDOS
    DOS_CHEB[CAVi,:,2] = DOS_CHEB[CAVi,:,2] / PDOS

    plt.plot( (EGRID-E0)/A0/np.sqrt(N_MOL), DOS_CHEB[CAVi,:,1], "-", lw=2, alpha=1, c=color_list[CAVi], label="$\\frac{\\Gamma}{A_0}$ = %1.3f" % (CAV/A0) )
    plt.plot( (EGRID-E0)/A0/np.sqrt(N_MOL), DOS_CHEB[CAVi,:,2], "--", lw=2, alpha=1, c=color_list[CAVi] )

    plt.xlim(-2, 2)
    plt.title("$A_0$ = %1.2f a.u" % A0, fontsize=15)
    plt.xlabel("Energy, $\\frac{E}{A_0\\sqrt{N_\\mathrm{mol}}}$", fontsize=15)
    plt.ylabel("Molecular Density of States", fontsize=15)

    plt.legend()
    plt.tight_layout()
    plt.savefig( "DOS.png", dpi=300 )

    np.savetxt( "%s/DOS_A0_%1.3f_WC_%1.3f_dH_%1.3f_NCHEB_%1.3f_GAM_%1.3f_CAVLOSS_%1.3f.dat" % (DATA_DIR, A0, WC, dH, N_CHEB, GAM, CAV), np.c_[(EGRID-E0)/A0/np.sqrt(N_MOL), DOS_CHEB[CAVi,:,0], DOS_CHEB[CAVi,:,1], DOS_CHEB[CAVi,:,2]], fmt="%1.4f", header="Energy, TDOS, MDOS, PDOS" )