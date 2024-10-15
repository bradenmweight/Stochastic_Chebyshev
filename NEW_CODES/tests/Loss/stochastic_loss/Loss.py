import numpy as np
from matplotlib import pyplot as plt
import sys

color_list = ["black", "red", "blue", "green", "orange", "purple", "cyan", "magenta"]

sys.path.insert(0, '../../../')
from main import Params

A0        = 0.10
WC        = 1.00 # Cavity Average Transition Energy
E0        = 1.0 # Molecular Average Transition Energy
LOSS_LIST = [0.0, 0.1, 0.2, 0.4, 0.8]

N_MOL = 1
EMIN  = 0.5 #np.min([WC,E0])-0.5
EMAX  = 1.5 #np.max([WC,E0])+0.5
dH    = EMAX - EMIN
EGRID = np.linspace( EMIN, EMAX, 200 )
dE    = EGRID[1] - EGRID[0]

for CAVi,CAV in enumerate( LOSS_LIST ):
    print( "Working on cavity loss rate: %1.3f a.u." % CAV )
    print("EMIN = %1.2f  EMAX = %1.2f" % ( EGRID[0], EGRID[-1] ) )
    print( "dH = %1.3f" % dH )
    print( "dE = %1.3f" % dE )
    
    DOS_CHEB         = np.zeros( ( len(EGRID), 3 ), dtype=np.complex128)
    DOS_EXACT        = np.zeros( ( len(EGRID), 3 ), dtype=np.complex128)
    for Ei,Ept in enumerate(EGRID):
        print( "Working on grid point %d of %d" % (Ei, len(EGRID)) )
        params = Params(    batch_size=5000, N_STOCHASTIC=5000, # (N_STOCHASTIC // batch_size) should be an integer
                            N_MOL=N_MOL,
                            F_type="lorentzian", # "gaussian" or "1_over_e" or "lorentzian"
                            N_CHEB=350, 
                            dH=dH, 
                            WC=WC,
                            E0=E0,
                            A0=A0, 
                            Ept=Ept, 
                            P_type="lorentzian", # "gaussian" or "lorentzian"
                            CAV_LOSS=CAV,
                            CAV_LOSS_TYPE="stochastic" ) # "stochastic" or "non-hermitian"
        params.run()
        DOS_CHEB[Ei]         = params.DOS
        DOS_EXACT[Ei]        = params.DOS_EXACT

    if ( params.F_type == "gaussian" ):
        DOS_CHEB  = np.real( DOS_CHEB )
        DOS_EXACT = np.real( DOS_EXACT )
    elif ( params.F_type == "1_over_e" ):
        DOS_CHEB  = -1 * np.imag( DOS_CHEB )
        DOS_EXACT = -1 * np.imag( DOS_EXACT )
    elif ( params.F_type == "lorentzian" ):
        DOS_CHEB  = np.real( DOS_CHEB )
        DOS_EXACT = np.real( DOS_EXACT )


    # Normalize the DOS
    TDOS          = np.sum( DOS_EXACT[:,0] ) * dE / (N_MOL + 1)
    MDOS          = np.sum( DOS_EXACT[:,1] ) * dE / (N_MOL)
    PDOS          = np.sum( DOS_EXACT[:,2] ) * dE
    DOS_EXACT[:,0] = DOS_EXACT[:,0] / TDOS
    DOS_EXACT[:,1] = DOS_EXACT[:,1] / MDOS
    DOS_EXACT[:,2] = DOS_EXACT[:,2] / PDOS

    TDOS          = np.sum( DOS_CHEB[:,0] ) * dE / (N_MOL + 1)
    MDOS          = np.sum( DOS_CHEB[:,1] ) * dE / (N_MOL)
    PDOS          = np.sum( DOS_CHEB[:,2] ) * dE
    DOS_CHEB[:,0] = DOS_CHEB[:,0] / TDOS
    DOS_CHEB[:,1] = DOS_CHEB[:,1] / MDOS
    DOS_CHEB[:,2] = DOS_CHEB[:,2] / PDOS

    #plt.plot( EGRID, DOS_EXACT[:,1],          "-", lw=6, alpha=0.5, c="black" )
    #plt.plot( EGRID, DOS_EXACT[:,2],          "-", lw=6, alpha=0.5, c="red" )
    #plt.plot( EGRID, DOS_CHEB[:,1], "-", lw=2, c='black', label="$\\frac{\\gamma}{A_0}$ = %1.3f" % (CAV/A0) )
    plt.plot( EGRID, DOS_CHEB[:,1], "-", lw=6, alpha=0.4, c=color_list[CAVi], label="$\\frac{\\gamma}{A_0}$ = %1.3f" % (CAV/A0) )
    plt.plot( EGRID, DOS_CHEB[:,2], "-",lw=2, c=color_list[CAVi] )

    plt.xlim(EMIN, EMAX)
    plt.title("$A_0$ = %1.2f a.u" % A0, fontsize=15)
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Density of States", fontsize=15)

    plt.legend()
    plt.savefig( "DOS.png", dpi=300 )