import numpy as np
from matplotlib import pyplot as plt
import sys
import subprocess as sp


DATA_DIR = "data/"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

color_list = ["black", "red", "blue", "green", "orange", "purple", "cyan", "magenta"]

sys.path.insert(0, '../')
from main import Params
from chebyshev import get_coeffs



A0_LIST       = [0.01] # np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
DISORDER_LIST = ["rectangular", "gaussian", "lorentzian"]
SIG_E_LIST    = np.arange(0.0, 4.0+0.25, 0.25) # np.array([0.0, 0.5, 1.0, 2.0, 4.0]) # Fractions of A0

counter = 0
for A0 in A0_LIST:
    for MOL_DISORDER in DISORDER_LIST:
        for SIG_E_FAC in SIG_E_LIST:
            SIG_E = SIG_E_FAC * A0

            N_MOL     = 2   # Number of molecules
            #A0        = 0.05 # Light-matter coupling strength
            WC        = 1.0  # Cavity Average Transition Energy
            E0        = 1.0  # Molecular Average Transition Energy

            N_STOCHASTIC = 5000 # Number of stochastic realizations
            GAM    = 0.001 # A0 / 50 / np.sqrt(N_MOL)
            EMIN   = E0 - 8*A0 # np.max([2*A0, 3*GAM, 3*SIG_E]) # * np.sqrt(N_MOL)
            EMAX   = E0 + 8*A0 # np.max([2*A0, 3*GAM, 3*SIG_E]) # * np.sqrt(N_MOL)
            dH     = EMAX - EMIN
            print( "2*A0 = %1.3f  3*GAM = %1.4f  3*SIGE = %1.4f" % (2*A0, 3*GAM, 3*SIG_E) )
            print( "EMIN = %1.3f  EMAX = %1.3f  dH = %1.3f" % (EMIN, EMAX, dH) )
            #ratio1  = 40 # Keep factor constant -- These numbers are good for N_CHEB = 300
            ratio2  = dH / GAM # This should be larger than ratio1, this N_CHEB > 300
            N_CHEB  = 2000 # int(300 * ratio2 / ratio1) + 1  # Keep N_CHEB=300 per GAM=0.005 and dH=0.2
            EGRID   = np.linspace( E0 - 2*A0, E0 + 2*A0, 101 ) # np.linspace( EMIN, EMAX, 101 )
            dE      = EGRID[1] - EGRID[0]

            print( "Gamma  Parameter: %1.6f" % GAM )
            print( "N_CHEB Parameter: %d" % N_CHEB )
            print( "Ratio = dH/GAM: %1.2f" % (ratio2) )


            DOS_CHEB  = np.zeros( ( len(EGRID), 3 ), dtype=np.complex128)
            DOS_EXACT = np.zeros( ( len(EGRID), 3 ), dtype=np.complex128)

            params = Params(    batch_size=N_STOCHASTIC, N_STOCHASTIC=N_STOCHASTIC, # (N_STOCHASTIC // batch_size) should be an integer
                                N_MOL=N_MOL,
                                F_type="gaussian", # "gaussian" or "1_over_e" or "lorentzian"
                                N_CHEB=N_CHEB,
                                GAM=GAM,
                                EGRID=EGRID, # Get DOS for these energies
                                dH=dH, 
                                WC=WC,
                                MOL_DISORDER=MOL_DISORDER,
                                SIG_E=SIG_E,
                                E0=E0,
                                A0=A0, 
                                CAV_LOSS=None,
                                CAV_LOSS_TYPE=None, # None, "stochastic", or "non-hermitian"
                                P_type="lorentzian" ) # "gaussian" or "lorentzian"
            if ( counter == 0 ): 
                coeffs = get_coeffs( params )
            elif ( counter > 0 and np.allclose(EGRID,EGRID_old) == False ):
                coeffs = get_coeffs( params )
            params.coeffs = coeffs
            params.run()
            EGRID_old = EGRID
            counter += 1

            if ( params.F_type == "gaussian" or params.F_type == "lorentzian" ):
                DOS_CHEB[:,:]  = np.real( params.DOS )
            elif ( params.F_type == "1_over_e" ):
                DOS_CHEB[:,:]  = -1 * np.imag( params.DOS )
            DOS_CHEB = DOS_CHEB.real


            # Normalize the DOS
            TDOS          = np.sum( DOS_CHEB[:,0] ) * dE / (N_MOL + 1)
            MDOS          = np.sum( DOS_CHEB[:,1] ) * dE / (N_MOL)
            PDOS          = np.sum( DOS_CHEB[:,2] ) * dE
            # TDOS          = np.max( DOS_CHEB[:,0] )
            # MDOS          = np.max( DOS_CHEB[:,1] )
            # PDOS          = np.max( DOS_CHEB[:,2] )
            DOS_CHEB[:,0] = DOS_CHEB[:,0] / TDOS
            DOS_CHEB[:,1] = DOS_CHEB[:,1] / MDOS
            DOS_CHEB[:,2] = DOS_CHEB[:,2] / PDOS

            fig, ax1 = plt.subplots()
            # ax1.plot( (EGRID-E0)/A0, DOS_CHEB[:,0], "-",  c="black",  lw=8, alpha=0.5, label="TDOS" )
            # ax1.plot( (EGRID-E0)/A0, DOS_CHEB[:,1], "-",  c="red",    lw=2, alpha=1, label="MDOS" )
            ax1.plot( EGRID, DOS_CHEB[:,0], "-",  c="black",  lw=8, alpha=0.5, label="TDOS" )
            ax1.plot( EGRID, DOS_CHEB[:,1], "-",  c="red",    lw=2, alpha=1, label="MDOS" )
            ax1.set_ylabel('Total/Molecular Density of States', fontsize=15)

            ax2 = ax1.twinx()
            # ax2.plot( (EGRID-E0)/A0, DOS_CHEB[:,2], "--", c="blue",   lw=2, alpha=1, label="PDOS" )
            ax2.plot( EGRID, DOS_CHEB[:,2], "--", c="blue",   lw=2, alpha=1, label="PDOS" )
            ax2.set_ylabel('Transmission Spectra', fontsize=15)

            # plt.xlim(-2, 2)
            plt.xlim(EGRID[0], EGRID[-1])
            # ax1.set_xlabel("Energy, $\\frac{E}{A_0\\sqrt{N_\\mathrm{mol}}}$", fontsize=15)
            ax1.set_xlabel("Energy, $E$", fontsize=15)

            plt.legend()
            plt.tight_layout()
            plt.savefig( "DOS.png", dpi=300 )

            plt.savefig( "%s/DOS_A0_%1.3f_WC_%1.3f_dH_%1.3f_N_STOCHASTIC_%d_NCHEB_%1.3f_GAM_%1.3f_NMOL_%1.3f_MOL_DISORDER_%s_SIGE_%1.3f.jpg" % (DATA_DIR, A0, WC, dH, N_STOCHASTIC, N_CHEB, GAM, N_MOL, MOL_DISORDER, SIG_E), dpi=300 )
            np.savetxt( "%s/DOS_A0_%1.3f_WC_%1.3f_dH_%1.3f_N_STOCHASTIC_%d_NCHEB_%1.3f_GAM_%1.3f_NMOL_%1.3f_MOL_DISORDER_%s_SIGE_%1.3f.dat" % (DATA_DIR, A0, WC, dH, N_STOCHASTIC, N_CHEB, GAM, N_MOL, MOL_DISORDER, SIG_E), np.c_[(EGRID-E0)/A0/np.sqrt(N_MOL), DOS_CHEB[:,0], DOS_CHEB[:,1], DOS_CHEB[:,2]], fmt="%1.4f", header="Energy, TDOS, MDOS, PDOS" )