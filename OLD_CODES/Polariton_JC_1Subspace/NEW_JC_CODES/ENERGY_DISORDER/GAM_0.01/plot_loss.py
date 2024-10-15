import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

DATA_DIR = "PLOTS_DATA" # "PLOTS_DATA__N_10_5"
EXACT = True # True or False

NR        = 5000
A0_LIST   = np.array([0.05])
SIGE_LIST = np.arange( 0.0, 0.26, 0.02 ) # np.arange( 0.0, 0.26, 0.02 ) # np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
CAVL_LIST = np.arange( 0.0, 0.26, 0.02 ) # np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05]) # np.arange( 0.0, 0.26, 0.02 )

NC   = 200
N    = 1000 # 1000 # 100000
GAM  = 0.01
WC   = 1.0
SIGG = 0.0
NPTS = 100

color_list = ["black", "red", "blue", "green", "orange"]
EGRID      = np.linspace( 0.9, 1.1, 1001 )
DOS        = np.zeros( (len(A0_LIST), len(SIGE_LIST), len(CAVL_LIST), len(EGRID)) )
if ( EXACT == True ):
    DOS_EXACT        = np.zeros( (len(A0_LIST), len(SIGE_LIST), len(CAVL_LIST), len(EGRID)) )

def Ham_2x2( WC, A0, CAVL_LIST, SIGE_LIST ):
    E_NH_LOSS = np.zeros( (len(CAVL_LIST),len(SIGE_LIST),4), dtype=np.complex128 )
    a         = np.diag( np.sqrt(np.ones(2-1)),k=1)
    MU_minus  = np.array([[0.,1],[0,0.]])
    MU_minus  = np.array([[0.,1],[0,0.]])
    MU_plus   = MU_minus.T
    for SIGEi,SIGE in enumerate(SIGE_LIST):
        for CLi,CL in enumerate(CAVL_LIST):
            WC_FREQ  = WC * np.arange(2) * (1 + 0.5*CL*1j)
            MOL_FREQ = WC * np.arange(2) * (1 + SIGE*1j)
            H_PF = np.zeros( (4,4), dtype=np.complex128 )
            H_PF +=               np.kron( np.diag( MOL_FREQ ), np.eye(2) )
            H_PF +=               np.kron( np.eye(2), np.diag( WC_FREQ ) )
            H_PF += WC * A0 *     np.kron( MU_minus, a.T )
            H_PF += WC * A0 *     np.kron( MU_plus,  a )
            E_NH_LOSS[CLi,SIGEi,:],_  = np.linalg.eig(H_PF)
            inds                      = E_NH_LOSS[CLi,SIGEi,:].real.argsort()   
            E_NH_LOSS[CLi,SIGEi,:]    = E_NH_LOSS[CLi,SIGEi,inds]
    return E_NH_LOSS.real - WC


def normalize( GRID, FUNC ):
    dx   = GRID[1]-GRID[0]
    NORM = np.sum( FUNC ) * dx
    FUNC  = FUNC / NORM # Set sum(DOS) = 1.0 photon
    return FUNC

for A0i,A0 in enumerate( A0_LIST ):
    for SIGEi,SIGE in enumerate( SIGE_LIST ):
        for CALi,CAVL in enumerate( CAVL_LIST ):
            try:
                if ( EXACT == True ):
                    ETMP, DOSTMP                = np.loadtxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVL), dtype=np.complex64).T
                    ETMP                        = ETMP.real
                    FUNC                        = interp1d( ETMP, np.abs(DOSTMP), kind='cubic', bounds_error=False, fill_value=0.0 )
                    DOS_EXACT[A0i,SIGEi,CALi,:] = FUNC(EGRID)
                    DOS_EXACT[A0i,SIGEi,CALi,:] = normalize( EGRID, DOS_EXACT[A0i,SIGEi,CALi,:] )
                ETMP, DOSTMP          = np.loadtxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVL), dtype=np.complex64).T
                ETMP                  = ETMP.real
                FUNC                  = interp1d( ETMP, np.abs(DOSTMP), kind='cubic', bounds_error=False, fill_value=0.0 )
                DOS[A0i,SIGEi,CALi,:] = FUNC(EGRID)
                DOS[A0i,SIGEi,CALi,:] = normalize( EGRID, DOS[A0i,SIGEi,CALi,:] )
            except OSError:
                print("Skipping:")
                print( A0,SIGE,CAVL )
                DOS[A0i,SIGEi,CALi,:] = np.zeros( len(EGRID) )*float("nan")

        CL_LIST_FINE    = np.linspace( CAVL_LIST[0],CAVL_LIST[-1],200 )
        SIGE_LIST_FINE  = np.linspace( SIGE_LIST[0],SIGE_LIST[-1],200 )
        E_NH_LOSS       = Ham_2x2( WC, A0, CL_LIST_FINE, SIGE_LIST_FINE )

        cmap = "jet" # "ocean_r"
        extent = [(EGRID[0]-WC)/A0,(EGRID[-1]-WC)/A0,CAVL_LIST[0]/A0,CAVL_LIST[-1]/A0]
        
        plt.imshow( DOS[A0i,SIGEi,:,:] / np.max(DOS[A0i,SIGEi,:,:]), origin='lower', cmap=cmap, extent=extent, aspect='auto', interpolation='spline16')
        plt.plot( E_NH_LOSS[:,SIGEi,1]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
        plt.plot( E_NH_LOSS[:,SIGEi,2]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
        plt.colorbar(pad=0.01)
        plt.xlabel("Energy $(E - \\omega_\\mathrm{c}) / A_0$", fontsize=15)
        plt.ylabel("Cavity Loss ($\\Gamma_\\mathrm{c} / A_0$)", fontsize=15)
        plt.xlim(-2,2)
        plt.title("$A_0$ = %1.3f, $\\sigma_\\mathrm{E}$ = %1.3f,  $N_\\mathrm{mol}$ = %d" % (A0, SIGE, N), fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_EXACT.jpg" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), dpi=300)
        plt.clf()
        np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), DOS[A0i,SIGEi,:,:] / np.max(DOS[A0i,SIGEi,:,:]), fmt="%1.4f")
        np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_EGRID_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), (EGRID-WC)/A0, fmt="%1.4f")
        np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_LOSSGRID_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), CAVL_LIST/A0, fmt="%1.4f")

        if ( EXACT == True ):
            plt.imshow( DOS_EXACT[A0i,SIGEi,:,:] / np.max(DOS_EXACT[A0i,SIGEi,:,:]), origin='lower', cmap=cmap, extent=extent, aspect='auto', interpolation='spline16')
            plt.plot( E_NH_LOSS[:,SIGEi,1]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
            plt.plot( E_NH_LOSS[:,SIGEi,2]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
            plt.colorbar(pad=0.01)
            plt.xlabel("Energy $(E - \\omega_\\mathrm{c}) / A_0$", fontsize=15)
            plt.ylabel("Cavity Loss ($\\Gamma_\\mathrm{c} / A_0$)", fontsize=15)
            plt.xlim(-2,2)
            plt.title("$A_0$ = %1.3f, $\\sigma_\\mathrm{E}$ = %1.3f,  $N_\\mathrm{mol}$ = %d" % (A0, SIGE, N), fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_EXACT.jpg" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), dpi=300)
            plt.clf()
            np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), DOS_EXACT[A0i,SIGEi,:,:] / np.max(DOS_EXACT[A0i,SIGEi,:,:]), fmt="%1.4f")
            np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_EGRID_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), (EGRID-WC)/A0, fmt="%1.4f")
            np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_LOSSGRID_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), CAVL_LIST/A0, fmt="%1.4f")
        

            plt.imshow( np.abs(DOS_EXACT[A0i,SIGEi,:,:] - DOS[A0i,SIGEi,:,:]), origin='lower', cmap=cmap, extent=extent, aspect='auto', interpolation='spline16', vmin=0.0, vmax=5)
            plt.plot( E_NH_LOSS[:,SIGEi,1]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
            plt.plot( E_NH_LOSS[:,SIGEi,2]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
            plt.colorbar(pad=0.01)
            plt.xlabel("Energy $(E - \\omega_\\mathrm{c}) / A_0$", fontsize=15)
            plt.ylabel("Cavity Loss ($\\Gamma_\\mathrm{c} / A_0$)", fontsize=15)
            plt.xlim(-2,2)
            plt.title("$A_0$ = %1.3f, $\\sigma_\\mathrm{E}$ = %1.3f,  $N_\\mathrm{mol}$ = %d" % (A0, SIGE, N), fontsize=15)
            plt.tight_layout()
            plt.savefig(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_ERROR.jpg" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), dpi=300)
            plt.clf()
            np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_ERROR.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), DOS_EXACT[A0i,SIGEi,:,:] / np.max(DOS_EXACT[A0i,SIGEi,:,:]) - DOS[A0i,SIGEi,:,:] / np.max(DOS[A0i,SIGEi,:,:]), fmt="%1.4f")
            np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_EGRID_ERROR.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), (EGRID-WC)/A0, fmt="%1.4f")
            np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_SCAN_LOSSGRID_ERROR.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG), CAVL_LIST/A0, fmt="%1.4f")
        





# # Save data to neat text file
# for A0i, A0 in enumerate( A0_LIST ):
#     output = [EGRID]
#     for SIGEi,SIGE in enumerate( SIGE_LIST ):
#         output.append( DOS[A0i,SIGEi,:] )

#     file = "PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f_CLEAN.dat" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL)
#     np.savetxt(file, np.c_[ output ].T , fmt="%1.6f", header="EGRID    DOS_STOCHASTIC")
    

    
