import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

DATA_DIR = "PLOTS_DATA__N_10_5"

NR        = 5000
A0_LIST   = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
#SIGE_LIST = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
SIGE_LIST = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])

NC   = 200
N    = 100000
GAM  = 0.01
WC   = 1.0
SIGG = 0.0
CAVL = 0.0
NPTS = 100

color_list = ["black", "red", "blue", "green", "orange"]
EGRID      = np.linspace( 0.9, 1.1, 1000 )
DOS        = np.zeros( (len(A0_LIST), len(SIGE_LIST), len(EGRID)) )
DOS_EXACT  = np.zeros( (len(A0_LIST), len(SIGE_LIST), len(EGRID)) )

for A0i,A0 in enumerate( A0_LIST ):

    # Plot stochastic DOS
    for SIGEi,SIGE in enumerate( SIGE_LIST ):
        ETMP, DOSTMP      = np.loadtxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVL), dtype=np.complex64).T
        ETMP              = ETMP.real
        FUNC              = interp1d( ETMP, np.abs(DOSTMP), kind='cubic', bounds_error=False, fill_value=0.0 )
        DOS[A0i,SIGEi,:]  = FUNC(EGRID)

    cmap = "jet" # "ocean_r"
    extent = [(EGRID[0]-WC)/A0,(EGRID[-1]-WC)/A0,SIGE_LIST[0]/A0,SIGE_LIST[-1]/A0]

    plt.imshow( DOS[A0i,:,:], origin='lower', cmap=cmap, interpolation='spline16', extent=extent, aspect='auto')
    plt.colorbar(pad=0.01,label="DOS")
    #plt.colorbar(pad=0.01,label="ln[DOS]")
    plt.xlabel("Energy $(E - \\omega_\\mathrm{c}) / A_0$", fontsize=15)
    plt.ylabel("Disorder Width ($\\sigma / A_0$)", fontsize=15)
    plt.title("$A_0$ = %1.3f, $\\omega_\\mathrm{c}$ = %1.3f,  $N_\\mathrm{mol}$ = %1.0f" % (A0, WC, N), fontsize=15)
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL), dpi=300)
    plt.clf()
    np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL), DOS[A0i,:,:], fmt="%1.4f")
    np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f_EGRID.dat" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL), (EGRID-WC)/A0, fmt="%1.4f")
    np.savetxt(f"{DATA_DIR}/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f_SIGEGRID.dat" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL), SIGE_LIST/A0, fmt="%1.4f")
    



# # Save data to neat text file
# for A0i, A0 in enumerate( A0_LIST ):
#     output = [EGRID]
#     for SIGEi,SIGE in enumerate( SIGE_LIST ):
#         output.append( DOS[A0i,SIGEi,:] )

#     file = "PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f_CLEAN.dat" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL)
#     np.savetxt(file, np.c_[ output ].T , fmt="%1.6f", header="EGRID    DOS_STOCHASTIC")
    

    
