import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


NR_LIST = np.array([ 5000 ])
A0_LIST = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.15, 0.20])

NC   = 500
N    = 1000
GAM  = 0.01
WC   = 1.0
SIGE = 0.0
SIGG = 1.0
CAVL = 0.0
NPTS = 500

color_list = ["black", "red", "blue", "green", "orange", "cyan", "magenta", "gray", "purple"]
DOS        = np.zeros( (len(NR_LIST), len(A0_LIST), NPTS) )
DOS_EXACT  = np.zeros( (len(NR_LIST), len(A0_LIST), NPTS) )

for NRi,NR in enumerate( NR_LIST ):

    # Plot EXACT DOS
    for A0i,A0 in enumerate( A0_LIST ):
        EGRID, TMP             = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVL), dtype=np.complex64).T
        EGRID                  = EGRID.real
        dE                     = EGRID[1]-EGRID[0]
        DOS_EXACT[NRi,A0i,:]   = np.abs(TMP)

        if ( A0i == 0 ):
            plt.plot( EGRID, DOS_EXACT[NRi,A0i,:], "-", lw=6, alpha=0.3, c=color_list[A0i], label="EXACT" )
        else:
            plt.plot( EGRID, DOS_EXACT[NRi,A0i,:], "-", lw=6, alpha=0.3, c=color_list[A0i] )

    # Plot stochastic DOS
    for A0i,A0 in enumerate( A0_LIST ):
        EGRID, TMP        = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVL), dtype=np.complex64).T
        EGRID             = EGRID.real
        dE                = EGRID[1]-EGRID[0]
        DOS[NRi,A0i,:]  = np.abs(TMP)

        if ( A0i == 0 ):
            plt.plot( EGRID, DOS[NRi,A0i,:], "-", lw=1, c=color_list[A0i], label="Stochastic" )
        else:
            plt.plot( EGRID, DOS[NRi,A0i,:], "-", lw=1, c=color_list[A0i] )
        

    plt.plot( EGRID, 1000*(EGRID-WC), "--", c="black", lw=1 )
    plt.legend()
    plt.xlim(0.7,1.3)
    plt.ylim(0,0.55)
    #plt.ylim(0,0.045)
    #plt.ylim(0.001,0.05)
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("Transmission Spectra (Arb. Units)", fontsize=15)
    plt.title("$A_0$ = 0.10 a.u., $\omega_\mathrm{c}$ = 1.00 a.u.,  $N_\mathrm{mol}$ = 1000", fontsize=15)
    plt.tight_layout()
    plt.savefig("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_SCAN_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (N,NR,NC,GAM,WC,SIGE,SIGG,CAVL), dpi=300)
    plt.clf()


    # PLOT ERROR w.r.t. EXACT
    for A0i,A0 in enumerate( A0_LIST ):
        plt.plot( EGRID, DOS_EXACT[NRi,A0i,:] - DOS[NRi,A0i,:], "-", lw=1, c=color_list[A0i] )
        
    plt.xlim(0.7,1.3)
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("EXACT - STO.", fontsize=15)
    plt.title("$A_0$ = 0.10 a.u., $\omega_\mathrm{c}$ = 1.00 a.u.,  $N_\mathrm{mol}$ = 1000", fontsize=15)
    plt.tight_layout()
    plt.savefig("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_SCAN_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_ERROR.jpg" % (N,NR,NC,GAM,WC,SIGE,SIGG,CAVL), dpi=300)
    plt.clf()



# Save data to neat text file
for NRi, NR in enumerate( NR_LIST ):
    output = [EGRID]
    for A0i,A0 in enumerate( A0_LIST ):
        output.append( DOS_EXACT[NRi,A0i,:] )
        output.append( DOS[NRi,A0i,:] )

    file = "PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_SCAN_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_CLEAN.dat" % (N,NR,NC,GAM,WC,SIGE,SIGG,CAVL)
    np.savetxt(file, np.c_[ output ].T , fmt="%1.6f", header="EGRID    DOS_EXACT    DOS_STOCHASTIC")
    
for NRi, NR in enumerate( NR_LIST ):
    output = [EGRID]
    for A0i,A0 in enumerate( A0_LIST ):
        output.append( DOS_EXACT[NRi,A0i,:] - DOS[NRi,A0i,:] )

    file = "PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_SCAN_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_CLEAN_ERROR.dat" % (N,NR,NC,GAM,WC,SIGE,SIGG,CAVL)
    np.savetxt(file, np.c_[ output ].T , fmt="%1.6f", header="EGRID    ERROR (EXACT - Stoch.)")
    
