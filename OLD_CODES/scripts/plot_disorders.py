import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


NR_LIST = np.array([ 10, 100, 2000, 5000 ])
SIGE_LIST = np.array([0.0, 0.05, 0.1, 0.15, 0.2])

NC   = 500
N    = 1000
GAM  = 0.01
WC   = 1.0
A0   = 0.1
SIGG = 0.0
CAVL = 0.0
NPTS = 500

color_list = ["black", "red", "blue", "green", "orange"]
DOS        = np.zeros( (len(NR_LIST), len(SIGE_LIST), NPTS) )
DOS_EXACT  = np.zeros( (len(NR_LIST), len(SIGE_LIST), NPTS) )

for NRi,NR in enumerate( NR_LIST ):

    # Plot EXACT DOS
    for SIGEi,SIGE in enumerate( SIGE_LIST ):
        EGRID, TMP             = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EXACT.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVL), dtype=np.complex64).T
        EGRID                  = EGRID.real
        dE                     = EGRID[1]-EGRID[0]
        DOS_EXACT[NRi,SIGEi,:] = np.abs(TMP)

        if ( SIGEi == 0 ):
            plt.plot( EGRID, DOS_EXACT[NRi,SIGEi,:], "-", lw=6, alpha=0.3, c=color_list[SIGEi], label="EXACT" )
        else:
            plt.plot( EGRID, DOS_EXACT[NRi,SIGEi,:], "-", lw=6, alpha=0.3, c=color_list[SIGEi] )

    # Plot stochastic DOS
    for SIGEi,SIGE in enumerate( SIGE_LIST ):
        EGRID, TMP        = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVL), dtype=np.complex64).T
        EGRID             = EGRID.real
        dE                = EGRID[1]-EGRID[0]
        DOS[NRi,SIGEi,:]  = np.abs(TMP)

        if ( SIGEi == 0 ):
            plt.plot( EGRID, DOS[NRi,SIGEi,:], "-", lw=1, c=color_list[SIGEi], label="Stochastic" )
        else:
            plt.plot( EGRID, DOS[NRi,SIGEi,:], "-", lw=1, c=color_list[SIGEi] )
        

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
    plt.savefig("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL), dpi=300)
    plt.clf()


    # PLOT ERROR w.r.t. EXACT
    for SIGEi,SIGE in enumerate( SIGE_LIST ):
        plt.plot( EGRID, DOS_EXACT[NRi,SIGEi,:] - DOS[NRi,SIGEi,:], "-", lw=1, c=color_list[SIGEi] )
        
    plt.xlim(0.7,1.3)
    plt.xlabel("Energy (a.u.)", fontsize=15)
    plt.ylabel("EXACT - STO.", fontsize=15)
    plt.title("$A_0$ = 0.10 a.u., $\omega_\mathrm{c}$ = 1.00 a.u.,  $N_\mathrm{mol}$ = 1000", fontsize=15)
    plt.tight_layout()
    plt.savefig("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f_ERROR.jpg" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL), dpi=300)
    plt.clf()



# Save data to neat text file
for NRi, NR in enumerate( NR_LIST ):
    output = [EGRID]
    for SIGEi,SIGE in enumerate( SIGE_LIST ):
        output.append( DOS_EXACT[NRi,SIGEi,:] )
        output.append( DOS[NRi,SIGEi,:] )

    file = "PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f_CLEAN.dat" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL)
    np.savetxt(file, np.c_[ output ].T , fmt="%1.6f", header="EGRID    DOS_EXACT    DOS_STOCHASTIC")
    
for NRi, NR in enumerate( NR_LIST ):
    output = [EGRID]
    for SIGEi,SIGE in enumerate( SIGE_LIST ):
        output.append( DOS_EXACT[NRi,SIGEi,:] - DOS[NRi,SIGEi,:] )

    file = "PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_SCAN_SIGG_%1.3f_CAVLOSS_%1.3f_CLEAN_ERROR.dat" % (N,NR,NC,GAM,A0,WC,SIGG,CAVL)
    np.savetxt(file, np.c_[ output ].T , fmt="%1.6f", header="EGRID    ERROR (EXACT - Stoch.)")
    
