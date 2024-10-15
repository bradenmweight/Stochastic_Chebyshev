import numpy as np
from matplotlib import pyplot as plt

N        = 1000
NPTS     = 1000
NR       = 200
NC       = 200
SIGE     = 0.0
SIGG     = 0.0
GAM      = 0.05
A0_LIST   = [0.1, 0.2]
WC_LIST   = np.arange( 0.5, 1.5+0.1,0.1 )
ERxN_LIST = np.arange( 0.5, 1.5+0.1,0.1 )

DOS_PHOT = np.zeros( (len(A0_LIST), len(WC_LIST), len(ERxN_LIST), 1000) )
DOS_MATT = np.zeros( (len(A0_LIST), len(WC_LIST), len(ERxN_LIST), 1000) )
EGRID    = None

for A0i,A0 in enumerate( A0_LIST ):
    for WCi,WC in enumerate( WC_LIST ):
        for ERxNi,ERxN in enumerate( ERxN_LIST ):
            A0   = round(A0,4)
            WC   = round(WC,4)
            ERxN = round(ERxN,4)
            EGRID, DOS_PHOT[A0i,WCi,ERxNi,:] = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_ERxN_%1.3f_ERxN_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,ERxN,ERxN)).T
            EGRID, DOS_MATT[A0i,WCi,ERxNi,:] = np.loadtxt("PLOTS_DATA/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_ERxN_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,ERxN)).T


for WCi,WC in enumerate( WC_LIST ):
    print(WC)
    ###### PHOTON ######
    for A0i,A0 in enumerate( A0_LIST ):
        #X,Y = np.meshgrid( np.sqrt(0.5**2 + (WC_LIST-0.5)**2), EGRID )
        X,Y = np.meshgrid( ERxN_LIST, EGRID )
        plt.contourf( X,Y,DOS_PHOT[A0i,WCi,:].T, cmap="Greys", levels=np.linspace(0,1,1001), vmin=0.0, vmax=1.0 )
        #plt.colorbar(pad=0.01)
        plt.xlim(ERxN_LIST[0],ERxN_LIST[-1])
        plt.ylim(EGRID[0],EGRID[-1])
        plt.xlabel("Cavity Frequency (a.u.)", fontsize=15)
        plt.ylabel("Energy (a.u.)", fontsize=15)
        plt.title("$A_0$ = %1.2f a.u.  $N$ = %1.0f  $\sigma_\mathrm{E}$ = %1.1f a.u.  $\sigma_\mathrm{G}$ = %1.1f a.u." % (A0,N,SIGE,SIGG), fontsize=15)
        plt.tight_layout()
        plt.savefig("PLOTS_DATA/Dispersion_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_WC_%1.3f.jpg" % (N,NR,NC,GAM,A0,SIGE,SIGG,WC), dpi=300 )
        plt.clf()

    ###### MATTER ######
    for A0i,A0 in enumerate( A0_LIST ):
        #X,Y = np.meshgrid( np.sqrt(0.5**2 + (WC_LIST-0.5)**2), EGRID )
        X,Y = np.meshgrid( ERxN_LIST, EGRID )
        plt.contourf( X,Y,DOS_MATT[A0i,WCi,:].T, cmap="Greys", levels=np.linspace(0,1,1001), vmin=0.0, vmax=1.0 )
        #plt.colorbar(pad=0.01)
        plt.xlim(ERxN_LIST[0],ERxN_LIST[-1])
        plt.ylim(EGRID[0],EGRID[-1])
        plt.xlabel("Cavity Frequency (a.u.)", fontsize=15)
        plt.ylabel("Energy (a.u.)", fontsize=15)
        plt.title("$A_0$ = %1.2f a.u.  $N$ = %1.0f  $\sigma_\mathrm{E}$ = %1.1f a.u.  $\sigma_\mathrm{G}$ = %1.1f a.u." % (A0,N,SIGE,SIGG), fontsize=15)
        plt.tight_layout()
        plt.savefig("PLOTS_DATA/Dispersion_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_WC_%1.3f.jpg" % (N,NR,NC,GAM,A0,SIGE,SIGG,WC), dpi=300 )
        plt.clf()