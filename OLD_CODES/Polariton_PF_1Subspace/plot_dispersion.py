import numpy as np
from matplotlib import pyplot as plt

N        = 10_000
NPTS     = 1000
NR       = 200
NC       = 300
SIG      = 0.0
GAM      = 0.02
A0_LIST  = [0.0, 0.1, 0.2, 0.3]
WC_LIST  = np.arange( 0.5, 1.5+0.05, 0.05 )

DOS_PHOT = np.zeros( (len(A0_LIST), len(WC_LIST), 1000) )
DOS_MATT = np.zeros( (len(A0_LIST), len(WC_LIST), 1000) )
EGRID    = None

for A0i,A0 in enumerate( A0_LIST ):
    for WCi,WC in enumerate( WC_LIST ):
        A0 = round(A0,4)
        WC = round(WC,4)
        try:
            EGRID, DOS_PHOT[A0i,WCi,:] = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIG_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIG)).T
            EGRID, DOS_MATT[A0i,WCi,:] = np.loadtxt("PLOTS_DATA/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIG_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIG)).T
        except OSError:
            continue


###### PHOTON ######
for A0i,A0 in enumerate( A0_LIST ):
    #X,Y = np.meshgrid( np.sqrt(0.5**2 + (WC_LIST-0.5)**2), EGRID )
    X,Y = np.meshgrid( WC_LIST, EGRID )
    plt.contourf( X,Y,DOS_PHOT[A0i,:,:].T, cmap="Greys", levels=np.linspace(0,1,1001), vmin=0.0, vmax=1.0 )
    #plt.colorbar(pad=0.01)
    plt.xlim(WC_LIST[0],WC_LIST[-1])
    plt.ylim(EGRID[0],EGRID[-1])
    plt.xlabel("Cavity Frequency (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$A_0$ = %1.2f a.u.  $N$ = %1.0f  $\sigma$ = %1.1f a.u." % (A0,N,SIG), fontsize=15)
    plt.tight_layout()
    plt.savefig("PLOTS_DATA/Dispersion_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIG_%1.3f.jpg" % (N,NR,NC,GAM,A0,SIG), dpi=300 )
    plt.clf()

###### MATTER ######
for A0i,A0 in enumerate( A0_LIST ):
    #X,Y = np.meshgrid( np.sqrt(0.5**2 + (WC_LIST-0.5)**2), EGRID )
    X,Y = np.meshgrid( WC_LIST, EGRID )
    plt.contourf( X,Y,DOS_MATT[A0i,:,:].T, cmap="Greys", levels=np.linspace(0,1,1001), vmin=0.0, vmax=1.0 )
    #plt.colorbar(pad=0.01)
    plt.xlim(WC_LIST[0],WC_LIST[-1])
    plt.ylim(EGRID[0],EGRID[-1])
    plt.xlabel("Cavity Frequency (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$A_0$ = %1.2f a.u.  $N$ = %1.0f  $\sigma$ = %1.1f a.u." % (A0,N,SIG), fontsize=15)
    plt.tight_layout()
    plt.savefig("PLOTS_DATA/Dispersion_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIG_%1.3f.jpg" % (N,NR,NC,GAM,A0,SIG), dpi=300 )
    plt.clf()