import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp2d

N        = 1000
NPTS     = 100
NR       = 200
NC       = 300
SIGE     = 0.2
SIGG     = 0.0
CAVLOSS  = 0.0
GAM      = 0.05
A0_LIST  = [0.1, 0.2]
WC_LIST  = np.arange( 0.5, 1.5+0.05, 0.05 )

DOS_PHOT = np.zeros( (len(A0_LIST), len(WC_LIST), NPTS) )
DOS_MATT = np.zeros( (len(A0_LIST), len(WC_LIST), NPTS) )
EGRID    = None

for A0i,A0 in enumerate( A0_LIST ):
    for WCi,WC in enumerate( WC_LIST ):
        A0 = round(A0,4)
        WC = round(WC,4)
        EGRID, DOS_PHOT[A0i,WCi,:] = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVLOSS), dtype=np.complex64).real.T
        EGRID, DOS_MATT[A0i,WCi,:] = np.loadtxt("PLOTS_DATA/DOS_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_WC_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,WC,SIGE,SIGG,CAVLOSS), dtype=np.complex64).real.T

        # Normalize each DOS
        DOS_PHOT[A0i,WCi,:] /= np.max( DOS_PHOT[A0i,WCi,:] )
        DOS_MATT[A0i,WCi,:] /= np.max( DOS_MATT[A0i,WCi,:] )

# Interpolating energy grid
WC_FINE    = np.linspace( WC_LIST[0], WC_LIST[-1], 1001 )
EGRID_FINE = np.linspace( EGRID[0], EGRID[-1], 1000 )


###### PHOTON ######
for A0i,A0 in enumerate( A0_LIST ):
    F   = interp2d( WC_LIST, EGRID, DOS_PHOT[A0i,:,:].T, kind='cubic' )
    X,Y = np.meshgrid( WC_FINE, EGRID_FINE )
    plt.contourf( X,Y,F(WC_FINE,EGRID_FINE) / np.max(F(WC_FINE,EGRID_FINE)), cmap="Greys", levels=np.linspace(0,1,1001), vmin=0.0, vmax=1.0 )
    plt.colorbar(pad=0.01)
    plt.xlim(WC_LIST[0],WC_LIST[-1])
    plt.ylim(EGRID[0],EGRID[-1])
    plt.xlabel("Cavity Frequency (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$A_0$ = %1.2f a.u.  $N$ = %1.0f  $\sigma_\mathrm{E}$ = %1.1f a.u.  $\sigma_\mathrm{G}$ = %1.1f a.u." % (A0,N,SIGE,SIGG), fontsize=15)
    plt.tight_layout()
    plt.savefig("PLOTS_DATA/Dispersion_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), dpi=300 )
    np.savetxt("PLOTS_DATA/Dispersion_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), F(WC_FINE,EGRID_FINE) / np.max(F(WC_FINE,EGRID_FINE)), fmt="%1.4f", header="WC_GRID (1001 PTS), E_GRID (1000 PTS)" )
    np.savetxt("PLOTS_DATA/Dispersion_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_WCGRID.dat" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), WC_FINE, fmt="%1.4f" )
    np.savetxt("PLOTS_DATA/Dispersion_PHOTON_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EGRID.dat" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), EGRID_FINE, fmt="%1.4f" )
    plt.clf()

###### MATTER ######
for A0i,A0 in enumerate( A0_LIST ):
    F   = interp2d( WC_LIST, EGRID, DOS_MATT[A0i,:,:].T, kind='cubic' )
    X,Y = np.meshgrid( WC_FINE, EGRID_FINE )
    plt.contourf( X,Y,F(WC_FINE,EGRID_FINE) / np.max(F(WC_FINE,EGRID_FINE)), cmap="Greys", levels=np.linspace(0,1,1001), vmin=0.0, vmax=1.0 )
    #plt.colorbar(pad=0.01)
    plt.xlim(WC_LIST[0],WC_LIST[-1])
    plt.ylim(EGRID[0],EGRID[-1])
    plt.xlabel("Cavity Frequency (a.u.)", fontsize=15)
    plt.ylabel("Energy (a.u.)", fontsize=15)
    plt.title("$A_0$ = %1.2f a.u.  $N$ = %1.0f  $\sigma_\mathrm{E}$ = %1.1f a.u.  $\sigma_\mathrm{G}$ = %1.1f a.u." % (A0,N,SIGE,SIGG), fontsize=15)
    plt.tight_layout()
    plt.savefig("PLOTS_DATA/Dispersion_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.jpg" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), dpi=300 )
    np.savetxt("PLOTS_DATA/Dispersion_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f.dat" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), F(WC_FINE,EGRID_FINE) / np.max(F(WC_FINE,EGRID_FINE)), fmt="%1.4f", header="WC_GRID (1001 PTS), E_GRID (1000 PTS)" )
    np.savetxt("PLOTS_DATA/Dispersion_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_WCGRID.dat" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), WC_FINE, fmt="%1.4f" )
    np.savetxt("PLOTS_DATA/Dispersion_MATTER_N_%1.0f_Nr_%1.0f_NC_%1.0f_GAM_%1.3f_A0_%1.3f_SIGE_%1.3f_SIGG_%1.3f_CAVLOSS_%1.3f_EGRID.dat" % (N,NR,NC,GAM,A0,SIGE,SIGG,CAVLOSS), EGRID_FINE, fmt="%1.4f" )
    plt.clf()