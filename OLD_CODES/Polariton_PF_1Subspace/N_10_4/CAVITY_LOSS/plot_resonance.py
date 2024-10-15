import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

CL_LIST = np.array([0.0, 0.1, 0.2, 0.3, 0.4])

for CLi,CL in enumerate( CL_LIST ):
    EGRID, TMP = np.loadtxt("PLOTS_DATA/DOS_PHOTON_N_10000_Nr_500_NC_500_GAM_0.050_A0_0.100_WC_1.000_SIGE_0.000_SIGG_0.000_CAVLOSS_%1.3f.dat" % CL, dtype=np.complex64).T
    EGRID = EGRID.real
    dE    = EGRID[1]-EGRID[0]
    DOS   = np.abs(TMP)
    DOS   = DOS - np.average(DOS[:10])
    DOS   = DOS / np.sum(DOS)
    DOS   = savgol_filter(DOS, 25, 3) # window size 51, polynomial order 3
    #plt.plot( EGRID, DOS, lw=4, label="$\gamma_\mathrm{c}$ = %1.2f a.u." % (CL/2) )
    plt.plot( EGRID, DOS, lw=4, label="$\mathrm{Im}[\omega_\mathrm{c}]$ = %1.2f a.u." % (CL/2) )

#plt.legend()
plt.xlim(0.7,1.3)
plt.ylim(0)
plt.xlabel("Energy (a.u.)", fontsize=15)
plt.ylabel("Density of States (Arb. Units)", fontsize=15)
plt.savefig("DOS_RES.jpg", dpi=300)
plt.clf()


