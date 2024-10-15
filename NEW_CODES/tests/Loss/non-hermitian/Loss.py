import numpy as np
from matplotlib import pyplot as plt
import sys

sys.path.insert(0, '../../../')
from main import Params

A0        = 0.05
LOSS_LIST = [0.0, 0.10]# [0.0, 0.10, 0.15, 0.20] # np.arange( 0.0, 0.05+0.01, 0.01 )

for CAVi,CAV in enumerate( LOSS_LIST ):
    print( "Working on cavity loss rate: %1.3f a.u." % CAV )

    EGRID            = np.linspace( 0.5, 1.5, 100 )
    dH               = EGRID[-1] - EGRID[0]
    dE               = EGRID[1]  - EGRID[0]
    DOS_CHEB         = np.zeros( ( len(EGRID) ), dtype=np.complex128)
    for Ei,Ept in enumerate(EGRID):
        print( "Working on grid point %d of %d" % (Ei, len(EGRID)) )
        params = Params(    batch_size=100, N_STOCHASTIC=100, # (N_STOCHASTIC // batch_size) should be an integer
                            #N_MOL=10_002,
                            N_MOL=1,
                            F_type="1_over_e", 
                            #F_type="gaussian", 
                            N_CHEB=300, 
                            dH=dH, 
                            WC=1.0, 
                            A0=A0, 
                            Ept=Ept, 
                            CAV_LOSS=CAV )
        params.run()
        DOS_CHEB[Ei]         = params.DOS[-1]


    # DOS_CHEB = np.real( DOS_CHEB ) # RE[F(E,\gamma)]
    # DOS_CHEB = np.imag( DOS_CHEB ) # IM[F(E,\gamma)]
    DOS_CHEB = np.abs( DOS_CHEB )    # |F(E,\gamma)|

    phase = np.sign( DOS_CHEB[:].sum() )
    plt.plot( EGRID, phase * DOS_CHEB[:] / np.max(phase * DOS_CHEB[:]), "-", label='$\\frac{\\gamma_\\mathrm{c}}{A_0}$ = %1.3f' % (CAV/A0) )


plt.legend()
plt.savefig( "DOS.png", dpi=300 )