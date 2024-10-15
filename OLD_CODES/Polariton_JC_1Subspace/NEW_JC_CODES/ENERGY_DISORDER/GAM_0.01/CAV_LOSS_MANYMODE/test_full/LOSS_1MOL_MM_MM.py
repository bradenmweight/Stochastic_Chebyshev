import numpy as np
from matplotlib import pyplot as plt
import sys
import subprocess as sp
from scipy.interpolate import interp1d

from nH_LOSS import main as nH_LOSS

DATA_DIR = "PLOTS"
sp.call(f"mkdir -p {DATA_DIR}", shell=True)

NMODE = int( sys.argv[1] ) # Choose an odd number
NMOL  = int( sys.argv[2] ) # Choose an odd number
NPOL  = NMOL + NMODE

A0             = 0.01
WC             = 1.00
CL_LIST        = np.linspace( 1e-10,8*A0,10 )
SIGE           = 0.01
if ( NMOL > 1 ):
    #EMOL_ARRAY     = np.linspace( 1-5*SIGE,1+5*SIGE,NMOL ) # Rectangular distribution
    EMOL_ARRAY     = np.random.normal( 1.0, SIGE, size=NMOL ) # Gaussian distribution
else:
    EMOL_ARRAY     = np.array( [1.0] )

E         = np.zeros( (len(CL_LIST),NPOL) )
PHOT      = np.zeros( (len(CL_LIST),NPOL) )
MATT      = np.zeros( (len(CL_LIST),NPOL) )

N_OP_TOTAL = np.zeros( (NPOL,NPOL) )
M_OP_TOTAL = np.zeros( (NPOL,NPOL) )
inds       = ( np.arange(NMOL),np.arange(NMOL) )
M_OP_TOTAL[inds] = 1.0


def get_H( WC_ARRAY, A0_ARRAY ):
    H_PF       = np.zeros( (NPOL,NPOL) )
    inds       = ( np.arange(NMOL),np.arange(NMOL) )
    H_PF[inds] = EMOL_ARRAY # Molecule Energy
    inds       = ( np.arange(NMOL,NPOL),np.arange(NMOL,NPOL) )
    H_PF[inds] = WC_ARRAY # Cavity Energies
    for mol in range( NMOL ):
        inds       = ( mol, np.arange(NMOL,NPOL) )
        H_PF[inds] = WC_ARRAY * A0_ARRAY # Coupling Elements
        inds       = ( np.arange(NMOL,NPOL), mol )
        H_PF[inds] = WC_ARRAY * A0_ARRAY # Coupling Elements
    return H_PF

def get_Cavity_Modes( CL ):
    if ( NMODE == 1 ):
        WC_ARRAY = np.array( [WC] )
        A0_ARRAY = np.array( [A0] )
        N_OP_TOTAL[ -1,-1 ] = 1.0
    else:
        WC_ARRAY   = np.linspace( WC-5*CL,WC+5*CL,NMODE )
        dwc        = WC_ARRAY[1] - WC_ARRAY[0]
        #L          = dwc * np.sqrt(1/2/np.pi)/CL * np.exp( -(WC-WC_ARRAY)**2 / 2 / CL**2 )
        L          = dwc * 1/np.pi * CL/2/( (WC-WC_ARRAY)**2 + CL**2/4 )
        A0_ARRAY   = A0 * np.sqrt( L ) # Eq. 3 of Wang et al., J. Chem. Phys. 154, 104109 (2021)
        #print( A0**2, (A0_ARRAY**2).sum() ) # Sum Rule. Eq. 5 of Wang et al., J. Chem. Phys. 154, 104109 (2021)
        inds       = ( np.arange(NMOL,NPOL),np.arange(NMOL,NPOL) )
        N_OP_TOTAL[inds] = 1.0
    return WC_ARRAY, A0_ARRAY, N_OP_TOTAL

def plot_H( H_PF, CL ):
    plt.imshow( H_PF - np.diag( np.diagonal( H_PF ) ), cmap="bwr" ) # Remove diagonal
    plt.colorbar(pad=0.01)
    plt.savefig(f"{DATA_DIR}/H_NMODE_{NMODE}_NMOL_{NMOL}_SIGE_{round(SIGE,4)}_CL_{round(CL,4)}.jpg", dpi=300)
    plt.clf()


for CLi,CL in enumerate(CL_LIST):
    WC_ARRAY, A0_ARRAY, N_OP_TOTAL = get_Cavity_Modes( CL )
    plt.plot( WC_ARRAY, A0_ARRAY, label="CL = %.4f" % CL )
plt.xlabel("Mode Label, $\\alpha$", fontsize=15)
plt.ylabel("Mode Coupling Strength, $A_0^\\alpha$", fontsize=15)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/A0_NMODE_{NMODE}.jpg", dpi=300)
plt.clf()


for CLi,CL in enumerate(CL_LIST):
    print( CLi, "of", len(CL_LIST) )

    WC_ARRAY, A0_ARRAY, N_OP_TOTAL = get_Cavity_Modes( CL )
    H_PF         = get_H( WC_ARRAY, A0_ARRAY )
    E[CLi,:],U   = np.linalg.eigh( H_PF )
    PHOT[CLi,:]  = np.einsum( "aj,ab,bj->j", np.conjugate(U), N_OP_TOTAL, U )
    MATT[CLi,:]  = np.einsum( "aj,ab,bj->j", np.conjugate(U), M_OP_TOTAL, U )
    #plot_H( H_PF, CL )


# Renormalize the eigenvalues w.r.t. the number of molecules/modes and shift to zero
print( "MIN E, MAX E", np.min(E), np.max(E) )
E = (E-WC) / np.sqrt( NMOL * NMODE ) / A0
print( "MIN E, MAX E", np.min(E), np.max(E) )


NPTS   = 1001
SIG    = 0.0025/A0
# print( SIG )
EMIN   =  np.min(E)*1.5
EMAX   =  np.max(E)*1.5
BOUNDS = np.array([EMIN,EMAX])
EMIN   = -BOUNDS[np.argmax(BOUNDS)]
EMAX   =  BOUNDS[np.argmax(BOUNDS)]
# print("BOUNDS:", EMIN,EMAX)
DOS    = np.zeros( (len(CL_LIST),NPTS) )
TM     = np.zeros( (len(CL_LIST),NPTS) )
ABS    = np.zeros( (len(CL_LIST),NPTS) )
EGRID  = np.linspace(EMIN,EMAX,NPTS)
for CLi,CL in enumerate(CL_LIST):
    for pt in range(NPTS):
        DOS[CLi,pt] += np.sum( 1.000000000 *  np.exp( -(EGRID[pt] - E[CLi,:])**2 / 2 / SIG**2 ) )
        TM[CLi,pt]  += np.sum( PHOT[CLi,:] *  np.exp( -(EGRID[pt] - E[CLi,:])**2 / 2 / SIG**2 ) )
        ABS[CLi,pt] += np.sum( MATT[CLi,:] *  np.exp( -(EGRID[pt] - E[CLi,:])**2 / 2 / SIG**2 ) )













# Calculate the non-Hermitian loss eigenvalues
CL_LIST_FINE   = np.linspace( CL_LIST[0],CL_LIST[-1],1000 )
E_NH_LOSS      = nH_LOSS(WC,A0,CL_LIST_FINE,EMOL_ARRAY).real
E_NH_LOSS     -= WC


cmap = "ocean_r"
extent = [EMIN,EMAX,0,CL_LIST[-1]/A0]


plt.imshow( DOS, interpolation="spline16", origin='lower', cmap=cmap, extent=extent, aspect='auto' )
plt.colorbar(pad=0.01)
plt.plot( E_NH_LOSS[:,1]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
plt.plot( E_NH_LOSS[:,2]/A0, CL_LIST_FINE/A0, "--", c='black' )
plt.plot( CL_LIST*0 - 1, CL_LIST/A0, "--", c='red' )
plt.plot( CL_LIST*0 + 1, CL_LIST/A0, "--", c='red' )
plt.xlabel('Energy, $(E-\\omega_\\mathrm{c})~/~A_0$', fontsize=15)
plt.ylabel('Normalized Cavity Loss Rate, $\\Gamma_\\mathrm{c}~/~A_0$', fontsize=15)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/LOSS_1MOL_MM_DOS_NMODE_{NMODE}_NMOL_{NMOL}_SIGE_{SIGE}.jpg', dpi=300)
plt.clf()

plt.imshow( TM, interpolation="spline16", origin='lower', cmap=cmap, extent=extent, aspect='auto' )
plt.colorbar(pad=0.01)
plt.plot( E_NH_LOSS[:,1]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
plt.plot( E_NH_LOSS[:,2]/A0, CL_LIST_FINE/A0, "--", c='black' )
plt.plot( CL_LIST*0 - 1, CL_LIST/A0, "--", c='red' )
plt.plot( CL_LIST*0 + 1, CL_LIST/A0, "--", c='red' )
plt.xlabel('Energy, $(E-\\omega_\\mathrm{c})~/~A_0$', fontsize=15)
plt.ylabel('Normalized Cavity Loss Rate, $\\Gamma_\\mathrm{c}~/~A_0$', fontsize=15)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/LOSS_1MOL_MM_TM_NMODE_{NMODE}_NMOL_{NMOL}_SIGE_{SIGE}.jpg', dpi=300)
plt.clf()

plt.imshow( ABS, interpolation="spline16", origin='lower', cmap=cmap, extent=extent, aspect='auto' )
plt.colorbar(pad=0.01)
plt.plot( E_NH_LOSS[:,1]/A0, CL_LIST_FINE/A0, "--", c='black', label="nh-QED" )
plt.plot( E_NH_LOSS[:,2]/A0, CL_LIST_FINE/A0, "--", c='black' )
plt.plot( CL_LIST*0 - 1, CL_LIST/A0, "--", c='red' )
plt.plot( CL_LIST*0 + 1, CL_LIST/A0, "--", c='red' )
plt.xlabel('Energy, $(E-\\omega_\\mathrm{c})~/~A_0$', fontsize=15)
plt.ylabel('Normalized Cavity Loss Rate, $\\Gamma_\\mathrm{c}~/~A_0$', fontsize=15)
plt.tight_layout()
plt.savefig(f'{DATA_DIR}/LOSS_1MOL_MM_ABS_NMODE_{NMODE}_NMOL_{NMOL}_SIGE_{SIGE}.jpg', dpi=300)
plt.clf()
