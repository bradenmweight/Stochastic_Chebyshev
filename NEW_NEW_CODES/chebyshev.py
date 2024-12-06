import numpy as np
from numba import njit
from time import time

import cmath

from scipy.integrate import quad_vec

@njit
def F_Gaussian( theta, chebs, EGRID, E0, dH, GAM ):
    dE    = dH*np.cos(theta) - (EGRID - E0)
    F     = np.exp(1j * theta * chebs[None,:] ) * np.exp( -dE[:,None,]**2 / 2 / GAM**2 ) / np.sqrt(2 * np.pi) / GAM
    return F

@njit
def F_1_over_E( theta, chebs, EGRID, E0, dH, GAM ):
    dE    = dH*np.cos(theta) - (EGRID - E0)
    F     = np.exp(1j * theta * chebs[None,:] ) * GAM / ( dE[:,None,] + 1j*GAM )
    return F

@njit
def F_Lorentzian( theta, chebs, EGRID, E0, dH, GAM ):
    dE    = dH*np.cos(theta) - (EGRID - E0)
    F     = np.exp(1j * theta * chebs[None,:] ) * GAM / ( dE[:,None,]**2 + GAM**2 ) / np.pi
    return F

@njit
def do_Riemann( c_l, FUNC, chebs, EGRID, E0, dH, GAM ):
    theta_list = np.linspace(0, 2*np.pi, 10000)
    dtheta     = theta_list[1] - theta_list[0]
    for theta in theta_list:
        c_l[:,:] += FUNC( theta, chebs, EGRID, E0, dH, GAM )
    c_l[:,:] = c_l * dtheta
    

    # tmp = np.zeros_like( c_l )
    # for l in chebs:
    #     z     = (EGRID[:] - E0 + 1j*GAM) / dH
    #     sqrt  = (z**2 - 1) ** (1/2) # (z**2 - 1) ** (1/2) # np.sqrt( z**2 - 1 )
    #     check = np.abs(z + sqrt)
    #     for ei,e in enumerate( EGRID ):
    #         if ( check[ei] > 1 ):
    #             tmp[ei,l] = 1 / sqrt[ei] / ( z[ei] + sqrt[ei] )**l
    #         else:
    #             tmp[ei,l] = 1 / sqrt[ei] / ( z[ei] - sqrt[ei] )**l
    # tmp[:,1:] *= 2
    # #tmp[:,:]  *= dH / 2.5

    # tmp = np.zeros_like( c_l )
    # tmp[:,0] = c_l[:,0]
    # tmp[:,1] = c_l[:,1]
    # z        = (EGRID[:] - E0) / dH
    # for l in range( 2, len(chebs) ):
    #     tmp[:,l] = 2 * z.conj() * tmp[:,l-1] - tmp[:,l-2]

    # c_l[:,1:] *= 2
    # tmp[:,1:] *= 2

    #print("Num.:\n", c_l[len(EGRID)//2,:4].imag )
    #print("Ana.:\n", tmp[len(EGRID)//2,:4].imag )
    # print("Ana./Num.:\n", np.round(tmp[len(EGRID)//2,:6].real / c_l[len(EGRID)//2,:6].real,4) )
    # print("Ana./Num.:\n", np.round(tmp[len(EGRID)//2,:6].imag / c_l[len(EGRID)//2,:6].imag,4) )
    # print( "Error:", np.linalg.norm( c_l - tmp ) )

    #exit()
    #return tmp
    return c_l

def get_DFT( params ):
    use_scipy_integrate = False
    # Using scipy's quad function
    chebs = np.arange(params.N_CHEB)
    c_l   = np.zeros( (len(params.EGRID), params.N_CHEB), dtype=np.complex128 )
    T0    = time()
    if ( params.F_type == "gaussian" ):
        FUNC  = F_Gaussian
    elif ( params.F_type == "1_over_e" ):
        FUNC  = F_1_over_E
    elif ( params.F_type == "lorentzian" ):
        FUNC  = F_Lorentzian
    if ( use_scipy_integrate == True ):
        print("\tComputing %d Chebyshev coefficients with scipy.integrate.quad_vec" % (params.N_CHEB))
        c_l[:,:], error = quad_vec(FUNC, 0, 2*np.pi, limit=5000, workers=1, quadrature="trapezoid", epsrel=1e-3, args=(chebs, params.EGRID, params.E0, params.dH, params.GAM))
        c_l[:,1:] *= 2
        print("\tError in Chebyshev coefficients: %1.8f" % (error) )
    else:
        print("\tComputing %d Chebyshev coefficients with Riemann sum" % (params.N_CHEB))
        c_l[:,:] = do_Riemann( c_l, FUNC, chebs, params.EGRID, params.E0, params.dH, params.GAM )

    print("\t%d Chebyshev coefficients calculated in %1.2f seconds" % (params.N_CHEB, time()-T0))
    return c_l






def get_coeffs( params ):
    """
    EGRID (float): Energies at which DOS will be evaluated
    E0  (float): Center of the DOS
    dH  (float): Width of the DOS
    N_CHEB (int): Number of Chebyshev coefficients
    type (str): Type of regularization ("Gaussian" or "1_over_E")
    """
    return get_DFT( params )


def get_Inner_Product_batch_einsum( L_vec, R_vec ):
    return np.einsum( "aR,aR->R", L_vec, R_vec ) # (a,R) -> (R)

@njit
def get_Inner_Product_batch( L_vec, R_vec ):
    return np.sum( L_vec * R_vec, axis=0 ) # (a,R) -> (R)

def normalize_coeffs( coeffs ):
    """
    Normalize the Chebyshev coefficients
    For plotting purposes only
    """
    return coeffs / np.sqrt( np.sum( np.abs(coeffs)**2 ) )

if ( __name__ == "__main__" ):
    import matplotlib.pyplot as plt
    from main import Params

    N_CHEB = 300
    dH     = 1.0
    GAM    = 0.01

    params = Params( F_type='Gaussian', N_CHEB=N_CHEB, dH=dH, GAM=GAM )
    coeffs = get_coeffs( params )
    coeffs = normalize_coeffs( coeffs )
    L_EFF   = np.sum(np.arange( N_CHEB ) * np.abs(coeffs)**2)
    L2_EFF  = np.sqrt(np.sum(np.arange( N_CHEB )**2 * np.abs(coeffs)**2))
    plt.plot( np.abs(coeffs)**2, "o", ms=4, c='black', label='Gaussian ($\\langle L \\rangle$ = %1.1f  $\\sqrt{\\langle L^2 \\rangle}$ = %1.1f)' % (L_EFF, L2_EFF) )
    params = Params( F_type='lorentzian', N_CHEB=N_CHEB, dH=dH, GAM=GAM )
    coeffs = get_coeffs( params )
    coeffs = normalize_coeffs( coeffs )
    L_EFF  = np.sum(np.arange( N_CHEB ) * np.abs(coeffs)**2)
    L2_EFF  = np.sqrt(np.sum(np.arange( N_CHEB )**2 * np.abs(coeffs)**2))
    plt.plot( np.abs(coeffs)**2, "o", ms=4, c="red", label='Lorentzian ($\\langle L \\rangle$ = %1.1f  $\\sqrt{\\langle L^2 \\rangle}$ = %1.1f)' % (L_EFF, L2_EFF) )
    params = Params( F_type='1_over_E', N_CHEB=N_CHEB, dH=dH, GAM=GAM )
    coeffs = get_coeffs( params ).imag
    coeffs = normalize_coeffs( coeffs )
    L_EFF  = np.sum(np.arange( N_CHEB ) * np.abs(coeffs)**2)
    L2_EFF  = np.sqrt(np.sum(np.arange( N_CHEB )**2 * np.abs(coeffs)**2))
    plt.plot( np.abs(coeffs)**2, "o", ms=2, c="blue", label='1_over_E ($\\langle L \\rangle$ = %1.1f  $\\sqrt{\\langle L^2 \\rangle}$ = %1.1f)' % (L_EFF, L2_EFF) )


    plt.legend()
    plt.xlabel("Chebyshev Expansion Index, $l$", fontsize=15)
    plt.ylabel("Chebyshev Coefficient, $|C_l|^2$", fontsize=15)
    plt.tight_layout()
    plt.savefig("chebyshev_expansion.png", dpi=300)