import numpy as np
from numba import njit


def F_Gaussian( params, x ):
    """
    Gaussian-regularized delta-function in angle space
    """
    F = np.exp( -x**2 / 2 / params.GAM**2 ) / np.sqrt(2 * np.pi) / params.GAM
    return F + 0.j

def F_1_over_E( params, x ):
    """
    Gaussian-regularized delta-function in angle space
    """
    F = params.GAM / ( x + 1j*params.GAM )
    return F + 0.j

def F_Lorentzian( params, x ):
    """
    Gaussian-regularized delta-function in angle space
    """
    F = params.GAM / ( x**2 + params.GAM**2 ) / np.pi
    return F + 0.j

@njit
def get_DFT( F, dth, theta, N_CHEB ):
    DFT      = np.exp(1j * np.outer(theta,np.arange(N_CHEB))) # Fourier Kernel
    c_l      =  F @ DFT * dth
    c_l[1:] *= 2
    return c_l

def get_coeffs( params ):
    """
    Ept (float): Energy at which DOS will be evaluated
    E0  (float): Center of the DOS
    dH  (float): Width of the DOS
    N_CHEB (int): Number of Chebyshev coefficients
    type (str): Type of regularization ("Gaussian" or "1_over_E")
    """
    x = params.dH*np.cos(params.theta) - (params.Ept-params.E0)
    if ( params.F_type.lower() == 'gaussian' ):
        F        = F_Gaussian( params, x )
    elif ( params.F_type.lower() == '1_over_e' ):
        F        = F_1_over_E( params, x )
    elif ( params.F_type.lower() == 'lorentzian' ):
        F        = F_Lorentzian( params, x )
    return get_DFT( F, params.dth, params.theta, params.N_CHEB )

def get_Inner_Product( L_vec, R_vec ):
    return np.einsum( "a,a->", L_vec, R_vec )

def get_Inner_Product_batch_einsum( L_vec, R_vec ):
    return np.einsum( "aR,aR->R", L_vec, R_vec )

@njit
def get_Inner_Product_batch( L_vec, R_vec ):
    return np.sum( L_vec * R_vec, axis=0 )

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