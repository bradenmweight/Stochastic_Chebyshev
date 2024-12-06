import numpy as np
from matplotlib import pyplot as plt
from random import random, gauss
from numba import njit

from chebyshev import get_coeffs, get_Inner_Product_batch
from Hamiltonian import H_JC_on_vec_batch

@njit
def myprod(shape):
    p = 1
    for s in shape:
        p *= s
    return p

@njit
def get_random_vector_complex(shape):
    r_vec = np.array([ random() for _ in range(myprod(shape)) ])
    r_vec = np.exp( 1j * 2 * np.pi * r_vec )
    return r_vec.reshape(shape) # |r| = 1, angle \in [0,2\pi)

@njit
def _sample( MEAN, WIDTH, MIN, MAX, P_type ):
    count = 0
    if ( P_type.lower() == "gaussian" ):
        rand = gauss(MEAN, WIDTH)
    elif ( P_type.lower() == "lorentzian" ):
        rand = MEAN + WIDTH * np.random.standard_cauchy()
    elif ( P_type.lower() == "rectangular" ):
        rand = np.random.uniform( MEAN-0.5*WIDTH, MEAN+0.5*WIDTH )
    else:
        print("Error: Unknown probability type")
        return None

    ##### THIS IS WHAT KILLS US #####
    if ( rand < MIN ):
        return MIN + 1e-5
    elif ( rand > MAX ):
        return MAX - 1e-5
    else:
        return rand


@njit
def _sample_random_frequency( MEAN, WIDTH, dH, size, P_type ):
    TMP = np.zeros( size, dtype=np.complex128 )
    for i in range( size ):
        TMP[i] = _sample( MEAN, WIDTH, -dH/2, dH/2, P_type )
    return TMP

class Params():
    def __init__(self,N_MOL=1,N_CHEB=100,N_STOCHASTIC=100,EGRID=np.linspace(0,2,100),
                 GAM=0.01,E0=1.0,dH=0.4,F_type="gaussian",P_type="Lorentzian",
                 A0=0.0,WC=0.0,CAV_LOSS=None,CAV_LOSS_TYPE=None,HAM="JC",
                 batch_size=1,SIG_E=0.0,doEXACT=False,MOL_DISORDER=None
                 ):
        
        self.doEXACT       = doEXACT
        self.N_MOL         = N_MOL
        #self.N_EL          = 2
        self.SIG_E         = SIG_E
        self.MOL_DISORDER  = MOL_DISORDER # Type of molecular disorder (None, "gaussian" or "lorentzian" or "rectangular")
        self.F_type        = F_type.lower() # Type of regularization ("Gaussian" or "1_over_E" or "lorentzian")
        self.N_CHEB        = N_CHEB
        self.N_STOCHASTIC  = N_STOCHASTIC
        self.dH            = dH # Width of DOS
        self.E0            = E0 # Center of DOS
        self.A0            = A0 / np.sqrt( self.N_MOL ) # Normalized Light-matter coupling strength (a.u.)
        self.CAV_LOSS_TYPE = CAV_LOSS_TYPE # Type of cavity loss ("stochastic" or "non-hermitian")
        self.CAV_LOSS      = CAV_LOSS # Cavity loss (a.u.)
        self.P_type        = P_type.lower() # if CAV_LOSS_TYPE == stochastic, Type of photon loss ("gaussian" or "lorentzian")
        self.WC            = WC # Cavity frequency (a.u.)
        self.HAM           = HAM # Hamiltonian type ("JC")
        self.GAM           = GAM # Regularization parameter
        self.EGRID         = EGRID # Energy grid on which to evaluate the DOS

        self.batch_size    = batch_size
        self.nbatch        = self.N_STOCHASTIC // self.batch_size

        self._build()

    def _build(self):

        if ( self.HAM == "JC" ):
            self.DIM_H       = self.N_MOL + 1
            self.MATTER_PROJ = ( np.arange( self.N_MOL ) ) # Matter indices
            self.PHOTON_PROJ = ( np.arange( self.N_MOL, self.DIM_H ) ) # Photon indices

            MU_MOL             = np.ones( self.N_MOL, dtype=np.complex128 )
            self.MU_MOL_SCALED = self.WC * self.A0 * MU_MOL
        else:
            print(f"Error: Unknown Hamiltonian type: {self.HAM}")
            exit()
            
        #self.c_l   = np.zeros( (len(self.EGRID), self.N_CHEB), dtype=np.complex128)
        self.DOS   = np.zeros( (len(self.EGRID),3), dtype=np.complex128)

        if ( self.batch_size == -1 ):
            self.Hvec  = np.zeros( (self.DIM_H), dtype=np.complex128)
            self.r_vec = np.zeros( (self.DIM_H), dtype=np.complex128)
            self.v0    = np.zeros( (self.DIM_H), dtype=np.complex128)
            self.v1    = np.zeros( (self.DIM_H), dtype=np.complex128)
            self.v2    = np.zeros( (self.DIM_H), dtype=np.complex128)
        else:
            self.Hvec  = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.r_vec = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.v0    = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.v1    = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.v2    = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )

    def get_current_WC(self):
        if ( self.CAV_LOSS_TYPE is None or abs(CAV_LOSS) < 1e-8 ):
            return self.WC - self.E0
        elif ( self.CAV_LOSS_TYPE == "stochastic" ):
            return _sample_random_frequency( self.WC - self.E0, 0.500 * self.CAV_LOSS, self.dH, self.batch_size, self.P_type )
        elif ( self.CAV_LOSS_TYPE == "non-hermitian" ):
            return (self.WC - self.E0) - 1j * 0.500 * self.CAV_LOSS

    def get_current_EMOL(self):
        if ( self.MOL_DISORDER is None or abs(self.SIG_E) < 1e-8 ):
            return np.zeros( (self.N_MOL, self.batch_size) )
        else:
            E_MOL = _sample_random_frequency( self.E0 - self.E0, 0.500*self.SIG_E, self.dH, self.N_MOL * self.batch_size, self.MOL_DISORDER )
            return E_MOL.reshape( (self.N_MOL, self.batch_size) )

    def evaluate_DOS_initial(self, c0, c1):
        TOTAL   = c0[:,None] * get_Inner_Product_batch( self.r_vec[:,:].conj(),                self.v0[:,:]                )[None,:]
        MATTER  = c0[:,None] * get_Inner_Product_batch( self.r_vec[self.MATTER_PROJ,:].conj(), self.v0[self.MATTER_PROJ,:] )[None,:]
        PHOTON  = c0[:,None] * get_Inner_Product_batch( self.r_vec[self.PHOTON_PROJ,:].conj(), self.v0[self.PHOTON_PROJ,:] )[None,:]
        TOTAL  += c1[:,None] * get_Inner_Product_batch( self.r_vec[:,:].conj(),                self.v1[:,:]                )[None,:]
        MATTER += c1[:,None] * get_Inner_Product_batch( self.r_vec[self.MATTER_PROJ,:].conj(), self.v1[self.MATTER_PROJ,:] )[None,:]
        PHOTON += c1[:,None] * get_Inner_Product_batch( self.r_vec[self.PHOTON_PROJ,:].conj(), self.v1[self.PHOTON_PROJ,:] )[None,:]
        return np.array([TOTAL, MATTER, PHOTON])

    def evaluate_DOS( self, c ):
        TOTAL   = c[:,None] * get_Inner_Product_batch( self.r_vec[:,:].conj(),                self.v2[:,:]                )[None,:]
        MATTER  = c[:,None] * get_Inner_Product_batch( self.r_vec[self.MATTER_PROJ,:].conj(), self.v2[self.MATTER_PROJ,:] )[None,:]
        PHOTON  = c[:,None] * get_Inner_Product_batch( self.r_vec[self.PHOTON_PROJ,:].conj(), self.v2[self.PHOTON_PROJ,:] )[None,:]
        return np.array([TOTAL, MATTER, PHOTON])

    def exact_diagonalization(self):
        from Hamiltonian import build_Exact
        H = build_Exact(self)
        E,U = np.linalg.eig( H )

        MATT_CHAR = np.einsum("aj->j", U[self.MATTER_PROJ,:]**2).real
        PHOT_CHAR = np.einsum("aj->j", U[self.PHOTON_PROJ,:]**2).real

        MATT_WIDTH     = MATT_CHAR * np.max([self.GAM, self.SIG_E])
        PHOT_WIDTH     = PHOT_CHAR * np.max([self.GAM, self.CAV_LOSS]) if ( self.CAV_LOSS_TYPE is not None ) else self.GAM
        TOT_WIDTH      = MATT_WIDTH + PHOT_WIDTH
        self.DOS_EXACT = np.zeros( 3, dtype=np.complex128 )
        if ( self.F_type.lower() == 'gaussian' ):
            self.DOS_EXACT[0] = np.sum( 1.0000000 * np.exp( -(self.EGRID[:,None] - E[None,:].real)**2 / 2 / TOT_WIDTH**2 ) )
            self.DOS_EXACT[1] = np.sum( MATT_CHAR * np.exp( -(self.EGRID[:,None] - E[None,:].real)**2 / 2 / TOT_WIDTH**2 ) )
            self.DOS_EXACT[2] = np.sum( PHOT_CHAR * np.exp( -(self.EGRID[:,None] - E[None,:].real)**2 / 2 / TOT_WIDTH**2 ) )
        elif ( self.F_type.lower() == '1_over_e' ):
            self.DOS_EXACT[0] =  np.sum( 1.0000000 * TOT_WIDTH  / ( self.EGRID[:,None] - E[None,:].real + 1j*TOT_WIDTH ) )
            self.DOS_EXACT[1] =  np.sum( MATT_CHAR * MATT_WIDTH / ( self.EGRID[:,None] - E[None,:].real + 1j*MATT_WIDTH ) )
            self.DOS_EXACT[2] =  np.sum( PHOT_CHAR * PHOT_WIDTH / ( self.EGRID[:,None] - E[None,:].real + 1j*PHOT_WIDTH ) )
        elif ( self.F_type.lower() == 'lorentzian' ):
            self.DOS_EXACT[0] =  np.sum( 1.0000000 * TOT_WIDTH  / np.pi / ( (self.EGRID[:,None] - E[None,:].real)**2 + TOT_WIDTH **2 ) )
            self.DOS_EXACT[1] =  np.sum( MATT_CHAR * MATT_WIDTH / np.pi / ( (self.EGRID[:,None] - E[None,:].real)**2 + MATT_WIDTH**2 ) )
            self.DOS_EXACT[2] =  np.sum( PHOT_CHAR * PHOT_WIDTH / np.pi / ( (self.EGRID[:,None] - E[None,:].real)**2 + PHOT_WIDTH**2 ) )

    def stochastic_chebyshev(self):
        #try:
        coeffs = self.coeffs
        #except:
        #    coeffs = get_coeffs( self )
        self.Hvec  = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
        for batchi in range( self.nbatch ):
            DOS_TMP            = np.zeros( (3, len(self.EGRID), self.batch_size), dtype=np.complex128 )
            self.r_vec[:,:]    = get_random_vector_complex( (self.DIM_H, self.batch_size) )
            self.WC_SHIFTED    = self.get_current_WC()
            self.E_MOL_SHIFTED = self.get_current_EMOL()
            self.v0[:,:]       = self.r_vec[:,:] * 1
            self.v1[:,:]       = H_JC_on_vec_batch( self.Hvec*0, self.v0, self.N_MOL, self.E_MOL_SHIFTED, self.MU_MOL_SCALED, self.WC_SHIFTED, self.dH )
            DOS_TMP           += self.evaluate_DOS_initial( coeffs[:,0], coeffs[:,1] )
            for n in range( 2, self.N_CHEB ):
                print(n, "of", self.N_CHEB)
                self.v2[:,:]     = 2 * H_JC_on_vec_batch( self.Hvec*0, self.v1, self.N_MOL, self.E_MOL_SHIFTED, self.MU_MOL_SCALED, self.WC_SHIFTED, self.dH ) - self.v0
                DOS_TMP         += self.evaluate_DOS( coeffs[:,n] )
                self.v0[:,:]     = self.v1[:,:] * 1
                self.v1[:,:]     = self.v2[:,:] * 1
            self.DOS += np.sum(DOS_TMP[:,:,:],axis=-1).swapaxes(0,1) # Sum over random vectors and swap ((3,EGRID) --> EGRID,3)
        self.DOS = self.DOS / self.N_STOCHASTIC # Average over random vectors

    def run(self):
        if ( self.doEXACT == True and self.DIM_H <= 10_001 ):
           self.exact_diagonalization()
        else:
            self.DOS_EXACT = np.array(['nan','nan','nan'])
            self.DOS_EXACT_APPROX = np.array(['nan','nan','nan'])
        
        self.stochastic_chebyshev()



if ( __name__ == "__main__" ):
    pass