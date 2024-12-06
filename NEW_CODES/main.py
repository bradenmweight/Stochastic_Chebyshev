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
def get_random_vector_real(shape):
    r_vec = np.array([ random() for _ in range(myprod(shape)) ])
    return r_vec.reshape(shape).astype(np.complex128) * 2 - 1 # r \in [-1,1)

# @njit
# def _sample( MEAN, WIDTH, MIN, MAX, P_type ):
#     count = 0
#     while ( True ):
#         if ( P_type.lower() == "gaussian" ):
#             rand = gauss(MEAN, WIDTH)
#         elif ( P_type.lower() == "lorentzian" ):
#             rand = MEAN + WIDTH * np.random.standard_cauchy()
#         else:
#             print("Error: Unknown probability type")
#             return None

#         if ( rand > MIN and rand < MAX ):
#             return rand
#         count += 1

@njit
def _sample( MEAN, WIDTH, MIN, MAX, P_type ):
    count = 0
    if ( P_type.lower() == "gaussian" ):
        rand = gauss(MEAN, WIDTH)
    elif ( P_type.lower() == "lorentzian" ):
        rand = MEAN + WIDTH * np.random.standard_cauchy()
    else:
        print("Error: Unknown probability type")
        return None

    if ( rand < MIN ):
        return MIN + 1e-5
    elif ( rand > MAX ):
        return MAX - 1e-5
    else:
        return rand


@njit
def _sample_random_frequency( WC, E0, CAV_LOSS, dH, batch_size, P_type ):
    TMP = np.zeros( batch_size, dtype=np.complex128 )
    for i in range(batch_size):
        TMP[i] = _sample( WC - E0, 0.500 * CAV_LOSS, -dH/2, dH/2, P_type ) # 0.500 here
    return TMP

class Params():
    def __init__(self,N_MOL=1,N_CHEB=100,N_STOCHASTIC=100,Ept=1.0,
                 GAM=0.01,E0=1.0,dH=0.4,F_type="gaussian",P_type="Lorentzian",
                 A0=0.0,WC=0.0,CAV_LOSS=0.0,CAV_LOSS_TYPE="non-hermitian",HAM="JC",
                 batch_size=1,SIG_E=0.0,doEXACT=True
                 ):
        
        self.doEXACT       = doEXACT
        self.N_MOL         = N_MOL
        #self.N_EL          = 2
        self.SIG_E         = SIG_E
        self.MOL_DISORDER  = "gaussian"
        self.F_type        = F_type # Type of regularization ("Gaussian" or "1_over_E" or "lorentzian")
        self.P_type        = P_type # Type of regularization ("Gaussian" or "Lorentzian")
        self.N_CHEB        = N_CHEB
        self.N_STOCHASTIC  = N_STOCHASTIC
        self.Ept           = Ept # Energy at which to evaluate DOS
        self.dH            = dH # Width of DOS
        self.E0            = E0 # Center of DOS
        self.A0            = A0 / np.sqrt( self.N_MOL ) # Normalized Light-matter coupling strength (a.u.)
        self.CAV_LOSS      = CAV_LOSS # Cavity loss (a.u.)
        self.CAV_LOSS_TYPE = CAV_LOSS_TYPE # Type of cavity loss ("stochastic" or "non-hermitian")
        self.WC            = WC # Cavity frequency (a.u.)
        self.HAM           = HAM # Hamiltonian type ("JC")
        self.GAM           = GAM # Regularization parameter

        self.batch_size    = batch_size
        self.nbatch        = self.N_STOCHASTIC // self.batch_size

        self._build()

    def _build(self):

        if ( self.HAM == "JC" ):
            self.DIM_H       = self.N_MOL + 1
            self.MATTER_PROJ = ( np.arange( self.N_MOL ) ) # Matter indices
            self.PHOTON_PROJ = ( np.arange( self.N_MOL, self.DIM_H ) ) # Photon indices

            if ( self.MOL_DISORDER == 'gaussian' ):
                self.E_MOL = np.random.normal( loc=self.E0, scale=0.500*self.SIG_E, size=self.N_MOL )
            elif ( self.MOL_DISORDER == 'lorentzian' ):
                self.E_MOL = self.E0 + 0.500*self.SIG_E * np.random.standard_cauchy( size=self.N_MOL )
            elif ( self.MOL_DISORDER == 'rectangular' ):
                self.E_MOL = np.random.uniform( self.E0-0.500*self.SIG_E, self.E0+0.500*self.SIG_E, size=self.N_MOL )
            self.E_MOL         = self.E_MOL.astype(dtype=np.complex128)
            self.E_MOL_SHIFTED = self.E_MOL - self.E0
            self.MU_MOL        = np.ones( self.N_MOL, dtype=np.complex128 )
            self.MU_MOL_SCALED = self.WC * self.A0 * self.MU_MOL
            
        self.theta = np.linspace(0,2*np.pi,2000, dtype=np.complex128)
        self.dth   = self.theta[1] - self.theta[0]
        self.c_l   = np.zeros(self.N_CHEB, dtype=np.complex128)
        self.DOS   = np.zeros(3, dtype=np.complex128)

        if ( self.batch_size == -1 ):
            self.Hvec  = np.zeros(self.DIM_H, dtype=np.complex128)
            self.r_vec = np.zeros(self.DIM_H, dtype=np.complex128)
            self.v0    = np.zeros(self.DIM_H, dtype=np.complex128)
            self.v1    = np.zeros(self.DIM_H, dtype=np.complex128)
            self.v2    = np.zeros(self.DIM_H, dtype=np.complex128)
        else:
            self.Hvec  = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.r_vec = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.v0    = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.v1    = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
            self.v2    = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )

    def get_current_WC(self):
        if ( self.CAV_LOSS_TYPE == "stochastic" ):
            return _sample_random_frequency( self.WC, self.E0, self.CAV_LOSS, self.dH, self.batch_size, self.P_type )
        elif ( self.CAV_LOSS_TYPE == "non-hermitian" ):
            return (self.WC - self.E0) - 1j * 0.500 * self.CAV_LOSS

    def evaluate_DOS_initial(self, c0, c1):
        TOTAL   = c0 * get_Inner_Product_batch( self.r_vec[:,:],                self.v0[:,:]                )
        MATTER  = c0 * get_Inner_Product_batch( self.r_vec[self.MATTER_PROJ,:], self.v0[self.MATTER_PROJ,:] )
        PHOTON  = c0 * get_Inner_Product_batch( self.r_vec[self.PHOTON_PROJ,:], self.v0[self.PHOTON_PROJ,:] )
        TOTAL  += c1 * get_Inner_Product_batch( self.r_vec[:,:],                self.v1[:,:]                )
        MATTER += c1 * get_Inner_Product_batch( self.r_vec[self.MATTER_PROJ,:], self.v1[self.MATTER_PROJ,:] )
        PHOTON += c1 * get_Inner_Product_batch( self.r_vec[self.PHOTON_PROJ,:], self.v1[self.PHOTON_PROJ,:] )
        return np.array([TOTAL, MATTER, PHOTON])

    def evaluate_DOS( self, c ):
        TOTAL   = c * get_Inner_Product_batch( self.r_vec[:,:],                self.v2[:,:]                )
        MATTER  = c * get_Inner_Product_batch( self.r_vec[self.MATTER_PROJ,:], self.v2[self.MATTER_PROJ,:] )
        PHOTON  = c * get_Inner_Product_batch( self.r_vec[self.PHOTON_PROJ,:], self.v2[self.PHOTON_PROJ,:] )
        return np.array([TOTAL, MATTER, PHOTON])

    def exact_diagonalization(self):
        from Hamiltonian import build_Exact
        H = build_Exact(self)
        E,U = np.linalg.eig( H )

        MATT_CHAR = np.einsum("aj->j", U[self.MATTER_PROJ,:]**2).real
        PHOT_CHAR = np.einsum("aj->j", U[self.PHOTON_PROJ,:]**2).real

        MATT_WIDTH     = MATT_CHAR * np.max([self.GAM, self.SIG_E])
        PHOT_WIDTH     = PHOT_CHAR * np.max([self.GAM, self.CAV_LOSS])
        TOT_WIDTH      = MATT_WIDTH + PHOT_WIDTH
        self.DOS_EXACT = np.zeros( 3, dtype=np.complex128 )
        if ( self.F_type.lower() == 'gaussian' ):
            self.DOS_EXACT[0] = np.sum( 1.0000000 * np.exp( -(self.Ept - E[:].real)**2 / 2 / TOT_WIDTH**2 ) )
            self.DOS_EXACT[1] = np.sum( MATT_CHAR * np.exp( -(self.Ept - E[:].real)**2 / 2 / TOT_WIDTH**2 ) )
            self.DOS_EXACT[2] = np.sum( PHOT_CHAR * np.exp( -(self.Ept - E[:].real)**2 / 2 / TOT_WIDTH**2 ) )
        elif ( self.F_type.lower() == '1_over_e' ):
            self.DOS_EXACT[0] =  np.sum( 1.0000000 * TOT_WIDTH  / ( self.Ept - E[:].real + 1j*TOT_WIDTH ) )
            self.DOS_EXACT[1] =  np.sum( MATT_CHAR * MATT_WIDTH / ( self.Ept - E[:].real + 1j*MATT_WIDTH ) )
            self.DOS_EXACT[2] =  np.sum( PHOT_CHAR * PHOT_WIDTH / ( self.Ept - E[:].real + 1j*PHOT_WIDTH ) )
        elif ( self.F_type.lower() == 'lorentzian' ):
            self.DOS_EXACT[0] =  np.sum( 1.0000000 * TOT_WIDTH  / np.pi / ( (self.Ept - E[:].real)**2 + TOT_WIDTH **2 ) )
            self.DOS_EXACT[1] =  np.sum( MATT_CHAR * MATT_WIDTH / np.pi / ( (self.Ept - E[:].real)**2 + MATT_WIDTH**2 ) )
            self.DOS_EXACT[2] =  np.sum( PHOT_CHAR * PHOT_WIDTH / np.pi / ( (self.Ept - E[:].real)**2 + PHOT_WIDTH**2 ) )




    def stochastic_chebyshev_serial(self):
        from chebyshev import get_coeffs, get_DOS_PHOTON
        from Hamiltonian import H_JC_on_vec
        coeffs     = get_coeffs( self )
        for _ in range( self.N_STOCHASTIC ):
            self.r_vec[:]  = get_random_vector_complex(self.DIM_H)
            if ( self.CAV_LOSS_TYPE == "stochastic" ): self.WC_SHIFTED = get_random_cavity_frequency( self.WC, self.CAV_LOSS )
            self.v0[:]     = self.r_vec[:]
            self.v1[:]     = H_JC_on_vec( self.Hvec*0, self.v0, self.N_MOL, self.E_MOL_SHIFTED, self.MU_MOL_SCALED, self.WC_SHIFTED, self.dH )
            self.DOS[-1]  += get_DOS_PHOTON( coeffs[0], self.r_vec[self.PHOTON_PROJ], self.v0[self.PHOTON_PROJ] )
            self.DOS[-1]  += get_DOS_PHOTON( coeffs[1], self.r_vec[self.PHOTON_PROJ], self.v1[self.PHOTON_PROJ] )
            for n in range( 2, self.N_CHEB ):
                self.v2[:]     = 2 * H_JC_on_vec( self.Hvec*0, self.v1, self.N_MOL, self.E_MOL_SHIFTED, self.MU_MOL_SCALED, self.WC_SHIFTED, self.dH ) - self.v0
                self.DOS[-1]  += get_DOS_PHOTON( coeffs[n], self.r_vec, self.v2 )
                self.v0[:]     = self.v1[:] * 1
                self.v1[:]     = self.v2[:] * 1
        self.DOS[-1] = self.DOS[-1] / self.N_STOCHASTIC

    def stochastic_chebyshev_batch(self):
        coeffs     = get_coeffs( self )
        self.Hvec  = np.zeros( (self.DIM_H, self.batch_size), dtype=np.complex128 )
        for batchi in range( self.nbatch ):
            #print("\tWorking on batch %d of %d" % (batchi, self.nbatch) )
            DOS_TMP          = np.zeros( (3, self.batch_size), dtype=np.complex128 )
            #self.r_vec[:,:]  = get_random_vector_complex( (self.DIM_H, self.batch_size) )
            self.r_vec[:,:]  = get_random_vector_real( (self.DIM_H, self.batch_size) )
            self.WC_SHIFTED  = self.get_current_WC()
            if ( abs(np.min(self.WC_SHIFTED) - 0.5*self.dH ) < 1e-3 or abs(np.max(self.WC_SHIFTED) - 0.5*self.dH ) < 1e-3 ):
                print( "MIN: %1.3f AVE: %1.3f STD %1.3f MAX %1.3f" % (np.min(self.WC_SHIFTED), np.average(self.WC_SHIFTED), np.std(self.WC_SHIFTED), np.max(self.WC_SHIFTED)) )
            self.v0[:,:]     = self.r_vec[:,:] * 1
            self.v1[:,:]     = H_JC_on_vec_batch( self.Hvec*0, self.v0, self.N_MOL, self.E_MOL_SHIFTED, self.MU_MOL_SCALED, self.WC_SHIFTED, self.dH )
            DOS_TMP         += self.evaluate_DOS_initial( coeffs[0], coeffs[1] )
            for n in range( 2, self.N_CHEB ):
                self.v2[:,:]     = 2 * H_JC_on_vec_batch( self.Hvec*0, self.v1, self.N_MOL, self.E_MOL_SHIFTED, self.MU_MOL_SCALED, self.WC_SHIFTED, self.dH ) - self.v0
                DOS_TMP         += self.evaluate_DOS( coeffs[n] )
                self.v0[:,:]     = self.v1[:,:] * 1
                self.v1[:,:]     = self.v2[:,:] * 1
            self.DOS += np.sum(DOS_TMP[:,:],axis=1) # Sum over random vectors
        self.DOS = self.DOS / self.N_STOCHASTIC

    def run(self):
        if ( self.doEXACT == True and self.DIM_H <= 10_001 ):
           self.exact_diagonalization()
        else:
            self.DOS_EXACT = np.array(['nan','nan','nan'])
            self.DOS_EXACT_APPROX = np.array(['nan','nan','nan'])
        
        #if ( self.batch_size == 1):
        #self.stochastic_chebyshev_serial()
        #else:
        self.stochastic_chebyshev_batch()



if ( __name__ == "__main__" ):
    pass