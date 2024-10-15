import numpy as np


def Ham_2x2( WC, A0, CL_LIST ):
    E_NH_LOSS = np.zeros( (len(CL_LIST),4), dtype=np.complex128 )
    a         = np.diag( np.sqrt(np.ones(2-1)),k=1)
    MU_minus  = np.array([[0.,1],[0,0.]])
    MU_minus  = np.array([[0.,1],[0,0.]])
    MU_plus   = MU_minus.T
    for CLi,CL in enumerate(CL_LIST):
        WC_FREQ  = WC * np.arange(2) * (1 + 0.5*CL*1j)
        H_PF = np.zeros( (4,4), dtype=np.complex128 )
        H_PF +=               np.kron( WC*np.diag([0,1]), np.eye(2) )
        H_PF +=               np.kron( np.eye(2), np.diag( WC_FREQ ) )
        H_PF += WC * A0 *     np.kron( MU_minus, a.T )
        H_PF += WC * A0 *     np.kron( MU_plus,  a )
        E_NH_LOSS[CLi,:],_ = np.linalg.eig(H_PF)
        inds                = E_NH_LOSS[CLi,:].real.argsort()   
        E_NH_LOSS[CLi,:]    = E_NH_LOSS[CLi,inds]
    return E_NH_LOSS
    
def Ham_NxN( WC, A0, CL_LIST, EMOL_ARRAY ):
    NPOL      = len(EMOL_ARRAY) + 1
    E_NH_LOSS = np.zeros( (len(CL_LIST),4), dtype=np.complex128 )
    a         = np.diag( np.sqrt(np.ones(2-1)),k=1)
    MU_minus  = np.array([[0.,1],[0,0.]])
    MU_minus  = np.array([[0.,1],[0,0.]])
    MU_plus   = MU_minus.T
    for CLi,CL in enumerate(CL_LIST):
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
        E_NH_LOSS[CLi,:],_ = np.linalg.eig(H_PF)
        inds                = E_NH_LOSS[CLi,:].real.argsort()   
        E_NH_LOSS[CLi,:]    = E_NH_LOSS[CLi,inds]
    return E_NH_LOSS

def main( WC, A0, CL_LIST, EMOL_ARRAY ):
    E_NH_LOSS = Ham_2x2( WC, A0, CL_LIST )
    #E_NH_LOSS = Ham_NxN( WC, A0, CL_LIST, EMOL_ARRAY )
    return E_NH_LOSS