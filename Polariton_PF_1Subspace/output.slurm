[-0.039925    0.82285065  1.21712234]
0 of 200
Time to get coefficients: 1.118 s
Traceback (most recent call last):
  File "stochastic_Aggregate.py", line 237, in <module>
    main()
  File "stochastic_Aggregate.py", line 225, in main
    DOS_STOCHASTIC = do_STOCHASTIC_DOS()
  File "stochastic_Aggregate.py", line 75, in do_STOCHASTIC_DOS
    DOS[pt,:] = do_Stochastic_Chebyshev( DOS[pt,:], c_l, N, EGS, E, pt, MU, DSE_AA, DSE_GS, WC, A0_SCALED, dH, E0, M, P )
  File "/scratch/bweight/anaconda3/lib/python3.8/site-packages/numba/core/dispatcher.py", line 482, in _compile_for_args
    error_rewrite(e, 'typing')
  File "/scratch/bweight/anaconda3/lib/python3.8/site-packages/numba/core/dispatcher.py", line 423, in error_rewrite
    raise e.with_traceback(None)
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
Failed in nopython mode pipeline (step: nopython frontend)
Failed in nopython mode pipeline (step: nopython frontend)
Use of unsupported NumPy function 'numpy.einsum' or unsupported use of the function.

File "Hamiltonian.py", line 114:
def get_H_vec_norm( N, EGS, E, MU, DSE_AA, DSE_GS, WC, A0, dH, E0, vec ):
    <source elided>
    Hvec[0]     += WC * A0**2 * np.sum( DSE_AA[:,0,1] * vec[1:N+1] )
    Hvec[0]     += 2 * WC * A0**2 * (np.einsum( "A,B,B->", MU[:,0,0], MU[:,0,1], vec[1:N+1] ) \
    ^

During: typing of get attribute at /gpfs/fs2/scratch/bweight/software/Stochastic_Chebyshev/Polariton_PF_1Subspace/Hamiltonian.py (114)

File "Hamiltonian.py", line 114:
def get_H_vec_norm( N, EGS, E, MU, DSE_AA, DSE_GS, WC, A0, dH, E0, vec ):
    <source elided>
    Hvec[0]     += WC * A0**2 * np.sum( DSE_AA[:,0,1] * vec[1:N+1] )
    Hvec[0]     += 2 * WC * A0**2 * (np.einsum( "A,B,B->", MU[:,0,0], MU[:,0,1], vec[1:N+1] ) \
    ^

During: resolving callee type: type(CPUDispatcher(<function get_H_vec_norm at 0x2b491cdfcf70>))
During: typing of call at stochastic_Aggregate.py (120)

During: resolving callee type: type(CPUDispatcher(<function get_H_vec_norm at 0x2b491cdfcf70>))
During: typing of call at stochastic_Aggregate.py (120)


File "stochastic_Aggregate.py", line 120:
def get_T0_T1_vec( DOS, pt, c_l, N, EGS, E, MU, DSE_AA, DSE_GS, WC, A0_SCALED, dH, E0, r_vec, M, P ):
    <source elided>
    v0 = r_vec
    v1 = get_H_vec_norm( N, EGS, E, MU, DSE_AA, DSE_GS, WC, A0_SCALED, dH, E0, r_vec )
    ^

During: resolving callee type: type(CPUDispatcher(<function get_T0_T1_vec at 0x2b491ce64550>))
During: typing of call at stochastic_Aggregate.py (110)

During: resolving callee type: type(CPUDispatcher(<function get_T0_T1_vec at 0x2b491ce64550>))
During: typing of call at stochastic_Aggregate.py (110)


File "stochastic_Aggregate.py", line 110:
def do_Stochastic_Chebyshev( DOS, c_l, N, EGS, E, pt, MU, DSE_AA, DSE_GS, WC, A0_SCALED, dH, E0, M, P ):
    <source elided>
        r_vec = np.array([ random.randint(0,1)*2.-1. for n in range(N+2) ]) # np.random.randint(0,high=2,size=N+2)*2. - 1
        v0, v1, DOS = get_T0_T1_vec( DOS, pt, c_l, N, EGS, E, MU, DSE_AA, DSE_GS, WC, A0_SCALED, dH, E0, r_vec, M, P )
        ^

