import numpy as np
import matplotlib.pyplot as plt

E = np.linspace( -0.3, 0.3, 75 )

REF  = np.exp( -(E - 0.1 * 1j)**2 / 2 / 0.1**2 )

color_list = ["black", "red", "blue", "green", "orange"]
for CLi,CL in enumerate([0.0, 0.1, 0.2]):
    F  = np.exp( -(E - CL * 1j)**2 / 2 / 0.1**2 )
    if ( CLi == 0 ):
        plt.plot( E, np.abs(F) / np.max(np.abs(REF)), "-", alpha=0.25,  lw=8, c=color_list[CLi], label="Abs" )
        plt.plot( E, F.real    / np.max(np.abs(REF)),    "--", lw=2, c=color_list[CLi], label="Re" )
        plt.plot( E, F.imag    / np.max(np.abs(REF)),    "o",  lw=2, c=color_list[CLi], label="Im" )
    else:
        plt.plot( E, np.abs(F)  / np.max(np.abs(REF))  , "-", alpha=0.25,  lw=8, c=color_list[CLi] )
        plt.plot( E, F.real     / np.max(np.abs(REF))     ,    "--", lw=2, c=color_list[CLi] )
        plt.plot( E, F.imag     / np.max(np.abs(REF))     ,    "o",  lw=2, c=color_list[CLi] )
    VAL = np.exp( E**2 )
    print( CL, np.max(np.abs(F)), VAL, VAL / np.max(np.abs(F)) )

plt.legend()
plt.savefig("Gaussian.jpg", dpi=300)





