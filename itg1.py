import NGKD
import sys
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

################ Parameter Initialization ###########################
num     = 200           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion

rho_H   = 1e-3
kxrho   = np.logspace(-6.0,-5,num)
kprho   = np.logspace(-2.0,2.0,num) # k_prp*rho_i
fzeta   = np.ones(num,dtype=complex)  # store zeta
dlnt_dlnp=np.linspace(1.0,0.3,num)

plotx = dlnt_dlnp
######################################################################
################### Main Code Starts Here ############################
for i in range(0,num):
  print( 'i=',i)
  if i==0:
    zeta = complex(0.5,2.9)
  else:
    zeta = fzeta[i-1]
  data  = (kxrho[0],kprho[0],tau,z,mu,rho_H,phi,dlnt_dlnp[i])
  sol   = root(NGKD.SOLVE1,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-8)
  fzeta[i] = complex(sol.x[0],sol.x[1])
  print fzeta[i]


######################## Plot the results #############################
# Plot zeta as a function of k_perp*rho_i or k_prl*rho_i
#fzeta = fzeta*kxrho/rho_H
flag  = (fzeta.imag)>0
plt.semilogy(plotx[flag],np.abs(fzeta.imag[flag])\
        ,'k-',linewidth=2,label='imag')

plt.legend(loc=0,prop={'size':20})
plt.xlabel(r'$dlnt/dlnp$')
plt.ylabel(r'$\zeta_i$')

plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

font = {'family' : 'sans-serif',
        'sans-serif' : 'Helvetica',
        'weight' : 'light',
        'size'   : 24}
matplotlib.rc('font', **font)

plt.axis([plotx[0],plotx[-1],1e-4,1e2])
fig = plt.gcf()
fig.set_size_inches(8, 5)
#plt.savefig('itg.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()


