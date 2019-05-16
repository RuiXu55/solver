import NGKD
import sys
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

################ Parameter Initialization ###########################
num     = 500           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion

rho_H   = 1e-2
kxrho   = np.logspace(-5,-5,num)
kprho   = np.logspace(-1.0,2.0,num) # k_prp*rho_i
fzeta   = np.ones(num,dtype=complex)  # store zeta

plotx = kprho
dlnt_dlnp=1.0
######################################################################
################### Main Code Starts Here ############################
for i in range(0,num):
  print( 'i=',i)
  if i==0:
    zeta = complex(0.1,1.9)
  else:
    zeta = fzeta[i-1]
  data  = (kxrho[0],kprho[i],tau,z,mu,rho_H,phi)
  sol   = root(NGKD.SOLVE3,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-8)
  fzeta[i] = complex(sol.x[0],sol.x[1])
  print fzeta[i]*kxrho[0]/rho_H


######################## Plot the results #############################
# Plot zeta as a function of k_perp*rho_i or k_prl*rho_i
#fzeta = fzeta*kxrho/rho_H
flag  = (fzeta.imag)>0
plt.loglog(plotx[flag],np.abs(fzeta.imag[flag])\
        ,'k-',linewidth=2,label='imag')
plt.loglog(plotx,np.abs(fzeta.real)\
        ,'r-',linewidth=2,label='real')

plt.legend(loc=0,prop={'size':20})
plt.xlabel(r'$k_\perp \rho_i$')
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


