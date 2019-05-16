import DK1
import sys
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, font_manager
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable



##################################################################### 
################ Parameter Initialization ###########################
#ind=0 for homogeneous plasma,ind=1 for dlnt/dz<0,ind=2 for dlnt/dz>0
ind     = 0             # Index for plasma state 
num     = 200            # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e10           # Ion plasma beta
d_H     = 1e-6         # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kprho  = np.logspace(-4.0,-2.0,num) # k_prp*rho_i
kxrho  = 1e-6
#fzeta   = np.ones((num,num),dtype=complex)  # store zeta
fzeta   = np.ones(num,dtype=complex)  # store zeta
plotx   = kprho

######################################################################
################### Main Code Starts Here ############################
# loop over kxrho
for i in range(0,num):
  # loop over kprho
  if i==0:
    zeta = complex(1e-6,0.4)
  else:
    zeta = fzeta[i-1]
  data = (beta,kxrho,kprho[i],tau,z,mu,rho_H,ind,phi)
  sol = root(DK1.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
  fzeta[i] = complex(sol.x[0],sol.x[1])
  print 'fz=',fzeta[i]*kxrho/rho_H
######################## Plot the results #############################
ypos  = 1e-1
size  = 24 
flag  = fzeta.imag>0
plt.loglog(plotx[flag]/rho_H,np.abs(fzeta.imag[flag])*kxrho/rho_H\
      ,'k-',linewidth=2.5,label='Imag')
flag  = fzeta.imag<0
plt.loglog(plotx[flag]/rho_H,np.abs(fzeta.imag[flag])*kxrho/rho_H\
      ,'k--',linewidth=2.5)
flag  = fzeta.real>0
plt.loglog(plotx[flag]/rho_H,np.abs(fzeta.real[flag])*kxrho/rho_H\
      ,'r-',linewidth=2.5,label='Real')
flag  = fzeta.real<0
plt.loglog(plotx[flag]/rho_H,np.abs(fzeta.real[flag])*kxrho/rho_H\
      ,'r--',linewidth=2.5)

plt.xlabel(r'$k_\parallel H$',fontsize=size)
plt.ylabel(r'$\omega/(v_{th}/H)$',fontsize=size)
#plt.title(r'$kH=10,\beta=10^4$',fontsize=28,y=1.02)
plt.axis([plotx[0]/rho_H,plotx[-1]/rho_H,1e-4,1e2])
#plt.legend(loc=0)
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : size}
matplotlib.rc('font', **font)
plt.tick_params(pad=10)

fig = plt.gcf()
plt.show()

