# This code is used to calculate plasma dispersion relation. 1
# Equation solved is Full Vlasov and GKMHD equations
# in thermally stratified plasma.
# FILE:

  # main.py 
  # main file, set up the code and plot the results

  # GKD.py
  # dispersion relation in gk limit

  # FD.py
  # dispersion relation for full vlasov
 
  # DK.py
  # calculate the dispersion relation for full vlasov
  # at k_perp = 0

  # DK1.py
  # calculate the dispersion relation for full vlasov
  # at k_perp != 0

  # f.py
  # store subsidize functions: zeta function, parallel integral
  # and its derivative, derivative of exponentially modified bessel
  # function

# To run the code, just type: python main.py 
# Written by Rui Xu. Jan. 2016

import DK1
import numpy as np
#import Newton as F  
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatterMathtext



##################################################################### 
################ Parameter Initialization ###########################
solver  = 'DK1'
#ind=0 for homogeneous plasma,ind=1 for dlnt/dz<0,ind=2 for dlnt/dz>0
ind     = 1             # Index for plasma state 
num     = 20            # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e4           # Ion plasma beta
d_H     = 1e-8          # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kprho   = np.logspace(-7.0,-4.0,num) # k_prp*rho_i
kxrho   = np.logspace(-7.0,-4.0,num) # k_prp*rho_i
fzeta   = np.ones((num,num),dtype=complex)  # store zeta
plotx   = kprho

######################################################################
################### Main Code Starts Here ############################
# loop over kxrho
for i in range(0,num):
  print 'i=',i
  # loop over kprho
  for j in range(0,num):
    if i==0 and j==0:
      zeta = complex(1e-1,0.4)
    elif i==0:
      zeta = fzeta[i,j-1]+complex(-1e-3,-1e-3)
    else:
      zeta = fzeta[i-1,j]
    data = (beta,kxrho[i],kprho[j],tau,z,mu,rho_H,ind,phi)
    sol = root(DK1.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
    fzeta[i,j] = complex(sol.x[0],sol.x[1])
    print 'fz=',fzeta[i,j]*kxrho[i]/rho_H

######################## Plot the results #############################
# plot growth rate 
ax = plt.gca()
#myplt = plt.imshow(fzeta.imag,extent=[kxrho[0]/rho_H,kxrho[-1]/rho_H,
#   kprho[0]/rho_H,kprho[-1]/rho_H],cmap='jet',norm=LogNorm(),
#   interpolation='bilinear',origin='lower')
myplt = plt.imshow(fzeta.imag.transpose()*kxrho/rho_H,extent=[kxrho[0]/rho_H,kxrho[-1]/rho_H,
      kprho[0]/rho_H,kprho[-1]/rho_H],cmap='jet',
      aspect='auto',origin='lower',norm=LogNorm(),vmin=1e-1,vmax=6e-1)
#myplt = plt.imshow(fzeta.imag.transpose()*kxrho/rho_H,extent=[kxrho[0]/rho_H,kxrho[-1]/rho_H,
#      kprho[0]/rho_H,kprho[-1]/rho_H],cmap='seismic',
#      aspect='auto',origin='lower',vmin=-0.7,vmax=0.7)
plt.xscale('log')
plt.yscale('log')
font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 24}
matplotlib.rc('font', **font)
plt.xlabel(r'$k_\parallel H$')
plt.ylabel(r'$k_y H$')
plt.tick_params(pad=10)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)
#labels = np.array([5e-2,1e-1,2e-1,3e-1,4e-1])
#cbar.set_ticks(labels)
fig = plt.gcf()
#plt.gca().tight_layout()
plt.savefig('ky.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
#fig.set_size_inches([14., 14.])


