import DK
import time
import GKD
import NGKD
import sys
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

##################################################################### 
################ Parameter Initialization ###########################
#ind=0 for homogeneous plasma,ind=1 for dlnt/dz<0,ind=2 for dlnt/dz>0
ind     = 1             # Index for plasma state 
num     = 300           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion

kxH     = 1.0
rho_H   = 1e-4
kxrho   = kxH*rho_H
beta    = 1e1           # Ion plasma beta


kprho   = np.logspace(-3.0,2,num) # k_prp*rho_i
fzeta   = np.ones(num,dtype=complex)  # store zeta
fzeta1   = np.ones(num,dtype=complex)  # store zeta
fzeta2   = np.ones(num,dtype=complex)  # store zeta
plotx = kprho

######################################################################
################### Main Code Starts Here ############################
for i in range(0,num):
  print( 'i=',i)
  if i==0:
    zeta = complex(1e-3,0.4)
  else:
    zeta = fzeta[i-1]
  data  = (beta,kxrho,kprho[i],tau,z,mu,rho_H,ind,phi)
  sol = root(GKD.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-8)
  fzeta[i] = complex(sol.x[0],sol.x[1])
  print 'fz=',fzeta[i]*kxrho/rho_H*np.sqrt(3.0)
  omp = -kprho[i]/kxrho/(2.0)*rho_H
  omt = -kprho[i]/kxrho/(2.0)*rho_H*1.0/3.0

  fzeta2[i] = complex(1.,1.)*(np.sqrt(np.pi*mu)/2.0*omt**2*
      (omt/2.0-omp))**(1./2.)
  fzeta2[i] = fzeta2[i]*kxrho/rho_H*np.sqrt(3.0)
  print 'fz=',fzeta2[i]

######################## Plot the results #############################
# Plot zeta as a function of k_perp*rho_i or k_prl*rho_i
fzeta  = fzeta*kxrho/rho_H*np.sqrt(3.0)
flag  = (fzeta.imag)>0
plt.loglog(plotx[flag],np.abs(fzeta.imag[flag])\
        ,'k-',linewidth=2,label=r'$\rm Imaginary$')
flag  = (fzeta.imag)<0
plt.loglog(plotx[flag],np.abs(fzeta.imag[flag])\
        ,'k--',linewidth=2)
plt.loglog(plotx,np.abs(fzeta.real)\
        ,'r-',linewidth=2,label=r'$\rm Real$')
plt.loglog(plotx,fzeta2.real\
    ,'b:',linewidth=2)

#plt.text(3.0,4.0, r'$m_e/m_i=1/183600$', fontsize=20)

plt.legend(loc=2,prop={'size':20})
plt.xlabel(r'$k_\perp \rho_i$')
plt.ylabel(r'$\omega$')


plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

font = {'family' : 'sans-serif',
        'sans-serif' : 'Helvetica',
        'weight' : 'light',
        'size'   : 24}
matplotlib.rc('font', **font)

plt.axis([plotx[0],plotx[-1],1e-2,1e1])
fig = plt.gcf()
#fig.set_size_inches(8, 5)
plt.savefig('gk.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()


