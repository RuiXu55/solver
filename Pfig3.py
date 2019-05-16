import FD1
import sys
from time import sleep
import numpy as np
import matplotlib
from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
################ Parameter Initialization ###########################
ind     = 1             # Index for plasma state 
num     = 100           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e6           # Ion plasma beta
d_H     = 1e-10          # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kprho   = np.logspace(-1.0,0.4,num) # k_prp*rho_i
fzeta   = np.ones(num,dtype=complex)  # store zeta
plotx = kprho
######################################################################
################### Main Code Starts Here ############################
for i in range(0,num):
  print 'i=',i
  # Initial guess accorind to alfven waves
  if i==0:
    #zeta = complex(1.0/np.sqrt(beta),0.5)
    zeta = complex(1e-8,1e-3)
  else:
    zeta = fzeta[i-1]
  data  = (beta,kprho[i],tau,z,mu,rho_H,ind,phi,1)
  sol = root(FD1.SOLVE1,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-16)
  fzeta[i] = complex(sol.x[0],sol.x[1])
  print 'fz=',fzeta[i]

######################## Plot the results #############################
fz = fzeta*kprho/rho_H
colors = ['r','k']
lab = r'$k_\perp$,'
flag  = (fzeta.imag)>0
plt.loglog(plotx[flag],np.abs(fz.imag[flag])\
        ,linestyle='-',\
        label=lab+'Imaginary',color=colors[0],linewidth=2)
flag  = (fzeta.imag)<0
plt.loglog(plotx[flag],np.abs(fz.imag[flag])\
        ,linestyle='--',\
        color=colors[0],linewidth=2)

flag  = (fzeta.real)>0
plt.loglog(plotx[flag],np.abs(fz.real[flag])\
        ,linestyle='-',\
        label=lab+'Real',color=colors[1],linewidth=2)
flag  = (fzeta.real)<0
plt.loglog(plotx[flag],np.abs(fz.real[flag])\
       ,linestyle='--',\
      color=colors[1],linewidth=2)

sizeOfFont = 22
fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'light', 'size' : sizeOfFont}
rc('text', usetex=True)
rc('font',**fontProperties)
plt.legend(loc=0,prop={'size':18})
plt.xlabel(r'$k_\perp \rho_i$')
plt.ylabel(r'$\omega/(v_{th}/H)$')
plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

plt.axis([plotx[0],plotx[-1],1e-5,1e3])
fig = plt.gcf()
#plt.savefig('full.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()


