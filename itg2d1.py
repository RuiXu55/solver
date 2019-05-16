import NGKD
import sys
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
from pylab import *
#from matplotlib.colors import SymLogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import f
import scipy.special as sp

################ Parameter Initialization ###########################
ind     = 1
num     = 800          # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number

beta    = 1e-10
rho_H   = 1e-2
kxrho1   = np.logspace(-4.0,-5.0,num)
kxrho   = np.logspace(-4.0,-2.0,2*num)
kprho   = np.logspace(-1.0,3.0,2*num) # k_prp*rho_i
kprho1   = np.logspace(-2.0,-1.0,50) # k_prp*rho_i
fzeta   = np.ones((3*num,2*num),dtype=complex)  # store zeta
fzeta1   = np.ones((3*num,2*num),dtype=complex)  # store zeta

dlnt_dlnp=3.0/3.0
mu = 1.0/1836.0
######################################################################
################### Main Code Starts Here ############################
for j in range(0,num):
   print 'j=',j
   for i in range(0,2*num):
    if i==0 and j==0:
      zeta = complex(0.1,1.9)
      for k in range(0,50):
        data  = (kxrho1[0],kprho1[k],tau,z,mu,rho_H,phi,dlnt_dlnp)
        sol   = root(NGKD.SOLVE1,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-8)
        zeta = complex(sol.x[0],sol.x[1])
    elif j==0:
      zeta = fzeta[j,i-1]
    else:
      zeta = fzeta[j-1,i]

    data  = (kxrho1[j],kprho[i],tau,z,mu,rho_H,phi,dlnt_dlnp)
    sol   = root(NGKD.SOLVE3,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-6)
    fzeta[j,i] = complex(sol.x[0],sol.x[1])
    zeta = complex(sol.x[0],sol.x[1])
    fzeta1[num-1-j,i] = fzeta[j,i]*kxrho1[j]/rho_H
    if kprho[i]<1.0 and kxrho1[j]<2e-5 and fzeta[j,i]>fzeta[j-1,i]: 
      fzeta1[num-1-j,i] = complex(0,-1e-10)
      print '1',fzeta1[num-1-j,i]
    if fzeta[num-1-j,i].imag<0:
      fzeta1[num-1-j,i] = complex(0,1e-10)

for j in range(0,num):
   for i in range(0,2*num):
     if i<num-1:
       zeta = fzeta[j,num-i]
     else:
       zeta = fzeta[j,num-1-i]

     data       = (kxrho1[j],kprho[num-1-i],tau,z,mu,rho_H,phi,dlnt_dlnp)
     sol        = root(NGKD.SOLVE3,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-6)
     fzeta[j,num-1-i] = complex(sol.x[0],sol.x[1])
     fzeta1[num-1-j,num-1-i] = fzeta[j,num-1-i]*kxrho1[j]/rho_H



for j in range(0,2*num):
   print 'j=',j
   for i in range(0,2*num):
    if i==0 and j==0:
      zeta = complex(0.1,1.9)
      for k in range(0,50):
        data  = (kxrho[0],kprho1[k],tau,z,mu,rho_H,phi,dlnt_dlnp)
        sol   = root(NGKD.SOLVE1,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-8)
        zeta = complex(sol.x[0],sol.x[1])
    elif i==0:
      zeta = fzeta[j-1,i]
    else:
      zeta = fzeta[j,i-1]
    if kprho[i]>1.0:
      zeta = fzeta[j,i-1]
    if kprho[i]>4.0 and kxrho[j]>1.5e-4:
      zeta = fzeta[j-1,i]
      if kxrho[j]>5e-4 and kprho[i]<20.0:
        zeta = fzeta[j-1,i-1]

    data  = (kxrho[j],kprho[i],tau,z,mu,rho_H,phi,dlnt_dlnp)
    sol   = root(NGKD.SOLVE3,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
    fzeta[j,i] = complex(sol.x[0],sol.x[1])

    fzeta1[num+j,i] = fzeta[j,i]*kxrho[j]/rho_H
    if fzeta[num+j,i].imag<0:
      fzeta1[num+j,i] = complex(0,1e-10)

######################## Plot the results #############################
extents = (1e-5/rho_H,1e-2/rho_H,kprho[0],kprho[-1])
#cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
cmap = cm.get_cmap('jet')
dlnp_dlnt=1.0/dlnt_dlnp
kxH = np.linspace(0,0,2*num)
for i in range(0,2*num):
  al = kprho[i]**2/2.0
  kxH[i] = kprho[i]/dlnp_dlnt*\
     np.sqrt((1.5-dlnp_dlnt-al*f.dive(0,al,1)/sp.ive(0,al))\
     /((2.0*(1.0+tau)/sp.ive(0,al)-1.0)**2-1.0))

######################## Plot the results #############################
plt.loglog(kxH,kprho,'w--',linewidth=1.5)

for i in range(0,2*num):
  al = (kprho[i]*np.sqrt(mu*z))**2/2.0
  kxH[i] = kprho[i]*np.sqrt(mu*z)/dlnp_dlnt*\
     np.sqrt((1.5-dlnp_dlnt-al*f.dive(0,al,1)/sp.ive(0,al))\
     /((2.0*(1.0+tau)/sp.ive(0,al)-1.0)**2-1.0))

######################## Plot the results #############################
plt.loglog(kxH,kprho,'w-.',linewidth=1.5)
plt.axis(extents)

fz = np.ma.masked_where(abs(fzeta1.imag<0),fzeta1.imag)
Z = fz.data.transpose()#*kxrho/rho_H*np.sqrt(3.0)
myplt = plt.imshow(Z,extent= extents,cmap=cmap,
      aspect='auto',origin='lower',vmin=0.0,vmax=1.5)
#myplt = plt.imshow(Z,extent= extents,cmap=cmap,
#     aspect='auto',origin='lower',vmin=1e-2,vmax=5.0,
#     norm=matplotlib.colors.LogNorm(vmin=1e-2, vmax=5.0))
plt.xscale('log')
plt.yscale('log')

sizeOfFont = 12
fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'normal', 'size' : sizeOfFont}
ticks_font = font_manager.FontProperties(family='Helvetica', style='normal',
        size=sizeOfFont, weight='normal', stretch='normal')
rc('text', usetex=True)
rc('font',**fontProperties)
font = {'size'  : 24}
rc('font', **font)
plt.xlabel(r'$k_\parallel H$')
plt.ylabel(r'$k_y\rho_i$')
plt.tick_params(pad=10,direction ='in')
plt.tick_params(axis='y',direction='out',which='both')
plt.tick_params(axis='x',direction='out',which='both')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)

fig = plt.gcf()
#plt.gca().tight_layout()
plt.savefig('itg.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
