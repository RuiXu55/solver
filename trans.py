import NGKD
import sys
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pylab import *
import f
import scipy.special as sp

################ Parameter Initialization ###########################
num     = 100           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
ind     = 1

beta    = 1e4
rho_H   = 1e-5
kprho   = np.logspace(-1.0,2.0,num) # k_prp*rho_i
fzeta   = np.ones((4*num,num),dtype=complex)  # store zeta
fz      = np.ones((4*num,num),dtype=complex)  # store zeta
tf      = np.ones(2*num,dtype=complex)

dlnt_dlnp=1.0
######################################################################
################### Main Code Starts Here ############################
rho_H1 = 1e-6
kxrho1  = 1e-5
for i in range(0,num):
  if i==0:
    zeta = complex(1e-4,0.4)
  else:
    zeta = tf[i-1]
    if tf[i-1].imag*kxrho1/rho_H1<0.1:
      zeta = complex(tf[i-1].real,-200.*abs(tf[i-1].imag))

  data  = (beta,kxrho1,-kprho[i],tau,z,mu,rho_H1,ind,phi)
  sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
  tf[i] = complex(sol.x[0],sol.x[1])
  if abs(sol.x[1])>1e2:
    tf[i] = complex(sol.x[0],-100.0)

kxrho  = np.logspace(-5,-8,2*num)
for j in range(0,2*num):
  print( 'j=',j)
  for i in range(0,num):
    if i==0 and j==0:
      zeta = complex(1e-4,0.4)
    elif j==0:
      zeta = fz[j,i-1]
      if kprho[i]>40:
        zeta = tf[i]
    else:
      zeta = fz[j-1,i]

    data  = (beta,kxrho[j],-kprho[i],tau,z,mu,rho_H,ind,phi)
    sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-6)
    fz[j,i] = complex(sol.x[0],sol.x[1])

    fzeta[2*num-j-1,i] = fz[j,i]*kxrho[j]/rho_H
    if fzeta[2*num-j-1,i].imag>0.1 and fzeta[2*num-j-1,i-1].imag<0.1 and i>0:
      fzeta[2*num-j-1,i] = complex(1.0,-0.1)
      print 'fz',fzeta[2*num-j-1,i]

kxrho  = np.logspace(-5,-3,2*num)
for j in range(0,2*num):
  print( 'j=',j)
  for i in range(0,num):
    if i==0 and j==0:
      zeta = complex(1e-4,0.4)
    elif j==0:
      zeta = fz[j,i-1]
      if kprho[i]>20:
        zeta = tf[i]
    else:
      zeta = fz[j-1,i]
      #if kxrho[j]/rho_H>20.0 and kprho[i]>3.0 and fz[j-1,i].imag*kxrho[j]/rho_H<0.1:
      #  zeta = complex(fz[j-1,i].real,-3.0*abs(fz[j-1,i].imag))

    data  = (beta,kxrho[j],-kprho[i],tau,z,mu,rho_H,ind,phi)
    sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
    fz[j,i] = complex(sol.x[0],sol.x[1])
    #if abs(sol.x[1])>1e2:
    #  fz[j,i] = complex(sol.x[0],-100.0)
    #if kprho[i]>10. and kxrho[j]/rho_H>10.0 and fz[j,i].imag>fz[j-1,i].imag:
    #  if fz[j,i].imag>=fz[j-1,i].imag or fz[j,i].imag>=fz[j,i-1]:
    #    zeta = complex(fz[j,i].real,-4.0*fz[j,i].imag)
    #    data  = (beta,kxrho[j],-kprho[i],tau,z,mu,rho_H,ind,phi)
    #    sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
    #    fz[j,i] = complex(sol.x[0],sol.x[1])
    fzeta[2*num+j,i] = fz[j,i]*kxrho[j]/rho_H

######################################################################
######################## Plot the results #############################
extents = (1e-7/rho_H,1e-3/rho_H,kprho[0],kprho[-1])
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors

fz = np.ma.masked_where(abs(fzeta.imag<0),fzeta.imag)
Z = fz.data.transpose()*np.sqrt(3.0)
myplt = plt.imshow(Z,extent= extents,cmap=cmap,
      aspect='auto',origin='lower',vmin=0.0,vmax=1.5)
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
plt.savefig('mticmp.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
