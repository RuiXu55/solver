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
num     = 20           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
ind     = 1

beta    = 1e6
d_H     = 1e-6
rho_H   = d_H*np.sqrt(beta)
kprho   = np.logspace(-3.0,2.0,5*num) # k_prp*rho_i
fzeta   = np.ones((5*num,5*num),dtype=complex)  # store zeta
fz      = np.ones((5*num,5*num),dtype=complex)  # store zeta
tf      = np.ones(5*num,dtype=complex)

dlnt_dlnp=1.0/3.0
######################################################################
################### Main Code Starts Here ############################
rho_H1 = 1e-6
kxrho1  = 1e-5
for i in range(0,5*num):
  if i==0:
    zeta = complex(1e-4,0.6)
  else:
    zeta = tf[i-1]
    if tf[i-1].imag*kxrho1/rho_H1<0.0:
      zeta = complex(tf[i-1].real,-6.*abs(tf[i-1].imag))

  data  = (beta,kxrho1,-kprho[i],tau,z,mu,rho_H1,ind,phi)
  sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
  tf[i] = complex(sol.x[0],sol.x[1])
  print 'tf',tf[i]
#sys.exit()

kxrho  = np.logspace(-5,-7,2*num)
for j in range(0,2*num):
  print( 'j=',j)
  for i in range(0,5*num):
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
    if fzeta[2*num-j-1,i].imag>0.15 and fzeta[2*num-j,i].imag<0.15 and j>0:
      fzeta[2*num-j-1,i] = fzeta[2*num-j-1,i-1]
    print 'fz',fzeta[2*num-j-1,i]

kxrho  = np.logspace(-5,-2,3*num)
for j in range(0,3*num):
  print( 'j=',j)
  for i in range(0,5*num):
    if i==0 and j==0:
      zeta = complex(1e-4,0.4)
    elif j==0:
      zeta = fz[j,i-1]
      if kprho[i]>40:
        zeta = tf[i]
    else:
      zeta = fz[j-1,i]
      if kxrho[j]/rho_H>20.0 and kprho[i]>3.0 and fz[j-1,i].imag*kxrho[j]/rho_H<0.1:
        zeta = complex(fz[j-1,i].real,-3.0*abs(fz[j-1,i].imag))

    data  = (beta,kxrho[j],-kprho[i],tau,z,mu,rho_H,ind,phi)
    sol = root(NGKD.SOLVE4,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
    fz[j,i] = complex(sol.x[0],sol.x[1])
    fzeta[2*num+j,i] = fz[j,i]*kxrho[j]/rho_H
    if fzeta[2*num+j,i].imag>0.15 and fzeta[2*num+j-1,i].imag<0.15 and j>0:
      fzeta[2*num+j,i] = fzeta[2*num+j-1,i-1]

    if fzeta[2*num+j,i].imag>fzeta[2*num+j-1,i] and kprho[i]>40 and kxrho[j]>1e-4:
      fzeta[2*num+j,i] = fzeta[2*num+j-1,i]
    print 'fz',fzeta[num-j-1,i]

######################################################################
######################## Plot the results #############################
extents = (1e-7/rho_H,1e-2/rho_H,kprho[0],kprho[-1])
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
#cmap = cm.get_cmap('jet',20)

#plt.loglog(kxH,1,'w--',linewidth=1.5)
xposition = 5.0
plt.axvline(xposition,color='r',linestyle='--',linewidth=1.5)
plt.axis(extents)

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
plt.savefig('mticmp6.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
