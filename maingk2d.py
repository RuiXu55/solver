import DK
import GKD
import sys
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, font_manager
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

##################################################################### 
################ Parameter Initialization ###########################
#ind=0 for homogeneous plasma,ind=1 for dlnt/dz<0,ind=2 for dlnt/dz>0
ind     = 1             # Index for plasma state 
num     = 100           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0.0
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e8           # Ion plasma beta
d_H     = 1e-12
rho_H   = d_H*np.sqrt(beta)
kxrho1   = np.logspace(-10,-8,num)
kyrho   = np.logspace(-2.0,2.0,num) # k_prp*rho_i
kzrho   = np.logspace(-4.0,2.0,num) # k_prp*rho_i
fzeta   = np.ones(num,dtype=complex)  # store zeta
fz   = np.ones((num,num),dtype=complex)  # store zeta


plotx = kyrho
fz0 = -1e100
for i in range(0,num):
  if i==0:
    zeta = complex(1e-1,0.4)
  else:
    zeta = fzeta[i-1]#+complex(-1e-8,-1e-8)
  if kxrho1[i]/rho_H>20:
    zeta += complex(-1e-8,-1e-8)
  data = (beta,kxrho1[i],tau,z,mu,rho_H,ind)
  sol = root(DK.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
  fzeta[i] = complex(sol.x[0],sol.x[1])
  if fz0<sol.x[1]*kxrho1[i]:
    kxrho = kxrho1[i]
    fz0 = sol.x[1]*kxrho1[i]
print("kxrho=",kxrho/rho_H)
 

######################################################################
################### Main Code Starts Here ############################
for i in range(0,num):
  for j in range(0,num):
    print( 'i=',i)
    if i==0 and j==0:
      zeta = complex(1e-1,0.4)
    elif j==0:
      zeta = fz[i-1,j]
    else:
      zeta = fz[i,j-1]
      if fz[i,j-1].imag<0:
        zeta = complex(fz[i,j-1].real,4.0*fz[i,j-1].imag)
    kprho = np.sqrt(kyrho[i]**2+kzrho[j]**2)
    phi = np.arcsin(kzrho[j]/kprho)
    data  = (beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi)
    sol = root(GKD.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-8)
    fz[i,j] = complex(sol.x[0],sol.x[1])

######################## Plot the results #############################
Z = fz.transpose()*kxrho/rho_H*np.sqrt(3.0)
f = open('gk2d.tab','w')
for i in range(0,num):
  for j in range(0,num):
    f.write('%f ' % Z[i,j].imag) 
  f.write('\n')
sys.exit()

#Z = np.matrix(np.loadtxt('gk2d.tab'))

extents = (kyrho[0]/rho_H,kyrho[-1]/rho_H,kzrho[0]/rho_H,kzrho[-1]/rho_H)
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
myplt = plt.imshow(Z,extent= extents,cmap=cmap,
      aspect='auto',origin='lower',vmin=0.0,vmax=1.0)
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
plt.xlabel(r'$k_y H$')
plt.ylabel(r'$k_z H$')
plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)
#labels = np.array([5e-2,1e-1,2e-1,3e-1,4e-1])


fig = plt.gcf()
#fig.set_size_inches([14., 14.])
plt.savefig('gk2d.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
