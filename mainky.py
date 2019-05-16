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
num     = 500            # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e4           # Ion plasma beta
d_H     = 1e-8         # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kprho  = np.logspace(-4.0,-2.0,num) # k_prp*rho_i
kxrho  = np.logspace(-6.0,-4.0,num) # k_prp*rho_i
#kprho   = np.linspace(kH*rho_H,1,num)
#kxrho   = np.linspace(kH*rho_H,1,num)
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
      zeta = complex(1e-2,0.4)
    elif i==0:
      zeta = fzeta[i,j-1]#+complex(-1e-5,-1e-5)
    else:
      zeta = fzeta[i-1,j]
    data = (beta,kxrho[i],kprho[j],tau,z,mu,rho_H,ind,phi)
    sol = root(DK1.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-8)
    fzeta[i,j] = complex(sol.x[0],sol.x[1])
    print 'fz=',fzeta[i,j]*kxrho[i]/rho_H*np.sqrt(3.0)
######################## Plot the results #############################
# plot growth rate 
extents = (kxrho[0]/rho_H,kxrho[-1]/rho_H,kprho[0]/rho_H,kprho[-1]/rho_H)
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
#plt.loglog(kxrho/rho_H,1.0*np.sqrt(kxrho/rho_H),'w--',linewidth=1.5)
#plt.axis(extents)
ax = plt.gca()
fz = np.ma.masked_where(fzeta.imag<0,fzeta.imag)
Z = fz.data.transpose()*kxrho/rho_H*np.sqrt(3.0)

f = open('kz.tab','w')
for i in range(0,num):
  for j in range(0,num):
    f.write('%f ' % Z[i,j]) 
  f.write('\n')

#Z = np.matrix(np.loadtxt('kz.tab'))
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
plt.xlabel(r'$k_\parallel H$')
plt.ylabel(r'$k_z H$')
plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(axis='y',direction='out',which='both')
plt.tick_params(length=5, width=1, which='minor')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax,ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
#labels = np.array([5e-2,1e-1,2e-1,3e-1,4e-1])
#cbar.set_ticks(labels)
fig = plt.gcf()
#plt.gca().tight_layout()
plt.savefig('kz.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
#fig.set_size_inches([14., 14.])


