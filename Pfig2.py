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
ind     = 1             # Index for plasma state 
num     = 500           # number of iteration
phi     = 0.0*np.pi/2.0
tau     = 1.0           # Temperature ratio ion/e
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e6           # Ion plasma beta
d_H     = 1e-10         # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kxrho  = np.logspace(-7.0,-4.0,num) # k_prp*rho_i
kprho  = np.logspace(-7.0,-2.0,num) # k_prp*rho_i
fzeta   = np.ones((num,num),dtype=complex)  # store zeta

######################################################################
################### Main Code Starts Here ############################
# loop over kyrho
for i in range(0,num):
  print 'i=',i
  # loop over kzrho
  for j in range(0,num):
    if i==0 and j==0:
      zeta = complex(1e-3,0.4)
    elif i==0:
      zeta = fzeta[i,j-1]#+complex(-1e-5,-1e-5)
    else:
      zeta = fzeta[i-1,j]
    data = (beta,kxrho[i],kprho[j],tau,z,mu,rho_H,ind,phi)
    sol = root(DK1.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-15)
    fzeta[i,j] = complex(sol.x[0],sol.x[1])
    print 'fz=',fzeta[i,j]*kxrho[i]/rho_H*np.sqrt(3.0)

######################## Plot the results #############################
extents = (kxrho[0]/rho_H,kxrho[-1]/rho_H,kprho[0]/rho_H,kprho[-1]/rho_H)
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors

#plt.loglog(kxrho/rho_H,1.0*np.sqrt(kxrho/rho_H),'w--',linewidth=1.5)
#plt.axis(extents)
fz = np.ma.masked_where(fzeta.imag<0,fzeta.imag)
Z = fz.data.transpose()*kxrho/rho_H*np.sqrt(3.0)


f = open('ky1.tab','w')
for i in range(0,num):
  for j in range(0,num):
    f.write('%f ' % Z[i,j]) 
  f.write('\n')

#Z = np.matrix(np.loadtxt('ky.tab'))

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
plt.ylabel(r'$k_y H$')
plt.tick_params(pad=10,direction ='in')
plt.tick_params(axis='y',direction='out',which='both')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)
#labels = np.array([5e-2,1e-1,2e-1,3e-1,4e-1])
#cbar.set_ticks(labels)


fig = plt.gcf()
#plt.gca().tight_layout()
plt.savefig('ky1.eps',format='eps',dpi=300,bbox_inches='tight')
#plt.show()
#fig.set_size_inches([14., 14.])

