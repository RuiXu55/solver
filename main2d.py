import FD1
import FD3
import GKD1 
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, font_manager
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#########################################################################
def Iter(zeta,beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,err,cap):
  maxz    = 3 
  start = 2
  fz = np.zeros(maxz+1,dtype=complex) 
  # when k*rho_i>=1, start with large n
  for n in range(start,maxz+1):
    print 'n=',n
    data  = (beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,n)
    sol = root(FD3.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-4)
    fz[n] = complex(sol.x[0],sol.x[1])
    print 'fz=',fz[n]
    # check answer if it is accurate enough
    if n>start and (np.abs((fz[n].real-fz[n-1].real)/fz[n-1])<err\
       or np.abs(fz[n].real-fz[n-1].real)<cap)\
       and (np.abs((fz[n].imag-fz[n-1].imag)/fz[n-1].imag)<err\
       or np.abs(fz[n].imag-fz[n-1].imag)<cap): 
       return fz[n]
    if n==maxz:
       return fz[n]
       sys.warnning('Error Message:Needs higher N in Bessel function!')

##################################################################### 
################ Parameter Initialization ###########################
#ind=0 for homogeneous plasma,ind=1 for dlnt/dz<0,ind=2 for dlnt/dz>0
ind     = 1             # Index for plasma state 
num     = 300           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
err     = 1e-2          # relative tolerance for Newton iteration
cap     = 1e-7          # actual tolereance for Newton iteration
phi     = 0*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e8           # Ion plasma beta
d_H     = 1e-8          # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kxrho   = np.logspace(-4.0,-2.0,num)
kprho   = np.logspace(-3.0,0.0,num) # k_prp*rho_i
fzeta   = np.zeros((num,num),dtype=complex)  # store zeta
fgk     = np.zeros((num,num),dtype = complex)

######################################################################
################### Main Code Starts Here ############################
# GK solver
for j in range(0,num):
  for i in range(0,num):
    if i==0 and j==0:
      zeta = complex(-1.0/np.sqrt(beta),0.02)
    elif j==0:
      zeta = fgk[0,i-1]
    else:
      zeta = fgk[j-1,i]
    data  = (beta,kxrho[j],kprho[i],tau,z,mu,rho_H,ind,phi)
    sol = root(GKD1.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-5)
    fgk[j,i] = complex(sol.x[0],sol.x[1])
    print 'fgk=',fgk[j,i]

# Full solver
for j in range(0,num):
  for i in range(0,num):
    zeta = fgk[j,i]
    n = 10
    data  = (beta,kxrho[j],kprho[i],tau,z,mu,rho_H,ind,phi,n)
    sol = root(FD3.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-15)
    fzeta[j,i] = complex(sol.x[0],sol.x[1])
    print 'fz=',fzeta[j,i]*kxrho[j]/rho_H

######################## Plot the results #############################
# plot growth rate 
plt.figure(1)
#extents = (kxrho[0]/rho_H,kxrho[-1]/rho_H,kprho[0]/rho_H,kprho[-1]/rho_H)
extents = (kxrho[0],kxrho[-1],kprho[0],kprho[-1])
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
ax = plt.gca()
fz = np.ma.masked_where(fzeta.imag<0,fzeta.imag)
Z = fz.data.transpose()*kxrho/rho_H
myplt = plt.imshow(Z,extent= extents,cmap=cmap,
      aspect='auto',origin='lower',vmin=0.0,vmax=6e-1)
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
plt.xlabel(r'$k_\parallel \rho$')
plt.ylabel(r'$k_\perp \rho$')
plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=5, width=1, which='major')
plt.tick_params(length=4, width=1, which='minor')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)
#labels = np.array([5e-2,1e-1,2e-1,3e-1,4e-1])
#cbar.set_ticks(labels)
fig = plt.gcf()
#plt.gca().tight_layout()
plt.savefig('2dgk.eps',format='eps',dpi=300,bbox_inches='tight')
#fig.set_size_inches([14., 14.])

##############################
##############################
# plot growth rate 
plt.figure(2)
extents = (kxrho[0]/rho_H,kxrho[-1]/rho_H,kprho[0]/rho_H,kprho[-1]/rho_H)
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
ax = plt.gca()
fz = np.ma.masked_where(fgk.imag<0,fgk.imag)
Z = fz.data.transpose()*kxrho/rho_H
myplt = plt.imshow(Z,extent= extents,cmap=cmap,
      aspect='auto',origin='lower',vmin=0.0,vmax=6e-1)
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
plt.xlabel(r'$k_\parallel \rho$')
plt.ylabel(r'$k_\perp \rho$')
plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=5, width=1, which='major')
plt.tick_params(length=4, width=1, which='minor')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)
#labels = np.array([5e-2,1e-1,2e-1,3e-1,4e-1])
#cbar.set_ticks(labels)
fig = plt.gcf()
#plt.gca().tight_layout()
plt.savefig('2dfull.eps',format='eps',dpi=300,bbox_inches='tight')
#plt.show()
#fig.set_size_inches([14., 14.])


