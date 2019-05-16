import DK
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
num     = 1000            # number of iteration
tau     = 1.0           # Temperature ratio ion/e
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e6           # Ion plasma beta
d_H     = 1e-10          # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kyrho   = np.logspace(-7.0,-5.0,num) # k_prp*rho_i
kzrho   = np.logspace(-7.0,-5.0,num) # k_prp*rho_i

kxrho1  = np.logspace(-7,-4,num)
fzeta   = np.ones((num,num),dtype=complex)  # store zeta
fzeta1   = np.ones(num,dtype=complex)  # store zeta
fzeta2   = np.ones((num,num),dtype=complex)  # store zeta

######################################################################
################### Main Code Starts Here ############################

fz = -1e100
phi = 0.0   
for i in range(0,num):
  if i==0:
    zeta = complex(1e-3,0.4)
  else:
    zeta = fzeta1[i-1]#+complex(-1e-8,-1e-8)
  if kxrho1[i]/rho_H>20:
    zeta += complex(-1e-6,-1e-6)
  data = (beta,kxrho1[i],tau,z,mu,rho_H,ind)
  sol = root(DK.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
  fzeta1[i] = complex(sol.x[0],sol.x[1])
  if fz<sol.x[1]*kxrho1[i]:
    kxrho = kxrho1[i]
    fz = sol.x[1]*kxrho1[i]
print("kxrho=",kxrho/rho_H)

sys.exit()
# loop over kyrho
for i in range(0,num):
  print 'i=',i
  # loop over kzrho
  for j in range(0,num):
    if i==0 and j==0:
      zeta = complex(1e-5,0.4)
    elif i==0:
      zeta = fzeta[i,j-1]
    else:
      zeta = fzeta[i-1,j]
    kprho = np.sqrt(kyrho[i]**2+kzrho[j]**2)
    phi     = np.arcsin(kzrho[j]/kprho)
    data = (beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi)
    sol = root(DK1.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
    fzeta[i,j] = complex(sol.x[0],sol.x[1])
    print 'fz=',fzeta[i,j]
    #if kprho*np.cos(phi)/rho_H>50.0 and kprho*np.sin(phi)/rho_H>50.0:
    #zeta = fzeta[i,j]
    sol = root(DK1.SOLVE1,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
    fzeta2[i,j] = complex(sol.x[0],sol.x[1])
    print 'FZ=',fzeta2[i,j]
    #sys.exit()
######################## Plot the results #############################
extents = (kyrho[0]/rho_H,kyrho[-1]/rho_H,kzrho[0]/rho_H,kzrho[-1]/rho_H)
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors

fz = np.ma.masked_where(fzeta.imag<0,fzeta.imag)
Z = fz.data.transpose()*kxrho/rho_H*np.sqrt(3.0)

fz1 = np.ma.masked_where(fzeta2.imag<0,fzeta.imag)
Z1 = fz.data.transpose()*kxrho/rho_H*np.sqrt(3.0)

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
plt.tick_params(axis='y',direction='out',which='both')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

ax = plt.gca()
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)

plt.figure(2)
fz1 = np.ma.masked_where(fzeta2.imag<0,fzeta.imag)
Z1 = fz1.data.transpose()*kxrho/rho_H*np.sqrt(3.0)

myplt = plt.imshow(Z1,extent= extents,cmap=cmap,
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
#plt.tick_params(axis='y', colors='w', width=5, which='minor')
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
#plt.savefig('general.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
#fig.set_size_inches([14., 14.])
