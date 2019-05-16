import FD3
import GKD1 
import numpy as np
import matplotlib
from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc, font_manager
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

#########################################################################
def Iter(zeta,beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,err,cap):
  maxz    = 10
  fz = np.zeros(maxz+1,dtype=complex) 
  # when k*rho_i>=1, start with large n
  if kprho>0.1:
    start = 10
  else:
    start = 10
  c = zeta
  for n in range(start,maxz+1):
    print 'n=',n
    data  = (beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,n)
    bnds = ((0.9*c.real,1.1*c.real), (0.9*c.imag,1.1*c.imag))
    x0 = np.array([zeta.real,zeta.imag])
    #res = minimize(FD0.SOLVE,x0,args=data,bounds = bnds,options={'gtol': 1e-45, 'disp':True})
    #res = minimize(FD0.SOLVE,x0,args=data,bounds=bnds,options={'gtol': 1e-15, 'disp':True})
    #fz[n] = complex(res.x[0],res.x[1])

    sol = root(FD3.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-15)
    fz[n] = complex(sol.x[0],sol.x[1])
    return fz[n]
    # check answer if it is accurate enough
    if n>start and (np.abs((fz[n].real-fz[n-1].real)/fz[n-1])<err\
       or np.abs(fz[n].real-fz[n-1].real)<cap)\
       and (np.abs((fz[n].imag-fz[n-1].imag)/fz[n-1].imag)<err\
       or np.abs(fz[n].imag-fz[n-1].imag)<cap): 
       return fz[n]
    if n==maxz:
       return fz[n]
       #sys.warnning('Error Message:Needs higher N in Bessel function!')

##################################################################### 
################ Parameter Initialization ###########################
#ind=0 for homogeneous plasma,ind=1 for dlnt/dz<0,ind=2 for dlnt/dz>0
ind     = 1             # Index for plasma state 
num     = 30           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
err     = 1e-2          # relative tolerance for Newton iteration
cap     = 1e-7          # actual tolereance for Newton iteration
phi     = 0.*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e4           # Ion plasma beta

kxrho   = np.logspace(-6,-5,num)
kprho   = np.logspace(-3.0,-1.0,num) # k_prp*rho_i
fzeta   = np.ones((num,num),dtype=complex)  # store zeta
rho_H   = kxrho/10.0
d_H     = rho_H/np.sqrt(beta)          # ion skin_depth/scale height
plotx = kxrho

######################################################################
################### Main Code Starts Here ############################
# solver=0 for KMHD, solver=1 for GKMHD, solver=2 for Full Vlasov
for j in range(0,num):
  print 'j=',j
  for i in range(0,num):
    print 'i=',i
    # Initial guess accorind to alfven waves
    if i==0:
      zeta = complex(-1e-5,0.1)
    elif j==0:
      zeta = fzeta[0,i-1]
    else:
      zeta = fzeta[j-1,i]
    #fzeta[i] = Iter(zeta,beta,kxrho,kprho[i],tau,z,mu,rho_H,ind,phi,err,cap)
    fzeta[j,i] = Iter(zeta,beta,kxrho[i],kprho[j],tau,z,mu,rho_H[i],ind,phi,err,cap)
    if fzeta[j,i].imag>1e3:
      sys.exit("WRONG!")
    print 'fz=',fzeta[j,i]

######################## Plot the results #############################

extents = (kxrho[0],kxrho[-1],kprho[0],kprho[-1])
cmap = cm.get_cmap('Greys_r',20)    # 11 discrete colors
cmap = cm.get_cmap('jet')
ax = plt.gca()
fz = np.ma.masked_where(fzeta.imag<0,fzeta.imag)
Z = fz.data.transpose()*kxrho/rho_H
myplt = plt.imshow(Z,extent= extents,cmap=cmap,
      aspect='auto',origin='lower',vmin=0.0,vmax=6e-1)
#myplt = plt.imshow(Z,extent= extents,cmap=cmap,
#      aspect='auto',origin='lower',norm=LogNorm())
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
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="8%", pad=0.2)
cbar = plt.colorbar(myplt, cax=cax)
#labels = np.array([5e-2,1e-1,2e-1,3e-1,4e-1])
#cbar.set_ticks(labels)
fig = plt.gcf()
#plt.gca().tight_layout()
plt.savefig('full2d1.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()
#fig.set_size_inches([14., 14.])


