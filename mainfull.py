import DK
import FD3
import numpy as np
import matplotlib
from scipy.optimize import root
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

#########################################################################
def Iter(zeta,beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,err,cap):
  maxz    = 20
  fz = np.zeros(maxz+1,dtype=complex) 
  # when k*rho_i>=1, start with large n
  if kprho>1.0:
    start =10
  else:
    start = 5
  c = zeta
  for n in range(start,maxz+1):
    print 'n=',n
    data  = (beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,n)
    bnds = ((0.9*c.real,1.1*c.real), (0.9*c.imag,1.1*c.imag))
    x0 = np.array([zeta.real,zeta.imag])
    #res = minimize(FD0.SOLVE,x0,args=data,bounds = bnds,options={'gtol': 1e-45, 'disp':True})
    #res = minimize(FD0.SOLVE,x0,args=data,bounds=bnds,options={'gtol': 1e-15, 'disp':True})
    #fz[n] = complex(res.x[0],res.x[1])

    sol = root(FD3.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-5)
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
num     = 100           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
err     = 1e-2          # relative tolerance for Newton iteration
cap     = 1e-7          # actual tolereance for Newton iteration
phi     = 0.*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e4           # Ion plasma beta
d_H     = 1e-4          # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kxrho1   = np.logspace(-2,0,num)
rho_H   = d_H*np.sqrt(beta)
kprho   = np.logspace(-2.0,1.5,num) # k_prp*rho_i
#kxH = 10.0
#kxrho = kxH*rho_H
fzeta   = np.ones(num,dtype=complex)  # store zeta
plotx = kprho

fz = -1e100
for i in range(0,num):
  if i==0:
    zeta = complex(1e-1,0.4)
  else:
    zeta = fzeta[i-1]#+complex(-1e-8,-1e-8)
  if kxrho1[i]/rho_H>20:
    zeta += complex(-1e-6,-1e-6)
  data = (beta,kxrho1[i],tau,z,mu,rho_H,ind)
  sol = root(DK.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
  fzeta[i] = complex(sol.x[0],sol.x[1])
  if fz<sol.x[1]*kxrho1[i]:
    kxrho = kxrho1[i]
    fz = sol.x[1]*kxrho1[i]
print("kxrho=",kxrho/rho_H)
######################################################################
################### Main Code Starts Here ############################
# solver=0 for KMHD, solver=1 for GKMHD, solver=2 for Full Vlasov
for j in range(1,2):
  for i in range(0,num):
    print 'i=',i
    # Initial guess accorind to alfven waves
    if i==0:
      zeta = complex(0.01,0.4)
    else:
      zeta = fzeta[i-1]
      if fzeta[i-1].imag<0:
        zeta = complex(fzeta[i-1].real,-6.0*abs(fzeta[i-1].imag))
    fzeta[i] = Iter(zeta,beta,kxrho,kprho[i],tau,z,mu,rho_H,ind,phi,err,cap)
    print 'fz=',fzeta[i]

######################## Plot the results #############################
# Plot zeta as a function of k_perp*rho_i or k_prl*rho_i
  colors = ['k','b']
  lab = 'Full,'
  flag  = (fzeta.imag)>0
  plt.loglog(plotx[flag],np.abs(fzeta.imag[flag])*kxrho/rho_H*np.sqrt(3.0)\
          ,linestyle='-',\
          label=lab+'Imaginary',color=colors[0],linewidth=2)
  flag  = (fzeta.imag)<0
  plt.loglog(plotx[flag],np.abs(fzeta.imag[flag])*kxrho/rho_H*np.sqrt(3.0)\
          ,linestyle='--',\
          color=colors[0],linewidth=2)
  
  flag  = (fzeta.real)>0
  plt.loglog(plotx[flag],np.abs(fzeta.real[flag])*kxrho/rho_H*np.sqrt(3.0)\
          ,linestyle='-',\
          label=lab+'Real',color=colors[1],linewidth=2)
  flag  = (fzeta.real)<0
  plt.loglog(plotx[flag],np.abs(fzeta.real[flag])*kxrho/rho_H*np.sqrt(3.0)\
         ,linestyle='--',\
        color=colors[1],linewidth=2)

sizeOfFont = 22
fontProperties = {'family':'sans-serif','sans-serif':['Helvetica'],
        'weight' : 'light', 'size' : sizeOfFont}
rc('text', usetex=True)
rc('font',**fontProperties)
plt.legend(loc=0,prop={'size':18})
plt.xlabel(r'$k_\perp \rho_i$')
plt.ylabel(r'$\omega$')

plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')

plt.axis([plotx[0],plotx[-1],1e-2,1e2])
plt.axis([1e-1,plotx[-1],1e-2,1e1])
fig = plt.gcf()

#plt.savefig('full.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()


