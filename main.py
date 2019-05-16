import FD1
import GKD 
import numpy as np
#import Newton as F  
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt


#########################################################################
def Iter(zeta,beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,err,cap):
  maxiter = 1000          # max iteration step
  maxz    = 50
  fz = np.zeros(maxz+1,dtype=complex) 

  # when k*rho_i>=1, start with large n
  if kprho>0.1:
    start= 5
  else:
    start = 5

  for n in range(start,maxz+1):
    print 'n=',n
    data  = (beta,kxrho,kprho,tau,z,mu,rho_H,ind,phi,n)
    sol = root(FD2.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-6)
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
solver  = 'GK'
#ind=0 for homogeneous plasma,ind=1 for dlnt/dz<0,ind=2 for dlnt/dz>0
ind     = 1             # Index for plasma state 
num     = 200           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
err     = 1e-1          # relative tolerance for Newton iteration
cap     = 1e-5          # actual tolereance for Newton iteration
phi     = 0*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e2           # Ion plasma beta
d_H     = 1e-8          # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kprho   = np.logspace(-2.0,2.0,num) # k_prp*rho_i
fzeta   = np.ones(num,dtype=complex)  # store zeta

if solver=='DK':
  kxrho   = np.logspace(-8,2,num)      # k_prl*rho_i 
else:
  kxrho   = 10*rho_H      # k_prl*rho_i 
######################################################################
################### Main Code Starts Here ############################
# solver=0 for KMHD, solver=1 for GKMHD, solver=2 for Full Vlasov
for solver in range(1,3):
  for i in range(0,num):
    print 'Iter=',i 
    # Initial guess accorind to alfven waves
    if i==0:
      zeta = complex(1.0/np.sqrt(beta),0.02)
    else:
      zeta = fzeta[i-1]
    if solver == 1:
      data  = (beta,kxrho,kprho[i],tau,z,mu,rho_H,ind,phi)
      sol = root(GKD.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-5)
      fzeta[i] = complex(sol.x[0],sol.x[1])
      print 'fz=',fzeta[i]
    elif solver== 2:
      fzeta[i] = Iter(zeta,beta,kxrho,kprho[i],tau,z,mu,rho_H,ind,phi,err,cap)
      print 'fz=',fzeta[i]


  ######################## Plot the results #############################
  # Plot zeta as a function of k_perp*rho_i or k_prl*rho_i
  # Imaginary part( solid line for positive and dashed line for negative)
  if solver == 'GK' or solver=='FV':
    plotx = kprho
    colors = ['k','r']
    flag  = (fzeta.imag)>0
    fig1, = plt.loglog(plotx[flag],np.abs(fzeta.imag[flag])*\
            np.sqrt(beta),linestyle='-',\
            label=solver+',Imag',color=colors[0],linewidth=2)
    flag  = (fzeta.imag)<0
    fig1, = plt.loglog(plotx[flag],np.abs(fzeta.imag[flag])*\
            np.sqrt(beta),linestyle='--',\
            color=colors[0],linewidth=2)
    
    flag  = (fzeta.real)>0
    fig1, = plt.loglog(plotx[flag],np.abs(fzeta.real[flag])*\
            np.sqrt(beta),linestyle='-',\
            label=solver+',Real',color=colors[1],linewidth=2)
    flag  = (fzeta.real)<0
    fig1, = plt.loglog(plotx[flag],np.abs(fzeta.real[flag])*\
           np.sqrt(beta),linestyle='--',\
          color=colors[1],linewidth=2)
    plt.xlabel(r'$k_\perp \rho_i$',fontsize=20)
    plt.ylabel(r'$\omega$',fontsize=20)
    #plt.title(r'$kH=10,\beta=10^4$',fontsize=28,y=1.02)
    plt.axis([kprho[0],kprho[-1],1e-2,1e5])
    plt.legend(loc=0)



font = {'family' : 'sans-serif',
        'weight' : 'light',
        'size'   : 20}
matplotlib.rc('font', **font)
plt.show()

#fig = plt.gcf()
#fig.set_size_inches([22., 14.])
#plt.savefig('2000.eps',format='eps',dpi=240)
#plt.savefig('1e6.pdf',format='pdf',dpi=240)


