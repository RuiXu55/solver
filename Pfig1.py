import DK
import DK1
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

ind     = 1             # Index for plasma state 
num     = 200
tau     = 0.5           # Temperature ratio ion/e
phi     = 0*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e6           # Ion plasma beta
d_H     = 1e-10          # ion skin_depth/scale height
rho_H   = d_H*np.sqrt(beta)
kxH     = np.logspace(0,3,num)
kxrho   = kxH*rho_H
#kxrho   = np.logspace(-6.0,-4.0,num) # k_prp*rho_i
plotx = kxrho
fzeta   = np.ones(num,dtype=complex)  # store zeta

######################################################################
################### Main Code Starts Here ############################
for j in range(0,1):
  for i in range(0,num):
    if i==0 and j==0:
      zeta = complex(1e-1,0.4)
    elif i==0:
      zeta = fzeta[i]
    else:
      zeta = fzeta[i-1]#+complex(-1e-8,-1e-8)
    if j==0:
      if kxrho[i]/rho_H>200:
        zeta += complex(-1e-8,-1e-8)
      data = (beta,kxrho[i],tau,z,mu,rho_H,ind)
      sol = root(DK.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
      fzeta[i] = complex(sol.x[0],sol.x[1])
      print fzeta[i]
    elif j==1:
      da = (beta,kxrho[i]/rho_H)
      sol = root(DK.SOL2,(zeta.real,zeta.imag),args=da,method='hybr',tol=1e-10)
      fzeta[i] = complex(sol.x[0],sol.x[1])
    elif j==2:
      da = (beta,kxrho[i]/rho_H)
      sol = root(DK.SOL3,(zeta.real,zeta.imag),args=da,method='hybr',tol=1e-10)
      fzeta[i] = complex(sol.x[0],sol.x[1])
      #print 'fz3=',fzeta[i]
    else:
      da = (beta,kxrho[i]/rho_H)
      sol = root(DK.SOL4,(zeta.real,zeta.imag),args=da,method='hybr',tol=1e-10)
      fzeta[i] = complex(sol.x[0],sol.x[1])
######################## Plot the results #############################
  if j==0:
    labels = r'$\rm Kinetic$'
    style = '-'
  elif j==1:
    labels = r'$H/\lambda_{mfp}=10$'
    style = '--'
  elif j==2:
    labels = r'$H/\lambda_{mfp}=100$'
    style = ':'
  elif j==3:
    labels = r'$H/\lambda_{mfp}=1$'
    style = '-.'

  flag  = fzeta.imag>0 
  plt.semilogx(plotx[flag]/rho_H,fzeta.imag[flag]*kxrho[flag]/rho_H\
        ,linestyle=style,color='k',linewidth=2.,label=labels)
  # plt.loglog(plotx[flag]/rho_H,fzeta.imag[flag]*kxrho[flag]/rho_H*np.sqrt(3.0)\
  #      ,linestyle=style,color='k',linewidth=2.,label=labels)

#plt.loglog(plotx[flag]/rho_H,plotx[flag]/rho_H/2.0,'b')
#plt.loglog(plotx[flag]/rho_H,(plotx[flag]/rho_H)**2/2.0,'y')
plt.axis([plotx[0]/rho_H,plotx[-1]/rho_H,0.0,1.2])

font = {'family':['sans-serif'],'sans-serif':['Helvetica'],'weight' : 'light', 'size' : 22}
rc('font', **font)

plt.xlabel(r'$k_\parallel H$')
plt.ylabel(r'$\rm Im(\omega)$')
plt.legend(loc=1,frameon=False,prop={'size':18})
plt.tick_params(right="off",labeltop="off",pad=10)
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=4, width=1, which='minor')

#fig = plt.gcf()
plt.savefig('fig1_05.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()


