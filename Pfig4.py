import DK
import numpy as np
import matplotlib
from scipy.optimize import root
import matplotlib.pyplot as plt
from matplotlib import rc, font_manager

ind     = 1             # Index for plasma state 
num     = 500           # number of iteration
tau     = 1.0           # Temperature ratio ion/e
phi     = 0*np.pi/2.0   # k_y = k_perp*cos(phi)
z       = 1.0           # Ion charge number
mu      = 1.0/1836.0    # Mass ratio e/ion
beta    = 1e4           # Ion plasma beta
kxH     = np.logspace(-1,2,num)
plotx = kxH

######################################################################
################### Main Code Starts Here ############################
for j in range(0,2):
  if j==0:
    d_H     = 1e-12          # ion skin_depth/scale height
  elif j==1:
    d_H    = 1e-8

  rho_H   = d_H*np.sqrt(beta)
  kxrho   = kxH*rho_H
  fzeta   = np.ones(num,dtype=complex)  # store zeta
  for i in range(0,num):
    print 'Iter=',i 
    if i==0:
      zeta = complex(1e-2,0.4)
    else:
      zeta = fzeta[i-1]
    if i>1 and fzeta[i-1].imag*kxrho[i-1]/rho_H<0.01:
      if j==1:
        zeta = fzeta[i-1]+complex(1e-6,1e-6)
      else:
        zeta = fzeta[i-1]+complex(1e-3,1e-3)

    data = (beta,kxrho[i],tau,z,mu,rho_H,ind)
    sol = root(DK.SOLVE,(zeta.real,zeta.imag),args=data,method='hybr',tol=1e-10)
    fzeta[i] = complex(sol.x[0],sol.x[1])
    print 'fz=',fzeta[i]

######################## Plot the results #############################
  size  = 20 
  if j==0:
    colors ='r'
    labels = r'$d_i/H=10^{-12}$'
  elif j==1:
    colors = 'k'
    labels = r'$d_i/H=10^{-8}$'

  flag1  = fzeta.imag>0 
  flag2  = abs(fzeta.imag)>abs(fzeta.real)
  flag = np.array(flag1) & np.array(flag2)
  plt.semilogx(plotx[flag],np.abs(fzeta.imag[flag])*kxrho[flag]/rho_H*np.sqrt(3.0)\
        ,linestyle='-',color=colors,linewidth=2.,label=labels)
  flag  = fzeta.imag<0 
  plt.semilogx(plotx[flag],np.abs(fzeta.real[flag])*kxrho[flag]/rho_H*np.sqrt(3.0)\
        ,linestyle='--',color=colors,linewidth=2.)
  #plt.axvline(1./rho_H,color = colors,linestyle='-.',linewidth=1.5)


plt.axis([plotx[0],plotx[-1],0.2,1.2])
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
plt.ylabel(r'$\omega$')
plt.legend(loc=2,prop={'size':18})
plt.tick_params(pad=10,direction ='in')
plt.tick_params(length=8, width=1, which='major')
plt.tick_params(length=5, width=1, which='minor')
plt.tick_params(pad=10)

fig = plt.gcf()
plt.savefig('kprl1.eps',format='eps',dpi=300,bbox_inches='tight')
plt.show()


